import torch
import torch.nn as nn
import math
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp.autocast_mode import autocast
from parameters import *

from functools import partial
from einops import repeat, rearrange, einsum
from mamba_ssm import Mamba
from mamba_ssm.modules.mamba_simple import  Block
from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn

def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=True,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, d_state=4, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    # mixer_cls = partial(Mamba,layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls=mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block



class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )



    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, hidden_states,src_key_padding_mask = None,pos= None , inference_params=None):
        hidden_states = rearrange(hidden_states, 't b c -> b t c')
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        hidden_states = rearrange(hidden_states, 'b t c -> t b c')
        return hidden_states

class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SingleHeadAttention, self).__init__()
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = embedding_dim
        self.key_dim = self.value_dim
        self.tanh_clipping = 10
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q

        batch_size, target_size, input_dim = h.size()
        n_query = q.size(1)  # n_query = target_size in tsp

        # assert q.size(0) == batch_size
        # assert q.size(2) == input_dim
        # assert input_dim == self.input_dim

        h_flat = h.reshape(-1, input_dim)  # (batch_size*graph_size)*input_dim
        q_flat = q.reshape(-1, input_dim)  # (batch_size*n_query)*input_dim

        shape_k = (batch_size, target_size, -1)
        shape_q = (batch_size, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(
            shape_q
        )  # batch_size*n_query*key_dim
        K = torch.matmul(h_flat, self.w_key).view(
            shape_k
        )  # batch_size*targets_size*key_dim

        U = self.norm_factor * torch.matmul(
            Q, K.transpose(1, 2)
        )  # batch_size*n_query*targets_size
        U = self.tanh_clipping * torch.tanh(U)

        if mask is not None:
            mask = mask.view(batch_size, 1, target_size).expand_as(
                U
            )  # copy for n_heads times
            # U = U-1e8*mask  # ??
            # U[mask.bool()] = -1e8
            U[mask.bool()] = -1e4
        attention = torch.log_softmax(U, dim=-1)  # batch_size*n_query*targets_size

        out = attention

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = self.embedding_dim // self.n_heads
        self.key_dim = self.value_dim
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(
            torch.Tensor(self.n_heads, self.input_dim, self.key_dim)
        )
        self.w_key = nn.Parameter(
            torch.Tensor(self.n_heads, self.input_dim, self.key_dim)
        )
        self.w_value = nn.Parameter(
            torch.Tensor(self.n_heads, self.input_dim, self.value_dim)
        )
        self.w_out = nn.Parameter(
            torch.Tensor(self.n_heads, self.value_dim, self.embedding_dim)
        )

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q

        batch_size, target_size, input_dim = h.size()
        n_query = q.size(1)  # n_query = target_size in tsp
        # assert q.size(0) == batch_size
        # assert q.size(2) == input_dim
        # assert input_dim == self.input_dim

        h_flat = h.contiguous().view(-1, input_dim)  # (batch_size*graph_size)*input_dim
        q_flat = q.contiguous().view(-1, input_dim)  # (batch_size*n_query)*input_dim
        shape_v = (self.n_heads, batch_size, target_size, -1)
        shape_k = (self.n_heads, batch_size, target_size, -1)
        shape_q = (self.n_heads, batch_size, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(
            shape_q
        )  # n_heads*batch_size*n_query*key_dim
        K = torch.matmul(h_flat, self.w_key).view(
            shape_k
        )  # n_heads*batch_size*targets_size*key_dim
        V = torch.matmul(h_flat, self.w_value).view(
            shape_v
        )  # n_heads*batch_size*targets_size*value_dim

        U = self.norm_factor * torch.matmul(
            Q, K.transpose(2, 3)
        )  # n_heads*batch_size*n_query*targets_size

        if mask is not None:
            mask = mask.view(1, batch_size, -1, target_size).expand_as(
                U
            )  # copy for n_heads times
            # U = U.masked_fill(mask == 1, -np.inf)
            U[mask.bool()] = -np.inf
        attention = torch.softmax(U, dim=-1)  # n_heads*batch_size*n_query*targets_size

        if mask is not None:
            attnc = attention.clone()
            attnc[mask.bool()] = 0
            # attnc = attnc.masked_fill(mask == 1, 0)
            attention = attnc

        heads = torch.matmul(attention, V)  # n_heads*batch_size*n_query*value_dim

        out = torch.mm(
            heads.permute(1, 2, 0, 3).reshape(-1, self.n_heads * self.value_dim),
            # batch_size*n_query*n_heads*value_dim
            self.w_out.view(-1, self.embedding_dim),
            # n_heads*value_dim*embedding_dim
        ).view(batch_size, n_query, self.embedding_dim)

        return out  # batch_size*n_query*embedding_dim


class Normalization(nn.Module):
    def __init__(self, embedding_dim):
        super(Normalization, self).__init__()
        self.normalizer = nn.LayerNorm(embedding_dim)

    def forward(self, input):
        return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(EncoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim),
        )
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, src, mask=None):
        h0 = src
        h = self.normalization1(src)
        h = self.multiHeadAttention(q=h, mask=mask)
        h = h + h0
        h1 = h
        h = self.normalization2(h)
        h = self.feedForward(h)
        h2 = h + h1
        return h2


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(DecoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim),
        )
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, tgt, memory, mask=None):
        h0 = tgt
        tgt = self.normalization1(tgt)
        memory = self.normalization1(memory)
        h = self.multiHeadAttention(q=tgt, h=memory, mask=mask)
        h = h + h0
        h1 = h
        h = self.normalization2(h)
        h = self.feedForward(h)
        h2 = h + h1
        return h2


class Encoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=3):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            EncoderLayer(embedding_dim, n_head) for i in range(n_layer)
        )

    def forward(self, src, mask=None):
        for layer in self.layers:
            src = layer(src, mask)
        return src


class Decoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(embedding_dim, n_head) for i in range(n_layer)]
        )

    def forward(self, tgt, memory, mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, mask)
        return tgt


class AttentionNet(nn.Module):
    def __init__(self, input_dim, embedding_dim, device="cuda:1"):
        super(AttentionNet, self).__init__()
        self.initial_embedding = nn.Linear(
            input_dim, embedding_dim
        )  # layer for non-end position
        self.end_embedding = nn.Linear(
            input_dim, embedding_dim
        )  # embedding layer for end position
        self.budget_embedding = nn.Linear(embedding_dim + 2, embedding_dim)
        self.value_output = nn.Linear(embedding_dim, 1)
        self.pos_embedding = nn.Linear(32, embedding_dim)

        # self.nodes_update_layers = nn.ModuleList([DecoderLayer(embedding_dim, 8) for i in range(3)])

        self.current_embedding = nn.Linear(embedding_dim * 2, embedding_dim)

        # self.encoder = Encoder(embedding_dim=embedding_dim, n_head=4, n_layer=1)
        self.encoder = MixerModel(d_model=embedding_dim, n_layer=1, device=device)
        self.decoder = Decoder(embedding_dim=embedding_dim, n_head=4, n_layer=1)
        self.pointer = SingleHeadAttention(embedding_dim)

        self.LSTM = nn.LSTM(embedding_dim, embedding_dim, batch_first=True)

    def graph_embedding(self, node_inputs, edge_inputs, pos_encoding, mask=None):
        # current_position (batch, 1, 2)
        # end_position (batch, 1,2)
        # node_inputs (batch, sample_size+2, 2) end position and start position are the first two in the inputs
        # edge_inputs (batch, sample_size+2, k_size)
        # mask (batch, sample_size+2, k_size)
        end_position = node_inputs[:, 0, :].unsqueeze(1)
        embedding_feature = torch.cat(
            (
                self.end_embedding(end_position),
                self.initial_embedding(node_inputs[:, 1:, :]),
            ),
            dim=1,
        )

        pos_encoding = self.pos_embedding(pos_encoding)
        embedding_feature = embedding_feature + pos_encoding

        sample_size = embedding_feature.size()[1]
        embedding_dim = embedding_feature.size()[2]

        # for layer in self.nodes_update_layers:
        #    updated_node_feature_list = []
        #    for i in range(sample_size):
        #        # print(embedding_feature)
        #        if i==0:
        #            updated_node_feature_list.append(embedding_feature[:,i,:].unsqueeze(1))
        #        else:
        #            connected_nodes_feature = torch.gather(input=embedding_feature, dim=1,
        #                                                   index=edge_inputs[:, i, :].unsqueeze(-1).repeat(1, 1,embedding_dim))
        # (batch, k_size, embedding_size)
        # print(connected_nodes_feature)
        #            if mask is not None:
        #                node_mask = mask[:,i,:].unsqueeze(1)
        #            else:
        #                node_mask = None
        #            updated_node_feature_list.append(
        #                layer(tgt=embedding_feature[:, i, :].unsqueeze(1), memory=connected_nodes_feature,mask=node_mask))
        #    updated_node_feature = torch.cat(updated_node_feature_list,dim=1)
        #    embedding_feature = updated_node_feature
        # print(embedding_feature.size())
        embedding_feature = self.encoder(embedding_feature)

        return embedding_feature

    def select_next_node(
        self,
        embedding_feature,
        edge_inputs,
        budget_inputs,
        current_index,
        LSTM_h,
        LSTM_c,
        mask,
        i=0,
    ):
        LSTM_h = LSTM_h.permute(1, 0, 2)
        LSTM_c = LSTM_c.permute(1, 0, 2)

        batch_size = edge_inputs.size()[0]
        sample_size = edge_inputs.size()[1]
        k_size = edge_inputs.size()[2]
        # print('ks',k_size)
        # quit()
        current_edge = torch.gather(edge_inputs, 1, current_index.repeat(1, 1, k_size))
        # print(current_edge)
        current_edge = current_edge.permute(0, 2, 1)
        embedding_dim = embedding_feature.size()[2]
        th = (
            torch.FloatTensor([ADAPTIVE_TH])
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch_size, sample_size, 1)
            .to(embedding_feature.device)
        )

        embedding_feature = self.budget_embedding(
            torch.cat((embedding_feature, budget_inputs, th), dim=-1)
        )
        connected_nodes_feature = torch.gather(
            embedding_feature, 1, current_edge.repeat(1, 1, embedding_dim)
        )

        # print("=======budeget_inputs, ", budget_inputs.shape)
        connected_nodes_budget = torch.gather(budget_inputs, 1, current_edge)
        # print(embedding_feature)
        # print(connected_nodes_feature)
        current_node_feature = torch.gather(
            embedding_feature, 1, current_index.repeat(1, 1, embedding_dim)
        )
        current_node_feature, (LSTM_h, LSTM_c) = self.LSTM(
            current_node_feature, (LSTM_h, LSTM_c)
        )

        end_node_feature = embedding_feature[:, 0, :].unsqueeze(1)
        current_node_feature = torch.cat(
            (current_node_feature, end_node_feature), dim=-1
        )
        current_node_feature = self.current_embedding(current_node_feature)
        # print(current_node_feature)
        if mask is not None:
            # print('mask', mask.size())
            current_mask = torch.gather(mask, 1, current_index.repeat(1, 1, k_size)).to(
                embedding_feature.device
            )
            # print('current mask', current_mask)
        else:
            current_mask = None
            current_mask = torch.zeros((batch_size, 1, k_size), dtype=torch.int64).to(
                embedding_feature.device
            )
        one = torch.ones_like(current_mask, dtype=torch.int64).to(
            embedding_feature.device
        )
                
        current_mask = torch.where(
            connected_nodes_budget.permute(0, 2, 1) > 0, current_mask, one
        )
        current_mask[:, :, 0] = 1  # don't stay at current position
        try:
            assert 0 in current_mask
        except:
            print("-------------------------", connected_nodes_budget.permute(0, 2, 1))
            print("----------current mask with i =", i)
            assert 0 in current_mask

        # connected_nodes_feature = self.encoder(connected_nodes_feature, current_mask)
        current_feature_prime = self.decoder(
            current_node_feature, connected_nodes_feature, current_mask
        )
        logp_list = self.pointer(
            current_feature_prime, connected_nodes_feature, current_mask
        )
        logp_list = logp_list.squeeze(1)
        value = self.value_output(current_feature_prime)

        LSTM_h = LSTM_h.permute(1, 0, 2)
        LSTM_c = LSTM_c.permute(1, 0, 2)

        return logp_list, value, LSTM_h, LSTM_c

    def forward(
        self,
        node_inputs,
        edge_inputs,
        budget_inputs,
        current_index,
        LSTM_h,
        LSTM_c,
        pos_encoding,
        mask=None,
        i=0,
    ):
        with autocast():
            embedding_feature = self.graph_embedding(
                node_inputs, edge_inputs, pos_encoding, mask=None
            )
            logp_list, value, LSTM_h, LSTM_c = self.select_next_node(
                embedding_feature,
                edge_inputs,
                budget_inputs,
                current_index,
                LSTM_h,
                LSTM_c,
                mask,
                i,
            )
        return logp_list, value, LSTM_h, LSTM_c


def padding_inputs(inputs):
    seq = pad_sequence(inputs, batch_first=False, padding_value=1)
    seq = seq.permute(2, 1, 0)
    mask = torch.zeros_like(seq, dtype=torch.int64)
    ones = torch.ones_like(seq, dtype=torch.int64)
    mask = torch.where(seq != 1, mask, ones)
    # print(mask)
    # print(seq.size())
    return seq, mask


if __name__ == "__main__":
    model = AttentionNet(2, 8, greedy=True)
    node_inputs = torch.torch.rand((128, 10, 2))
    # print(node_inputs)
    edge_inputs = torch.randint(0, 10, (128, 10, 5))
    edge_inputs_list = []
    # for i in range(edge_inputs.size()[1]):
    #     edge_inputs_list.append(edge_inputs[:,i].permute(1,0))
    # edge_inputs_list.append(torch.randint(0, 10, (8, 1)))
    # edge_inputs, mask = padding_inputs(edge_inputs_list)
    current_index = torch.ones(size=(128, 1, 1), dtype=torch.int64)
    next_node, logp_list, value = model(node_inputs, edge_inputs, current_index)
    print(next_node.size())
    print(logp_list.size())
    print(value.size())
