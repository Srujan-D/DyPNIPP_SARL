# # import torch
# # import torch.nn as nn
# # import math
# # import numpy as np
# # from torch.nn.utils.rnn import pad_sequence
# # from torch.cuda.amp.autocast_mode import autocast


# # class PredictNextBelief(nn.Module):
# #     def __init__(self, device="cuda"):
# #         super(PredictNextBelief, self).__init__()
# #         self.device = device
# #         # self.conv encoder --> what features of GP to use? predicted mean, uncertainty, 
# #                             # --> do we want to encode history (just regress GP over time) explicitly?
# #         # self.lstm layer
# #         # self.MLP layer

# #         self.conv_encoder = nn.Sequential(
# #             nn.Conv2d(1, 8, kernel_size=(3, 3)),
# #             nn.ReLU(),
# #             nn.MaxPool2d(kernel_size=(2, 2)),
# #             nn.Conv2d(8, 4, kernel_size=(3, 3)),
# #             nn.ReLU(),
# #             nn.MaxPool2d(kernel_size=(2, 2)),
# #         )

# #     def forward(self, x):
# #         x = self.conv_encoder(x)
# #         return x
    

# # if __name__ == "__main__":
# #     model = PredictNextBelief()
# #     dummy_ip = torch.randn(1, 1, 30, 30)
# #     out = model(dummy_ip)
# #     print(out.shape)

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.cluster import KMeans

# # Load MNIST dataset
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# train_loader = DataLoader(mnist_train, batch_size=128, shuffle=True)

# class Autoencoder(nn.Module):
#     def __init__(self, input_dim, encoding_dim):
#         super(Autoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 256),
#             nn.ReLU(True),
#             nn.Linear(256, 64),
#             nn.ReLU(True),
#             nn.Linear(64, encoding_dim)
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(encoding_dim, 64),
#             nn.ReLU(True),
#             nn.Linear(64, 256),
#             nn.ReLU(True),
#             nn.Linear(256, input_dim),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x

#     def encode(self, x):
#         return self.encoder(x)

# def clustering_loss(encoded, kmeans):
#     cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
#     distances = torch.cdist(encoded, cluster_centers)
#     loss = torch.mean(torch.min(distances, dim=1)[0])
#     return loss


# # Model parameters
# input_dim = 28 * 28  # MNIST images are 28x28
# encoding_dim = 10
# num_epochs = 50
# learning_rate = 0.001

# # Initialize the autoencoder model
# model = Autoencoder(input_dim, encoding_dim).to('cuda')
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# # Training loop
# for epoch in range(num_epochs):
#     for data in train_loader:
#         images, _ = data
#         images = images.view(-1, 28 * 28).to('cuda')
        
#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, images)
        
#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
    
#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# # Encode the entire dataset
# encoded_data = []
# labels = []

# with torch.no_grad():
#     for data in train_loader:
#         images, y = data
#         images = images.view(-1, 28 * 28).to('cuda')
#         encoded = model.encode(images).cpu().numpy()
#         encoded_data.append(encoded)
#         labels.append(y.numpy())

# encoded_data = np.concatenate(encoded_data)
# labels = np.concatenate(labels)

# # Apply k-means clustering on the encoded data
# kmeans = KMeans(n_clusters=10)
# kmeans.fit(encoded_data)

# # Further training with clustering loss
# for epoch in range(num_epochs):
#     for data in train_loader:
#         images, _ = data
#         images = images.view(-1, 28 * 28).to('cuda')
        
#         # Forward pass
#         encoded = model.encode(images)
#         outputs = model(images)
        
#         # Calculate losses
#         reconstruction_loss = criterion(outputs, images)
#         cluster_loss = clustering_loss(encoded.cpu(), kmeans)
#         total_loss = reconstruction_loss + cluster_loss
        
#         # Backward pass and optimization
#         optimizer.zero_grad()
#         total_loss.backward()
#         optimizer.step()
    
#     print(f'Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss.item():.4f}, Reconstruction Loss: {reconstruction_loss.item():.4f}, Cluster Loss: {cluster_loss.item():.4f}')

# # Final encoding after training
# encoded_data = []

# with torch.no_grad():
#     for data in train_loader:
#         images, _ = data
#         images = images.view(-1, 28 * 28).to('cuda')
#         encoded = model.encode(images).cpu().numpy()
#         encoded_data.append(encoded)

# encoded_data = np.concatenate(encoded_data)

# plt.figure(figsize=(10, 8))
# scatter = plt.scatter(encoded_data[:, 0], encoded_data[:, 1], c=labels, cmap='plasma', s=50)
# cbar = plt.colorbar(scatter)
# cbar.set_label('Labels')
# plt.title('Autoencoder with Clustering Loss Visualization')
# plt.xlabel('Latent Dimension 1')
# plt.ylabel('Latent Dimension 2')
# plt.show()


import torch
import torch.nn as nn

class ConvLSTMAutoencoder(nn.Module):
    def __init__(self):
        super(ConvLSTMAutoencoder, self).__init__()
        
        # Encoder
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(8, 4, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        self.lstm = nn.LSTM(
            input_size=4 * 6 * 6, hidden_size=32, num_layers=1, batch_first=True
        )

        # Decoder
        self.fc = nn.Linear(32, 4 * 6 * 6)
        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 8, kernel_size=3, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoding
        batch_size, seq_len, C, H, W = x.size()
        x = x.view(batch_size * seq_len, C, H, W)  # Flatten batch and sequence dimensions
        x = self.conv_encoder(x)
        x = x.view(batch_size, seq_len, -1)  # Reshape for LSTM input

        # LSTM
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take the output of the last time step

        # Decoding
        x = self.fc(x)
        x = x.view(batch_size, 4, 6, 6)
        x = self.conv_decoder(x)
        
        return x

# Example usage
model = ConvLSTMAutoencoder()
input_tensor = torch.randn(10, 1, 1, 30, 30)  # Batch size 10, sequence length 1, 1 channel, 30x30 images
output_tensor = model(input_tensor)
print(output_tensor.shape)  # Should print torch.Size([10, 1, 30, 30])
