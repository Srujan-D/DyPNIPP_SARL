# import numpy as np
# import GPy
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Define the spatiotemporal Gaussian process class
# class SpatioTemporalGP:
#     def __init__(self):
#         self.model = None
#         self.X_train = None
#         self.Y_train = None

#     def add_points(self, X_new, Y_new):
#         if self.X_train is None:
#             self.X_train = X_new
#             self.Y_train = Y_new
#         else:
#             self.X_train = np.vstack((self.X_train, X_new))
#             self.Y_train = np.vstack((self.Y_train, Y_new))

#     def train(self):
#         kernel = GPy.kern.Matern52(input_dim=3, ARD=True)
#         self.model = GPy.models.GPRegression(self.X_train, self.Y_train, kernel)
#         self.model.optimize(messages=True)

#     def predict(self, X_test):
#         mean, var = self.model.predict(X_test)
#         return mean, var

# # Generate some example observed points
# np.random.seed(0)
# num_observed_points = 10
# observed_points = np.random.rand(num_observed_points, 2)  # Random points in the range (0,1) for X, Y, and time
# observed_points = np.hstack((observed_points, np.random.randint(0, 50, (num_observed_points, 1))))  # Add time

# # Create a spatiotemporal Gaussian process
# gp = SpatioTemporalGP()

# # Add observed points to the GP
# X_observed = observed_points  # Extract X and Y coordinates
# Y_observed = np.random.rand(num_observed_points, 1)  # Random values for the observed points
# gp.add_points(X_observed, Y_observed)

# # Train the GP model
# gp.train()

# # Define the range of X, Y, and time for visualization
# X_range = np.linspace(0, 1, 50)
# Y_range = np.linspace(0, 1, 50)
# time_range = np.linspace(0, 51, 50)

# # Create a meshgrid for visualization
# X_grid, Y_grid, time_grid = np.meshgrid(X_range, Y_range, time_range)
# X_test = np.vstack((X_grid.flatten(), Y_grid.flatten(), time_grid.flatten())).T

# # Predict at a specific timestamp
# timestamp_index = 49  # index of time_range
# mean, var = gp.predict(X_test)
# mean_at_timestamp = mean.reshape(len(X_range), len(Y_range), len(time_range))[:, :, timestamp_index]
# var_at_timestamp = var.reshape(len(X_range), len(Y_range), len(time_range))[:, :, timestamp_index]

# # Visualize the mean and uncertainty
# plt.figure()

# # Plot the mean
# plt.imshow(mean_at_timestamp, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
# plt.colorbar(label='Mean')

# # Scatter observed points
# plt.scatter(X_observed[:, 0], X_observed[:, 1], color='r', label='Observations')
# for i in range(len(X_observed)):
#     plt.text(X_observed[i, 0], X_observed[i, 1], f"({X_observed[i, 0]:.2f}, {X_observed[i, 1]:.2f}, {X_observed[i, 2]:.2f} == {Y_observed[i, 0]:.2f})", fontsize=8)

# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Spatiotemporal Gaussian Process at timestamp {}'.format(time_range[timestamp_index]))
# plt.legend()
# plt.show()


# import matplotlib.pyplot as plt

# data = [0.001, 0.127, 2.126, 0.668, 0.005, 0.004, 0.077, 0.612, 0.936, 1.856, 1.786, 3.322,
#         1.73, 2.024, 0.759, 1.71, 2.273, 1.078, 0.407, 0., 0.172, 0.23, 0.845, 0.375,
#         0.75, 1.076, 0.102, 1.482, 1.03, 1.705, 0.25, 0.092, 0.304, 0.684, 0.978, 0.18,
#         0., 0.449, 1.058, 1.121, 0.044, 0.254, 0.189, 1.109, 0.241, 0.111, 0.233, 0.103,
#         0.186, 0.147, 0.049, 0.021, 1.292, 0.222, 0.035, 0.041, 0.06, 0.21, 0.137, 1.086,
#         0.558, 1.46, 0.675, 0.64, 0.701, 0., 0.694, 0.429, 0.187, 1.029, 0.142, 0.122,
#         0.539, 0.393, 0.494, 0.502, 0.238, 0.611, 0.169, 0.202, 0.205, 0.239, 0.088, 0.354,
#         0.269, 0.286, 0.31, 0.467, 0.274, 0.408, 0.277, 0.242, 0.412, 0.051, 0.209, 0.,
#         0.662, 0., 0.362, 0.111, 0.041]

# plt.plot(data)
# plt.xlabel('Timestamp')
# plt.ylabel('Value')
# plt.title('Norm (Ot+1 - Ot)')
# plt.savefig('data_plot.png')

# from scipy.stats import ks_2samp
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# def get_named_parameters(obj):
#     if isinstance(obj, torch.nn.Module):
#         return obj.named_parameters()
#     elif isinstance(obj, dict):
#         return [param for name, param in obj.items() if isinstance(param, torch.Tensor) and param.requires_grad]
#     else:
#         raise ValueError("Input should be a PyTorch model or a dictionary of parameters")


# def compare_weight_distributions(model1, model2):
#     named_params1 = get_named_parameters(model1)
#     named_params2 = get_named_parameters(model2)

#     for (name1, param1), (name2, param2) in zip(named_params1, named_params2):
#         if param1.requires_grad and param2.requires_grad:
#             # Access the tensors within the dictionary
#             stat, p_value = ks_2samp(param1.cpu().numpy().flatten(), param2.cpu().numpy().flatten())
#             print(f"Layer {name1}: KS Statistic: {stat}, p-value: {p_value}")

# def get_flattened_weights(model):
#     weights = dict()
#     for key in model.keys():
#         weights[key] = model[key].flatten()
#     return torch.cat(list(weights.values()))


# def compute_cosine_similarity(model1, model2):
#     # weights1 = get_flattened_weights(model1)
#     # weights2 = get_flattened_weights(model2)

#     for key in model1.keys():
#         if model1[key].shape != model2[key].shape:
#             raise ValueError(f"Shapes of the models do not match for key: {key}")
#         else:
#             similarity = F.cosine_similarity(model1[key].flatten(), model2[key].flatten(), dim=0)
#             print(f"Similarity for key {key}: {similarity}")

# # load two models
# model1 = torch.load('/data/srujan/research/catnipp/model/robust_1/belief_checkpoint.pth')['model']
# model2 = torch.load('/data/srujan/research/catnipp/model/robust_lambda100/belief_checkpoint.pth')['model']

# similarity = compute_cosine_similarity(model1, model2)
# print(f"Cosine similarity between the models: {similarity}")

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# og = 'robust_eval/robust_lambda_100_allfrom1_10_seed1_5'
eval = 'robust_eval/robust_lambda_100_allfrom1_10_seed1_5/'
vae = 'robust_vae/no_lambda/'
vae_16 = 'robust_vae/'

X = None
y = None
indices = []
seed = []

# for j in range(1, 5):
#     # for i in range(1, 11):
#     for i in [1, 5, 10]:
#         # X_i = np.load(
#         #     f"{eval}features_lambda100_{i}_seed{j}.npy"
#         # )
#         # y_i = np.load(
#         #     f"{eval}labels_lambda100_{i}_seed{j}.npy"
#         # )

#         X_i = np.load(
#             f"{vae_16}features_vae_nolambda_{i}_seed{j}.npy"
#         )
#         y_i = np.load(
#             f"{vae_16}labels_vae_nolambda_{i}_seed{j}.npy"
#         )

#         print(f"y_{i} shape:", y_i.shape)

#         seed_i = np.full((len(y_i), 1), j)
#         seed.extend(seed_i)

#         indices_i = np.arange(len(y_i))
#         indices.extend(indices_i)

#         if X is None:
#             X = X_i
#             y = y_i
#         else:
#             X = np.concatenate((X, X_i), axis=0)
#             y = np.concatenate((y, y_i), axis=0)


# Assuming your feature array is called X and label array is y
X_1 = np.load('robust_vae_meanstd/same_belief/features_samebelief_1.npy')
y_1 = np.load('robust_vae_meanstd/same_belief/labels_samebelief_1.npy')

print('x1 shape:', X_1.shape)
print('y1 shape:', y_1.shape)
X_5 = np.load('robust_vae_meanstd/same_belief/features_samebelief_5.npy')
y_5 = np.load('robust_vae_meanstd/same_belief/labels_samebelief_5.npy')

print('x5 shape:', X_5.shape)
print('y5 shape:', y_5.shape)

X_10 = np.load('robust_vae_meanstd/same_belief/features_samebelief_10.npy')
# X_10 = X_10[:-1]
y_10 = np.load('robust_vae_meanstd/same_belief/labels_samebelief_10.npy')
print('x10 shape:', X_10.shape)
print('y10 shape:', y_10.shape)

# X_20 = np.load('robust_vae/features_cur_vae_20.npy')
# y_20 = np.load('robust_vae/labels_cur_vae_20.npy')

# print('x20 shape:', X_20.shape)
# print('y20 shape:', y_20.shape)

X = np.concatenate((X_1, X_5, X_10), axis=0)
y = np.concatenate((y_1, y_5, y_10), axis=0)

# X_1 = np.load('robust_eval/curriculum/features_crl_1_250.npy')
# y_1 = np.load('robust_eval/curriculum/labels_crl_1_250.npy')

# print('x1 shape:', X_1.shape)
# print('y1 shape:', y_1.shape)

# X_5 = np.load('robust_eval/curriculum/features_crl_5_250.npy')
# y_5 = np.load('robust_eval/curriculum/labels_crl_5_250.npy')

# print('x5 shape:', X_5.shape)
# print('y5 shape:', y_5.shape)

# X_10 = np.load('robust_eval/curriculum/features_crl_10_250.npy')
# X_10 = X_10[:-1]
# y_10 = np.load('robust_eval/curriculum/labels_crl_10_250.npy')

# print('x10 shape:', X_10.shape)
# print('y10 shape:', y_10.shape)

# X = np.concatenate((X_1, X_5, X_10), axis=0)
# y = np.concatenate((y_1, y_5, y_10), axis=0)

# X = np.concatenate((X_1, X_10), axis=0)
# y = np.concatenate((y_1, y_10), axis=0)


# # # Step 1: Apply PCA to reduce dimensions from 128 to 50
# pca = PCA(n_components=10)
# X = pca.fit_transform(X)

# # Print the explained variances
# print("Explained variances (variances of the principal components):")
# print(pca.explained_variance_ratio_)
            
import time
t1 = time.time()
# Step 2: Apply t-SNE to reduce dimensions from 50 to 2
tsne = TSNE(n_components=2, random_state=42 , perplexity=60)
X = tsne.fit_transform(X)

t2 = time.time()
print(f"Time taken for t-SNE in seconds: {t2-t1:.2f}")

# Step 3: Plot the t-SNE result
plt.figure(figsize=(10, 8))
# cmap = plt.get_cmap('RdBu', 10)
# cmap = 'tab10'
cmap = "viridis"
scatter = plt.scatter(
    X[:, 0], X[:, 1], c=y, cmap=cmap, s=50
)  # , vmin=-0.5, vmax=10.5)
plt.colorbar(scatter, ticks=range(11))
plt.title("same belief t-SNE visualization")
plt.xlabel("t-SNE component 1")
plt.ylabel("t-SNE component 2")

# print(len(X))
# print(len(y))
# print(len(seed))
# for i in range(len(X)):
#     plt.text(X[i, 0] + 0.5, X[i, 1] + 0.5, str(seed[i][0]), fontsize=8)

# texts = []
# for i in range(len(X)):
#     texts.append(plt.text(X[i, 0], X[i, 1], str(y[i]), fontsize=8))

# from adjustText import adjust_text
# adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray'))

# plt.show()
plt.savefig('vae_same_belief_2.png')