import warnings
from itertools import product
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from classes.Gaussian2D import Gaussian2D
# from parameters import *
from mpl_toolkits.axes_grid1 import make_axes_locatable



class GaussianProcessForIPP():
    def __init__(self, node_coords):
        # self.kernel = C(1.0, (1e-3, 1e3)) * RBF([5,5], (1e-2, 1e2))
        # self.kernel = RBF(0.2)
        self.kernel = Matern(length_scale=0.45)
        # self.gp = GaussianProcessRegressor(kernel=self.kernel, optimizer=None, n_restarts_optimizer=0, normalize_y=True)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=10, normalize_y=True)
        self.observed_points = []
        self.observed_value = []
        self.node_coords = node_coords
        self.env_size = 30

    def add_observed_point(self, point_pos, value):
        self.observed_points.append(point_pos)
        self.observed_value.append(value)

    def update_gp(self):
        if self.observed_points:
            X = np.array(self.observed_points).reshape(-1,2)
            y = np.array(self.observed_value).reshape(-1,1)
            # print("Y : ", y)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.gp.fit(X, y)

    def update_node(self):
        y_pred, std = self.gp.predict(self.node_coords, return_std=True)
        # if len(self.observed_value) > 2:
        #     print(">>> y_true min max: ", min(self.observed_value), max(self.observed_value))
        # print(">>> y_pred min max: ",y_pred.min(), y_pred.max())
        return y_pred, std

    def evaluate_RMSE(self, y_true):
        x1 = np.linspace(0, 1, self.env_size)
        x2 = np.linspace(0, 1, self.env_size)
        x1x2 = np.array(list(product(x1, x2)))
        y_pred, std = self.gp.predict(x1x2, return_std=True)
        RMSE = np.sqrt(mean_squared_error(y_pred, y_true))
        return RMSE

    def evaluate_F1score(self, y_true):
        x1 = np.linspace(0, 1, self.env_size)
        x2 = np.linspace(0, 1, self.env_size)
        x1x2 = np.array(list(product(x1, x2)))
        score = self.gp.score(x1x2,y_true)
        return score

    def evaluate_cov_trace(self, X=None):
        if X is None or X.shape[0] == 0:
            x1 = np.linspace(0, 1, self.env_size)
            x2 = np.linspace(0, 1, self.env_size)
            X = np.array(list(product(x1, x2)))
        _, std = self.gp.predict(X, return_std=True)
        trace = np.sum(std*std)
        return trace

    def evaluate_mutual_info(self, X=None):
        if X is None or X.shape[0] == 0:
            x1 = np.linspace(0, 1, self.env_size)
            x2 = np.linspace(0, 1, self.env_size)
            X = np.array(list(product(x1, x2)))
        n_sample = X.shape[0]
        _, cov = self.gp.predict(X, return_cov=True)
        
        mi = (1 / 2) * np.log(np.linalg.det(0.01*cov.reshape(n_sample, n_sample) + np.identity(n_sample)))
        return mi

    def get_high_info_area(self, t=0, beta=1):
        x1 = np.linspace(0, 1, self.env_size)
        x2 = np.linspace(0, 1, self.env_size)
        x1x2 = np.array(list(product(x1, x2)))
        y_pred, std = self.gp.predict(x1x2, return_std=True)
        
        high_measurement_area = []
        for i in range(900):
            if y_pred[i] + beta * std[i] >= t:
                high_measurement_area.append(x1x2[i])
        high_measurement_area = np.array(high_measurement_area)
        # print(">>> high_measurement_area: ", high_measurement_area.shape, y_pred.max(), std.max(), t)
        if high_measurement_area.shape[0] == 0:
            high_measurement_area = x1x2
        return high_measurement_area

    def plot(self, y_true):
        x1 = np.linspace(0, 1, self.env_size)
        x2 = np.linspace(0, 1, self.env_size)

        x1x2 = np.array(list(product(x1, x2)))
        y_pred, std = self.gp.predict(x1x2, return_std=True)

        X0p, X1p = x1x2[:, 0].reshape(self.env_size, self.env_size), x1x2[:, 1].reshape(self.env_size, self.env_size)
        y_pred = np.reshape(y_pred, (self.env_size, self.env_size))
        std = std.reshape(self.env_size,self.env_size)

        X = np.array(self.observed_points)

        fig = plt.figure(figsize=(6,6))
        #if self.observed_points:
        #    plt.scatter(X[:, 0].reshape(1, -1), X[:, 1].reshape(1, -1), s=10, c='r')
        plt.subplot(2, 2, 2) # ground truth
        plt.title('Ground truth')
        fig.colorbar(plt.pcolormesh(X0p, X1p, y_true.reshape(self.env_size, self.env_size), shading='auto', vmin=0, vmax=y_true.max()))
        plt.subplot(2, 2, 3) # stddev
        plt.title('Predict std')
        fig.colorbar(plt.pcolormesh(X0p, X1p, std, shading='auto', vmin=std.min(), vmax=std.max()))
        plt.subplot(2, 2, 1) # mean
        plt.title('Predict mean')
        fig.colorbar(plt.pcolormesh(X0p, X1p, y_pred, shading='auto', vmin=y_pred.min(), vmax=y_pred.max()))
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        # if self.observed_points:
        #     plt.scatter(X[:, 0].reshape(1, -1), X[:, 1].reshape(1, -1), s=2, c='r')
        # plt.show()
        return y_pred

if __name__ == '__main__':
    example = Gaussian2D()
    x1 = np.linspace(0, 1)
    x2 = np.linspace(0, 1)
    x1x2 = np.array(list(product(x1, x2)))
    y_true = example.distribution_function(X=x1x2)
    # print(y_true.shape)
    node_coords = np.random.uniform(0,1,(100,2))
    gp_ipp = GaussianProcessForIPP(node_coords)
    gp_ipp.plot(y_true.reshape(50,50))
    for i in range(node_coords.shape[0]):
        y_observe = example.distribution_function(node_coords[i].reshape(-1,2))
        # print(node_coords[i], y_observe)
        gp_ipp.add_observed_point(node_coords[i], y_observe)
        gp_ipp.update_gp()
        y_pre, std = gp_ipp.update_node()
        print(gp_ipp.evaluate_cov_trace())
    gp_ipp.plot(y_true)
    print(gp_ipp.evaluate_F1score(y_true))
    print(gp_ipp.gp.kernel_)
