import numpy as np
import math
import matplotlib.pyplot as plt
from itertools import product

class Gaussian3D():
    def __init__(self):
        self.rangeDistribs = (8, 12)
        self.mean = []
        self.sigma_x = []
        self.sigma_y = []
        self.sigma_t = []
        self.cov_xy = []
        self.cov_xt = []
        self.cov_yt = []
        self.max_value = None
        self.createProbability()

    def addGaussian(self):
        gaussian_mean = np.random.rand(3)
        gaussian_var = np.zeros((3, 3))
        gaussian_var[([0, 1, 2], [0, 1, 2])] = np.random.uniform(0.00005, 0.0002, 3)
        
        SigmaX = np.sqrt(gaussian_var[0][0])
        SigmaY = np.sqrt(gaussian_var[1][1])
        SigmaT = np.sqrt(gaussian_var[2][2])
        
        CovXY = gaussian_var[0][1]
        CovXT = gaussian_var[0][2]
        CovYT = gaussian_var[1][2]
        
        self.mean.append(gaussian_mean)
        self.sigma_x.append(SigmaX)
        self.sigma_y.append(SigmaY)
        self.sigma_t.append(SigmaT)
        self.cov_xy.append(CovXY)
        self.cov_xt.append(CovXT)
        self.cov_yt.append(CovYT)

    def createProbability(self):
        numDistribs = np.random.randint(self.rangeDistribs[0], self.rangeDistribs[1] + 1)
        for _ in range(numDistribs):
            self.addGaussian()

    def distribution_function(self, X):
        y = np.zeros(X.shape[0])
        row_mat, col_mat, time_mat = X[:, 0], X[:, 1], X[:, 2]
        
        for gaussian_mean, SigmaX, SigmaY, SigmaT, CovXY, CovXT, CovYT in zip(self.mean, self.sigma_x, self.sigma_y, self.sigma_t, self.cov_xy, self.cov_xt, self.cov_yt):
            covariance_matrix = np.array([[SigmaX**2, CovXY, CovXT],
                                          [CovXY, SigmaY**2, CovYT],
                                          [CovXT, CovYT, SigmaT**2]])
            inv_cov_matrix = np.linalg.inv(covariance_matrix)
            diff = X - gaussian_mean
            exponent = -0.5 * np.einsum('ij,ij->i', diff, np.dot(inv_cov_matrix, diff.T).T)
            coefficients = 1 / (np.sqrt((2 * np.pi)**3 * np.linalg.det(covariance_matrix)))
            distribution_matrix = coefficients * np.exp(exponent)
            y += distribution_matrix
            
        if self.max_value is None:
            self.max_value = np.max(y)
            y /= self.max_value
        else:
            y /= self.max_value
            
        return y

    @staticmethod
    def plot(img):
        plt.imshow(img)
        plt.colorbar()
        plt.show()

if __name__ == '__main__':
    example = Gaussian3D()
    print(len(example.mean))
    x1 = np.linspace(0, 1, 50)
    x2 = np.linspace(0, 1, 50)
    t = np.linspace(0, 9, 10)  # Adding time axis
    x1x2t = np.array(list(product(x1, x2, t)))
    y = example.distribution_function(x1x2t)
    for i in range(10):
        example.plot(y.reshape(50, 50, 10)[:,:,i])
    # example.plot(y.reshape(50, 50, 10)[:,:,0])
