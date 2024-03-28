import numpy as np
from scipy.interpolate import griddata
from itertools import product

class FireMap:
    def __init__(self, fire_map):
        self.fire_map = np.array(fire_map)
    
    def get_ground_truth(self, scale=1):
        # Extracting fire map coordinates and intensities
        if self.fire_map.shape[0] > 0:
            x_fire = self.fire_map[:, 0]
            y_fire = self.fire_map[:, 1]
            fire_intensity = self.fire_map[:, 2]
        
        # Scaling fire map coordinates between (0,1)
        x_fire_scaled = x_fire / scale
        y_fire_scaled = y_fire / scale
        
        # Creating a grid for ground truth
        x1 = np.linspace(0, 1, 30)
        x2 = np.linspace(0, 1, 30)
        x1x2 = np.array(list(product(x1, x2)))
        
        # Interpolating fire intensity values onto the grid
        ground_truth = griddata((x_fire_scaled, y_fire_scaled), fire_intensity, x1x2, method='linear')
        
        # Reshape ground_truth to match x1x2 shape
        ground_truth = ground_truth.reshape((len(x1), len(x2)))
        
        return ground_truth

# Example usage:
fire_map = [[1,4,500],[4,5,1000]]
fire_map_obj = FireMap(fire_map)
ground_truth = fire_map_obj.get_ground_truth(scale=10)  # Assuming scale factor is 10
print(ground_truth)
