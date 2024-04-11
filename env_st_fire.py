import numpy as np
from itertools import product
from utils.graph_controller import GraphController
from classes import Graph, PRMController
# from utils.target_controller import VTSPGaussian
from matplotlib import pyplot as plt
from gp_st_ipp import GaussianProcessWrapper
from arguments import arg
from parameters import ADAPTIVE_AREA
from scipy.interpolate import griddata
from fire_commander.catnipp_2d_fire import FireCommanderExtreme as Fire


def add_t(X, t: float):
    return np.concatenate((X, np.zeros((X.shape[0], 1)) + t), axis=1)

class Env:
    def __init__(self, graph_size, k_size, budget_size=None, start=None, destination=None, obstacle=None, budget_range=None, save_image=False, seed=None, fixed_env=None):
        self.graph_size = graph_size
        self.k_size = k_size
        self.budget = self.budget_init = budget_size
        if start is None:
            self.start = np.random.rand(1, 2)
        else:
            self.start = np.array([start])
        if destination is None:
            self.destination = np.random.rand(1, 2)
        else:
            self.destination = np.array([destination])

        self.obstacles = obstacle
        self.seed = seed
        self.curr_t = 0.0
        self.n_targets = 1

        # # generate Fire environment
        self.fire = Fire(world_size=30, online_vis=True, start=self.start[0])
        self.fire.env_init()
        
        # self.graph_ctrl = GraphController(self.graph_size, self.start, self.k_size, self.obstacles)
        self.graph_ctrl = PRMController(self.graph_size, self.obstacles, self.start, self.destination, self.k_size)
        self.node_coords, self.graph = self.graph_ctrl.runPRM(seed=self.seed)
        
        # underlying distribution
        self.underlying_distrib = None
        self.ground_truth = None
        self.high_info_idx = None
        
        # GP
        self.gp_wrapper = None
        self.node_feature = None
        self.RMSE = None
        self.JS, self.JS_init, self.JS_list, self.KL, self.KL_init, self.KL_list = None, None, None, None, None, None
        self.cov_trace, self.cov_trace_init = None, None
        self.unc, self.unc_list, self.unc_init, self.unc_sum, self.unc_sum_list = None, None, None, None, None
        
        # start point
        self.current_node_index = 0
        self.dist_residual = 0
        self.sample = self.start.copy()
        # self.random_speed_factor = None
        # self.d_to_target = None
        self.route = []

        self.save_image = save_image
        self.frame_files = []


    def reset(self, seed=None):
        # underlying distribution
        if seed:
            np.random.seed(seed)

        # # generate Fire environment
        self.fire = Fire(online_vis=True, start=self.start[0])
        self.fire.env_init()

        self.underlying_distrib = self.fire

        self.ground_truth = self.get_ground_truth()  # (1600, n_targets)
        self.high_info_idx = self.get_high_info_idx() if ADAPTIVE_AREA else None

        # initialize GP
        self.curr_t = 0.0

        # self.visit_t = [[] for _ in range(self.n_targets)]
        self.gp_wrapper = GaussianProcessWrapper(self.n_targets, self.node_coords)
        if arg.prior_measurement:
            node_prior = self.underlying_distrib.mean
            self.gp_wrapper.add_init_measures(add_t(node_prior, self.curr_t))
            self.gp_wrapper.update_gps()

        self.node_feature = self.gp_wrapper.update_node_feature(self.curr_t)

        # initialize evaluations
        self.RMSE = self.gp_wrapper.eval_avg_RMSE(self.ground_truth, self.curr_t)
        self.cov_trace = self.gp_wrapper.eval_avg_cov_trace(self.curr_t, self.high_info_idx)
        self.unc, self.unc_list = self.gp_wrapper.eval_avg_unc(self.curr_t, self.high_info_idx, return_all=True)
        self.JS, self.JS_list = self.gp_wrapper.eval_avg_JS(self.ground_truth, self.curr_t, return_all=True)
        self.KL, self.KL_list = self.gp_wrapper.eval_avg_KL(self.ground_truth, self.curr_t, return_all=True)
        self.unc_sum, self.unc_sum_list = self.gp_wrapper.eval_avg_unc_sum(self.unc_list, self.high_info_idx, return_all=True)
        self.JS_init = self.JS
        self.KL_init = self.KL
        self.cov_trace_init = self.cov_trace
        self.unc_init = self.unc
        self.budget = self.budget_init

        # start point
        self.current_node_index = 0
        self.dist_residual = 0
        self.sample = self.start.copy()
        self.random_speed_factor = np.random.rand()
        self.d_to_target = np.linalg.norm(self.sample - self.underlying_distrib.mean, axis=1)
        self.route = []
        return self.node_coords, self.graph, self.node_feature, self.budget

    def step(self, next_node_index, global_step=0, sample_length=0.1, measurement=True, eval_speed=None):
        reward = 0
        sample_length = sample_length
        metrics = {'budget': [], 'dtotarget': [], 'rmse': [], 'jsd': [], 'jsdall': [], 'jsdstd': [], 'unc': [], 'uncall': [], 'uncstd': []}
        alpha = min(global_step // 1000 * 0.1, 1) if arg.curriculum else 1  # 10k episodes
        d_len = np.linalg.norm(self.node_coords[next_node_index] - self.node_coords[self.current_node_index])
        remain_length = d_len
        next_length = sample_length - self.dist_residual
        no_sample = True

        while remain_length > next_length:
            if no_sample:
                self.sample = (self.node_coords[next_node_index] - self.node_coords[self.current_node_index]) * \
                             next_length / d_len + self.node_coords[self.current_node_index]
            else:
                self.sample = (self.node_coords[next_node_index] - self.node_coords[self.current_node_index]) * \
                             next_length / d_len + self.sample
            # if not eval_speed:
            #     steplen = 0.1 * sample_length * alpha * self.random_speed_factor  # target speed at least 10x slower
            # else:
            #     steplen = eval_speed * sample_length
            
            if measurement:
                observed_value = self.underlying_distrib.return_fire_at_location(
                    self.sample.reshape(-1, 2)) # + np.random.normal(0, 1e-10)
            else:
                observed_value = np.array([0])
            # self.gp_ipp.add_observed_point(self.sample, observed_value)

            self.curr_t += sample_length
            self.budget -= sample_length

            self.underlying_distrib.single_agent_state_update(self.sample.reshape(-1, 2))
            fire_state, reward_perception, done_fire, perception_complete, action_complete = self.fire.env_step()
            reward += reward_perception
            self.get_ground_truth()

            # target_mean = self.underlying_distrib.mean
            # self.d_to_target = np.linalg.norm(self.sample - target_mean, axis=1)
            for idx in range(self.n_targets):
            #     if self.d_to_target[idx] < 0.1:  # FOV
            #         measure_coord = target_mean[idx]
            #         measure_value = 1.0
            #         self.visit_t[idx] += [self.curr_t]
            #     else:
            #         measure_coord = self.sample
            #         measure_value = 0.0
                self.gp_wrapper.GPs[idx].add_observed_point(add_t(self.sample.reshape(-1, 2), self.curr_t), observed_value)

            remain_length -= next_length
            next_length = sample_length
            no_sample = False

            if eval_speed and self.gp_wrapper.GPs[0].observed_points:  # only in testing
                self.gp_wrapper.update_gps()
                metrics['budget'] += [self.budget_init - self.budget]
                metrics['dtotarget'] += [self.d_to_target]
                metrics['rmse'] += [self.gp_wrapper.eval_avg_RMSE(self.ground_truth, self.curr_t)]
                JS, JS_list = self.gp_wrapper.eval_avg_JS(self.ground_truth, self.curr_t, return_all=True)
                metrics['jsd'] += [JS]
                metrics['jsdall'] += [JS_list]
                metrics['jsdstd'] += [np.std(JS_list)]
                unc, unc_list = self.gp_wrapper.eval_avg_unc(self.curr_t, self.high_info_idx, return_all=True)
                metrics['unc'] += [unc]
                metrics['uncall'] += [unc_list]
                metrics['uncstd'] += [np.std(unc_list)]

        if self.gp_wrapper.GPs[0].observed_points:
            self.gp_wrapper.update_gps()
        self.dist_residual = self.dist_residual + remain_length if no_sample else remain_length
        actual_t = self.curr_t + self.dist_residual
        actual_budget = self.budget - self.dist_residual
        self.node_feature = self.gp_wrapper.update_node_feature(actual_t)

        self.ground_truth = self.get_ground_truth()
        self.high_info_idx = self.get_high_info_idx() if arg.high_info_thre else None
        self.RMSE = self.gp_wrapper.eval_avg_RMSE(self.ground_truth, actual_t)
        cov_trace = self.gp_wrapper.eval_avg_cov_trace(actual_t, self.high_info_idx)
        unc, unc_list = self.gp_wrapper.eval_avg_unc(actual_t, self.high_info_idx, return_all=True)
        unc_sum, unc_sum_list = self.gp_wrapper.eval_avg_unc_sum(self.unc_list, self.high_info_idx, return_all=True)
        JS, JS_list = self.gp_wrapper.eval_avg_JS(self.ground_truth, actual_t, return_all=True)
        KL, KL_list = self.gp_wrapper.eval_avg_KL(self.ground_truth, actual_t, return_all=True)

        # TODO: find in STAMP paper what this piece of code is for
        # r = 0
        # for i in range(self.n_targets):
        #     r += max(self.unc_list[i] - unc_list[i], 0)
        # reward += 5 * r - 0.1

        if next_node_index in self.route[-2:]:
            reward += -0.1

        elif self.cov_trace > cov_trace:
            reward += (self.cov_trace - cov_trace) / self.cov_trace
        self.cov_trace = cov_trace

        if done:
            reward -= cov_trace/900
            
        self.JS, self.JS_list = JS, JS_list
        self.KL, self.KL_list = KL, KL_list
        # self.cov_trace = cov_trace
        self.unc, self.unc_list = unc, unc_list
        self.unc_sum, self.unc_sum_list = unc_sum, unc_sum_list
        self.route += [next_node_index]
        self.current_node_index = next_node_index
        done = True if actual_budget <= 0 else False
        return reward, done, self.node_feature, actual_budget, metrics

    # def get_ground_truth(self):
    #     x1 = np.linspace(0, 1, 40)
    #     x2 = np.linspace(0, 1, 40)
    #     x1x2 = np.array(list(product(x1, x2)))
    #     ground_truth = self.underlying_distrib.fn(x1x2)
    #     return ground_truth

    def get_ground_truth(self, scale=1):
        # Extracting fire map coordinates and intensities
        if self.underlying_distrib.fire_map.shape[0] > 0:
            x_fire = self.underlying_distrib.fire_map[:, 0]
            y_fire = self.underlying_distrib.fire_map[:, 1]
            fire_intensity = self.underlying_distrib.fire_map[:, 2]
        
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

    def get_high_info_idx(self):
        high_info_idx = []
        for i in range(self.n_targets):
            idx = np.argwhere(self.ground_truth[:, i] > arg.high_info_thre)
            high_info_idx += [idx.squeeze(1)]
        return high_info_idx

    # def plot(self, route, n, step, path, budget_list, rew_list, div_list):
    #     # Plotting shorest path
    #     div_list = np.array(div_list)
    #     y_pred_sum = []
    #     plt.switch_backend('agg')
    #     plt.figure(figsize=(self.n_targets*2.8+3.6, 6))
    #     target_cmap = ['r', 'g', 'b', 'm', 'y', 'c', 'lightcoral', 'lightgreen', 'lightblue', 'orange', 'gold', 'pink']
    #     assert len(target_cmap) >= self.n_targets
    #     target_mean = self.underlying_distrib.mean
    #     for i, gp in enumerate(self.gp_wrapper.GPs):
    #         y_pred = gp.plot(self.ground_truth, target_id=i, target_num=self.n_targets, target_loc=target_mean,
    #                          all_pred=y_pred_sum, high_idx=self.high_info_idx, agent_loc=self.node_coords[self.current_node_index])
    #         y_pred_sum.append(y_pred)
    #     # plt.scatter(self.start[:, 0], self.start[:, 1], c='r', s=15, zorder=10)
    #     points_display = [(self.graph_ctrl.findPointsFromNode(path)) for path in route]
    #     x = [item[0] for item in points_display]
    #     y = [item[1] for item in points_display]
    #     plt.scatter(x[-1], y[-1], c='c', s=15, zorder=10)
    #     for i in range(len(x) - 1):
    #         alpha = max(0.02 * (i-len(x)) + 1, 0.1)
    #         plt.plot(x[i:i + 2], y[i:i + 2], c='white', linewidth=2, zorder=5, alpha=alpha)
    #     if target_mean[0] is not None:  # target location
    #         for i, mean in enumerate(target_mean):
    #             plt.scatter(*mean, c=target_cmap[i], s=10, marker='s')
    #         plt.subplot(2, self.n_targets+1, self.n_targets+3)
    #         for i, mean in enumerate(target_mean):
    #             plt.scatter(*mean, c=target_cmap[i], s=10, marker='s')
    #     ax1 = plt.subplot(2, self.n_targets+1, 2*self.n_targets+2)
    #     plt.grid(linestyle='--')
    #     plt.xlim(0, self.budget_init)
    #     plt.ylim(0, 1.4)
    #     for target_div in range(div_list.shape[1]):  # chart
    #         plt.plot(budget_list, div_list[:, target_div], alpha=0.5, c=target_cmap[target_div])
    #     # plt.plot(budget_list, div_list.mean(axis=1), 'k--', alpha=0.7)
    #     plt.ylabel('JSDiv')
    #     plt.title('{:g}/{:g} Reward:{:.3f}'.format(self.budget_init - self.budget, self.budget_init, rew_list[-1]))
    #     # ax2 = ax1.twinx()
    #     # ax2.plot(budget_list, rew_list, 'r--', alpha=0.5)
    #     # ax2.set_ylim([-2, 2])
    #     # ax2.set_ylabel('Reward(r)')
    #     plt.tight_layout()
    #     plt.savefig('{}/{}_{}_samples.png'.format(path, n, step, self.graph_size), dpi=150)
    #     frame = '{}/{}_{}_samples.png'.format(path, n, step, self.graph_size)
    #     self.frame_files.append(frame)

    def plot(self, route, n, step, path, testID=0, CMAES_route=False, sampling_path=False):
        # Plotting shorest path
        plt.switch_backend('agg')
        self.gp_ipp.plot(self.ground_truth)
        # plt.subplot(1,3,1)
        # plt.scatter(self.start[:,0], self.start[:,1], c='r', s=15)
        # plt.scatter(self.destination[:,0], self.destination[:,1], c='r', s=15)
        if CMAES_route:
            pointsToDisplay = route
        else:
            pointsToDisplay = [(self.prm.findPointsFromNode(path)) for path in route]
        x = [item[0] for item in pointsToDisplay]
        y = [item[1] for item in pointsToDisplay]
        for i in range(len(x)-1):
            plt.plot(x[i:i+2], y[i:i+2], c='black', linewidth=4, zorder=5, alpha=0.25+0.6*i/len(x))
        if sampling_path:
            pointsToDisplay2 = [(self.prm.findPointsFromNode(path)) for path in sampling_path]
            x0 = [item[0] for item in pointsToDisplay2]
            y0 = [item[1] for item in pointsToDisplay2]
            x1 = [item[0] for item in pointsToDisplay2[:3]]
            y1 = [item[1] for item in pointsToDisplay2[:3]]
            for i in range(len(x0) - 1):
                plt.plot(x0[i:i + 2], y0[i:i + 2], c='white', linewidth=4, zorder=5, alpha=1- 0.2 * i / len(x0))
            for i in range(len(x1) - 1):
                plt.plot(x1[i:i + 2], y1[i:i + 2], c='red', linewidth=4, zorder=6)

        plt.subplot(2,2,4)
        plt.title('High interest area')
        xh = self.high_info_area[:,0]
        yh = self.high_info_area[:,1]
        plt.hist2d(xh, yh, bins=30, range=[[0,1], [0,1]], vmin=0, vmax=1, rasterized=True)
        # plt.scatter(self.start[:,0], self.start[:,1], c='r', s=15)
        # plt.scatter(self.destination[:,0], self.destination[:,1], c='r', s=15)

        # x = [item[0] for item in pointsToDisplay]
        # y = [item[1] for item in pointsToDisplay]

        for i in range(len(x)-1):
            plt.plot(x[i:i+2], y[i:i+2], c='black', linewidth=4, zorder=5, alpha=0.25+0.6*i/len(x))
        plt.suptitle('Budget: {:.4g}/{:.4g},  Cov trace: {:.4g}'.format(
            self.budget, self.budget0, self.cov_trace))
        plt.tight_layout()
        plt.savefig('{}/{}_{}_{}_samples.png'.format(path, n, testID, step, self.sample_size), dpi=150)
        # plt.show()
        frame = '{}/{}_{}_{}_samples.png'.format(path, n, testID, step, self.sample_size)
        self.frame_files.append(frame)

if __name__ == '__main__':
    pass

