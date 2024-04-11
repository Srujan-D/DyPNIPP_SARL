import copy, pdb
import csv
import sys
# sys.path.append('..')
sys.path.append('/data/srujan/research/catnipp')
import os
import ray
import torch
import time
from multiprocessing import Pool
import numpy as np
import time
from test_attention_net import AttentionNet
from test_runner import Runner
# from test_worker import WorkerTest
from test_worker_real import WorkerTestReal as WorkerTest
from test_parameters import *

import cProfile
# import signal

# def signal_handler(sig, frame):
#     print("\nCtrl+C detected. Exiting.")
#     ray.shutdown()
#     sys.exit(0)

def run_test(seed, global_network, checkpoint, device, local_device, result_path_, model_idx=0):
    time0 = time.time()
    if not os.path.exists(result_path_):
        os.makedirs(result_path_)
    # device = torch.device('cuda') if USE_GPU_GLOBAL else torch.device('cpu')
    # local_device = torch.device('cuda') if USE_GPU else torch.device('cpu')
    # global_network = AttentionNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    # checkpoint = torch.load(f'{model_path}/checkpoint.pth')
    global_network.load_state_dict(checkpoint['model'])
    # print(f'Loading model: {FOLDER_NAME}...')
    # print(f'##### of episode for training: ', checkpoint['episode'])
    # print(f'Total budget range: {BUDGET_RANGE}')
    if 'MG' in result_path_:
        multigamma = True
    else:
        multigamma = False

    # init meta agents
    meta_agents = [RLRunner.remote(i, seed, multigamma) for i in range(NUM_META_AGENT)]
    weights = global_network.to(local_device).state_dict() if device != local_device else global_network.state_dict()
    curr_test = 1
    metric_name = ['budget', 'success_rate', 'RMSE', 'delta_cov_trace', 'MI', 'F1Score', 'cov_trace', 'planning_time']
    perf_metrics = {}
    for n in metric_name:
        perf_metrics[n] = []
    cov_trace_list = []
    time_list = []
    episode_number_list = []
    budget_history = []
    obj_history = []
    obj2_history = []
    
    # # signal.signal(signal.SIGINT, signal_handler)
    # profiler = cProfile.Profile()
    # profiler.enable()

    try:
        while True:
            jobList = []
            for i, meta_agent in enumerate(meta_agents):
                jobList.append(meta_agent.job.remote(weights, curr_test, budget_range=BUDGET_RANGE, sample_length=SAMPLE_LENGTH, model_idx=model_idx))
                curr_test += 1
            done_id, jobList = ray.wait(jobList, num_returns=NUM_META_AGENT)
            done_jobs = ray.get(done_id)

            for job in done_jobs:
                metrics, info = job
                episode_number_list.append(info['episode_number'])
                print(">>> Metrics keys : ", metrics.keys())
                cov_trace_list.append(metrics['cov_trace'])
                time_list.append(metrics['planning_time'])
                for n in metric_name:
                    perf_metrics[n].append(metrics[n])
                budget_history += metrics['budget_history']
                obj_history += metrics['obj_history']
                obj2_history += metrics['obj2_history']

            if curr_test > NUM_TEST:
                print('#Test sample:', NUM_SAMPLE_TEST, '|#Total test:', NUM_TEST, '|Budget range:', BUDGET_RANGE, '|Sample size:', SAMPLE_SIZE, '|K size:', K_SIZE)
                # print('Avg time per test:', (time.time()-time0)/NUM_TEST)
                perf_data = []
                for n in metric_name:
                    perf_data.append(np.nanmean(perf_metrics[n]))
                # for i in range(len(metric_name)):
                    # print(metric_name[i], ':\t', perf_data[i])
                
                idx = np.array(episode_number_list).argsort()
                cov_trace_list = np.array(cov_trace_list)[idx]
                time_list = np.array(time_list)[idx]

                if SAVE_TRAJECTORY_HISTORY:
                    idx = np.array(budget_history).argsort()
                    budget_history = np.array(budget_history)[idx]
                    obj_history = np.array(obj_history)[idx]
                    obj2_history = np.array(obj2_history)[idx]
                break

        Budget = int(perf_data[0])+1
        if SAVE_CSV_RESULT:
            if TRAJECTORY_SAMPLING:
                csv_filename = f'result/CSV/Budget_'+str(Budget)+'_ts_'+str(PLAN_STEP)+'_'+str(NUM_SAMPLE_TEST)+'_'+str(SAMPLE_SIZE)+'_'+str(K_SIZE)+'_results.csv'
                csv_filename3 = f'result/CSV3/Budget_'+str(Budget)+'_ts_'+str(PLAN_STEP)+'_'+str(NUM_SAMPLE_TEST)+'_'+str(SAMPLE_SIZE)+'_'+str(K_SIZE)+'_planning_time.csv'
            else:
                csv_filename = f'result/CSV/Budget_'+str(Budget)+'_greedy'+'_'+str(SAMPLE_SIZE)+'_'+str(K_SIZE)+'_results.csv'
                csv_filename3 = f'result/CSV3/Budget_'+str(Budget)+'_greedy'+'_'+str(SAMPLE_SIZE)+'_'+str(K_SIZE)+'_planning_time.csv'
            csv_data = [cov_trace_list]
            csv_data3 = [time_list]
            with open(csv_filename, 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(csv_data)
            with open(csv_filename3, 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(csv_data3)

        if SAVE_TRAJECTORY_HISTORY:
            if TRAJECTORY_SAMPLING:
                csv_filename2 = f'result/CSV2/Budget_'+str(Budget)+'_ts_'+str(PLAN_STEP)+'_'+str(NUM_SAMPLE_TEST)+'_'+str(SAMPLE_SIZE)+'_'+str(K_SIZE)+'_trajectory_result.csv'
            else:
                csv_filename2 = f'result/CSV2/Budget_'+str(Budget)+'_greedy_'+'_'+str(SAMPLE_SIZE)+'_'+str(K_SIZE)+'_trajectory_result.csv'
            new_file = False if os.path.exists(csv_filename2) else True
            field_names = ['budget','obj','obj2']
            with open(csv_filename2, 'a') as csvfile:
                writer = csv.writer(csvfile)
                if new_file:
                    writer.writerow(field_names)
                csv_data = np.concatenate((budget_history.reshape(-1,1), obj_history.reshape(-1,1), obj2_history.reshape(-1,1)), axis=-1)
                writer.writerows(csv_data)
        return cov_trace_list

    except KeyboardInterrupt:
        print(">>> CTRL_C pressed. Killing remote workers")
        # profiler.disable()
        # profiler.print_stats(sort='cumulative')

        for a in meta_agents:
            ray.kill(a)


@ray.remote(num_cpus=8/NUM_META_AGENT, num_gpus=NUM_GPU/NUM_META_AGENT)
class RLRunner(Runner):
    def __init__(self, metaAgentID, seed=0, multigamma=False):
        super().__init__(metaAgentID)
        self.seed = seed

    def singleThreadedJob(self, episodeNumber, budget_range, sample_length, model_idx=0):
        save_img = True # if episodeNumber % SAVE_IMG_GAP == 0 else False
        np.random.seed(self.seed + 100 * episodeNumber)
        print("seed : ", self.seed)
        #torch.manual_seed(SEED + 100 * episodeNumber)
        worker = WorkerTest(self.metaAgentID, self.localNetwork, episodeNumber, budget_range, sample_length, self.device, save_image=save_img, greedy=False, seed=self.seed + 100 * episodeNumber, multigamma=multigamma)
        worker.work(episodeNumber, 0, model_idx)
        perf_metrics = worker.perf_metrics
        return perf_metrics

    def multiThreadedJob(self, episodeNumber, budget_range, sample_length, model_idx=0):
        save_img = True #if (SAVE_IMG_GAP != 0 and episodeNumber % SAVE_IMG_GAP == 0) else False
        #save_img = False
        np.random.seed(self.seed + 100 * episodeNumber)
        #torch.manual_seed(SEED + 100 * episodeNumber)
        worker = WorkerTest(self.metaAgentID, self.localNetwork, episodeNumber, budget_range, sample_length, self.device, save_image=save_img, greedy=False, seed=self.seed + 100 * episodeNumber)
        subworkers = [copy.deepcopy(worker) for _ in range(NUM_SAMPLE_TEST)]
        p = Pool(processes=NUM_SAMPLE_TEST)
        results = []
        for testID, subw in enumerate(subworkers):
            results.append(p.apply_async(subw.work, args=(episodeNumber, testID+1)))
        p.close()
        p.join()
        all_results = []
        best_score = np.inf
        perf_metrics = None
        for res in results:
            metric = res.get()
            all_results.append(metric)
            if metric['cov_trace'] < best_score: # TODO
                perf_metrics = metric
                best_score = metric['cov_trace']
        return perf_metrics

    def job(self, global_weights, episodeNumber, budget_range, sample_length=None, model_idx=0):
        self.set_weights(global_weights)
        metrics = self.singleThreadedJob(episodeNumber, budget_range, sample_length, model_idx)

        info = {
            "id": self.metaAgentID,
            "episode_number": episodeNumber,
        }

        return metrics, info


if __name__ == '__main__':
    ray.init()
    
    device = torch.device('cuda') if USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if USE_GPU else torch.device('cpu')

    result_cov_all = []
    for j in range(len(model_path)):
        result_cov = np.array([])
        if 'MG' in model_path[j]:
            multigamma = True
        else:
            multigamma = False
        global_network = AttentionNet(INPUT_DIM, EMBEDDING_DIM, multigamma=multigamma).to(device)
        checkpoint = torch.load(f'{model_path}/checkpoint.pth')
        result_path_= result_path
        for i in range(5):
            result_cov_ = run_test(seed=SEED+i, global_network=global_network, checkpoint=checkpoint, device=device, local_device=local_device, result_path_=result_path_, model_idx=j)
            # print("result_cov_ shape : ", result_cov_.shape)
            result_cov = np.concatenate([result_cov, result_cov_])
            # pdb.set_trace()
        print("###############################################################")
        print("---------------model path : ", model_path[j], "   FIXED_ENV : ", FIXED_ENV)
        # print("FIXED_ENV : ", FIXED_ENV)
        print("---------------# of trained epi : ", checkpoint['episode'])
        print("---------------result cov : ", result_cov)
        print("---------------avg : ", np.mean(result_cov))
        print("---------------std : ", np.std(result_cov))
        print("---------------max : ", np.max(result_cov))
        print("---------------min : ", np.min(result_cov))
        result_cov_all.append(result_cov)

    print("############# FINAL REPORT #############")
    print("BUDGET_RANGE : ", BUDGET_RANGE)
    print("FIXED_ENV : ", FIXED_ENV)
    print("RANDOM_GAMMA : ", RANDOM_GAMMA)
    print("SPECIFIC_GAMMA : ", SPECIFIC_GAMMA)
    print("DECREASE_GAMMA : ", DECREASE_GAMMA)
    print("FIT_GAMMA : ", FIT_GAMMA)

    
    for i in range(len(result_cov_all)):
        print("###############################################################")
        print("---------------model path : ", model_path[i])
        print("---------------# of trained epi : ", checkpoint['episode'])
        print("---------------result cov : ", result_cov_all[i])
        print("---------------avg : ", np.mean(result_cov_all[i]))
        print("---------------std : ", np.std(result_cov_all[i]))
        print("---------------max : ", np.max(result_cov_all[i]))
        print("---------------min : ", np.min(result_cov_all[i]))
    

