import copy, pdb
import csv
import sys

# sys.path.append('..')
sys.path.append("/data/srujan/research/catnipp")
import os
import ray
import torch
import time
from multiprocessing import Pool
import numpy as np
import time

# from test_attention_net import AttentionNet
from test_attention_robust import AttentionNet, PredictNextBelief, EncoderSimParams
from test_runner import Runner

# from test_worker import WorkerTest
from test_worker_real import WorkerTestReal as WorkerTest
from test_parameters import *

import cProfile
import pprint

# import signal

# def signal_handler(sig, frame):
#     print("\nCtrl+C detected. Exiting.")
#     ray.shutdown()
#     sys.exit(0)


def run_test(
    seed,
    model_idx=0,
    belief_predictor=None,
    belief_checkpoint=None,
    counter=0,
):
    time0 = time.time()
    result_path_ = result_path
    if not os.path.exists(result_path_):
        os.makedirs(result_path_)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE[0])
    device = torch.device('cuda') if USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if USE_GPU else torch.device('cpu')
    global_network = AttentionNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    checkpoint = torch.load(f'{model_path}/checkpoint.pth')
    global_network.load_state_dict(checkpoint["model"])

    belief_predictor = PredictNextBelief(device).to(device)
    belief_checkpoint = torch.load(f"{model_path}/belief_checkpoint.pth")

    # print(f'Loading model: {FOLDER_NAME}...')
    # print(f'##### of episode for training: ', checkpoint['episode'])
    # print(f'Total budget range: {BUDGET_RANGE}')
    # init meta agents
    meta_agents = [RLRunner.remote(i, seed) for i in range(NUM_META_AGENT)]
    weights = (
        global_network.to(local_device).state_dict()
        if device != local_device
        else global_network.state_dict()
    )
    belief_weights = (
        belief_predictor.to(local_device).state_dict()
        if device != local_device
        else belief_predictor.state_dict()
    )
    curr_test = 1 + (counter * NUM_TEST)
    metric_name = [
        "avgrmse",
        "avgunc",
        "avgjsd",
        "avgkld",
        "stdunc",
        "stdjsd",
        "cov_trace",
        "f1",
        "mi",
        "js",
        "rmse",
        "scalex",
        "scaley",
        "scalet",
        "success_rate",
        "planning_time",
        "belief_loss_mean",
        "belief_loss_std",
        "belief_loss_total",
        "belief_loss_list",
    ]
    perf_metrics = {}
    for n in metric_name:
        perf_metrics[n] = []
    cov_trace_list = []
    time_list = []
    episode_number_list = []
    budget_history = []
    obj_history = []
    obj2_history = []
    avg_rmse = []
    cum_rmse = 0.0

    belief_loss = 0.0

    # # signal.signal(signal.SIGINT, signal_handler)
    # profiler = cProfile.Profile()
    # profiler.enable()

    try:
        while True:
            jobList = []
            for i, meta_agent in enumerate(meta_agents):
                jobList.append(
                    meta_agent.job.remote(
                        weights,
                        curr_test,
                        budget_range=BUDGET_RANGE,
                        sample_length=SAMPLE_LENGTH,
                        model_idx=model_idx,
                        belief_predictor_weights=belief_weights,
                    )
                )
                curr_test += 1
            done_id, jobList = ray.wait(jobList, num_returns=NUM_META_AGENT)
            done_jobs = ray.get(done_id)

            for job in done_jobs:
                metrics, info = job
                episode_number_list.append(info["episode_number"])
                # print(">>> Metrics keys : ", metrics.keys())
                cov_trace_list.append(metrics["cov_trace"])
                time_list.append(metrics["planning_time"])
                for n in metric_name:
                    perf_metrics[n].append(metrics[n])
                budget_history += metrics["budget_history"]
                obj_history += metrics["obj_history"]
                obj2_history += metrics["obj2_history"]
                avg_rmse += metrics["avgrmse"]
                cum_rmse = metrics["cum_rmse"]
                belief_loss = metrics["belief_loss_total"]
                belief_loss_list = metrics["belief_loss_list"]
                # print(avg_rmse)

            if curr_test > NUM_TEST:
                print(
                    "#Test sample:",
                    NUM_SAMPLE_TEST,
                    "|#Total test:",
                    NUM_TEST,
                    "|Budget range:",
                    BUDGET_RANGE,
                    "|Sample size:",
                    SAMPLE_SIZE,
                    "|K size:",
                    K_SIZE,
                )
                # print('Avg time per test:', (time.time()-time0)/NUM_TEST)
                perf_data = []
                for n in metric_name:
                    try:
                        perf_data.append(np.nanmean(perf_metrics[n]))
                    except:
                        print("metric_name : ", n)
                        print("perf_metrics : ", perf_metrics[n][0])
                        perf_data.append(np.array(perf_metrics[n][0]))
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
                    avg_rmse = np.array(avg_rmse)[idx]
                break

        Budget = int(perf_data[0]) + 1
        if SAVE_CSV_RESULT:
            if TRAJECTORY_SAMPLING:
                csv_filename = (
                    f"../result/CSV/Budget_"
                    + str(FOLDER_NAME)
                    +'-'
                    + str(Budget)
                    + "_ts_"
                    + str(PLAN_STEP)
                    + "_"
                    + str(NUM_SAMPLE_TEST)
                    + "_"
                    + str(SAMPLE_SIZE)
                    + "_"
                    + str(K_SIZE)
                    + "_results.csv"
                )
                csv_filename3 = (
                    f"../result/CSV/Budget_"
                    + str(FOLDER_NAME)
                    +'-'
                    + str(Budget)
                    + "_ts_"
                    + str(PLAN_STEP)
                    + "_"
                    + str(NUM_SAMPLE_TEST)
                    + "_"
                    + str(SAMPLE_SIZE)
                    + "_"
                    + str(K_SIZE)
                    + "_planning_time.csv"
                )
            else:
                csv_filename = (
                    f"../result/CSV/Budget_"
                    + str(FOLDER_NAME)
                    +'-'
                    + str(Budget)
                    + "_greedy"
                    + "_"
                    + str(SAMPLE_SIZE)
                    + "_"
                    + str(K_SIZE)
                    + "_results.csv"
                )
                csv_filename3 = (
                    f"../result/CSV/Budget_"
                    + str(FOLDER_NAME)
                    +'-'
                    + str(Budget)
                    + "_greedy"
                    + "_"
                    + str(SAMPLE_SIZE)
                    + "_"
                    + str(K_SIZE)
                    + "_planning_time.csv"
                )
            csv_data = [cov_trace_list]
            csv_data3 = [time_list]
            with open(csv_filename, "a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(csv_data)
            with open(csv_filename3, "a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(csv_data3)

        if SAVE_TRAJECTORY_HISTORY:
            if TRAJECTORY_SAMPLING:
                csv_filename2 = (
                    f"../result/CSV/Budget_"
                    + str(FOLDER_NAME)
                    +'-'
                    + str(Budget)
                    + "_ts_"
                    + str(PLAN_STEP)
                    + "_"
                    + str(NUM_SAMPLE_TEST)
                    + "_"
                    + str(SAMPLE_SIZE)
                    + "_"
                    + str(K_SIZE)
                    + "_trajectory_result.csv"
                )
            else:
                csv_filename2 = (
                    f"../result/CSV/Budget_"
                    + str(FOLDER_NAME)
                    +'-'
                    + str(Budget)
                    + "_greedy_"
                    + "_"
                    + str(SAMPLE_SIZE)
                    + "_"
                    + str(K_SIZE)
                    + "_trajectory_result.csv"
                )
            new_file = False if os.path.exists(csv_filename2) else True
            field_names = ["budget", "obj", "obj2"]
            with open(csv_filename2, "a") as csvfile:
                writer = csv.writer(csvfile)
                if new_file:
                    writer.writerow(field_names)
                csv_data = np.concatenate(
                    (
                        budget_history.reshape(-1, 1),
                        obj_history.reshape(-1, 1),
                        obj2_history.reshape(-1, 1),
                    ),
                    axis=-1,
                )
                writer.writerows(csv_data)
        # return cov_trace_list
        # return obj2_history
        return cum_rmse, obj2_history, belief_loss, belief_loss_list, cov_trace_list

    except KeyboardInterrupt:
        print(">>> CTRL_C pressed. Killing remote workers")
        # profiler.disable()
        # profiler.print_stats(sort='cumulative')

        for a in meta_agents:
            ray.kill(a)


@ray.remote(num_cpus=1 / NUM_META_AGENT, num_gpus=NUM_GPU / NUM_META_AGENT)
class RLRunner(Runner):
    def __init__(self, metaAgentID, seed=0):
        super().__init__(metaAgentID)
        self.seed = seed

    def singleThreadedJob(
        self, episodeNumber, budget_range, sample_length, model_idx=0
    ):
        save_img = False # episodeNumber % SAVE_IMG_GAP == 0 else False
        np.random.seed(self.seed + episodeNumber*100)
        print("seed : ", self.seed)
        # torch.manual_seed(SEED + 100 * episodeNumber)
        worker = WorkerTest(
            self.metaAgentID,
            self.localNetwork,
            episodeNumber,
            budget_range,
            sample_length,
            self.device,
            save_image=save_img,
            greedy=False,
            seed=self.seed + episodeNumber*100,
            belief_predictor=self.belief_predictor,
        )
        worker.work(episodeNumber, 0, model_idx)
        perf_metrics = worker.perf_metrics
        return perf_metrics

    def multiThreadedJob(self, episodeNumber, budget_range, sample_length, model_idx=0):
        save_img = (
            False #True if (SAVE_IMG_GAP != 0 and episodeNumber % SAVE_IMG_GAP == 0) else False
        )
        # save_img = False
        np.random.seed(self.seed + 100 * episodeNumber)
        # torch.manual_seed(SEED + 100 * episodeNumber)
        worker = WorkerTest(
            self.metaAgentID,
            self.localNetwork,
            episodeNumber,
            budget_range,
            sample_length,
            self.device,
            save_image=save_img,
            greedy=False,
            seed=self.seed + 100 * episodeNumber,
        )
        subworkers = [copy.deepcopy(worker) for _ in range(NUM_SAMPLE_TEST)]
        p = Pool(processes=NUM_SAMPLE_TEST)
        results = []
        for testID, subw in enumerate(subworkers):
            results.append(p.apply_async(subw.work, args=(episodeNumber, testID + 1)))
        p.close()
        p.join()
        all_results = []
        best_score = np.inf
        perf_metrics = None
        for res in results:
            metric = res.get()
            all_results.append(metric)
            if metric["cov_trace"] < best_score:  # TODO
                perf_metrics = metric
                best_score = metric["cov_trace"]
        return perf_metrics

    def job(
        self,
        global_weights,
        episodeNumber,
        budget_range,
        sample_length=None,
        model_idx=0,
        belief_predictor_weights=None,
    ):
        self.set_weights(global_weights)
        self.set_belief_predictor_weights(belief_predictor_weights)

        metrics = self.singleThreadedJob(
            episodeNumber, budget_range, sample_length, model_idx
        )

        info = {
            "id": self.metaAgentID,
            "episode_number": episodeNumber,
        }

        return metrics, info


if __name__ == "__main__":
    # # ray.init(num_cpus=NUM_META_AGENT)
    # # ray.init()
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE[0])
    # ray.init(num_cpus=NUM_META_AGENT)
    # # torch.cuda.set_device(CUDA_DEVICE[0])
    # pprint.pprint(ray.cluster_resources())
    # pprint.pprint(os.environ["CUDA_VISIBLE_DEVICES"])

    # device = torch.device("cuda") if USE_GPU_GLOBAL else torch.device("cpu")
    # local_device = torch.device("cuda") if USE_GPU else torch.device("cpu")

    # result_RMSE_all = []
    # result_cumRMSE_all = []
    # result_belief_loss_all = []
    # result_belief_loss_list_all = []
    # for j in range(1):
    #     result_RMSE = np.array([])
    #     # result_cumRMSE = np.array([])
    #     global_network = AttentionNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    #     checkpoint = torch.load(f"{model_path}/checkpoint.pth")

    #     belief_predictor = PredictNextBelief(device).to(device)
    #     belief_checkpoint = torch.load(f"{model_path}/belief_checkpoint.pth")

    #     result_path_ = result_path
    #     for i in range(1):
    #         result_cumRMSE_, result_rmse_, belief_loss, belief_loss_list = run_test(
    #             seed=SEED + i + 5,
    #             global_network=global_network,
    #             checkpoint=checkpoint,
    #             device=device,
    #             local_device=local_device,
    #             result_path_=result_path_,
    #             model_idx=j,
    #             belief_predictor=belief_predictor,
    #             belief_checkpoint=belief_checkpoint,
    #         )
    #         # print("result_rmse_ shape : ", result_rmse_, result_cumRMSE_)
    #         result_RMSE = np.concatenate([result_RMSE, result_rmse_])
    #         result_cumRMSE_all.append(result_cumRMSE_)
    #         result_belief_loss_all.append(belief_loss)
    #         result_belief_loss_list_all.append(belief_loss_list)
    #         # print(belief_loss_list)

    #         # result_cumRMSE = np.concatenate([result_cumRMSE, result_cumRMSE_])
    #         # pdb.set_trace()
    #     print("###############################################################")
    #     print(
    #         "---------------model path : ", model_path[j], "   FIXED_ENV : ", FIXED_ENV
    #     )
    #     # print("FIXED_ENV : ", FIXED_ENV)
    #     print("---------------# of trained epi : ", checkpoint["episode"])
    #     # print("---------------result rmse, avg RMSE : ", result_RMSE)
    #     print("---------------avg of all seeds final RMSE: ", np.mean(result_RMSE))
    #     print("---------------std : ", np.std(result_RMSE))
    #     print("---------------max : ", np.max(result_RMSE))
    #     print("---------------min : ", np.min(result_RMSE))

    #     # print("---------------mean cum_RMSE of all seeds: ", np.mean(result_cumRMSE))
    #     # print("---------------std cum_RMSE of all seeds: ", np.std(result_cumRMSE))
    #     # print("---------------max cum_RMSE of all seeds: ", np.max(result_cumRMSE))
    #     # print("---------------min cum_RMSE of all seeds: ", np.min(result_cumRMSE))
    #     result_RMSE_all.append(result_RMSE)

    # result_cumRMSE_all = np.array(result_cumRMSE_all)
    # result_belief_loss_all = np.array(result_belief_loss_all)
    # # print("result_belief_loss_list_all : ", result_belief_loss_list_all)
    # for i in result_belief_loss_list_all:
    #     print("shape : ", len(i))
    # # result_belief_loss_list_all = np.array(result_belief_loss_list_all)

    # # print("############# FINAL REPORT #############")
    # # print("BUDGET_RANGE : ", BUDGET_RANGE)
    # # print("FIXED_ENV : ", FIXED_ENV)
    # # print("RANDOM_GAMMA : ", RANDOM_GAMMA)
    # # print("SPECIFIC_GAMMA : ", SPECIFIC_GAMMA)
    # # print("DECREASE_GAMMA : ", DECREASE_GAMMA)
    # # print("FIT_GAMMA : ", FIT_GAMMA)

    # for i in range(len(result_RMSE_all)):
    #     print("###############################################################")
    #     # print("---------------model path : ", model_path[i])
    #     print("---------------# of trained epi : ", checkpoint["episode"])
    #     print("---------------avg of all seeds final RMSE: ", result_RMSE_all[i])
    #     print("---------------avg : ", np.mean(result_RMSE_all[i]))
    #     print("---------------std : ", np.std(result_RMSE_all[i]))
    #     print("---------------max : ", np.max(result_RMSE_all[i]))
    #     print("---------------min : ", np.min(result_RMSE_all[i]))

    # print("###############################################################")
    # print("---------------cum_RMSE of all seeds: ", result_cumRMSE_all)
    # print("---------------mean cum_RMSE of all seeds: ", np.mean(result_cumRMSE_all))
    # print("---------------std cum_RMSE of all seeds: ", np.std(result_cumRMSE_all))
    # print("---------------max cum_RMSE of all seeds: ", np.max(result_cumRMSE_all))
    # print("---------------min cum_RMSE of all seeds: ", np.min(result_cumRMSE_all))
    # print("###############################################################")
    # print("---------------belief_loss of all seeds: ", result_belief_loss_all)
    # print(
    #     "---------------mean belief_loss of all seeds: ",
    #     np.mean(result_belief_loss_all),
    # )
    # print(
    #     "---------------std belief_loss of all seeds: ", np.std(result_belief_loss_all)
    # )
    # print(
    #     "---------------max belief_loss of all seeds: ", np.max(result_belief_loss_all)
    # )
    # print(
    #     "---------------min belief_loss of all seeds: ", np.min(result_belief_loss_all)
    # )
    # # with np.printoptions(precision=3, suppress=True):
    # #     print("belief_loss_list_all : ", result_belief_loss_list_all)


    result_rmse = []
    cov_trace_list = []
    result_cumRMSE = []
    for i in range(4):
        ray.init()
        result_cumRMSE_, result_rmse_, _, _,  cov_trace_list_ = run_test(
                seed=SEED+i,
                # global_network=global_network,
                # checkpoint=checkpoint,
                # device=device,
                # local_device=local_device,
                # result_path_=result_path_,
                model_idx=i,
                counter=i
            )
        
        ray.shutdown()
        result_rmse.extend(result_rmse_)
        result_cumRMSE.append(result_cumRMSE_)
        cov_trace_list.extend(cov_trace_list_)
    
    print("###############################################################")

    print("---------------avg of all seeds final RMSE: ", np.mean(result_rmse))
    print("---------------std : ", np.std(result_rmse))
    print("---------------max : ", np.max(result_rmse))
    print("---------------min : ", np.min(result_rmse))

    print("---------------result cov_trace_list, avg cov_trace_list : ", cov_trace_list)
    print("---------------avg of all seeds final cov_trace_list: ", np.mean(cov_trace_list))
    print("---------------std : ", np.std(cov_trace_list))
    print("---------------max : ", np.max(cov_trace_list))

    print("###############################################################")
    # print("---------------cum_RMSE of all seeds: ", result_cumRMSE_)
    print("---------------mean cum_RMSE of all seeds: ", np.mean(result_cumRMSE))
    print("---------------std cum_RMSE of all seeds: ", np.std(result_cumRMSE))
    print("---------------max cum_RMSE of all seeds: ", np.max(result_cumRMSE))
    print("---------------min cum_RMSE of all seeds: ", np.min(result_cumRMSE))