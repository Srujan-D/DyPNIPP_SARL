INPUT_DIM = 4
EMBEDDING_DIM = 128
USE_GPU = True
USE_GPU_GLOBAL = True
NUM_GPU = 1
CUDA_DEVICE = [2] #[0, 1, 2, 3]
NUM_META_AGENT = 1

# BUDGET_RANGE = (7.99999, 8) #(9.99999, 10)


################################################################
GAMMA = 1
BUDGET_TRAINED = 10
BUDGET_RANGE = (14.9999, 15)
# BUDGET_RANGE = (6.9999, 7)
# BUDGET_RANGE = (10.9999, 11)
# BUDGET_RANGE = (1.49999, 1.5)  # (6, 8)   (2, 4)  (4, 6)  (6, 8) (8, 10)
MULTI_GAMMA = None #  [0.2, 0.6, 0.8, 0.9]  # [0, 0.2, 0.6, 0.8 , 0.9, 0.99] #  [0.2, 0.6, 0.8, 0.9]  # [0, 0.1, 0.5, 0.9, 0.99]  #  [0.1, 0.5, 0.9, 0.99] # [0.1, 0.5, 0.9, 0.99]  None

if MULTI_GAMMA is not None:
    GAMMA = MULTI_GAMMA[-1]


RANDOM_GAMMA = None
SPECIFIC_GAMMA = 3
DECREASE_GAMMA = None
FIT_GAMMA = None


SEED = 4912

FIXED_ENV= 2  # None 1 2 3 
################################################################

# FOLDER_NAME = 'ipp'  # ipp-4heads  ipp  ipp-4heads
# FOLDER_NAME = FOLDER_NAME + str(GAMMA) + '_bud' + str(BUDGET_TRAINED)
# if MULTI_GAMMA is not None:
#     FOLDER_NAME = FOLDER_NAME + '_MG' + str(len(MULTI_GAMMA))
# FOLDER_NAME = FOLDER_NAME + '_S' + str(SEED)
# print("FOLDER_NAME : ", FOLDER_NAME)


FOLDER_NAME_ALL = []


# model_path, result_path = [], []
# for name in FOLDER_NAME_ALL:
#     model_path.append(f'./model_final/{name}')
#     result_path.append(f'result/{name}')

# print("test for : ", model_path)

FOLDER_NAME = 'catnipp_3fires'
model_path = f'../model/{FOLDER_NAME}'
# model_path = 'result/veg_10'
result_path = f'result/{FOLDER_NAME}'



# model_path = f'./model/{FOLDER_NAME}'
# result_path = f'result/{FOLDER_NAME}'



NUM_TEST = 1
TRAJECTORY_SAMPLING = False
PLAN_STEP = 15
NUM_SAMPLE_TEST = 4 # do not exceed 99
SAVE_IMG_GAP = 15 #1
SAVE_CSV_RESULT = True
SAVE_TRAJECTORY_HISTORY = False
SAVE_TIME_RESULT = True


SAMPLE_SIZE = 200
K_SIZE = 20
SAMPLE_LENGTH = 0.1 # 0/None: sample at nodes
ADAPTIVE_AREA = True
ADAPTIVE_TH = 0.2