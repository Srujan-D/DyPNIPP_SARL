# INPUT_DIM = 4
# EMBEDDING_DIM = 128
# USE_GPU = False
# USE_GPU_GLOBAL = True
CUDA_DEVICE = [1]
# NUM_GPU = 1
# NUM_META_AGENT = 1
# GAMMA = 1
# MULTI_GAMMA = None
# FOLDER_NAME = 'ipp'
# model_path = f'../model/{FOLDER_NAME}'
# result_path = f'result/{FOLDER_NAME}'

# SEED = 1
# NUM_TEST = 1
# TRAJECTORY_SAMPLING = True
# PLAN_STEP = 15
# NUM_SAMPLE_TEST = 4 # do not exceed 99
# SAVE_IMG_GAP = 1
# SAVE_CSV_RESULT = False
# SAVE_TRAJECTORY_HISTORY = False
# SAVE_TIME_RESULT = False

# BUDGET_RANGE = (9.99999, 10)
# SAMPLE_SIZE = 400
# K_SIZE = 20
# SAMPLE_LENGTH = 0.2 # 0/None: sample at nodes

# # addition by woojun
FIXED_ENV= 2

INPUT_DIM = 4
EMBEDDING_DIM = 128
USE_GPU = True
USE_GPU_GLOBAL = True
NUM_GPU = 2
NUM_META_AGENT = 1
GAMMA = 1
# FOLDER_NAME = 'ipp-4heads'
FOLDER_NAME = 'ipp'
model_path = f'../model/{FOLDER_NAME}'
result_path = f'result/{FOLDER_NAME}'
# /data/srujan/research/catnipp/model/ipp-4heads/best_model_checkpoint.pth
SEED = 5
NUM_TEST = 1000
TRAJECTORY_SAMPLING = False
PLAN_STEP = 15
NUM_SAMPLE_TEST = 1 # do not exceed 99
SAVE_IMG_GAP = 1
SAVE_CSV_RESULT = False
SAVE_TRAJECTORY_HISTORY = False
SAVE_TIME_RESULT = False
BUDGET_RANGE = (5, 10)
SAMPLE_SIZE = 400
K_SIZE = 20
SAMPLE_LENGTH = 0.2 # 0/None: sample at nodes
ADAPTIVE_TH = 0.4