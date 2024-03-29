INPUT_DIM = 4
EMBEDDING_DIM = 128
USE_GPU = False
USE_GPU_GLOBAL = True
NUM_GPU = 1
CUDA_DEVICE = [0] #[0, 1, 2, 3]
NUM_META_AGENT = 1

# BUDGET_RANGE = (7.99999, 8) #(9.99999, 10)


################################################################
GAMMA = 1
BUDGET_TRAINED = 10
BUDGET_RANGE = (9.99999, 10)
# BUDGET_RANGE = (1.49999, 1.5)  # (6, 8)   (2, 4)  (4, 6)  (6, 8) (8, 10)
MULTI_GAMMA = None #  [0.2, 0.6, 0.8, 0.9]  # [0, 0.2, 0.6, 0.8 , 0.9, 0.99] #  [0.2, 0.6, 0.8, 0.9]  # [0, 0.1, 0.5, 0.9, 0.99]  #  [0.1, 0.5, 0.9, 0.99] # [0.1, 0.5, 0.9, 0.99]  None

if MULTI_GAMMA is not None:
    GAMMA = MULTI_GAMMA[-1]


RANDOM_GAMMA = None
SPECIFIC_GAMMA = 3
DECREASE_GAMMA = None
FIT_GAMMA = None


SEED = 2

FIXED_ENV= 2  # None 1 2 3 
################################################################

# FOLDER_NAME = 'ipp'  # ipp-4heads  ipp  ipp-4heads
# FOLDER_NAME = FOLDER_NAME + str(GAMMA) + '_bud' + str(BUDGET_TRAINED)
# if MULTI_GAMMA is not None:
#     FOLDER_NAME = FOLDER_NAME + '_MG' + str(len(MULTI_GAMMA))
# FOLDER_NAME = FOLDER_NAME + '_S' + str(SEED)
# print("FOLDER_NAME : ", FOLDER_NAME)


FOLDER_NAME_ALL = []
# FOLDER_NAME_ALL.append('ipp0.0_bud8_S1')
# FOLDER_NAME_ALL.append('ipp0.0_bud8_S2')

# FOLDER_NAME_ALL.append('ipp0.2_bud8_S1')
# FOLDER_NAME_ALL.append('ipp0.6_bud8_S1')
# FOLDER_NAME_ALL.append('ipp0.8_bud8_S1')

# FOLDER_NAME_ALL.append('ipp0.8_bud8_S2')
# FOLDER_NAME_ALL.append('ipp0.9_bud8_S0')
# FOLDER_NAME_ALL.append('ipp0.9_bud8_S1')

# FOLDER_NAME_ALL.append('ipp0.99_bud8_S1')
# FOLDER_NAME_ALL.append('ipp0.99_bud8_S2')
# FOLDER_NAME_ALL.append('ipp0.99_bud8_S3')

# FOLDER_NAME_ALL.append('ipp0.99_bud8_MG5_S1')

# FOLDER_NAME_ALL.append('ipp0.99_bud8_MG6_S5')
# FOLDER_NAME_ALL.append('ipp0.99_bud8_MG6_SG_S5')
# FOLDER_NAME_ALL.append('ipp0.99_bud8_MG6_FG_S2')
# FOLDER_NAME_ALL.append('ipp0.99_bud8_MG6_SG_FG_S2')
# FOLDER_NAME_ALL.append('ipp0.99_bud8_MG6_RG_S100')

# FOLDER_NAME_ALL.append('ipp0.99_bud8_MG4_RG_S100')
# FOLDER_NAME_ALL.append('ipp0.9_bud8_MG4_SPG3_S100')
# FOLDER_NAME_ALL.append('ipp0.9_bud8_MG4_VT_S1')

# FOLDER_NAME_ALL.append('ipp0.9_bud8_MG4_TO_S1')
# # FOLDER_NAME_ALL.append('ipp0.9_bud8_MG4_SG_TO_S1')






# model_path, result_path = [], []
# for name in FOLDER_NAME_ALL:
#     model_path.append(f'./model_final/{name}')
#     result_path.append(f'result/{name}')

# print("test for : ", model_path)

FOLDER_NAME = 'ipp'
model_path = f'../model/{FOLDER_NAME}'
result_path = f'result/{FOLDER_NAME}'



# model_path = f'./model/{FOLDER_NAME}'
# result_path = f'result/{FOLDER_NAME}'



NUM_TEST = 1
TRAJECTORY_SAMPLING = False
PLAN_STEP = 15
NUM_SAMPLE_TEST = 4 # do not exceed 99
SAVE_IMG_GAP = 10000000 #1
SAVE_CSV_RESULT = False
SAVE_TRAJECTORY_HISTORY = False
SAVE_TIME_RESULT = False


SAMPLE_SIZE = 400
K_SIZE = 20
SAMPLE_LENGTH = 0.2 # 0/None: sample at nodes
ADAPTIVE_AREA = True
ADAPTIVE_TH = 0.1