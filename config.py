import os

# set the batch size
BATCH_SIZE = 2**8  #256

REALIZATION = 64

EPOCH = 5

BASE_PATH = "data"

BASE_INPUT = "Comsol_output_Pe_u_Brinkman__Phi2_III__R8192_1.csv"

# set the path to the serialized model after training
INPUT_PATH = os.path.join(BASE_PATH, BASE_INPUT)
# data/Comsol_output_Pe_u_Brinkman__Phi2_III__R8192_1.csv

BASE_SAVE = "models"
# determine name for trained model
SAVE_MODEL_NAME = os.path.join(BASE_SAVE, BASE_INPUT[29:-6])+'_E'+str(EPOCH)
# models/Phi2_III__R8192

