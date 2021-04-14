import numpy as np
import pandas as pd
import config
from Auxiliary import relz, prepare_FE_data
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def prepare_learning_data():
    # features sequence: (1) epsl, (2) phi2, (3)Pe, (4)Dr, (5)c_f_avg_norm, (6)d_f_c_avg_norm,  
    # target: (7) eta_d_f
    
    # The actual number of realizatioins is 4x of following value, nb_realization
    
    nb_realization = relz(config.REALIZATION)
#     nb_epoch = config.EPOCH
#     batch_size = config.BATCH_SIZE
    input_file_name = config.INPUT_PATH
    
    print(input_file_name, '===> loaded for training')

    time_now = datetime.now()

    data = prepare_FE_data(config.INPUT_PATH)

    features = data[:,:-1]
    values   = data[:,-1]


    data_norm = normalize_data(data)
    features_norm = data_norm[:,:-1]
    values_norm = data_norm[:,-1]

    return features_norm, values, features


def split_shuffle_data(features, values):
    # Split the data
    (X_train, X_test, Y_train, Y_test) = train_test_split(features, values, test_size=0.2, random_state=42)
    print('train size', X_train.shape[0], Y_train.shape, 'which is:', (X_train.shape[0]/features.shape[0])*100, '% of data')
    print('test  size',  X_test.shape[0],  Y_test.shape, 'which is:', ( X_test.shape[0]/features.shape[0])*100, '% of data')
    return X_train, X_test, Y_train, Y_test


scaler = MinMaxScaler(feature_range=(0, 1))

def normalize_data(X):
    # Normalization
    scaler.fit(X)
    normalized_X = scaler.transform(X)
    return normalized_X