# import the necessary packages
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Activation, Input, Flatten, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import Callback, CSVLogger, ModelCheckpoint



def build_mlp_model(X_train):
    # building model
    model = Sequential()
    print('number of features =', X_train.shape[1])
    model.add(Dense(1024, input_dim=X_train.shape[1], activation='relu', kernel_initializer='uniform'))
    model.add(Dense(512, activation='relu',   kernel_initializer='uniform'))
    model.add(Dense(256, activation='relu',   kernel_initializer='uniform'))
    model.add(Dense(64,  activation='relu',   kernel_initializer='uniform'))
    model.add(Dense(16,  activation='relu',   kernel_initializer='uniform'))
    model.add(Dense(1,   activation='linear', kernel_initializer='uniform'))
#         self.model_summary = model.summary()
    return model