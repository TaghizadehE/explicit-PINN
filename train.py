import numpy as np
import config
from models import build_mlp_model
from save_results import saveLossHistory, saveCummLoss
from show_historyPrediction import showHistory, showPredictionXY, showPredictionSurf
from predictions import prediction_train_test, mse_test
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, CSVLogger,  ModelCheckpoint


def training_run(X_train, X_test, Y_train, Y_test, features_norm, values):
    model = get_model(X_train)
    history = train(model, X_train, Y_train, X_test, Y_test)

    predictions_train      = prediction_train_test(model, X_train, Y_train, 'only_train_data')
    predictions_test       = prediction_train_test(model, X_test,  Y_test,  'only_test_data')
    predictions_train_test = prediction_train_test(model, features_norm, values, 'train_test_data')
    MSE_test               = mse_test(model, X_test,  Y_test)

    showHistory(history, model)

    showPredictionXY(Y_train, predictions_train, Y_test, predictions_test)

    showPredictionSurf()
    
    #==================================================================================
    # very important: uncheck this onlt if you are saving the best mse with iterions
    save_model(model, config.SAVE_MODEL_NAME+'_final')
    #==================================================================================
    

def get_model(X_train):
    model = build_mlp_model(X_train)
    optimizer = Adam(lr=1e-3, decay=1e-3 / 200)
    # loss inside the compile is important, not metrics
    # metrics is only for printing purposes 
    model.compile(loss="mean_absolute_percentage_error", optimizer=optimizer, metrics=['mse','accuracy'])
    return model


def train(model, X_train, Y_train, X_test, Y_test):
    checkpoint = ModelCheckpoint(config.SAVE_MODEL_NAME, 
                                 monitor='mse', 
                                 verbose=0, 
                                 save_best_only=True,
                                 mode= 'auto')
    
    callbacks_list = [TestCallback((X_test, Y_test)), checkpoint]

    history = model.fit(x=X_train, 
                        y=Y_train, 
                        batch_size=config.BATCH_SIZE, 
                        verbose=False, 
                        epochs=config.EPOCH,
                        callbacks=callbacks_list, 
                        validation_split=0.05, 
                        shuffle=True)

    loss_history = history.history["loss"]
    saveLossHistory(loss_history)
    
    return history


class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data
#         self.model = model
        
    def on_epoch_begin(self, epoch, logs={}):
        X_test, Y_test = self.test_data
        predictions = self.model.predict(X_test)
        difference = predictions.flatten() - Y_test
        percent_difference = (difference / Y_test) * 100
        abs_percent_difference = np.abs(percent_difference)

        mean = np.mean(abs_percent_difference)
        std = np.std(abs_percent_difference)
        return
    

def save_model(model, name):
    model.save(name)

def load_model(name):
    model = tensorflow.keras.models.load_model(name)  
    return model