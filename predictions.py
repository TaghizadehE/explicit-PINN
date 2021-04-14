import numpy as np
from save_results import saveCummLoss

def prediction_train_test(model, X, Y, case):
    prediction = model.predict(X)
    difference  = prediction.flatten() - Y
    percent_difference = (difference / Y) * 100
    abs_percent_difference = np.abs(percent_difference)
    mean = np.mean(abs_percent_difference)
    std  = np.std(abs_percent_difference)
    
    saveCummLoss(model, mean, std, case)
    
    print("Mean: {:.5f}%, std: {:.5f}%".format(mean, std))
    return prediction


def mse_test(model, X_test,  Y_test):
    prediction_test = model.predict(X_test)
    difference = prediction_test.flatten() - Y_test
    MSE_test = np.mean(difference**2)

    print("MSE_test: {:.32f}".format(MSE_test))

    return MSE_test

