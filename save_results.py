import numpy as np
from datetime import datetime
from contextlib import redirect_stdout
import config

def saveLossHistory(loss_history):
    np.savetxt('model_history/loss_history_E'+str(config.EPOCH)+'.txt', loss_history, delimiter=",")
    return


def saveCummLoss(model, mean, std, case):
    ftxt=open('model_history/cummulative_loss_E'+str(config.EPOCH)+'.txt', 'a') # append mode 
    if case == 'only_train_data':
        print('---------- Prediction on Y_train (only)------')
        ftxt.write('-------------------------------------------------------------------\n')
        with redirect_stdout(ftxt):
            model.summary()
#             ftxt.write(model.summary(print_fn=lambda x: (x + '\n')))
        ftxt.write('\n')
        ftxt.write(config.INPUT_PATH)
        ftxt.write('  ('+str(datetime.now())+') ')
        ftxt.write('\n')
        ftxt.write(str(config.EPOCH))
        ftxt.write('\n')
        ftxt.write('Prediction on Y_train:\n')
        ftxt.write("Mean: {:.2f}%, std: {:.6f}%".format(mean, std))
    elif case == 'only_test_data':
        print('---------- Prediction on Y_test (only)------')
        ftxt.write('Prediction on Y_test:\n')
        ftxt.write("Mean: {:.2f}%, std: {:.6f}%".format(mean, std))
    elif case == 'train_test_data':
        print('---------- Prediction on Y_train_test (both)------')
        ftxt.write('Prediction on Y_train_test:\n')
        ftxt.write("Mean: {:.2f}%, std: {:.6f}%".format(mean, std))
    else:
        print('something is wrong')           

    ftxt.write('\n')
    ftxt.close()
    return
    
    
def pp():
    ftxt=open('model_history/datasize_loss_MSE.txt', 'a') # append mode 
    ftxt.write('------------------------------------------\n')
    ftxt.write("validation mse at last epoch for datasize of {}".format(self.dataslicing))
    ftxt.write(" = {:.16f}".format(np.array(history.history['val_mse'])[-1]))
    ftxt.write('\n')
    ftxt.close()