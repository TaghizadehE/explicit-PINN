import importlib
from datasets import prepare_learning_data, split_shuffle_data
from train import training_run
# importlib.reload(prepare_learning_data)

features_norm, values, features = prepare_learning_data()
X_train, X_test, Y_train, Y_test = split_shuffle_data(features_norm, values)
training_run(X_train, X_test, Y_train, Y_test, features_norm, values)

print('Execution finished!')