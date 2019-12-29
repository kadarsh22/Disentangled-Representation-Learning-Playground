from disentanglement_lib.data.ground_truth import util
from tensorflow import gfile
import numpy as np
import torch
from torch.autograd import Variable
import utils
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
import scipy

def compute_dci(ground_truth_data, representation_function, random_state,num_train,num_test,batch_size=70):

    scores = {}
    mus_train, ys_train = utils.generate_batch_factor_code(ground_truth_data, representation_function, num_train,random_state, batch_size)
    assert mus_train.shape[1] == num_train
    assert ys_train.shape[1] == num_train
    
    mus_test, ys_test = utils.generate_batch_factor_code(
      ground_truth_data, representation_function, num_test,
      random_state, batch_size)
    
    importance_matrix, train_err, test_err = compute_importance_gbt(
      mus_train, ys_train, mus_test, ys_test)
    assert importance_matrix.shape[0] == mus_train.shape[0]
    assert importance_matrix.shape[1] == ys_train.shape[0]
    
    scores["informativeness_train"] = train_err
    scores["informativeness_test"] = test_err
    scores["disentanglement"] = disentanglement(importance_matrix)
    scores["completeness"] = completeness(importance_matrix)
    return scores

def compute_importance_gbt(x_train, y_train, x_test, y_test):
    """Compute importance based on gradient boosted trees."""
    num_factors = y_train.shape[0]
    num_codes = x_train.shape[0]
    importance_matrix = np.zeros(shape=[num_codes, num_factors],
                               dtype=np.float64)
    train_loss = []
    test_loss = []
    for i in range(1,num_factors):
        parameters = {"alpha":[0.01,0.1,1]}
        model_xg = Lasso()
        model = GridSearchCV(model_xg,parameters,cv=5)
        model.fit(x_train.T, y_train[i, :])
        final_model = Lasso(alpha = model.best_params_['alpha'])
        final_model.fit(x_train.T, y_train[i, :])
        importance_matrix[:, i] = np.abs(final_model.coef_ )
        train_loss.append(np.mean(final_model.predict(x_train.T) == y_train[i, :]))
        test_loss.append(np.mean(final_model.predict(x_test.T) == y_test[i, :]))
    return importance_matrix, np.mean(train_loss), np.mean(test_loss)


def disentanglement_per_code(importance_matrix):
  # importance_matrix is of shape [num_codes, num_factors].
    return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11,
                                  base=importance_matrix.shape[1])


def disentanglement(importance_matrix):
    """Compute the disentanglement score of the representation."""
    per_code = disentanglement_per_code(importance_matrix)
    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)
    code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

    return np.sum(per_code*code_importance)

def completeness_per_code(importance_matrix):
    """Compute completeness of each code."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1. - scipy.stats.entropy(importance_matrix + 1e-11,
                                      base=importance_matrix.shape[0])


def completeness(importance_matrix):
    """"Compute completeness of the representation."""
    per_factor = completeness_per_code(importance_matrix)
    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)
    factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
    return np.sum(per_factor*factor_importance)
