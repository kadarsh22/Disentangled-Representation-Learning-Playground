from disentanglement_lib.data.ground_truth import ground_truth_data
from disentanglement_lib.data.ground_truth import util
from tensorflow import gfile
import numpy as np
from sklearn import linear_model
import torch
from sklearn.model_selection import GridSearchCV
from torch.autograd import Variable

device_id = 0
def compute_beta_vae_sklearn(ground_truth_data,representation_function,random_state,batch_size,num_train,num_eval):
  
    train_points, train_labels = _generate_training_batch(ground_truth_data, representation_function, batch_size, num_train,random_state)
    eval_points, eval_labels = _generate_training_batch(ground_truth_data, representation_function, batch_size, num_eval,random_state)
    
    parameters = {"C":[0.001,0.01,0.1,1]}
    
    model = linear_model.LogisticRegression(random_state=random_state)
    model_cv = GridSearchCV(model,parameters,cv=10)
    model_cv.fit(train_points, train_labels)

    best_c = model_cv.best_params_['C']
    
    model_test = linear_model.LogisticRegression(C= best_c,random_state=random_state)
    model_test.fit(train_points, train_labels)
    eval_accuracy = model_test.score(eval_points, eval_labels)

    scores_dict = {}
    scores_dict["train_accuracy"] = model_cv.best_score_
    scores_dict["eval_accuracy"] = eval_accuracy
    return scores_dict

def _generate_training_batch(ground_truth_data, representation_function,batch_size, num_points, random_state):

    points = None  # Dimensionality depends on the representation function.
    labels = np.zeros(num_points, dtype=np.int64)
    for i in range(num_points):
        labels[i], feature_vector = _generate_training_sample(
        ground_truth_data, representation_function, batch_size, random_state)
        if points is None:
            points = np.zeros((num_points, feature_vector.shape[0]))
        points[i, :] = feature_vector
    return points, labels


def _generate_training_sample(ground_truth_data, representation_function,
                              batch_size, random_state):

    # Select random coordinate to keep fixed.
    index = random_state.randint(low=1,high=3)
    # Sample two mini batches of latent variables.
    factors1 = ground_truth_data.sample_factors(batch_size, random_state)
    factors2 = ground_truth_data.sample_factors(batch_size, random_state)
    # Ensure sampled coordinate is the same across pairs of samples.
    factors2[:, index] = factors1[:, index]
    # Transform latent variables to observation space.
    observation1 = ground_truth_data.sample_observations_from_factors(
      factors1, random_state)
    observation2 = ground_truth_data.sample_observations_from_factors(
      factors2, random_state)
    # Compute representations based on the observations.
    representation1,_ = representation_function.encoder(Variable(torch.from_numpy(observation1)).cuda(device_id))
    representation2,_ = representation_function.encoder(Variable(torch.from_numpy(observation2)).cuda(device_id))
    representation1 = representation1.data.cpu().numpy()
    representation2 = representation2.data.cpu().numpy()
    # Compute the feature vector based on differences in representation.
    feature_vector = np.mean(np.abs(representation1 - representation2), axis=0)
    return index, feature_vector