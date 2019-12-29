# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions that are useful for the different metrics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from six.moves import range
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
import gin.tf
import torch
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_batch_factor_code(ground_truth_data, representation_function,
                               num_points, random_state, batch_size):
  """Sample a single training sample based on a mini-batch of ground-truth data.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observation as input and
      outputs a representation.
    num_points: Number of points to sample.
    random_state: Numpy random state used for randomness.
    batch_size: Batchsize to sample points.

  Returns:
    representations: Codes (num_codes, num_points)-np array.
    factors: Factors generating the codes (num_factors, num_points)-np array.
  """
  representations = None
  factors = None
  i = 0
  while i < num_points:
    num_points_iter = min(num_points - i, batch_size)
    current_factors = ground_truth_data.sample_factors(num_points_iter, random_state)
    current_observations = ground_truth_data.sample_observations_from_factors(current_factors,random_state)
    if i == 0:
      factors = current_factors
      representations,_ = representation_function.encoder(Variable(torch.Tensor(current_observations).to(device)))
      
      representations = representations.data.cpu().numpy()
    else:
      factors = np.vstack((factors, current_factors))
      a,_ = representation_function.encoder(Variable(torch.Tensor(current_observations).to(device)))
      a = a.data.cpu().numpy()
      representations = np.vstack((representations,a))
    i += num_points_iter
  return np.transpose(representations), np.transpose(factors)


def discrete_mutual_info(mus, ys):
  """Compute discrete mutual information."""
  num_codes = mus.shape[0]
  num_factors = ys.shape[0]
  m = np.zeros([num_codes, num_factors])
  for i in range(num_codes):
    for j in range(num_factors):
      m[i, j] = sklearn.metrics.mutual_info_score(ys[j, :], mus[i, :])
  return m


def discrete_entropy(ys):
  """Compute discrete mutual information."""
  num_factors = ys.shape[0]
  h = np.zeros(num_factors)
  for j in range(num_factors):
    h[j] = sklearn.metrics.mutual_info_score(ys[j, :], ys[j, :])
  return h


@gin.configurable(
    "discretizer", blacklist=["target"])
def make_discretizer(target, num_bins=10,
                     discretizer_fn=gin.REQUIRED):
  """Wrapper that creates discretizers."""
  return discretizer_fn(target, num_bins)


@gin.configurable("histogram_discretizer", blacklist=["target"])
def _histogram_discretize(target, num_bins=gin.REQUIRED):
  """Discretization based on histograms."""
  discretized = np.zeros_like(target)
  for i in range(target.shape[0]):
    discretized[i, :] = np.digitize(target[i, :], np.histogram(
        target[i, :], num_bins)[1][:-1])
  return discretized


def normalize_data(data, mean=None, stddev=None):
  if mean is None:
    mean = np.mean(data, axis=1)
  if stddev is None:
    stddev = np.std(data, axis=1)
  return (data - mean[:, np.newaxis]) / stddev[:, np.newaxis], mean, stddev





def make_predictor_fn():
  """Logistic regression with 5 folds cross validation."""
  return LogisticRegressionCV(Cs=10, cv=KFold(n_splits=5))


@gin.configurable("gradient_boosting_classifier")
def gradient_boosting_classifier():
  """Default gradient boosting classifier."""
  return GradientBoostingClassifier()
