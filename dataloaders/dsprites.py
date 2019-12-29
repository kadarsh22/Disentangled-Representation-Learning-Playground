from disentanglement_lib.data.ground_truth import ground_truth_data
from disentanglement_lib.data.ground_truth import util
from tensorflow import gfile
import numpy as np
import PIL
import sys
sys.path.insert(0,'../../utils')
SCREAM_PATH  = "/home/adarsh/Desktop/Disentangled-Representation-Learning-Playground/data/scream/scream.jpg"

class DSprites(ground_truth_data.GroundTruthData):


    def __init__(self, latent_factor_indices=None):
        # By default, all factors (including shape) are considered ground truth
        # factors.
        if latent_factor_indices is None:
            latent_factor_indices = list(range(6))
        self.latent_factor_indices = latent_factor_indices
        self.data_shape = [64, 64, 1]
        # Load the data so that we can sample from it.
        with gfile.Open("/home/adarsh/Desktop/Disentangled-Representation-Learning-Playground/data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz", "rb") as data_file:
          # Data was saved originally using python2, so we need to set the encoding.
            data = np.load(data_file, encoding="latin1",allow_pickle=True)
            self.images = np.array(data["imgs"])
            self.factor_sizes = np.array(data["metadata"][()]["latents_sizes"], dtype=np.int64)
            self.full_factor_sizes = [1, 3, 6, 40, 32, 32]
            self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(
            self.factor_sizes)
            self.state_space = util.SplitDiscreteStateSpace(self.factor_sizes,
                                                        self.latent_factor_indices)

    def num_factors(self):
        return self.state_space.num_latent_factors


    def factors_num_values(self):
        return [self.full_factor_sizes[i] for i in self.latent_factor_indices]

    def observation_shape(self):
        return self.data_shape

    def sample_factors(self, num, random_state):
        return self.state_space.sample_latent_factors(num, random_state)

    def sample_observations_from_factors(self, factors, random_state):
        return self.sample_observations_from_factors_no_color(factors, random_state)

    def sample_observations_from_factors_no_color(self, factors, random_state):
        all_factors = self.state_space.sample_all_factors(factors, random_state)
        indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)
        return np.expand_dims(self.images[indices].astype(np.float32), axis=3)

    def _sample_factor(self, i, num, random_state):
        return random_state.randint(low=2,high=6, size=num)
    
class ColorDSprites(DSprites):

    def __init__(self, latent_factor_indices=None):
        DSprites.__init__(self, latent_factor_indices)
        self.data_shape = [64, 64, 3]

    def sample_observations_from_factors(self,no_color_observations, random_state,Transform = False):
        if (Transform == False):
            no_color_observations = self.sample_observations_from_factors_no_color(no_color_observations, random_state)
        observations = np.repeat(no_color_observations, 3, axis=3)
        color = np.repeat(
            np.repeat(
                random_state.uniform(0.5, 1, [observations.shape[0], 1, 1, 3]),
                observations.shape[1],
                axis=1),
            observations.shape[2],
            axis=2)
        return observations * color


class NoisyDSprites(DSprites):
    """Noisy DSprites.
    This data set is the same as the original DSprites data set except that when
    sampling the observations X, the background pixels are replaced with random
    noise.
    The ground-truth factors of variation are (in the default setting):
    0 - shape (3 different values)
    1 - scale (6 different values)
    2 - orientation (40 different values)
    3 - position x (32 different values)
    4 - position y (32 different values)
    """

    def __init__(self, latent_factor_indices=None):
        DSprites.__init__(self, latent_factor_indices)
        self.data_shape = [64, 64, 3]

    def sample_observations_from_factors(self, no_color_observations, random_state,Transform = False):
        if (Transform ==False):
            no_color_observations = self.sample_observations_from_factors_no_color(no_color_observations, random_state)
        observations = np.repeat(no_color_observations, 3, axis=3)
        color = random_state.uniform(0, 1, [observations.shape[0], 64, 64, 3])
        return np.minimum(observations + color, 1.)


class ScreamDSprites(DSprites):

    def __init__(self, latent_factor_indices=None):
        DSprites.__init__(self, latent_factor_indices)
        self.data_shape = [64, 64, 3]
        with gfile.Open(SCREAM_PATH, "rb") as f:
          scream = PIL.Image.open(f)
          scream.thumbnail((350, 274, 3))
          self.scream = np.array(scream) * 1. / 255.

    def sample_observations_from_factors(self, factors, random_state):
        no_color_observations = self.sample_observations_from_factors_no_color(
            factors, random_state)
        observations = np.repeat(no_color_observations, 3, axis=3)

        for i in range(observations.shape[0]):
          x_crop = random_state.randint(0, self.scream.shape[0] - 64)
          y_crop = random_state.randint(0, self.scream.shape[1] - 64)
          background = (self.scream[x_crop:x_crop + 64, y_crop:y_crop + 64] +
                        random_state.uniform(0, 1, size=3)) / 2.
          mask = (observations[i] == 1)
          background[mask] = 1 - background[mask]
          observations[i] = background
        return observations
