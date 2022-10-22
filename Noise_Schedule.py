from numpy.random import normal, poisson, randint
from numpy import copy, ceil, unique, log2, zeros, clip
from math import exp, log, cos


class Noise:
    """
    Following are supported values for various parameters.
        strategy :  ['linear','exponential','log','cosine-linear','cosine-exponential']
        noise_type : ['gaussian','salt&pepper','poisson','speckle']
    """
    def __init__(self, data, config):
        self.noise_type = config["noise_type"]
        self.strategy = config["strategy"]
        self.n_steps = config["n_steps"]
        self.config = config
        self.schedule = []
        self.schedule.append(data)
        self.step_size = len(data)
    def adjust(self, value, i):
        if self.strategy == 'linear':
            return value * i
        elif self.startegy == 'exponential': 
            return value * exp(i)
        elif self.startegy == 'log':
            return value * log(i)
        elif self.startegy == 'cosine-linear':
            return value * cos(i)
        elif self.strategy == 'cosine-exponential':
            return value * exp(cos(i))
        else:
            raise Exception(f"The support for {self.startegy} growth strategy is not implemented!")
    def create_schedule(self):
        for i in range(1, self.n_steps):
            self.schedule.append(list(map(self.induce_noise, self.schedule[i - 1], [i] * self.step_size)))
    def induce_noise(self, x, i):
        h, w, ch = x.shape
        if self.noise_type == 'gaussian':
            mean = self.adjust(self.config["gaussian_noise_mean"], i)
            var = self.adjust(self.config["gaussian_noise_variance"], i)
            std = var ** 0.5
            gaussian_additive_noise = normal(mean, std, (h,w,ch))
            return self.clip(x + gaussian_additive_noise)
        elif self.noise_type == 'salt&pepper':
            p = self.config["salt_pepper_noise_salt_probability"]
            noise_density = self.adjust(self.config["salt_pepper_noise_density"], i)
            noised_image = copy(x)
            n_salt = ceil(noise_density * h * w * ch * p)
            n_pepper = ceil(noise_density * h * w * ch * (1 - p))
            idx_salt = [
                randint(0 , j - 1, n_salt) for j in x.shape
            ]
            idx_pepper = [
                randint(0 , j - 1, n_pepper) for j in x.shape
            ]
            noised_image[idx_salt] = 1
            noised_image[idx_pepper] = 0
            return self.clip(noised_image)
        elif self.noise_type == 'poisson':
            values = len(unique(x))
            values = 2 ** ceil(log2(values))
            return poisson(x * values) / float(values)
        elif self.noise_type == 'speckle':
            gaussian_multiplicative_noise = randn(h,w,ch)
            return self.clip(x + x * gaussian_multiplicative_noise)
        else:
            raise Exception(f"{self.noise_type} noise not supported.")
    def getSchedule(self, idx):
        assert idx >= 0 and idx < len(self.schedule), f"Index of schedule {idx} is out of bounds."
        return self.schedule[idx]
    def clip(self, x):
        return clip(x, a_min = 0, a_max = None)

if __name__ == '__main__':
    images = [zeros((5,5,1)) for i in range(2)]
    noise_handler = Noise(images)
    noise_handler.create_schedule()
    for i in range(2):
        print(noise_handler.getSchedule(i))