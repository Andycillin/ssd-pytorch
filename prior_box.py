import torch
from math import sqrt as sqrt
from itertools import product as product


class PriorBox(object):
    def __init__(self, config):
        super(PriorBox, self).__init__()
        self.img_size = config['dim']
        self.num_of_priors = len(config['aspect_ratios'])
        self.variance = config['variance'] or [0.1]
        self.feature_maps = config['feature_maps']
        self.min_sizes = config['min_sizes']
        self.max_sizes = config['max_sizes']
        self.steps = config['steps']
        self.aspect_ratios = config['aspect_ratios']
        self.clip = config['clip']
        self.version = config['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.img_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k] / self.img_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.img_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
        output = torch.Tensor(mean).view(-1, 4)

        if self.clip:
            output.clamp_(max=1, min=0)
        return output
