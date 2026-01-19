#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import qmc.sobol_seq as QMC

# from numpy.polynomial.legendre import leggauss
# from itertools import product, tee, islice
# class LegendreGaussSampler(object):
#     def __init__(self, M, degree):
#         pairs = zip(*leggauss(int((degree+1)/2)))
#         self.iter = product(*tee(pairs, M))
#     def __call__(self, num_samples, offset):
#         # use reversed samples s.t. the first KL coefficient get higher degrees faster than the lower ones
#         return np.array(list(map(reversed ,islice(self.iter, num_samples))))

class RVSampler(object):
    def __init__(self, M, distribution='normal', strategy='random'):
        if distribution == 'normal':
            if strategy == 'random':
                self.sampler = lambda num_samples, offset: np.random.randn(num_samples, M)
                self.generator = 'np.random.randn'
            elif strategy == 'sobol':
                self.sampler = lambda num_samples, offset: QMC.i4_sobol_generate_std_normal(dim_num=M, n=num_samples, skip=offset)
                self.generator = 'qmc.i4_sobol_generate_std_normal'
            else:
                raise ValueError("strategy must be 'random' or 'sobol'")
        elif distribution == 'uniform':
            if strategy == 'random':
                self.sampler = lambda num_samples, offset: 2*np.random.rand(num_samples, M)-1
                self.generator = 'np.random.rand'
            elif strategy == 'sobol':
                self.sampler = lambda num_samples, offset: 2*QMC.i4_sobol_generate(dim_num=M, n=num_samples, skip=offset)-1
                self.generator = 'qmc.i4_sobol_generate'
            # elif strategy == 'gauss':
            #     self.sampler = lambda
            else:
                raise ValueError("strategy must be 'random' or 'sobol'")
        else:
            raise ValueError("distribution must be 'uniform' or 'normal'")


        self.offset = 1

    def __call__(self, num_samples):
        samples = self.sampler(num_samples, self.offset)
        self.offset += num_samples
        return samples
