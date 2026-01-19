# -*- coding: utf-8 -*-
from __future__ import division
from dolfin import interpolate, Expression, Function
import numpy as np

class TestField:
    """ artificial M-term KL (plane wave Fourier modes as in EGSZ but w/o respective scaling).
    """
    def __init__(self, M, mean=1, expfield=False, scale=1.0, sigma=0):
        # type: (float, bool, int) -> None
        """
         initialise field with given mean.
         field is log-uniform/-normal for expfield=True.
        :param M: number of terms in expansion
        :param mean: mean value of the field
        :param expfield: Switch to go to lognormal field
        :param sigma: KL decay rate
        :return:
        """
        assert M > 0, 'number of terms in expansion has to be positive'
        # create a Fenics expression of the affine field
        self.a = Expression('C + S*sin(A*pi*F1*x[0]) * sin(A*pi*F2*x[1])', A=1, C=mean, F1=0, F2=0, S=1, degree=5)
        self.M = M
        self.mean = mean
        self.expfield = expfield
        self.scale = scale
        self.sigma = sigma

    def realisation(self, y, V):
        # type: (List[float], FunctionSpace) -> Function
        """
          evaluate realisation of random field subject to samples y and return interpolation onto FEM space V.
        :param y: list of samples of the RV
        :param V: FunctionSpace
        :return: FEniCS Function as field realisation
        """
        assert self.M == len(y), 'number of parameters differs from the number of terms in the expansion'

        def indexer(i):
            m1 = np.floor(i/2)
            m2 = np.ceil(i/2)
            return m1, m2
        a = self.a                                  # store affine field Expression
        sigma = self.sigma
        a.C, a.F1, a.F2 = self.mean, 0, 0           # get mean function as starting point
        x = interpolate(a, V).vector().get_local()      # interpolate constant mean on FunctionSpace
        a.C = 0                                     # set mean back to zero. From now on look only at amp_func
        #                                           # add up stochastic modes
        #TODO: im mc_exp2 ändern (wie im paper...)
        CM = 1 / self.M
        for m, ym in enumerate(y):              # loop through sample items
            a.F1, a.F2 = indexer(m+2)           # get running values in Expression
            a.S = self.scale
            #                                   # add current Expression value
            x += CM / ((m+1)**sigma) * ym * interpolate(a, V).vector().get_local()
        f = Function(V)                             # create empty function on FunctionSpace
        #                                           # set function coefficients to realisation coefficients
        f.vector()[:] = x if not self.expfield else np.exp(x)
        return f
