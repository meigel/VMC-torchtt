# -*- coding: utf-8 -*-
"""
Author: Philipp Trunschke (trunschk@math.hu-berlin.de)

Copyright: 2018-2019 Philipp Trunschke.
License: GNU Affero General Public License Version 3
"""
from __future__ import division, print_function
from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
from numpy.polynomial import Polynomial, Legendre, HermiteE as Hermite
from scipy.linalg import solve_triangular
from scipy.special import factorial

import os, logging, json
from logging.config import dictConfig
logging_config_path = os.path.join(os.path.dirname(__file__), 'logging_config.json')
with open(logging_config_path) as f:
    dictConfig(json.load(f))

TODO = 60
logging.addLevelName(TODO, "TODO")
def todo(self, message, *args, **kws):
    if self.isEnabledFor(TODO):
        self._log(TODO, message, args, **kws)
logging.Logger.todo = todo
module_logger = logging.getLogger(__name__)

module_logger.todo("Allow for multidimensional bases (in Generic). (Do you need the Interval class?)")


def cart(*arrs):
    return np.stack(np.meshgrid(*arrs), -1).reshape(-1, len(arrs))


class Interval(object):
    """
    Interval on the real line.

    Paramters
    ---------
    left : float
        Left boundary of the interval.
    right : float
        Left boundary of the interval.
    """
    def __init__(self, left, right):
        assert left < right
        self.__bounds = left, right

    def contains(self, point):
        """
        Return the truth value of `point` being contained in the interval (element-wise).

        Parameters
        ----------
        point : (N,) array_like
            Input array.

        Returns
        -------
        out : ndarray, shape (N,), dtype bool
            Output array, element-wise test for `point` to be contained in the interval.
        """
        return (self.__bounds[0] <= point) & (point <= self.__bounds[1])

    @property
    def diam(self):
        """
        The diameter (or length) of the interval.
        """
        return self.__bounds[1] - self.__bounds[0]

    def __str__(self): return "Interval({}, {})".format(*self.__bounds)
    def __repr__(self): return str(self)
    def __getitem__(self, key):
        """
        Parametrization of the interval over [0,1].

        Parameters
        ----------
        key : int, float or array_like
            Paramters.

        Returns
        -------
        out : ndarray, dtype float
            Values corresponding to the parameter.
        """
        if self.__bounds == (-np.inf, np.inf):
            return np.arctanh(2*key - 1)
        elif self.__bounds[0] == -np.inf:
            # -arctanh : [-1, 1] --> [-inf, inf];  -1 -->  inf;    1 --> -inf
            # -exp : [-inf,inf] --> [-inf, 0];    inf --> -inf; -inf --> 0
            return -np.exp(-np.arctanh(2*key - 1)) + self.__bounds[1]
        elif self.__bounds[1] == np.inf:
            # arctanh : [-1, 1] --> [-inf, inf];   -1 --> -inf;    1 --> inf
            # exp : [-inf,inf] --> [0, inf];     -inf -->    0;  inf --> inf
            return np.exp(np.arctanh(2*key - 1)) + self.__bounds[0]
        elif np.all(np.isfinite(self.__bounds)):
            return self.__bounds[0] + key*self.diam
        # ret = self.__bounds[0] + key*self.diam
        # # ret = np.array(self.__bounds[0] + key*self.diam, dtype=np.float64)
        # # ret[(key < 0) | (1 < key)] = np.nan
        # # if np.any(np.isnan(ret)):
        # #     raise KeyError(str(key))
        # return ret
    def find(self, x):
        """
        Find `t` such that `self[t] = x`.

        Parameters
        ----------
        x : int, float or array_like
            Values.

        Returns
        -------
        out : ndarray, dtype float
            Parameters corresponding to the values.
        """
        if self.__bounds == (-np.inf, np.inf):
            return (np.tanh(x) + 1)/2
        elif self.__bounds[0] == -np.inf:
            #     x = -np.exp(-np.arctanh(2*key - 1)) + self.__bounds[1]
            # --> np.log(-(x - self.bounds[1])) = -np.arctanh(2*key - 1)
            # --> np.tanh(-np.log(-(x - self.bounds[1]))) = 2*key - 1
            # --> (np.tanh(-np.log(-(x - self.bounds[1]))) + 1)/2 = key
            return (np.tanh(-np.log(-(x - self.bounds[1]))) + 1)/2
        elif self.__bounds[1] == np.inf:
            #     x = np.exp(np.arctanh(2*key - 1)) + self.__bounds[1]
            # --> np.log(x - self.bounds[1]) = np.arctanh(2*key - 1)
            # --> np.tanh(np.log(x - self.bounds[1])) = 2*key - 1
            # --> (np.tanh(np.log(x - self.bounds[1])) + 1)/2 = key
            return (np.tanh(np.log(x - self.bounds[1])) + 1)/2
        elif np.all(np.isfinite(self.__bounds)):
            return (x - self.__bounds[0]) / self.diam
        # ret = (x - self.__bounds[0]) / self.diam
        # # ret = np.array((x - self.__bounds[0]) / self.diam, dtype=np.float64)
        # # ret[(ret < 0) | (1 < ret)] = np.nan
        # # if np.any(np.isnan(ret)):
        # #     raise KeyError(str(key))
        # return ret
    def __iter__(self):
        """
        Iterator through the intervals bounds.

        Returns
        -------
        out : iterator
            Iterator.
        """
        return iter(self.__bounds)
    def __eq__(self, other):
        other = Interval(*other)  # allow `other` that is convertible to an interval
        return other[0] == self[0] and other[1] == self[1]


def transform(int1, int2):
    """
    Affine transformation from `int1` to `int2`.

    Parameters
    ----------
    int1 : Interval
        Interval to transform from.
    int2 : Interval
        Interval to transform to.

    Returns
    -------
    out : callable, signature (array_like -> ndarray)
        Function performing the affine transform.

    Notes
    -----
    `transfrom(int1, int2)(y)` is equivalent to `int2[int1.find(y)]`.
    """
    return lambda y: int2[int1.find(y)]


module_logger.todo("make EmptySet a singleton object like None")
# EmptySet = None  
class EmptySet(object):
    """
    Representation of the empty set.
    """
    pass


def intersection(int1, int2):
    """
    Return the intersection of two intervals.

    Parameters
    ----------
    int1 : (2,) array_like
        Bounds of the first interval.
    int2 : (2,) array_like
        Bounds of the second interval.

    Returns
    -------
    out: Interval or EmptySet
        The intersection of `int1` and `int2`.
    """
    a = max(int1[0], int2[0])
    b = min(int1[1], int2[1])
    if a < b:
        return Interval(a,b)
    else:
        return EmptySet


def gramian(basis, integral):
    """
    Return the Gramian matrix of the basis w.r.t. a specified integral.

    The output `out = gramian(basis, integral)` satisfies the identity
    `out[i,j] = integral(prod, dom)` where `prod` is the product of the
    `i`'th and `j`'th basis function and `dom` is the intersection of
    their domains.

    Parameters
    ----------
    basis : Basis
        Basis of which to compute the Gramian matrix.
    integral : callable, signature (function, domain -> float)
        Integration routine.

    Returns
    -------
    out : ndarray, shape (n,n)
        Gramian matrix.
    """
    ret = np.empty((len(basis), len(basis)))
    for i,b1 in enumerate(basis):
        for j,b2 in enumerate(basis):
            prod = lambda x: b1(x)*b2(x)
            dom = intersection(b1.domain, b2.domain)
            ret[i,j] = integral(prod, dom)
    return ret


def midpoint_integral(fnc, domain, density=lambda x: 1, num=2000):
    """
    Returns the approximate integral of `fnc` over `domain` with measure `density`.

    This integration routine implements the midpoint rule.

    Paramters
    ---------
    fnc : callable
        Function to integrate.
    domain : (2,) array_like
        Boundary of the interval to integrate over.
    density : callable, optional
        Density for the integration.  (default is (lambda x: 1))
    num : int, optional
        Number of uniform sub-intervals. (default is 2000)

    Returns
    -------
    out : float
        Integral of `fnc` over `domain` with measure `density`.
    """
    if domain is EmptySet:
        return 0
    else:
        x = np.linspace(*domain, num=num+1)
        y = fnc(x) * density(x)
        return np.sum((y[1:] + y[:-1])/2*np.diff(x))


def trapezoidal_integral(fnc, domain, density=lambda x: 1, num=2000):
    """
    Returns the approximate integral of `fnc` over `domain` with measure `density`.

    This integration routine implements the trapezoidal rule.

    Paramters
    ---------
    fnc : callable
        Function to integrate.
    domain : (2,) array_like
        Boundary of the interval to integrate over.
    density : callable, optional
        Density for the integration.  (default is (lambda x: 1))
    num : int, optional
        Number of uniform sub-intervals. (default is 2000)

    Returns
    -------
    out : float
        Integral of `fnc` over `domain` with measure `density`.
    """
    if domain is EmptySet:
        return 0
    else:
        x = np.linspace(*domain, num=num+1)
        return np.trapz(fnc(x) * density(x), x)


def scipy_integral(fnc, domain, density=lambda x: 1, **kwargs):
    """
    Returns the approximate integral of `fnc` over `domain` with measure `density`.

    This integration routine uses `scipy.integrate.quad`.

    Paramters
    ---------
    fnc : callable
        Function to integrate.
    domain : (2,) array_like
        Boundary of the interval to integrate over.
    density : callable, optional
        Density for the integration.  (default is (lambda x: 1))
    num : int, optional
        Number of uniform sub-intervals. (default is 2000)

    Returns
    -------
    out : float
        Integral of `fnc` over `domain` with measure `density`.
    """
    if domain is EmptySet:
        return 0
    else:
        from scipy import integrate
        prod = lambda x: fnc(x) * density(x)
        return integrate.quad(prod, *domain, **kwargs)[0]


class BasisFunction(object):
    """
    Base class for all basis functions.

    The instance represents a function on the given domain.

    Parameters
    ----------
    fnc : callable
        Callable object that evaluates the function.
    domain : (2,) array_like
        Domain of the function.

    Notes
    -----

    Methods `norm`, `normalized` and `orthogonal` are intentionally not enforced,
    since they depend on the norm (i.e. on the measure).
    """

    def __init__(self, fnc, domain):
        self.__fnc = fnc
        self.__domain = Interval(*domain)

    @property
    def domain(self):
        """
        The domain of a basis function is an `Interval` object.
        """
        return self.__domain

    def __call__(self, x):
        """
        Evaluates the basis function at `x`, element-wise.

        Paramters
        ---------
        x : (..., 1) array_like
            Points to evaluate the basis function at.

        Returns
        -------
        out : ndarray, shape (...)
            Function values. Points outside of the domain are evaluated to zero.
        """
        return self.domain.contains(x) * self.__fnc(x)
        return self.__fnc(x)

    @classmethod
    def __subclasshook__(cls, C):
        has = lambda s: any(s in B.__dict__ for B in C.__mro__)
        return has("domain") and has("__call__")


class Basis(object):
    """
    Abstract base for families of basis functions.

    Methods `norm`, `normalized` and `orthogonal` are intentionally not enforced,
    since they depend on the norm (i.e. on the measure).
    """
    __metaclass__ = ABCMeta

    @abstractproperty
    def domain(self):
        """
        The domain of a basis function is an `Interval` object.
        """
        pass

    @abstractproperty
    def dofmap(self):
        """
        An iterator over all keys that can be used to access the different `BasisFunction`s.
        """
        pass

    def __len__(self): return len(self.dofmap)
    def __contains__(self, key): return key in self.dofmap
    def __iter__(self):
        for dof in self.dofmap:
            yield self[dof]

    def eval(self, x):
        """
        Evaluate all basis functions at `x`, element-wise.

        .. deprecated:: 2.0.0
            `Basis.eval` will be removed in basis 2.0.0.
            You should use `Basis.__call__` instead.

        Paramters
        ---------
        x : (..., 1) array_like
            Points to evaluate the basis functions at.

        Returns
        -------
        out : ndarray, shape (..., len(self))
            Function values. Points outside of the domain are evaluated to zero.
        """
        return self(x)

    def __call__(self, x):
        """
        Evaluate all basis functions at `x`, element-wise.

        Paramters
        ---------
        x : (..., 1) array_like
            Points to evaluate the basis functions at.

        Returns
        -------
        out : ndarray, shape (..., len(self))
            Function values. Points outside of the domain are evaluated to zero.
        """
        return np.moveaxis([f(x) for f in self], 0, -1)

    @abstractmethod
    def __getitem__(self, key):
        """
        Return the `BasisFunction` for the key.

        Parameters
        ----------
        key : object
            The key to access.

        Returns
        -------
        out : BasisFunction
            Basis function assiociated with that key.
        """
        return NotImplemented

    @classmethod
    def __subclasshook__(cls, C):
        has = lambda s: any(s in B.__dict__ for B in C.__mro__)
        return has("domain") and has("dofmap") and has("eval") \
               and has("__iter__") and has("__getitem__") and has("__len__")


module_logger.todo("Merge Piecewise and Basis")
module_logger.todo("Polynomials are just particular piecewise Bases where each piece has the same domain.")


module_logger.todo("provide an iterator `pieces` that returns the bases on each subdomain (faster sampling)")
class Piecewise(Basis):
    """
    Piecewise basis.

    Combines several bases with disjoint domains into a new basis.

    Parameters
    ----------
    pieces : list of `Basis` objects with disjoint domains.
        List of `Basis` objects to combine.

    Notes
    -----
    The new dofmap indexes the `BasisFunction`s by the pairs
    of the position of their `Basis` in `pieces` and the key
    of the function in that `Basis`.
    """
    def __init__(self, pieces):
        self.__pieces = sorted(pieces, key=lambda p: p.domain[0])

        for i in range(len(self.__pieces)-1):
            assert self.__pieces[i].domain[1] <= self.__pieces[i+1].domain[0]

        a = min(p.domain[0] for p in self.__pieces)
        b = max(p.domain[1] for p in self.__pieces)
        self.__domain = Interval(a,b)

        dofmap = []
        for e,p in enumerate(self.__pieces):
            dofmap.extend([(e,b) for b in p.dofmap])
        self.__dofmap = sorted(dofmap, key = lambda eb: eb[::-1])

    @property
    def pieces(self):
        """
        List of all BasisFunctions. Sorted ascendingly by the left boundary of their domain.
        """
        return self.__pieces

    @property
    def domain(self):
        """
        The domain of a basis function is an `Interval` object.
        """
        return self.__domain

    @property
    def dofmap(self):
        """
        An iterator over all keys that can be used to access the different `BasisFunction`s.
        """
        return self.__dofmap

    def __getitem__(self, key):
        """
        Return the `BasisFunction` for the key.

        Parameters
        ----------
        key : object
            The key to access.

        Returns
        -------
        out : BasisFunction
            Basis function assiociated with that key.
        """
        if isinstance(key, int):
            return self.__pieces[key]
        else:
            cell, deg = key
            return self.__pieces[cell][deg]

    def orthonormalize(self, integral):
        """
        Return a new Basis that is orthonormalized with respect to the given integral.

        Parameters
        ----------
        integral : callable, signature (function, domain -> float)
            Integration routine.

        Returns
        -------
        out : Piecewise
            Basis that is orthonormal w.r.t. the given integral.
        """
        pieces = [p.orthonormalize(integral) for p in self.pieces]
        return Piecewise(pieces)

    @property
    def nodes(self):
        """
        List of the boundaries of all BasisFunctions. Sorted ascendingly.
        """
        try:
            return self.__nodes
        except AttributeError:
            nodes = [list(p.domain) for p in self.pieces]
            nodes = np.unique(np.concatenate(nodes))
            self.__nodes = nodes
            return nodes


class Generic(Basis):
    """
    Generic basis.

    The `basisFunction`s are given as linear combinations of functions.

    Parameters
    ----------
    functions : list of callables
        Functions from which to construct the basis functions.
    coefficients : (n, d) array_like
        Collection of coefficients.
    domain : (2,) array_like
        Domain of the basis.

    Notes
    -----
    The dofmap indexes the `BasisFunction`s by their position in `coefficients`.
    """
    def __init__(self, functions, coefficients, domain):
        self.__functions = functions
        self.__coefficients = list(coefficients)
        self.__domain = Interval(*domain)
        self.__dofmap = np.arange(len(self.__coefficients))

    @property
    def coefficients(self):
        """
        List of coefficients of the basis functions.
        """
        return self.__coefficients

    @property
    def functions(self):
        """
        List of basis functions.
        """
        return self.__functions

    @property
    def domain(self):
        """
        The domain of a basis is an `Interval` object.
        """
        return self.__domain

    @property
    def dofmap(self):
        """
        An iterator over all keys that can be used to access the different `BasisFunction`s.
        """
        return self.__dofmap

    def __getitem__(self, key):
        """
        Return the `BasisFunction` for the key.

        Parameters
        ----------
        key : object
            The key to access.

        Returns
        -------
        out : BasisFunction
            Basis function assiociated with that key.
        """
        functions = self.functions
        coeffs = self.coefficients[key]
        return BasisFunction(lambda x: sum(c*b(x) for c,b in zip(coeffs, functions)), self.domain)

    def orthonormalize(self, integral):
        """
        Return a new Basis that is orthonormalized with respect to the given integral.

        Parameters
        ----------
        integral : callable, signature (function, domain -> float)
            Integration routine.

        Returns
        -------
        out : Polynomial
            Basis that is orthonormal w.r.t. the given integral.
        """
        gram = gramian(self, integral)
        logger = logging.getLogger(__name__)
        logger.debug("computing Cholesky factorization")
        L = np.linalg.cholesky(gram)
        err = np.linalg.norm(np.dot(L, L.T) - gram)
        assert err < 1e-12, "%.2e"%err

        coeffs = self.coefficients
        logger.debug("inverting Cholesky factor")
        iLc = solve_triangular(L, coeffs, lower=True)
        return Generic(self.functions, iLc, self.domain)


module_logger.todo("change order of parameters for `Polynomials`")
class Polynomials(Generic):
    """
    Polynomial basis.

    Parameters
    ----------
    domain : (2,) array_like or Interval
        Boundary of the interval or the interval that is the basis domain.
    coefficients : (n, d) array_like
        Collection of polynomial coefficients.

    Notes
    -----
    The dofmap indexes the `BasisFunction`s by their position in `coefficients`.
    """
    def __init__(self, domain, coefficients):
        bfs = [Polynomial(coef) for coef in coefficients]
        cfs = np.eye(len(bfs))
        super(Polynomials, self).__init__(bfs, cfs, domain)


class HermitePolynomials(Polynomials):
    """
    Basis of Hermite polynomials.

    Parameters
    ----------
    degree : int
        Maximum degree of the polynomials.
    mean : float, optional
        Mean of the gaussian distribution.
        (default is 0)
    variance : float, optional
        Variance of the gaussian distribution.
        (default is 1)

    Notes
    -----
    The dofmap indexes the `BasisFunction`s by their polynomial degree.
    """
    def __init__(self, degree, mean=0, variance=1):
        # normaization factors to make the basis orthonormal (they are indeed independent of mean and variance)
        factors = 1/np.sqrt(factorial(np.arange(degree+1), exact=True))

        # exp(-(x-mean)**2/(2*variance)) == exp(-y**/2) ==> y = (x-mean)/sqrt(variance)
        # to_reference(x) = (x-mean)/variance = m*x + n
        m = 1/np.sqrt(variance)
        n = -mean/np.sqrt(variance)
        to_reference = Polynomial([n,m])

        # `q`-th normalized Hermite Polynomial with given mean and variance
        Herm = lambda q: Hermite([0]*q+[factors[q]])(to_reference)

        coefficients = [Herm(q).coef for q in range(degree+1)]
        super(HermitePolynomials, self).__init__((-np.inf, np.inf), coefficients)


class LegendrePolynomials(Polynomials):
    """
    Basis of Legendre polynomials.

    Parameters
    ----------
    domain : (2,) array_like or Interval
        Boundary of the interval or the interval that is the basis domain.
    degree : int
        Maximum degree of the polynomials.

    Notes
    -----
    The dofmap indexes the `BasisFunction`s by their polynomial degree.
    """
    def __init__(self, domain, degree):
        # normaization factors to make each basis orthonormal in L^2([-1,1], 1/2 dx)
        factors = np.sqrt(2*np.arange(degree+1) + 1)
        factors = factors / np.sqrt(Interval(*domain).diam / 2)

        # to_reference(x) = 2*(x-dom[0])/diam(dom)-1 = 2*x/diam(dom) - (2*dom[0]/diam(dom)+1)
        dom = Interval(*domain)
        to_reference = Interval(-1,1)[dom.find(Polynomial([0,1]))]

        # `q`-th normalized Legendre Polynomial on the domain `domain`
        LegDom = lambda q: Legendre([0]*q+[factors[q]])(to_reference)

        coefficients = [LegDom(q).coef for q in range(degree+1)]
        super(LegendrePolynomials, self).__init__(domain, coefficients)


class PiecewiseLegendrePolynomials(Piecewise):
    """
    Piecewise basis of Legendre polynomials.

    .. deprecated:: 2.0.0
          `PiecewiseLegendrePolynomials` will be removed in basis 2.0.0
          due to insufficient generality.
          You should use `Piecewise` instead.

    Parameters
    ----------
    nodes : (n,) array_like
        Boundaries of the sub-intervals.
    degrees : (n-1,) array_like
        Maximum degree of the polynomials on each sub-interval.

    Notes
    -----
    The dofmap indexes the `BasisFunction`s by pairs
    of the position of their domain and their polynomial degree.
    """
    def __init__(self, nodes, degrees):
        if isinstance(nodes, int):
            assert nodes >= 1
            nodes = np.linspace(-1, 1, nodes+1)
        else:
            nodes = np.asarray(nodes)
            assert nodes.ndim == 1 and len(nodes) >= 2 \
                   and np.all(nodes[:-1] < nodes[1:]) \
                   and nodes[0] == -1 and nodes[-1] == 1
        if isinstance(degrees, int):
            assert degrees >= 0
            degrees = np.full(len(nodes)-1, degrees)
        else:
            degrees = np.asarray(degrees)
            assert degrees.ndim == 1 and len(degrees) == len(nodes)-1 \
                   and np.all(degrees >= 0)

        pieces = [LegendrePolynomials((nd0, nd1), d) for nd0, nd1, d  in zip(nodes[:-1], nodes[1:], degrees)]

        super(PiecewiseLegendrePolynomials, self).__init__(pieces)


def to_function(coefficients, bases):
    """
    Return the function that is represented by the coefficient tensor for the given bases.

    Parameters
    ----------
    coefficients : array_like
        Coefficients of the function.
    bases : Basis of list of Basis
        Basis for the coefficients.

    Notes
    -----
    The number of dimensions of `coefficients` must be equal to the length of `bases`.
    """
    if isinstance(bases, Basis):
        bases = [bases]
    def function(xs):
        # ret = np.array(coefficients)
        xs = np.reshape(xs, (-1, len(bases)))
        ret = np.dot(bases[0].eval(xs[:, 0]), coefficients)
        for m, space in enumerate(bases[1:], 1):
            Bx = space.eval(xs[:, m])
            R = ret.reshape(ret.shape[:2] + (-1,))
            R = np.einsum('ijk,ij->ik', R, Bx)
            ret = R.reshape((ret.shape[0],) + ret.shape[2:])
            # R = np.rollaxis(ret, m, 0)
            # Rs = R.shape
            # R.shape = R.shape[0], -1
            # R = Bx @ R
            # R.shape = Rs
            # ret = np.rollaxis(R, 0, m+1)
            # ret = np.tensordot(ret, Bx, (m, 1))
        return ret
    return function


def project(function, basis, integral):
    """
    Return the coefficients of the projection onto the given basis w.r.t. the given integral.

    Parameters
    ----------
    function : callable
        Function to project.
    basis : Basis
        Basis on which to project.
    integral : callable, signature (function, domain -> float)
        Integration routine.

    Returns
    -------
    out : ndarray
        Coefficient tensor.
    """
    C = np.empty(len(basis))
    for i, b in enumerate(basis):
        prod = lambda x: function(x) * b(x)
        C[i] = integral(prod, b.domain)
    return C


module_logger.todo("wrong formula fro midpoint_integral_2d")
def midpoint_integral_2d(fnc, domain_x, domain_y, density=lambda xy: 1, num=2000):
    if domain_x is EmptySet or domain_y is EmptySet:
        return 0
    else:
        x = np.linspace(*domain_x, num=num)
        y = np.linspace(*domain_y, num=num)

        prod = lambda xys: fnc(xys) * density(xys)
        area = domain_x.diam * domain_y.diam
        return area * np.mean(prod(cart(x,y)))

        # return domain.diam * np.mean(fnc(x) * density(x))


def scipy_integral_2d(fnc, domain_x, domain_y, density=lambda xy: 1):
    if domain_x is EmptySet or domain_y is EmptySet:
        return 0
    else:
        def uncurry(f):
            def ret(x,y):
                x = np.reshape(x, (-1,1))
                y = np.reshape(y, (-1,1))
                xy = np.concatenate([x,y], axis=1)
                return f(xy)
            return ret

        from scipy import integrate
        prod_xy = lambda xy: fnc(xy) * density(xy)
        prod = lambda y,x: uncurry(prod_xy)(x,y)
        const = lambda c: lambda x: c
        return integrate.dblquad(prod, *domain_x, gfun=const(domain_y[0]), hfun=const(domain_y[1]))[0]


def project2d(function, basis_x, basis_y, integral):
    """
    Returns coefficients of projection onto the basis.
    """
    C = np.empty((len(basis_y), len(basis_x)))
    for i, b_y in enumerate(basis_y):
        for j, b_x in enumerate(basis_x):
            prod = lambda xy: f(xy) * b_x(xy[:,0]) * b_y(xy[:,1])
            C[i,j] = integral(prod, b_x.domain, b_y.domain)
    return C.T


def is_orthonormal(basis, integ, tol=1e-6):
    return np.linalg.norm(gramian(basis, integ) - np.eye(len(basis))) < tol


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from functools import partial


    def decart(fxy, x, y):
        if fxy.size == len(y)*len(x):
            return fxy.reshape(len(y), len(x))
        else:
            return fxy.reshape(len(y), len(x), -1)


    def discretize(function, nx=1000, ny=1000, xdom=(-1,1), ydom=(-1,1)):
        x = np.linspace(*xdom, num=nx)
        y = np.linspace(*ydom, num=ny)
        return x, y, decart(function(cart(x, y)), x, y)


    PLOT = True

    # unif = partial(midpoint_integral, density=lambda x: 0.5, num=100)
    # unif = partial(trapezoidal_integral, density=lambda x: 0.5, num=100)
    unif = partial(scipy_integral, density=lambda x: 0.5, limit=50)

    basis = PiecewiseLegendrePolynomials(8, 7)
    assert basis.domain == Interval(-1,1)
    assert is_orthonormal(basis, unif)

    basis = PiecewiseLegendrePolynomials([-1,-.5,-.25,-.124,0,.125,.25,.5,1], [3,2,1,1,1,1,2,3])
    assert basis.domain == Interval(-1,1)
    assert is_orthonormal(basis, unif)


    def plot_Piecewise(basis):
        assert isinstance(basis, Piecewise)

        colors = ['xkcd:navy blue', 'xkcd:reddish orange', 'xkcd:dark teal',
                  'xkcd:tomato red', 'xkcd:blue green', 'xkcd:scarlet', 'xkcd:jade', 'xkcd:crimson']
        styles = ['-', '--', '-.', ':']

        f, ax = plt.subplots(1, 1)
        for key in basis.dofmap:
            if key[0] >= 10 or key[1] >= 4:
                continue
            fnc = basis[key]
            x = np.linspace(*fnc.domain)
            y = fnc(x)
            ax.plot(x, y, styles[key[1]], color=colors[key[0]])

        ax.set_xlim([-1, 1])
        ax.set_ylim([-10, 10])

        nodes = np.array(basis.nodes)
        ax.set_xticks(nodes, minor=False)
        ax.grid(color='xkcd:grey', which='major', linestyle='--', linewidth=0.5)

        plt.show()


    if PLOT:
        plot_Piecewise(basis)

        f = lambda x: np.sin(2*np.pi*x)
        c = project(f, basis, unif)
        fc = to_function(c, basis)

        x = np.linspace(*basis.domain, num=1000)
        plt.plot(x, f(x), 'k--', label='original')
        plt.plot(x, fc(x), 'r', label='projection')
        plt.legend()
        plt.tight_layout()
        plt.show()


    sigma2 = 1/50
    density = lambda x: np.exp(-x**2/sigma2)
    # integ = partial(midpoint_integral, density=density, num=100)
    # integ = partial(trapezoidal_integral, density=density, num=100)
    integ = partial(scipy_integral, density=density)

    basis = PiecewiseLegendrePolynomials([-1,-.5,-.25,-.124,0,.125,.25,.5,1], [3,2,1,1,1,1,2,3])
    assert not is_orthonormal(basis, integ)

    basis = basis.orthonormalize(integ)
    assert is_orthonormal(basis, integ)


    if PLOT:
        plot_Piecewise(basis)

        f = lambda x: np.sin(2*np.pi*x)
        c = project(f, basis, integ)
        fc = to_function(c, basis)

        x = np.linspace(*basis.domain, num=2000)
        plt.plot(x, f(x), 'k--', label='original')
        plt.plot(x, fc(x), 'r', label='projection')
        plt.legend()
        plt.tight_layout()
        plt.show()

        _, ax = plt.subplots(1,1)
        ax.plot(x, abs(f(x) - fc(x)), 'r', label='error')
        ax.plot(x, abs(f(x) - fc(x))*np.sqrt(density(x)), 'b', label='weighted error')
        ax.plot(x, density(x), 'k--', label='density')
        plt.ylim(-0.08, 1.08)
        nodes = np.array(basis.nodes)
        ax.set_xticks(nodes, minor=False)
        ax.grid(color='xkcd:grey', which='major', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()


    unif2d = partial(midpoint_integral_2d, density=lambda xy: 0.25, num=100)
    # unif2d = partial(scipy_integral_2d, density=lambda xy: 0.25)

    basis = PiecewiseLegendrePolynomials([-1,-.5,-.25,-.124,0,.125,.25,.5,1], [3,2,1,1,1,1,2,3])


    if PLOT:
        f = lambda xys: np.sin(2*np.pi*xys[:,0]) * np.cos(2*np.pi*xys[:,1])
        c = project2d(f, basis, basis, unif2d)
        fc = to_function(c, [basis, basis])

        x, y, fxy = discretize(f)
        x, y, fcxy = discretize(fc)

        f, ax = plt.subplots(1,3)
        ax[0].contourf(x,y,fxy)
        ax[1].contourf(x,y,fcxy)
        im = ax[2].contourf(x,y,abs(fcxy-fxy))
        f.colorbar(im)
        plt.tight_layout()
        plt.show()


    density2d = lambda xy: np.exp(-np.linalg.norm(xy, ord=2, axis=-1)**2/sigma2)
    integ2d = partial(midpoint_integral_2d, density=density2d, num=100)
    # integ2d = partial(scipy_integral_2d, density=density2d)

    basis = PiecewiseLegendrePolynomials([-1,-.5,-.25,-.124,0,.125,.25,.5,1], [3,2,1,1,1,1,2,3])
    basis = basis.orthonormalize(integ)


    if PLOT:
        f = lambda xys: np.sin(2*np.pi*xys[:,0]) * np.cos(2*np.pi*xys[:,1])
        c = project2d(f, basis, basis, integ2d)
        fc = to_function(c, [basis, basis])

        x, y, fxy = discretize(f)
        x, y, fcxy = discretize(fc)
        x, y, dxy = discretize(lambda xy: np.sqrt(density2d(xy)))

        f, ax = plt.subplots(1,4)
        ax[0].contourf(x,y,fxy)
        ax[1].contourf(x,y,fcxy) # , vmin=np.min(fxy), vmax=np.max(fxy))
        ax[2].contourf(x,y,abs(fcxy-fxy))
        im = ax[3].contourf(x,y,abs(fcxy-fxy)*dxy)
        f.colorbar(im)
        plt.tight_layout()
        plt.show()


    for mean in [-np.e, 0, np.pi]:
        for variance in [0.1, 1, 2]:
            density = lambda x: np.exp(-(x-mean)**2/(2*variance)) / np.sqrt(2*np.pi*variance)
            assert np.allclose(scipy_integral(density, (-np.inf, np.inf)), 1)

            integ = partial(scipy_integral, density=density)
            basis = HermitePolynomials(4, mean, variance)
            assert is_orthonormal(basis, integ), (mean, variance)
