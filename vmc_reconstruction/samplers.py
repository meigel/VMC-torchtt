# coding: utf-8
"""
Author: Philipp Trunschke (trunschk@math.hu-berlin.de)

Copyright: 2018-2019 Philipp Trunschke.
License: GNU Affero General Public License Version 3
"""
from __future__ import division, print_function
import numpy as np
from scipy.integrate import quad as integrate
from scipy.optimize import fminbound as minimize
from scipy.stats import triang as triangular
maximize = lambda f, *domain: minimize(lambda x: -f(x), *domain)
from scipy.optimize import brentq
from scipy.integrate import cumtrapz
from operator import itemgetter
from functools import partial
from basis import is_orthonormal, scipy_integral
import abc

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

logger = logging.getLogger(__name__)
logger.todo("own repo + allow custom random source (pass a random-generator (default=None))")


def cart(*arrs):
    return np.stack(np.meshgrid(*arrs), -1).reshape(-1, len(arrs))


logger.todo("make `domain` a function accepting points")
class Sampler(object):
    """
    Base class for all continuous samplers.
    """
    __metaclass__ = abc.ABCMeta

    @property
    def dimension(self):
        """
        The dimension of the ambient space for the domain of definition.
        """
        return len(self.domain)

    @abc.abstractproperty
    def domain(self):
        """
        The domain of a sampler is a list of pairs representing
        the n-orthotope (hyperrectangle) where it is defined.
        """
        pass

    @abc.abstractmethod
    def density(self, point):
        """
        Density of the samplers distribution.

        Computes the density of the samplers underlying
        distribution at the given `point`.

        Parameters
        ----------
        point : (..., D) array_like
            List of points or single point.
            `D` is the objects dimension.

        Returns
        -------
        out : ndarray, shape (..., 1)
            Density at the point.
        """
        pass

    @abc.abstractmethod
    def sample(self, *shape):
        """
        Random values in a given shape.

        Create an array of the given shape and populate it with
        random samples from the samplers distribution.

        Parameters
        ----------
        shape : array_like, optional
            The dimensions of the returned array, should all be positive.
            If no argument is given a single Python float is returned.

        Returns
        -------
        out : ndarray, shape `shape`
            Random values.
        """
        pass

    @abc.abstractproperty
    def mass(self):
        """
        Mass of the samplers distribution.

        The integral of the samplers distribution over the
        domain of definition. If the density is normalised
        this value should be one.
        """
        pass

    @classmethod
    def __subclasshook__(cls, C):
        has = lambda s: any(s in B.__dict__ for B in C.__mro__)
        return has("domain") and has("density") \
                and has("sample") and has("dimension") \
                and has("mass")


# def sample_discrete(*shape, weights=None, cdf=False):
def sample_discrete(*shape, **kwargs):
    """
    Random values in a given shape.

    Create an integer array of the given shape and populate it
    with random samples from a discrete distribution in the
    "half-open" interval [0, `len(weights)`).
    If `cdf` is False, then `weights[i]` contains the probability
    of sampling `i`. If `cdf` is True it contains the cummulative
    probability of sampling any `0 <= j <= i`.

    Parameters
    ----------
    shape : array_like
        The dimensions of the returned array, should all be positive.
        If no argument is given a single Python float is returned.
    weights : list
        List of (cummulative) probabilities.
    cdf : bool, optional
        Boolean flag indicating if weights contains the PDF or the CDF.
        (default is False)

    Returns
    -------
    out : ndarray
        Random values.
    """
    weights = kwargs.pop('weights', None)
    cdf = kwargs.pop('cdf', False)
    if weights is None:
        raise ValueError("weights has to be a list of weights or the number of uniform weights")
    elif isinstance(weights, int):
        assert weights > 0
        weights = [1]*weights
        if cdf: raise ValueError("cannot interpret integer as a cdf")
    if not cdf:
        weights = np.cumsum(weights).astype(np.float64)
    weights = weights / weights[-1]
    ret = np.random.rand(*shape)
    return np.searchsorted(weights, ret)


class Uniform(Sampler):
    """
    Uniform sampler in one dimension.

    Paramters
    ---------
    domain : (..., 2) array_like
        Domain of definition. Must be of size 2.
    height : float, optional
        Height of the step function. (default is 1/(length of domain)
    """
    def __init__(self, domain, height=None):
        self.__domain = np.reshape(domain, (-1, 2))
        assert self.__domain.shape[0] == 1
        self.__length = self.__domain[0,1] - self.__domain[0,0]
        self.__height = 1/self.__length if height is None else height
    @property
    def domain(self): return np.array(self.__domain)  # make a copy
    def density(self, point):
        in_domain = (self.__domain[0,0] <= point) & (point <= self.__domain[0,1])
        return self.__height * in_domain.astype(np.float64)
    @property
    def mass(self): return self.__height*self.__length
    def sample(self, *shape):
        return self.__length*np.random.rand(*shape) + self.__domain[0,0]


class CartesianProductSampler(Sampler):
    """
    Cartesian product of multiple samplers.

    The object represents a sampler for an independent cartesian product
    of the given sampler distributions.

    Parameters
    ----------
    samplers : list
        List of samplers.
    """
    def __init__(self, samplers):
        self.__samplers = list(samplers)
        self.__domain = np.concatenate([s.domain for s in self.__samplers])
        self.__mass = np.prod([s.mass for s in self.__samplers])

    @property
    def domain(self): return np.array(self.__domain)  # make a copy
    def density(self, point):
        point.shape = -1, self.dimension
        denses = [s.density(p) for s,p in zip(self.__samplers, point.T)]
        ret = np.prod(denses, axis=0)
        return ret
    @property
    def mass(self): return self.__mass

    @property
    def componentSamplers(self): return list(self.__samplers)

    def sample(self, *shape):
        ret = [s.sample(*shape) for s in self.__samplers]
        return np.stack(ret, -1)


class SumSampler(Sampler):
    """
    Sum of multiple samplers.

    The object represents a sampler for the sum of
    the densities of the provided samplers.

    Parameters
    ----------
    samplers : list
        List of samplers.
    """
    def __init__(self, samplers):
        self.__samplers = samplers
        domain = samplers[0].domain
        for s in samplers[1:]:
            domain[:, 0] = np.minimum(domain[:, 0], s.domain[:, 0])
            domain[:, 1] = np.maximum(domain[:, 1], s.domain[:, 1])
        self.__domain = domain
        masses = [s.mass for s in samplers]
        self.__cum_masses = np.cumsum(masses)

    @property
    def domain(self): return np.array(self.__domain)  # make a copy
    def density(self, point):
        return sum(s.density(point) for s in self.__samplers)
    @property
    def mass(self): return self.__cum_masses[-1]

    def sample(self, *shape):
        #TODO: Why doesn't it work to reorder each sample in its initial position?
        #      (i.e. concatenate all arange(size)[part_indices == i] and use fancy indexing on ret)
        size = np.prod(shape)
        part_indices = sample_discrete(size, weights=self.__cum_masses, cdf=True)

        ret = []
        for i,s in enumerate(self.__samplers):
            count = np.sum(part_indices == i)
            ret.append(s.sample(count))
        ret = np.concatenate(ret)
        np.random.shuffle(ret)  #TODO: use random.shuffle(ret, gen.rand) (builtin) wobei gen ein generator-Objet ist, dass eine rand-methode besitzt (uniform sampling in [0,1))

        if self.dimension == 1:
            return ret.reshape(shape)
        else:
            return ret.reshape(shape+(self.dimension,))


def interpolate(density, domain, discr_nodes=1000, max_nodes=100, eps=1e-8):
    """
    Interpolate the density on the given domain.

    Parameters
    ----------
    density : callable
        Density to interpolate.
    domain : (2,) array_like of floats
        Domain of definition.
    discr_nodes : iterable of floats or int, optional
        Candidates for the interpolation nodes.
        It can be an iterable or an int.
        If an iterable it will be used directly as the array of nodes that are consider for interpolation nodes.
        If an int then an array of equidistant points is generated as candidate set.
        (default is 1000)
    max_nodes : int, optional
        Maximum number of nodes for the interpolation.
        (default is 100)
    eps : float, optional
        Target supremum norm.
        (default is 1e-8)

    Returns
    -------
    out : ndarray, shape (n,)
        Interpolation nodes (n < max_nodes).
    """
    domain = np.reshape(domain, (2,))
    if isinstance(discr_nodes, int):
        x = np.linspace(*domain, num=discr_nodes)
    else:
        x = np.asarray(discr_nodes)
        assert np.all(domain[0] <= x) and np.all(x <= domain[1])
    y = density(x)
    approx = lambda nodes: np.interp(x, nodes, density(np.array(nodes)), left=0, right=0)
    nodes = list(domain)
    for _ in range(max_nodes-2):
        res = abs(y - approx(nodes))
        idx = np.argmax(res)
        if res[idx] < eps:
            break
        nodes.append(x[idx])
        nodes.sort()
    return np.array(nodes)


def scan_AffineSampler(nodes, density):
    """
    A sampler that samples from a linear interpolation of `density` in the interpolation nodes given in `nodes`.

    Parameters
    ----------
    nodes : (n,) ndarray
        Interpolation nodes.
    density : callable
        Density to interpolate.

    Returns
    -------
    out : SumSampler
        Sampler for the linear interpolation.
    """
    dup = lambda arr: np.stack([arr[:-1], arr[1:]]).T
    new = lambda dom: AffineSampler(dom, density(dom[0]), density(dom[1]))
    return SumSampler([new(dom) for dom in dup(nodes)])


class TriangularSampler(Sampler):
    """
    Triangular distribution.

    Parameters
    ----------
    domain : (..., 2) array_like
        Domain of definition.
    peak : float
        Position of the peak.
    height : float
        Height of the peak.
    """
    def __init__(self, domain, peak, height):
        logger = logging.getLogger(__name__)
        logger.debug("Init TriangularSampler")
        domain = np.reshape(domain, (1, 2))
        assert domain[0][0] <= peak <= domain[0][1]
        self.__domain = domain

        self.__peak = peak

        left, right = domain[0]
        diam = right-left
        peak01 = (peak-left)/diam
        self.__pdf = lambda x: triangular.pdf(x, peak01, loc=left, scale=diam)

        self.__scale = height / self.__pdf(peak)
        self.__mass = height * diam / 2  #TODO: why /2 ?

        # Actually, this is not a problem. A mass 0 sampler should never be called and raises an exception otherwise.
        # But it makes the algorithm slower and therefore raises an exception.
        assert self.mass > 1e-9

    @property
    def domain(self): return np.array(self.__domain)  # make a copy
    def density(self, point): return self.__scale * self.__pdf(np.asarray(point))
    @property
    def mass(self): return self.__mass
    def sample(self, *shape):
        size = np.prod(shape)
        ret = np.random.triangular(self.domain[0][0], self.__peak, self.domain[0][1], size)
        ret.shape = shape
        return ret


def scan_TriangularSampler(nodes, density):
    """
    A sampler that samples from a linear interpolation of `density` in the interpolation nodes given in `nodes`.

    Parameters
    ----------
    nodes : (n,) ndarray
        Interpolation nodes.
    density : callable
        Density to interpolate.

    Returns
    -------
    out : SumSampler
        Sampler for the linear interpolation.
    """
    left = TriangularSampler((nodes[0], nodes[1]), nodes[0], density(nodes[0]))
    right = TriangularSampler((nodes[-2], nodes[-1]), nodes[-1], density(nodes[-1]))
    hat = lambda k: TriangularSampler((nodes[k-1], nodes[k+1]), nodes[k], density(nodes[k]))
    return SumSampler([left] + [hat(k) for k in range(1, len(nodes)-1)] + [right])


class AffineSampler(Sampler):
    """
    Distribution with an affine density in one dimension.

    Parameters
    ----------
    domain : (..., 2) array_like
        Domain of definition.
    left : float
        Value at the left boundary of the domain.
    right : float
        Value at the right boundary of the domain.
    """
    def __init__(self, domain, left, right):
        logger = logging.getLogger(__name__)
        logger.debug("Init AffineSampler")
        domain = np.reshape(domain, (1, 2))
        self.__domain = domain

        a,b = domain[0]
        m = (right-left)/(b-a)
        n = left

        if n < 1e-9 and abs(m) < 1e-9:
            # Actually, this only is a problem when sampling and is perfectly fine in the approximation phase.
            raise ValueError("The density is essentially zero on {}.".format(domain[0]))

        # if abs(m) < 2*n*tol:  # density is nearly constant ([abs(m)*(b-a)/2]/[n*(b-a)] < tol)
        #     # Uniform...
        #     self.__pdf = lambda x: np.full_like(x, n, dtype=np.float64)
        #     self.__cdf = lambda x: n*(x-a)
        #     self.__inv_cdf = lambda y: y/n + a
        # else:
        self.__pdf = lambda x: m*(x-a) + n
        self.__cdf = lambda x: 0.5*m*(x**2-a**2) + (n-m*a)*(x-a)
        def inv_cdf(y):
            p2 = n/m
            nq = 2*y/m
            s = p2**2 + nq
            assert np.all(s >= 0)
            # m > 0 ==> only the positive root is relevant
            # m < 0 ==> only the smaller of both roots is relevant
            if m > 0:
                z = np.sqrt(s) - p2
            else:
                z = -p2 - np.sqrt(s)
            x = z + a
            assert np.all(a <= x)
            assert np.all(x <= b)
            return x
        self.__inv_cdf = inv_cdf

        assert self.__cdf(a) == 0
        self.__mass = self.__cdf(b)
        # Actually, this is not a problem. A mass 0 sampler should never be called and raises an exception otherwise.
        # But it makes the algorithm slower and therefore raises an exception.
        assert self.mass > 1e-9

    @property
    def domain(self): return np.array(self.__domain)  # make a copy
    def density(self, point):
        point = np.asarray(point)
        in_domain = (self.__domain[0,0] <= point) & (point <= self.__domain[0,1])
        return self.__pdf(point) * in_domain.astype(np.float64)
    @property
    def mass(self): return self.__mass

    def sample(self, *shape):
        y = np.random.rand(*shape)
        return self.__inv_cdf(self.__mass*y)  # make y uniformly on [0, cdf(b)]


class RejectionSampler(Sampler):
    """
    Rejection sampler for the given density on the given domain.

    Parameters
    ----------
    domain : (..., 2) array_like
        Domain of definition.
    density : callable
        Density according to which to sample.
    trial_dist : Sampler
        Trial distribution.
    maxIter : int, optional
        Maximum number of iteration to perform before
        using samples from the trial distribution.
        (default is 1000)
    """
    def __init__(self, domain, density, trial_dist, maxIter=1000, likelihood_bound=None):
        logger = logging.getLogger(__name__)
        logger.debug("Init RejectionSampler")
        self.__domain = np.reshape(domain, (-1,2))
        self.__density = density
        self.__trial_dist = trial_dist

        def volume(domain):
            if len(domain) == 1:
                return domain[0][1] - domain[0][0]
            ld = len(domain)//2
            return volume(domain[:ld]) * volume(domain[ld:])

        if self.dimension == 1:
            self.__mass = integrate(density, *self.domain[0])[0]  # ignore error estimate
        else:
            xs = cart(*[np.linspace(*dom, num=1000) for dom in self.domain])
            dd = density(xs)
            dt = trial_dist.density(xs)
            self.__mass = np.mean(dd) * volume(self.domain)

        if likelihood_bound is None:
            if self.dimension == 1:
                # self.__mass = integrate(density, *self.domain[0])[0]  # ignore error estimate
                lhood_x = maximize(lambda x: density(x)/trial_dist.density(x), *self.domain[0])
                lhood = density(lhood_x)/trial_dist.density(lhood_x)
                likelihood_bound = max(lhood, 1)
            else:
                # normalize
                assert np.allclose(dd[dt==0], 0)
                rs = dd/dt
                rs[~np.isfinite(rs)] = 0
                likelihood_bound = max(np.max(rs), 1)
            assert np.isfinite(likelihood_bound), self.__domain
            logger.info("RejectionSampler: likelihood_bound: {}".format(likelihood_bound))
        self.__lhood_bound = likelihood_bound
        self.__maxIter = maxIter

    @property
    def trial_distribution(self): return self.__trial_dist

    @property
    def likelihood_bound(self): return self.__lhood_bound

    @property
    def domain(self): return np.array(self.__domain)  # make a copy
    def density(self, point):
        in_domain = np.all([(dom[0] <= point) & (point <= dom[1]) for dom in self.domain], axis=0)
        return self.__density(point) * in_domain
    @property
    def mass(self): return self.__mass

    def sample(self, *shape):
        logger = logging.getLogger(__name__)
        logger.debug("RejectionSampler")
        f = self.density
        g = self.__trial_dist.density
        M = self.__lhood_bound

        size = np.prod(shape)
        res = []
        for _ in range(self.__maxIter):
            tsize = int(np.ceil(1.1*M*size))  # acceptance probability == 1/M
            logger.info("RejectionSampler: sampling: {} = {} + {}".format(tsize, size, tsize-size))
            ys = self.__trial_dist.sample(tsize)  # candidates
            us = np.random.rand(tsize)
            mask = us*M*g(ys) <= f(ys)
            logger.info("RejectionSampler: rejecting {}".format(len(mask) - np.sum(mask)))
            res.append(ys[mask])
            size -= np.sum(mask)
            if size <= 0: break
        if size>0:
            logger.warning("RejectionSampler: maxIter exceeded")
            res.append(self.__trial_dist.sample(size))

        ret = np.concatenate(res)
        ret = ret[:np.prod(shape)]
        if self.dimension == 1:
            ret.shape = shape
        else:
            ret.shape = shape+(self.dimension,)
        return ret


def CMDensity(basis, density):
    """
    The optimal sampling density for empirical $L_2$ best approximation in the given basis [1]_.

    Parameters
    ----------
    basis : Basis
        Basis in which the best approximation is sought.
    density : callable
        Density of the $L_2$ space in which the best approximation is sought.

    Returns
    -------
    out : callable
        The density.

    References
    ----------
    .. [1] A. Cohen & G. Migliorati, "Optimal weighted least-squares
       methods", SMAI Journal of Computational Mathematics, vol. 3,
       pp. 181-203, 2017.
    """
    mass = len(basis)
    def cm_density(x):
        ret = 0
        for fnc in basis:
            ret = ret + fnc(x)**2
        return ret * density(x) / mass
    return cm_density


def CMWeights(cm_sampler, density, samples):
    """
    Weights for the CMSamples with given base density.
    """
    return density(samples) / cm_sampler.density(samples)


def ks_significance(density, domain, samples, test_nodes=1000):
    assert np.ndim(samples) == 1, str(np.shape(samples))
    domain = np.reshape(domain, (2,))

    idcs = np.argsort(samples)
    samples = samples[idcs]

    if isinstance(test_nodes, int):
        a = min(domain[0], np.min(samples))
        o = max(domain[1], np.max(samples))
        if a < domain[0] or o > domain[1]:
            import warnings
            warnings.warn("samples outside of the domain", RuntimeWarning)
        domain = a,o
        xs = np.linspace(*domain, num=test_nodes)
        # xs = interpolate(density, domain, discr_nodes, test_nodes)
    else:
        xs = np.asarray(test_nodes)
    assert np.all(domain[0] <= xs) and np.all(xs <= domain[1])

    cds = cumtrapz(density(xs), xs, initial=0)
    cds /= cds[-1]  # normalize
    ecds = np.sum(samples[:,None] <= xs[None], axis=0).astype(float)
    ecds /= ecds[-1]  # normalize

    D = np.max(abs(ecds-cds))  # test statistic

    n = len(xs)
    m = len(samples)
    f = np.sqrt((n+m)/(n*m))
    c = lambda alpha: np.sqrt(-0.5*np.log(alpha))

    alphas = np.linspace(0,1,1001)[1:]  #TODO: 1001...
    lim = c(alphas)*f
    alpha_value = alphas[np.sum(lim >= D)]
    assert D > c(alpha_value)*f
    return D, alpha_value


def test_CMSamples(cm_sampler, samples):
    assert 1 <= np.ndim(samples) <= 2
    assert isinstance(cm_sampler, CartesianProductSampler) == (np.ndim(samples) == 2)
    if np.ndim(samples) == 2:
        assert len(cm_sampler.componentSamplers) == np.shape(samples)[1]
    else:
        cm_sampler = CartesianProductSampler([cm_sampler])
        samples = np.reshape(samples, (len(samples), 1))

    sig = 1
    for component, marginal in zip(samples.T, cm_sampler.componentSamplers):
        sig = min(sig, ks_significance(marginal.density, marginal.domain, component)[1])

    return sig


def test_CMWeights(cm_sampler, density, samples, weights):
    assert 1 <= np.ndim(samples) <= 2
    assert isinstance(cm_sampler, CartesianProductSampler) == (np.ndim(samples) == 2)
    if np.ndim(samples) == 2:
        assert len(cm_sampler.componentSamplers) == np.shape(samples)[1]
    assert np.shape(weights) == (np.shape(samples)[0],)
    real_weights = CMWeights(cm_sampler, density, samples)
    return np.max(abs(weights - real_weights))  # L_inf error -- allows for error bound of the integrals wrt these weights


def CMSampler(basis, density, domain=None, check=True, tol=1e-1):
    """
    Create a sampler with the optimal density for empirical $L_2$ best approximation in the given basis [1]_.

    Parameters
    ----------
    basis : Basis
        Basis in which the best approximation is sought.
    density : callable
        Density of the $L_2$ space in which the best approximation is sought.
    domain : (2,) array_like or Interval, optional
        The interval containing (the main part of) the mass.
        This parameter is only necessary when the domain for the density
        does not coincide with the domain of the basis or the domain of the basis is infinite.
        In the latter case the domain must be finite and should contian `1-tol` parts of the mass.
        This is necessary since for aritrary functions the essential domain cannot be determined dynamically.
        (default is `basis.domain`)
    tol : float, optional
        Tolerance for the internal linear interpolation of the density. May effect performance.
        (default is 1e-1)

    Returns
    -------
    out : Sampler
        Sampler instance.

    References
    ----------
    .. [1] A. Cohen & G. Migliorati, "Optimal weighted least-squares
       methods", SMAI Journal of Computational Mathematics, vol. 3,
       pp. 181-203, 2017.
    """
    if domain is None:
        domain = np.reshape(list(basis.domain), (1,2))
    assert np.all(np.isfinite(list(domain)))

    if check:
        integ = partial(scipy_integral, density=density)
        assert is_orthonormal(basis, integ)

    cm_density = CMDensity(basis, density)
    nodes = interpolate(cm_density, domain, eps=tol)
    sampler = scan_AffineSampler(nodes, cm_density)
    # sampler = scan_TriangularSampler(nodes, cm_density)
    sampler = RejectionSampler(sampler.domain, cm_density, sampler)
    return sampler


def approx_quantiles(density, tol):
     """
     Computes the point `x` for which CDF(density, -x) = 1-CDF(density, x) < tol.

     Assumes that `density` is centered and symmetrical.
     """
     mass = scipy_integral(density, (-np.inf, np.inf))
     pos_mean = scipy_integral(lambda x: x*density(x)/mass, (0, np.inf))
     return brentq(lambda x: density(x)/mass - tol, 0, pos_mean/tol)



# ============ #
# UNDOCUMENTED #
# ============ #


class Exponential(Sampler):
    def __init__(self, lam):
        self.__lam = lam

    @property
    def mass(self): return 1
    @property
    def domain(self): return [[0, np.inf]]
    def density(self, point): return self.__lam*np.exp(-self.__lam*point)
    def sample(self, *shape):
        U = np.random.rand(*shape)
        return -np.log(U)/self.__lam


class Transformed(Sampler):
    def __init__(self, sampler, shift=0, scale=1, likelihood=1):
        """
        X --> scale*X + shift
        """
        assert sampler.dimension == 1
        self.__sampler = sampler
        self.__shift = shift
        self.__scale = scale
        self.__likelihood = likelihood

    @property
    def mass(self): return self.__likelihood * self.__sampler.mass
    @property
    def domain(self):
        dom = self.__scale*self.__sampler.domain + self.__shift
        if self.__scale < 0: dom[0] = dom[0][::-1]
        return dom
    def density(self, point): return self.__likelihood * self.__sampler.density((point-self.__shift)/self.__scale)/abs(self.__scale)
    def sample(self, *shape): return self.__scale*self.__sampler.sample(*shape) + self.__shift



# ========== #
# DEPRECATED #
# ========== #


def CMSampler2D(basis1, basis2):
    """
    Create a sampler with the optimal density for empirical best approximation in the product of the given bases [1]_.

    .. deprecated:: 2.0.0
          `CMSampler2D` will be removed in basis 2.0.0 due to insufficient generality.
          You should use `CartesianProductSampler` and `CMSampler` instead.

    Parameters
    ----------
    basis1 : Basis
        Basis in which the best approximation is sought.
    basis2 : Basis
        Basis in which the best approximation is sought.

    Returns
    -------
    out : Sampler
        Sampler instance.

    Notes
    -----
    This function is deprecated.
    It is a slower, less precise method for obtaining a `CartesianProductSampler(CMSampler(basis1), CMSampler(basis2))`.

    References
    ----------
    .. [1] A. Cohen & G. Migliorati, "Optimal weighted least-squares
       methods", SMAI Journal of Computational Mathematics, vol. 3,
       pp. 181-203, 2017.
    """
    logger = logging.getLogger(__name__)
    def sq(f): return lambda x: f(x)**2

    GENS1 = []
    for key in basis1.dofmap:
        logger.info("Creating Sampler {}".format(key))
        trial_dist = create_P1Sampler(list(basis1[key].domain), sq(basis1[key]), 10, 1e-3, 0.33)
        trial_dist = RejectionSampler(list(basis1[key].domain), sq(basis1[key]), trial_dist)
        GENS1.append(trial_dist)

    GENS2 = []
    for key in basis2.dofmap:
        logger.info("Creating Sampler {}".format(key))
        trial_dist = create_P1Sampler(list(basis2[key].domain), sq(basis2[key]), 10, 1e-3, 0.33)
        trial_dist = RejectionSampler(list(basis2[key].domain), sq(basis2[key]), trial_dist)
        GENS2.append(trial_dist)

    S = []
    for G1 in GENS1:
        for G2 in GENS2:
            S.append(CartesianProductSampler([G1, G2]))

    return SumSampler(S)


def partition(density, domain, num_cells, rel_err=1e-3):

    maxIter = 1000
    def bisect(density, quantile, left, right):
        """
        Denote the density by f and let F be its anti-derivative.
        Find the quantile c in dom=(left,right).

        That is, find the c that minimized E(c) = .5(F(c) - F(dom[0]) - quantile)**2.
        mid = lambda (a,o): (a+o)/2
        For this define c0 = mid(dom). Then, if
            E(c0) == 0  <-->  F(c0) - F(dom[0]) - quantile == 0
        return c0 else if
            E'(c0) > 0  <-->  (F(c0) - F(dom[0]) - quantile) * f(c0) > 0
        narrow to dom = (dom[0], c0) else if
            E'(c0) < 0
        narrow to dom = (c0, dom[1]).
        """
        abs_err = quantile*rel_err

        for _ in range(maxIter):
            # choose the center as a candidate for the quantile
            center = (left+right)/2
            candidate = integrate(density, left, center)[0]
            res = candidate - quantile
            if abs(res) < abs_err:
                return center
            # if res * density(center) > 0:
            if res > 0: # assume density >= 0 !!!
                right = center
            else:
                left = center
                quantile -= candidate
        raise RuntimeError("Iteration Limit exceeded")

    mass = integrate(density, *domain)[0]  # ignore error estimate
    dm = mass / num_cells

    a,o = domain
    ret = [a]
    for n in range(num_cells-1):
        a = bisect(density, dm, a, o)
        ret.append(a)
    ret.append(o)
    return mass, np.array(ret)



if __name__ == '__main__':
    from basis import PiecewiseLegendrePolynomials, HermitePolynomials
    import matplotlib.pyplot as plt


    def histogram(xys, nx=1000, ny=1000, xdom=(-1,1), ydom=(-1,1)):
        logger = logging.getLogger(__name__)

        xs = xys[:,0]
        ys = xys[:,1]
        if not np.all(xs[:-1] <= xs[1:]):
            logger.debug("Sorting xs")
            xs_sorter = np.argsort(xs)
            xs = xs[xs_sorter]
            ys = ys[xs_sorter]

        bins = np.empty((nx, ny))

        logger.debug("Splitting xs")
        xsplits = np.searchsorted(xs, np.linspace(*xdom, num=nx+1))
        xsplits[0] = 0

        for i in range(nx):
            logger.debug("{}/{}".format(i, nx))
            ys_slice = ys[xsplits[i]:xsplits[i+1]]
            logger.debug("Sorting ys-slice")
            ys_slice = np.sort(ys_slice)
            logger.debug("Splitting ys-slice")
            ysplits = np.searchsorted(ys_slice, np.linspace(*ydom, num=ny+1))
            ysplits[0] = 0
            bins[i] = np.diff(ysplits)

        return bins


    # # TEST CMSampler for gaussian density ...
    # for mean in [-np.e, 0, np.pi]:
    #     for variance in [0.1, 1, 2]:
    #         print(mean, variance)
    #         density = lambda x: np.exp(-(x-mean)**2/(2*variance)) / np.sqrt(2*np.pi*variance)
    #         assert np.allclose(scipy_integral(density, (-np.inf, np.inf)), 1)
    #         basis = HermitePolynomials(4, mean, variance)
    #         sampler_1d = CMSampler(basis, density, (mean-4*np.sqrt(variance), mean+4*np.sqrt(variance)))
    # exit()


    def decart(fxy, x, y):
        if fxy.size == len(y)*len(x):
            return fxy.reshape(len(y), len(x))
        else:
            return fxy.reshape(len(y), len(x), -1)


    def discretize(function, nx=1000, ny=1000, xdom=(-1,1), ydom=(-1,1)):
        x = np.linspace(*xdom, num=nx)
        y = np.linspace(*ydom, num=ny)
        return x, y, decart(function(cart(x, y)), x, y)


    # z = Uniform([-1,1])
    # import cProfile
    # pr = cProfile.Profile()
    # pr.enable()
    # z.sample(10000)
    # pr.disable()
    # print("Dump stats:", 'ziggurat.prof')
    # pr.dump_stats('ziggurat.prof')


    from contextlib import contextmanager
    import resource
    @contextmanager
    def timer(msg):
        r0 = resource.getrusage(resource.RUSAGE_SELF)
        try:
            yield
        finally:
            r1 = resource.getrusage(resource.RUSAGE_SELF)
            u = r1.ru_utime - r0.ru_utime
            s = r1.ru_stime - r0.ru_stime
            print(msg%(int(1000*(u+s))))


    def error(sampler, num, plot=True):
        with timer("Sampling time: %dms"):
            samples = sampler.sample(num)
        hist = histogram(samples, 200, 200)
        x, y, dens = discretize(sampler.density, 200, 200)

        dens /= np.max(dens)
        hist /= np.max(hist)
        err = abs(dens - hist)

        print("Max-Error: {0:.2e}".format(np.max(err)))
        print("L1-Error:  {0:.2e}".format(np.mean(err)))

        if plot:
            f, ax = plt.subplots(1,3)
            ax[0].contourf(x, y, dens)
            ax[1].contourf(x, y, hist)
            ax[1].scatter(samples[:,0], samples[:,1], s=1, color='xkcd:scarlet')
            ax[1].set_xlim(-1,1)
            ax[1].set_ylim(-1,1)
            im = ax[2].contourf(x, y, err)
            plt.colorbar(im)
            plt.tight_layout()

        return np.max(err), np.mean(err)


    basis = PiecewiseLegendrePolynomials(2, 3)
    density = lambda x: 1/2
    PLOT = False

    print("Convergence of CMSampler2D")
    print("--------------------------\n")

    sampler_2d = CMSampler2D(basis, basis)
    errs_1 = []
    for e in range(6, 12):
        print("Number of samples: 2**({})".format(2*e))
        err = error(sampler_2d, 2**(2*e), PLOT)  # err ~ 1/sqrt(num)
        errs_1.append(err)
        print()
    if PLOT: plt.show()

    if PLOT:
        errs_1 = np.array(errs_1).T
        f, ax = plt.subplots(1,1)
        ax.plot(errs_1[0], label='L_inf')
        ax.plot(errs_1[1], label='L_1')
        ax.set_yscale('log')
        ax.legend()
        plt.show()

    x, y, fxy = discretize(sampler_2d.density)
    fxy /= np.max(fxy)

    sampler_1d = CMSampler(basis, density)
    sampler = CartesianProductSampler([sampler_1d, sampler_1d])

    _, _, gxy = discretize(sampler.density)
    gxy /= np.max(gxy)

    err = abs(fxy - gxy)
    print("Absolute error of densities")
    print("---------------------------\n")

    print("Max-Error: {0:.2e}".format(np.max(err)))
    print("L1-Error:  {0:.2e}".format(np.mean(err)))
    print()

    print("Convergence of [CMSampler,CMSampler]")
    print("--------------------------\n")

    errs_2 = []
    for e in range(6, 12):
        print("Number of samples: 2**({})".format(2*e))
        err = error(sampler, 2**(2*e), PLOT)  # err ~ 1/sqrt(num)
        errs_2.append(err)
        print()
    if PLOT: plt.show()

    if PLOT:
        errs_2 = np.array(errs_2).T
        f, ax = plt.subplots(1,1)
        ax.plot(errs_2[0], label='L_inf')
        ax.plot(errs_2[1], label='L_1')
        ax.set_yscale('log')
        ax.legend()
        plt.show()
