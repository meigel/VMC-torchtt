# coding: utf-8
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad as integrate
from scipy.optimize import fminbound as minimize
maximize = lambda f, *domain: minimize(lambda x: -f(x), *domain)
from bisect import bisect
from numpy.polynomial.legendre import legval
from operator import itemgetter
import abc


class Sampler(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def domain(self): pass

    @abc.abstractmethod
    def density(self, point): pass

    @abc.abstractmethod
    def sample(self, *shape): pass

    @abc.abstractproperty
    def mass(self): pass

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Sampler:
            has = lambda s: any(s in B.__dict__ for B in C.__mro__)
            return has("domain") and has("density") and has("sample")
        return NotImplemented


class LinearInterpolation(Sampler):
    def __init__(self, density, domain):
        a = density(domain[0])
        b = density(domain[1])
        self.slope = (b-a)/(domain[1]-domain[0])
        self.offset = a - self.slope*domain[0]
        self.__domain = domain
        self.__target_density = density
        self.__density = lambda x: self.slope*x + self.offset

        a, b = self.domain
        m = self.slope
        n = self.offset
        cdf = lambda x: 0.5*m*(x**2-a**2) + n*(x-a)
        self.__mass = cdf(b)

        # print("LinearInterpolation:", m,n)

    @property
    def domain(self): return self.__domain
    def density(self, point): return self.__density(point)
    @property
    def mass(self): return self.__mass

    def sample(self, *shape):
        a, b = self.domain
        m = self.slope
        n = self.offset

        y = np.random.rand(*shape)
        if abs(m) > 1e-6:
            cdf = lambda x: 0.5*m*(x**2-a**2) + n*(x-a)
            l = cdf(0) - y*cdf(b)
            rt = np.sqrt(n**2 - 2*m*l)
            x1 = (-n + rt) / m
            x2 = (-n - rt) / m
            x = np.where((a-1e-4 <= x1) & (x1 <= b+1e-4), x1, x2)
            assert np.all((a-1e-4 <= x) & (x <= b+1e-4)), "{} <= {}, {} <= {}".format(a, x1, x2, b)
        else:
            # print(m, n)
            # print(self.mass)
            cdf = lambda x: n*(x-a)  # cdf(b)*y = n*(x-a) --> x = cdf(b)*y/n + a
            x = cdf(b)*y/n + a
        return x

    def abs_residuum(self, x):
        return abs(self.density(x) - self.__target_density(x))

    def error(self):  # L1 error
        return integrate(self.abs_residuum, *self.domain)[0]  # ignore error estimate

    @property
    def target_density(self): return self.__target_density

    def refine(self):
        # find the point of maximal L-inf error
        density = self.__target_density
        split_point = maximize(self.abs_residuum, *self.domain)
        domain_left = self.domain[0], split_point
        domain_right = split_point, self.domain[1]
        return LinearInterpolation(density, domain_left), LinearInterpolation(density, domain_right)


class P1Interpolation(Sampler): #TODO: use PW-Legendre class ?
    def __init__(self, density, domain, max_cells, rel_err, theta=0.33):
        # find linear interpolation of density
        parts = [LinearInterpolation(density, domain)]
        max_cells -= 1
        while max_cells > 0:
            errors = [part.error() for part in parts]
            total_error = sum(errors)
            if total_error <= rel_err: break
            indicators = sorted(enumerate(errors), key=itemgetter(1), reverse=True)
            marked_error = 0
            marked_parts = []
            idx = 0
            while marked_error <= theta*total_error and len(marked_parts) < max_cells:
                marked_error += indicators[idx][1]
                marked_parts.append(indicators[idx][0])
                idx += 1
            for idx in marked_parts:
                parts[idx:idx+1] = parts[idx].refine()
            max_cells -= len(marked_parts)

        self.__domain = domain
        self.parts = parts
        self.nodes = [part.domain[0] for part in parts] + [parts[-1].domain[1]]
        self.part_masses = [part.mass for part in parts]
        self.cum_part_masses = np.cumsum(self.part_masses)
        self.__mass = sum(self.part_masses)
        # print("Domain:", domain, "\tCells:", len(parts))

    @property
    def domain(self): return self.__domain
    def density(self, point):
        if np.isscalar(point):
            # print("scalar")
            point = float(point)
            idx = bisect(self.nodes, point) - 1
            return self.parts[idx].density(point)
        else:
            # print("array")
            ret = np.empty_like(point, dtype=float)
            it = np.nditer([point, ret], ["zerosize_ok"], [['readonly'], ['writeonly']])
            for pt,ot in it:
                ot[...] = self.density(float(pt))
            return ret

    @property
    def mass(self): return self.__mass

    def sample(self, *shape):
        # print("P1")
        us = self.mass * np.random.rand(np.prod(shape))
        ret = []
        for u in us:
            idx = bisect(self.cum_part_masses, u) - 1
            ret.append(self.parts[idx].sample(1))  #TODO: efficiency... first get all indices, then compute samples batch-wise
        return np.array(ret).reshape(shape)


class RejectionSampler(Sampler):
    def __init__(self, density, domain, trial_dist=None, max_cells=np.inf, rel_err=1e-3):
        # approximate the distribution by a triangular one and perform a rejection step
        self.__domain = domain
        self.__density = density
        if trial_dist is None:
            trial_dist = P1Interpolation(density, domain, max_cells, rel_err)
            #TODO: wtf???
            # if trial_dist[0].offset == trial_dist[0].slope == 0:
            #     trial_dist = Uniform(domain)
        lhood = lambda x: density(x)/trial_dist.density(x)
        self.__lhood_bound = lhood(maximize(lhood, *domain))
        if np.isnan(self.__lhood_bound):
            trial_dist = Uniform(domain)
            lhood = lambda x: density(x)/trial_dist.density(x)
            self.__lhood_bound = lhood(maximize(lhood, *domain))
        self.__trial_dist = trial_dist
        self.__mass = integrate(density, *domain)[0]

    @property
    def domain(self): return self.__domain
    def density(self, point): return self.__density(point)
    @property
    def mass(self): return self.__mass

    def sample(self, *shape):
        # print("RejectionSampler")
        f = self.density
        g = self.__trial_dist.density
        M = self.__lhood_bound

        size = np.prod(shape)
        res = []
        for _ in range(1000): # maxIter
            ys = self.__trial_dist.sample(size)  # candidates
            us = np.random.rand(size)
            mask = us*M*g(ys) <= f(ys)
            res.append(ys[mask])
            size -= sum(mask)
            if size <= 0: break
        if size>0:
            res.append(np.random.rand(size))  #TODO: fix...
            print("maxIter exceeded")

        ret = np.concatenate(res)
        return ret.reshape(shape)


class Uniform(Sampler):
    def __init__(self, domain, height=None):
        self.__left = domain[0]
        self.__length = length = domain[1] - domain[0]
        if height is None:
            height = 1/length
        else:
            height = max(height-1e-6, 0)
        self.__density = lambda x: np.full_like(x, height, dtype=float)
        self.__mass = height*length
        # print("Sample in:", [self.__left, self.__length+self.__left])
    @property
    def domain(self): return self.__left, self.__left+self.__length
    def density(self, point): return self.__density(point)
    @property
    def mass(self): return self.__mass
    def sample(self, *shape):
        # print("Unfiorm")
        return self.__length*np.random.rand(*shape) + self.__left


class HybridUP1I(Sampler):
    def __init__(self, density, domain):
        x_min = minimize(density, *domain)
        y_min = density(x_min)
        assert y_min >= 0
        self.__p0 = Uniform(domain, y_min)
        self.__res = RejectionSampler(lambda x: density(x) - y_min, domain)

    @property
    def domain(self): return self.__p0.domain
    def density(self, point): return self.__p0.density(point) + self.__res.density(point)
    @property
    def mass(self): return self.__p0.mass + self.__res.mass
    def sample(self, *shape):
        # print("Hybrid")
        size = np.prod(shape)

        us = self.mass*np.random.rand(size)
        if self.__p0.mass > 0:
            markers = (us - self.__p0.mass)/self.__p0.mass <= 1e-3  # relative error: 0.1%
        else:
            markers = np.full(us.shape, False)
        num_p0s = sum(markers)

        ret = np.empty(size)
        ret[markers] = self.__p0.sample(num_p0s)
        ret[~markers] = self.__res.sample(size - num_p0s)

        return ret.reshape(shape)


class Ziggurat(Sampler):
    def __init__(self, density, domain, num_cells):
        total_mass, nodes = partition(density, domain, num_cells)
        # total_mass = integrate(density, *domain)[0]  # ignore error estimate
        # nodes = np.linspace(*domain, num=num_cells+1)
        cell_mass = total_mass / num_cells

        parts = []
        for k in range(len(nodes)-1):
            cell = (nodes[k], nodes[k+1])
            parts.append(HybridUP1I(density, cell))

        self.__domain = domain
        self.__density = density
        self.__mass = sum(p.mass for p in parts)
        self.__parts = parts
        self.__num_cells = num_cells

    @property
    def num_cells(self): return self.__num_cells
    @property
    def domain(self): return self.__domain
    def density(self, point): return self.__density(point)
    @property
    def mass(self): return self.__mass

    def sample_fast(self, *shape):
        #TODO: do not reorder...
        size = np.prod(shape)

        #TODO: das geht schneller!
        cells = np.random.randint(0, self.num_cells, size)

        ret = np.empty(size)
        a = 0
        for e in range(self.num_cells):
            num_cells = np.sum(cells==e)
            o = a + num_cells
            ret[a:o] = self.__parts[e].sample(num_cells)
            a = o

        return ret.reshape(shape)

    def sample(self, *shape):
        # print("Ziggurat")
        size = np.prod(shape)

        cells = np.random.randint(0, self.num_cells, size)

        ret = np.empty(size)
        for e in range(self.num_cells):
            cell_mask = cells==e
            ret[cell_mask] = self.__parts[e].sample(np.sum(cell_mask))

        return ret.reshape(shape)


class ProductSampler(Sampler): pass


class SumSampler(Sampler):
    def __init__(self, densities, domain): pass

    @property
    def domain(self): return self.__domain
    def density(self, point): return self.__density(point)
    @property
    def mass(self): return self.__mass

    def sample_fast(self, *shape): return NotImplemented

    def sample(self, *shape): return NotImplemented


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


def sample_discrete(*shape, weights=None):
    if weights is None:
        raise ValueError("weights has to be a list of weights or the number of uniform weights")
    elif isinstance(weights, int):
        assert weights > 0
        weights = [1]*weights
    ret = np.random.rand(*shape)
    cdf = np.cumsum(weights).astype(np.float64)
    cdf /= cdf[-1]
    return np.searchsorted(cdf, ret)




def histogram(xys, nx=1000, ny=1000, xdom=(-1,1), ydom=(-1,1)):
    xs = xys[:,0]
    ys = xys[:,1]
    if not np.all(xs[:-1] <= xs[1:]):
        print("Sorting xs")
        xs_sorter = np.argsort(xs)
        xs = xs[xs_sorter]
        ys = ys[xs_sorter]

    bins = np.empty((nx, ny))

    print("Splitting xs")
    xsplits = np.searchsorted(xs, np.linspace(*xdom, num=nx+1))
    xsplits[0] = 0

    for i in range(nx):
        print("{}/{}".format(i, nx))
        ys_slice = ys[xsplits[i]:xsplits[i+1]]
        print("Sorting ys-slice")
        ys_slice = np.sort(ys_slice)
        print("Splitting ys-slice")
        ysplits = np.searchsorted(ys_slice, np.linspace(*ydom, num=ny+1))
        ysplits[0] = 0
        bins[i] = np.diff(ysplits)

    return bins

def cart(*arrs):
    return np.stack(np.meshgrid(*arrs), -1).reshape(-1, len(arrs))

def decart(fxy, x, y):
    return fxy.reshape(len(y), len(x))

def discretize(function, nx=1000, ny=1000, xdom=(-1,1), ydom=(-1,1)):
    x = np.linspace(*xdom, num=nx)
    y = np.linspace(*ydom, num=ny)
    return x, y, decart(function(cart(x, y)), x, y)

def density(xys, degree=3):
    def basisSq(d): return lambda x: legval(x, [0]*d+[1])**2
    xys = xys.reshape(-1, 2)
    ret = np.zeros(len(xys))
    for i in range(degree+1):
        for j in range(degree+1):
            Bx = basisSq(i)(xys[:,0])
            By = basisSq(j)(xys[:,1])
            ret += Bx * By
    return ret




if __name__ == '__main__':

    # density = lambda x: np.ones_like(x)
    # density = lambda x: np.exp(-x**2/1e-2)
    # mass, pts = partition(density, (-1,1), 20)
    # x = np.linspace(-1,1,1000)
    # plt.plot(x, density(x))
    # plt.plot(pts, density(pts), 'o')
    # plt.show()

    def plot_sampler(z, title=None):
        plt.hist(z.sample_fast(10000), bins=100, density=True, histtype='stepfilled', label='histogram')
        x = np.linspace(-1, 1, 1000)
        plt.plot(x, z.density(x)/z.mass, label='density')
        plt.legend()
        if title: plt.title(title)
        plt.tight_layout()
        plt.show()

    # density = lambda x: np.ones_like(x)
    # z = Ziggurat(density, (-1,1), 1)
    # plot_sampler(z)

    # density = lambda x: np.ones_like(x)
    # # z = Ziggurat(density, (-1,1), 5)
    # z = Ziggurat(density, (-1,1), 1064)
    # plot_sampler(z)

    # density = lambda x: 1+x
    # z = Ziggurat(density, (-1,1), 1)
    # plot_sampler(z)

    # density = lambda x: 1+x
    # # z = Ziggurat(density, (-1,1), 5)
    # z = Ziggurat(density, (-1,1), 1064)
    # plot_sampler(z)

    # density = lambda x: abs(x)
    # z = Ziggurat(density, (-1,1), 2)
    # plot_sampler(z)

    # density = lambda x: abs(x)
    # z = Ziggurat(density, (-1,1), 1064)
    # plot_sampler(z)

    # density = lambda x: np.exp(-x**2/1e-2)
    # z = Ziggurat(density, (-1,1), 1064)
    # plot_sampler(z)

    # import cProfile #, pstats, StringIO
    # pr = cProfile.Profile()
    # pr.enable()
    # z.sample(10000)
    # pr.disable()
    # print("Dump stats:", 'ziggurat.prof')
    # pr.dump_stats('ziggurat.prof')

    #TODO: Use dP1 instead of dP0 in Ziggurat?

    # #TODO: das geht schneller!
    # density = lambda x: np.exp(-x**2/2)
    # z = Ziggurat(density, (-1,1), 1064)
    # import timeit
    # def timez(): return z.sample_fast(1000)
    # tz = timeit.timeit(timez, number=10)
    # def timenp(): return np.random.randn(1000)
    # tnp = timeit.timeit(timenp, number=10)
    # print("Slowdown:", tz/tnp)

    sample_discrete(100, 100, weights=[1,2,3,4])


    # for d in range(5):
    #     print("Evaluating Legendre Density %d"%d)
    #     density = lambda x: (legval(x, [0]*d+[1]))**2
    #     z = Ziggurat(density, (-1,1), 1064)
    #     plot_sampler(z, "Legendre %d"%d)
    # plt.show()

    # D = 2
    # print("Evaluating Combined Legendre Density %d"%D)
    # density = lambda x: sum((legval(x, [0]*d+[1]))**2 for d in range(D))
    # z = Ziggurat(density, (-1,1), 1064)
    # plot_sampler(z, "Combined Legendre %d"%D)
    # plt.show()

    # D = 8
    # print("Evaluating Combined Legendre Density %d"%D)
    # density = lambda x: sum((legval(x, [0]*d+[1]))**2 for d in range(D))
    # z = Ziggurat(density, (-1,1), 1064)
    # plot_sampler(z, "Combined Legendre %d"%D)
    # plt.show()




    domain = (-1,1)
    degree = 5
    num = 5*10**7

    RESAMPLE = False

    if RESAMPLE:
        def basisSq(d): return lambda x: legval(x, [0]*d+[1])**2
        masses1d = np.array([integrate(basisSq(d), *domain)[0] for d in range(degree+1)])
        masses2d = masses1d[:,None] * masses1d

        #TODO: it is irrelevant if I mess up x and y -- both are symmetric in this case

        idcs = sample_discrete(num, weights=masses2d.reshape(-1))

        # to get them in random order again just use a shuffle!
        counts = [np.sum(idcs == dd) for dd in range(masses2d.size)]
        assert np.sum(counts) == len(idcs)

        GENS = []
        for i in range(degree+1):
            print("Creating Ziggurat {}".format(i))
            GENS.append(Ziggurat(basisSq(i), domain, 1064))

        ret = []
        for i in range(degree+1):
            for j in range(degree+1):
                print("Generating {} Samples from Product ({}, {})".format(counts[(degree+1)*i + j], i, j))
                XGEN = GENS[i]
                YGEN = GENS[j]
                xs = XGEN.sample_fast(counts[(degree+1)*i + j])
                ys = YGEN.sample_fast(counts[(degree+1)*i + j])
                assert len(xs) == len(ys) == counts[(degree+1)*i + j]
                np.random.shuffle(xs)
                np.random.shuffle(ys)
                ret.append(np.stack([xs, ys], -1))
                assert np.all(np.isfinite(ret[-1]))

        retc = np.concatenate(ret)
        # np.random.shuffle(retc)
        np.save('samples.npy', retc)

    else:
        retc = np.load('samples.npy')


    # # The order of the samples in retc is NOT random.
    # print("Shuffling")
    # np.random.shuffle(retc)
    # retc = retc[::50]
    # retc = retc[::10]
    retc = retc[::1]

    from functools import partial
    x, y, density = discretize(partial(density, degree=5), 200, 200)
    bins = histogram(retc, 200, 200)

    # normalization
    density /= np.max(density)
    bins /= np.max(bins)

    f, ax = plt.subplots(1,3)
    ax[0].contourf(x, y, density)
    ax[1].contourf(x, y, bins)
    im = ax[2].contourf(x, y, abs(density - bins))
    plt.colorbar(im)
    plt.tight_layout()

    f, ax = plt.subplots(1,1)
    ax.contour(x, y, abs(density - bins))
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False, top=False, left=False, right=False,
        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    plt.tight_layout()

    plt.show()

    exit()
