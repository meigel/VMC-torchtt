from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
import xerus as xe


# def TTapply(tt, x, basis):
#     tt = TTto_ndarray_components(tt)
#     phys = tt[0]
#     phys.shape = phys.shape[1:]
#     stoch = tt[1:]
#     assert len(x) == len(stoch)

#     stochastic_dim = max(c.shape[1] for c in stoch)

#     if basis == xe.PolynomBasis.Legendre:
#         from numpy.polynomial.legendre import legval
#         factors = np.sqrt(2*np.arange(stochastic_dim) + 1)  # normalization
#         poly = lambda x: (factors[:,None] * legval(x, np.eye(stochastic_dim))).T
#     elif basis == xe.PolynomBasis.Hermite:
#         from numpy.polynomial.hermite_e import hermeval
#         from scipy.special import factorial
#         factors = 1/np.sqrt(factorial(np.arange(stochastic_dim), exact=True))  # normalization
#         poly = lambda x: (factors[:,None] * hermeval(x, np.eye(stochastic_dim))).T
#     else:
#         raise NotImplementedError("ERROR: unknown basis '{}'".format(basis))

#     pos = poly(x)
#     assert pos.shape == (len(x), stoch[0].shape[1])

#     cum = np.ones((1,))
#     for y,c in reversed_zip(pos,stoch):
#         cum = np.tensordot(c, cum, (2,0))
#         cum = np.tensordot(cum, y, (1,0))
#         assert cum.ndim == 1
#     cum = np.tensordot(phys, cum, (1,0))
#     return cum


class Measure(object):
    __metaclass__ = ABCMeta
    @abstractmethod
    def __call__(self, points): pass
    @abstractproperty
    def dimensions(self): pass


class IdentityMeasure(Measure):
    def __init__(self, shape):
        self.shape = shape
    @property
    def dimensions(self): return self.shape[:1]
    def __call__(self, points):
        points = np.array(points)
        assert np.shape(points)[-len(self.shape):] == self.shape
        return points[None]


class FunctionMeasure(Measure):
    def __init__(self, function, dimension):
        """
        function : (...,) --> (...,d) where d = dimension
        """
        assert callable(function)
        assert isinstance(dimension, int)
        self.__dimensions = (dimension,)
        self.__function = function
    @property
    def dimensions(self): return self.__dimensions
    def __call__(self, points):
        """
        (...,) --> (1, ..., d) where d = self.dimensions[0]
        """
        ret = self.__function(points)
        assert np.shape(ret) == np.shape(points) + self.dimensions
        return ret[None]


class BasisMeasure(FunctionMeasure):
    def __init__(self, basis): super(BasisMeasure, self).__init__(basis, len(basis))


def log2(num):
    assert isinstance(num, int)
    assert num > 0
    ld = num.bit_length() - 1
    assert 2**ld == num
    return ld


class QTTMeasure(Measure):
    def __init__(self, measure):
        assert isinstance(measure, Measure)
        assert len(measure.dimensions) == 1
        assert measure.dimensions[0] > 0
        self.__measure = measure
        self.__order = log2(self.__measure.dimensions[0])

    @staticmethod
    def convert_vector(v):
        assert np.ndim(v) == 1
        d = log2(len(v))
        v = np.reshape(v, (2,)*d)
        t = xe.Tensor.from_ndarray(v)
        tt = xe.TTTensor(t)
        tt.round(1)
        assert xe.frob_norm(t - xe.Tensor(tt)) < 1e-12
        return [tt.get_component(k).to_ndarray()[0,:,0] for k in range(d)]

    @property
    def dimensions(self): return (2,)*self.__order

    def __call__(self, points):
        meas = self.__measure(points)
        mdims = self.__measure.dimensions
        assert np.shape(meas) == (1,) + np.shape(points) + mdims
        meas = [QTTMeasure.convert_vector(m) for m in meas.reshape(-1,mdims[0])]
        shape = np.shape(points) + (self.__order, 2)
        return np.moveaxis(np.reshape(meas, shape), -2, 0)


class MeasurementList(object):
    def __init__(self, measures=[]):
        assert all(isinstance(m, Measure) for m in measures)
        self.measures = measures

    def __call__(self, points):
        """
        Evaluates the vector of basis functions at the given point.

        points.shape == (M,N)  (points is a list of ndarrays of shape: shape(points[j]) == (N,) + self.dimensions[j])
        output.shape == (M,N,*)

        The shape of points and output where chosen for the following reasons:
            - points may not be a homogeneous array but the input to each mode is homogeneous.
              Example:
                  self = MeasurementList([IdentityMeasure(shape), BasisMeasure(basis)])
                  self(points)
                  len(points) == 2
                  np.shape(points[0]) == (N, 10)
                  np.shape(points[1]) == (N, 1)
        """
        # if len(points) < len(self.measures):
        #     raise ValueError("Number of measurements too small: {} vs {}".format(len(points), len(self.measures)))
        assert len(points) == len(self.measures), "{} vs {}".format(len(points), len(self.measures))
        return sum([list(m(cmp)) for m,cmp in zip(self.measures, points)], [])

    def evaluate(self, tensor, points):
        """
        contract(tensor, self(point))
        """
        #TODO: specialization for xe.TTTensor
        assert tuple(tensor.dimensions) == self.dimensions, "{} vs {}".format(tuple(tensor.dimensions), self.dimensions)

        def xe_kron(val):
            ret = xe.TensorNetwork(xe.Tensor.ones([]))
            a,e,i,o = xe.indices(4)  # [a]ccumulation, [e]xtra, [i]nput, [o]utput
            tensor = xe.Tensor.from_ndarray
            ed = 0
            for v in val:
                od = v.ndim - 1
                ret(a&(1+ed+od), i, e^ed, o^od) << ret(a&ed, e^ed) * tensor(v)(i, o^od)
                ed += od
            return ret

        def prod(t1, t2):
            i,e = xe.indices(2)
            ret = xe.Tensor()
            id = t1.order()
            ret(e&0) << t1(i&0) * t2(i^id,e&id)
            return ret

        assert len(points) == len(self)
        num = np.shape(points[0])[0]
        for pt in points[1:]:
            assert num == np.shape(pt)[0]
        transpose = lambda ls: list(zip(*ls))
        return [prod(tensor, xe_kron(v)) for v in transpose(self(points))]

    @property
    def dimensions(self): return sum((m.dimensions for m in self.measures), ())

    def __len__(self): return len(self.measures)


if __name__ == '__main__':
    from itertools import product
    from basis import Piecewise, LegendrePolynomials

    def P1Basis(nodes):
        dup = lambda arr: np.stack([arr[:-1], arr[1:]]).T
        return Piecewise([LegendrePolynomials(dom, 1) for dom in dup(nodes)])

    def map_to_qtt(bxs):
        ret = [[QTTMeasure.convert_vector(bxi) for bxi in bx] for bx in bxs]
        return np.moveaxis(ret, -2, 1).reshape(-1,np.shape(bxs)[1],2)

    es = np.eye(10)

    print("Testing vector-valued IdentityMeasure (Tensor) ...")
    for d in range(1,5):
        ml = MeasurementList([IdentityMeasure((10,))]*d)
        xs = np.moveaxis(list(product(es, repeat=d)), 0, 1)
        meas = ml(xs)
        assert np.shape(meas) == xs.shape
        assert np.all(meas == xs)
        t = xe.Tensor.random(ml.dimensions)
        vals = ml.evaluate(t, xs)
        vals = np.array([val.to_ndarray() for val in vals])
        assert np.allclose(t.to_ndarray().reshape(-1) - vals, 0)

    print("Testing vector-valued IdentityMeasure (TTTensor) ...")
    for d in range(1,5):
        ml = MeasurementList([IdentityMeasure((10,))]*d)
        xs = np.moveaxis(list(product(es, repeat=d)), 0, 1)
        meas = ml(xs)
        assert np.shape(meas) == xs.shape
        assert np.all(meas == xs)
        t = xe.TTTensor.random(ml.dimensions, [4]*(d-1))
        vals = ml.evaluate(t, xs)
        vals = np.array([val.to_ndarray() for val in vals])
        assert np.allclose(xe.Tensor(t).to_ndarray().reshape(-1) - vals, 0)

    print("Testing matrix-valued IdentityMeasure ...")
    for d in range(1,5):
        ml = MeasurementList([IdentityMeasure((10,10))]*d)
        xs = [[es, 2*es]]*d
        assert ml.dimensions == (10,)*d
        t = xe.Tensor.random(ml.dimensions)
        vals = ml.evaluate(t, xs)
        assert len(vals) == np.shape(xs)[1]
        for e,val in enumerate(vals, start=1):
            assert xe.frob_norm(e**d*t-val) < 1e-12

    print("Testing BasisMeasure ...")
    basis = LegendrePolynomials([-1,1], 15)
    for d in range(1,5):
        measures = [BasisMeasure(basis)]*d
        ml = MeasurementList(measures)
        xs = 2*np.random.rand(d,100) - 1
        meas = ml(xs)
        assert np.shape(meas) == np.shape(xs) + (len(basis),)
        assert np.all(meas == basis(xs))
        t = xe.TTTensor.random(ml.dimensions, [4]*(d-1))
        vals = ml.evaluate(t, xs)
        assert np.shape(vals) == (xs.shape[1],)

    print("Testing QTTMeasure ...")
    p = 6  # len(basis) == 2**p
    basis = P1Basis(np.linspace(-1,1, 2**(p-1)+1))
    for d in range(1,5):
        measures = [QTTMeasure(BasisMeasure(basis))]*d
        ml = MeasurementList(measures)
        assert ml.dimensions == (2,)*p*d
        xs = 2*np.random.rand(d,100) - 1
        meas = ml(xs)
        assert np.shape(meas) == (p*d, np.shape(xs)[1], 2)
        bxs_qtt = map_to_qtt(basis(xs))
        assert np.shape(meas) == np.shape(bxs_qtt)
        assert np.all(np.equal(bxs_qtt, meas))
        t = xe.TTTensor.random(ml.dimensions, [p]*(p*d-1))
        vals = ml.evaluate(t, xs)
        vals = np.array([val.to_ndarray() for val in vals])
        assert vals.shape == (xs.shape[1],)
        measures_ref = [BasisMeasure(basis)]*d
        ml_ref = MeasurementList(measures)
        t_ref = xe.Tensor(t)
        t_ref.reinterpret_dimensions(ml_ref.dimensions)
        vals_ref = ml_ref.evaluate(t_ref, xs)
        vals_ref = np.array([val.to_ndarray() for val in vals_ref])
        assert np.allclose(vals, vals_ref)

    print("Testing heterogeneous MeasurementList ...")
    N = 100
    fem_dofs = 600

    measures = [
        IdentityMeasure((fem_dofs,fem_dofs)),
        BasisMeasure(LegendrePolynomials([-1,1], 11)),
        QTTMeasure(BasisMeasure(P1Basis(np.linspace(-1,1, 2**7+1))))
    ]
    ml = MeasurementList(measures)

    id = np.eye(fem_dofs)
    xs = 2*np.random.rand(2, N) - 1
    pts = [[id]*N, xs[0], xs[1]]
    meas = ml(pts)
    t = xe.TTTensor.random(ml.dimensions, [4]*(len(ml.dimensions)-1))
    vals = ml.evaluate(t, pts)
    vals = np.array([val.to_ndarray() for val in vals])
    assert vals.shape == (N, fem_dofs)
