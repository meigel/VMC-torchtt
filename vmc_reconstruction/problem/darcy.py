# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
from dolfin import *
from field.testfield import TestField
import os, tempfile # fenics cannot create or write meshes from or to other things than files...
set_log_level(WARNING)
import copy_reg
import types

# def _pickle_method(method):
#     func_name = method.im_func.__name__
#     obj = method.im_self
#     cls = method.im_class
#     return _unpickle_method, (func_name, obj, cls)
# def _unpickle_method(func_name, obj, cls):
#     for cls in cls.mro():
#         try: func = cls.__dict__[func_name]
#         except KeyError: pass
#         else: break
#     return func.__get__(obj, cls)

def _pickle_method(method):
    obj = method.im_self
    name = method.im_func.__name__
    return _unpickle_method, (obj, name)

def _unpickle_method(obj, name):
    return getattr(obj, name)

def _pickle_mesh(mesh):
    with tempfile.NamedTemporaryFile(prefix='mesh_', suffix='.xml') as f:
        File(f.name) << mesh
        xml = f.read()
    return _unpickle_mesh, (xml,)

def _unpickle_mesh(xml):
    with tempfile.NamedTemporaryFile(prefix='mesh_', suffix='.xml') as f:
        f.write(xml)
        f.flush()
        mesh = Mesh(f.name)
    return mesh

copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)
copy_reg.pickle(Mesh, _pickle_mesh, _unpickle_mesh)


class Problem(object):
    def __init__(self, info):
        assert info['problem']['name'] == "darcy"
        mesh_info = info['fe']['mesh']
        if isinstance(mesh_info, int):
            mesh = UnitSquareMesh(mesh_info, mesh_info)
        else:
            mesh = Mesh(mesh_info)
        self.__setstate__({
            'info': info,
            'mesh': mesh,
            })

    def __getstate__(self):
        _dict = {
            'info': self.info,
            'mesh': self.mesh,
        }
        return _dict

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.name = self.info['problem']['name']

        # setup fe space
        self.degree = self.info['fe']['degree']
        self.space = FunctionSpace(self.mesh, 'CG', self.degree)

        # setup random field
        M = self.info['expansion']['size']
        mean = self.info['expansion']['mean']
        scale = self.info['expansion']['scale']
        sigma = self.info['expansion']['decay rate']
        # expfield = self.info['expansion']['lognormal']
        expfield = self.info['sampling']['distribution'] == 'normal'
        self.field = TestField(M, mean=mean, expfield=expfield, scale=scale, sigma=sigma)
        #TODO: assert self.info['sampling']['distribution'] == ['uniform', 'normal'][int(self.info['expansion']['lognormal'])]

        # define forcing term
        self.forcing = Constant(1)

    def solution(self, param):
        V = self.space
        f = self.forcing
        kappa = self.field.realisation(param, V)

        u = TrialFunction(V)
        v = TestFunction(V)
        a = kappa * inner(grad(u), grad(v)) * dx
        L = f * v * dx

        # define boundary condition
        bc = DirichletBC(V, Constant(0.0), 'on_boundary')

        u = Function(V)
        solve(a == L, u, bc)

        return u.vector().array()

    def residuum(self, y_u):
        y,u_vec = y_u
        M = self.info['expansion']['size']
        assert y.shape == (M,)

        V = self.space
        f = self.forcing
        kappa = self.field.realisation(y, V)

        u = TrialFunction(V)
        v = TestFunction(V)
        a = kappa * inner(grad(u), grad(v)) * dx
        L = f * v * dx

        bc = DirichletBC(V, Constant(0.0), 'on_boundary')

        A, b = assemble_system(a, L, bc)
        u = Function(V).vector()
        u[:] = u_vec
        res = (A*u - b).array()
        assert res.shape == u_vec.shape
        return res

    def residual_estimator(self, y_u):
        y,u_vec = y_u
        V = self.space
        u = Function(V)
        u.vector()[:] = u_vec
        f = self.forcing
        kappa = self.field.realisation(y, V)

        # setup indicator
        mesh = V.mesh()
        h = CellSize(mesh)
        DG0 = FunctionSpace(mesh, 'DG', 0)
        dg0 = TestFunction(DG0)

        kappa = self.field.realisation(y, V)
        R_T = -(f + div(kappa * grad(u)))
        R_dT = kappa * grad(u)
        J = jump(R_dT)
        indicator = h ** 2 * (1 / kappa) * R_T ** 2 * dg0 * dx + avg(h) * avg(1 / kappa) * J **2 * 2 * avg(dg0) * dS

        # prepare indicators
        eta_res_local = assemble(indicator, form_compiler_parameters={'quadrature_degree': -1})
        return y_u, eta_res_local.array()

    def refine_mesh(self, marked_cells):
        marker = MeshFunction("bool", self.mesh, self.mesh.topology().dim())
        marker.set_all(False)
        for idx in marked_cells: marker[idx] = True
        self.mesh = refine(self.mesh, marker)
        self.space = FunctionSpace(self.mesh, 'CG', self.degree)

    def dofs(self): return len(Function(self.space).vector())

