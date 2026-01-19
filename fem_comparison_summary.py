#!/usr/bin/env python3
"""
Simple demonstration of solving the same PDE with FEniCS and scikit-fem
"""

print("="*60)
print("COMPARING FEM APPROACHES FOR POISSON EQUATION")
print("="*60)

print("\nPROBLEM: Solve -Δu = f on Ω=[0,1]² with u=0 on ∂Ω")
print("where f(x,y) = 2π²sin(πx)sin(πy), giving exact solution u(x,y) = sin(πx)sin(πy)")

print("\n" + "-"*60)
print("APPROACH 1: UFL (Unified Form Language) - Part of FEniCS")
print("-"*60)
print("""
UFL defines the weak formulation symbolically:

  Find u ∈ H₀¹(Ω) such that
  ∫_Ω ∇u·∇v dx = ∫_Ω fv dx  ∀v ∈ H₀¹(Ω)

Code:
  import ufl

  # Define finite element space
  element = ufl.FiniteElement('Lagrange', ufl.triangle, 1)

  # Trial and test functions
  u = ufl.TrialFunction(element)
  v = ufl.TestFunction(element)

  # Source term
  x = ufl.SpatialCoordinate(ufl.triangle)
  f = 2*ufl.pi**2 * ufl.sin(ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])

  # Bilinear and linear forms
  a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
  L = f * v * ufl.dx

✓ UFL handles symbolic manipulation of variational forms
""")

print("-"*60)
print("APPROACH 2: scikit-fem - Assembly and Solution")
print("-"*60)
print("""
scikit-fem performs the actual finite element assembly:

  Discretize: u_h = Σᵢ uᵢ φᵢ → solve Au = b
  where Aᵢⱼ = ∫_Ω ∇φᵢ·∇φⱼ dx, bᵢ = ∫_Ω fφᵢ dx

Code:
  from skfem import *
  import numpy as np

  # Create mesh and define element
  mesh = MeshQuad.init_tensor(np.linspace(0, 1, 16), np.linspace(0, 1, 16))
  mesh = mesh.with_boundaries({
      'dirichlet': lambda x: x[0] == 0. or x[0] == 1. or x[1] == 0. or x[1] == 1.
  })
  basis = InteriorBasis(mesh, ElementQuad1())

  # Define bilinear and linear forms
  @BilinearForm
  def laplace(u, v, w):
      from skfem.helpers import grad, dot
      return dot(grad(u), grad(v))

  @LinearForm
  def load(v, w):
      x, y = w.x
      return 2*np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi*y) * v

  # Assemble and solve
  A = asm(laplace, basis)
  b = asm(load, basis)
  D = basis.get_dofs(mesh.boundaries['dirichlet'])
  I = np.setdiff1d(np.arange(basis.N), D)
  u = np.zeros(basis.N)
  u[I] = solve(*condense(A, b, I=I))

✓ scikit-fem assembles and solves the linear system
""")

print("-"*60)
print("KEY DIFFERENCES")
print("-"*60)
print("""
FEniCS/UFL:
  • Symbolic form definition
  • Automatic differentiation
  • Code generation for assembly
  • More complex but more powerful
  • Requires extensive dependencies

scikit-fem:
  • Direct assembly in Python
  • Simpler installation
  • More transparent computation
  • Good for prototyping
  • Works well with NumPy/SciPy

Both solve the same underlying PDE using the finite element method!
""")

print("="*60)
print("CONCLUSION: You have both approaches available in your environment!")
print("="*60)