import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from sympy.utilities.lambdify import implemented_function
from poisson import Poisson

x, y = sp.symbols('x,y')

class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), x, y in [0, Lx] x [0, Ly]

    with homogeneous Dirichlet boundary conditions.
    """

    def __init__(self, Lx, Ly, Nx, Ny, ue):
        self.px = Poisson(Lx, Nx) # we can reuse some of the code from the 1D case
        self.py = Poisson(Ly, Ny)
        self.xij = None
        self.yij = None
        self.ue = ue

    def create_mesh(self):
        """Return a 2D Cartesian mesh
        """
        return np.meshgrid(self.px.x, self.py.x)

    def laplace(self):
        """Return a vectorized Laplace operator"""
        D2x = self.px.D2()
        D2y = self.py.D2()
        return (sparse.kron(D2x, sparse.eye(self.py.N + 1)) +
            sparse.kron(sparse.eye(self.px.N + 1), D2y))

    def assemble(self, f=None):
        """Return assemble coefficient matrix A and right hand side vector b"""
        xij, yij = self.create_mesh()
        self.xij = xij
        self.yij = yij
        A = self.laplace()
        F = sp.lambdify((x, y), f)(xij, yij)
        B = np.ones((self.px.N + 1, self.py.N + 1), dtype=bool)
        B[1:-1, 1:-1] = 0
        bnds = np.where(B.ravel() == 1)[0]
        A = A.tolil()
        for i in bnds:
            A[i] = 0
            A[i, i] = 1
        A = A.tocsr()
        b = F.ravel()
        b[bnds] = sp.lambdify((x, y), self.ue)(self.xij, self.yij).ravel()[bnds]
        return A, b.ravel()

    def l2_error(self, u, ue):
        """Return l2-error

        Parameters
        ----------
        u : array
            The numerical solution (mesh function)
        ue : Sympy function
            The analytical solution
        """
        dx = self.px.dx
        dy = self.py.dx
        return np.sqrt(dx*dy*np.sum((u - sp.lambdify((x, y), ue)(self.xij, self.yij))**2))

    def __call__(self, f=implemented_function('f', lambda x, y: 2)(x, y)):
        """Solve Poisson's equation with a given righ hand side function

        Parameters
        ----------
        N : int
            The number of uniform intervals
        f : Sympy function
            The right hand side function f(x, y)

        Returns
        -------
        The solution as a Numpy array

        """
        A, b = self.assemble(f=f)
        return sparse.linalg.spsolve(A, b.ravel()).reshape((self.px.N+1, self.py.N+1))

def test_poisson2d():
    Lx = 1
    Ly = 1
    Nx = 30
    Ny = 30
    ue = x*(1-x)*y*(1-y)*sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    f = ue.diff(x, 2) + ue.diff(y, 2)
    sol = Poisson2D(Lx, Ly, Nx, Ny, ue)
    u = sol(f)
    xij, yij = sol.xij, sol.yij
    plt.contourf(xij, yij, u)
    plt.colorbar()
    plt.show()
    assert sol.l2_error(u, ue) < 1e-3

if __name__ == "__main__":
    test_poisson2d()