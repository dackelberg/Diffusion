import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Covariance matrices
SIGMA1 = np.matrix([[2.3, 0], [0, 1.5]])
SIGMA2 = np.matrix([[1, 0], [0, 3]])

# Mean vectors
MU1 = np.array([2, 3])
MU2 = np.array([0, 0])

# Small increment for numerical gradient approximation
EPSILON = 0.0001

def densite_mixture(x, mu1=MU1, mu2=MU2, sigma1=SIGMA1, sigma2=SIGMA2):
    """Computes a mixture density given two Gaussian distributions."""
    if len(x) != len(mu1) or len(x) != len(mu2):
        raise ValueError("Dimension mismatch between input and mean vectors.")
    if sigma1.shape != (len(x), len(x)) or sigma2.shape != (len(x), len(x)):
        raise ValueError("Covariance matrices must be square and match input dimension.")
    
    x_mu1 = np.matrix(x - mu1)
    x_mu2 = np.matrix(x - mu2)

    inv1 = sigma1.I
    inv2 = sigma2.I

    result = (math.exp(-0.5 * (x_mu1 @ inv1 @ x_mu1.T)) +
              math.exp(-0.5 * (x_mu2 @ inv2 @ x_mu2.T)))
    return result

def densite():
    """Returns a function for computing density values."""
    return lambda x: densite_mixture(x)

def numerical_gradient(f):
    """Computes the numerical gradient of a function f at a point x."""
    def grad_f(x):
        return [
            (f([x[0] + EPSILON, x[1]]) - f(x)) / EPSILON,
            (f([x[0], x[1] + EPSILON]) - f(x)) / EPSILON
        ]
    return grad_f

def get_gradient_function():
    """Returns the gradient function of the density."""
    return numerical_gradient(densite())

def affichage(points, grid_size=25, bounds=5):
    """Displays the gradient field with sample points."""
    absc = np.linspace(0, bounds, grid_size)
    ordo = np.linspace(0, bounds, grid_size)
    X, Y = np.meshgrid(absc, ordo)

    dens = densite()
    grad = get_gradient_function()

    points_array = np.column_stack([X.ravel(), Y.ravel()])
    Z = np.array([dens(pt) for pt in points_array]).reshape(grid_size, grid_size)
    U = np.array([grad(pt)[0] for pt in points_array]).reshape(grid_size, grid_size)
    V = np.array([grad(pt)[1] for pt in points_array]).reshape(grid_size, grid_size)

    # Create plot
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.quiver(X, Y, U, V, color="black", alpha=0.6)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_aspect('equal')

    ax.contour(Z, cmap=plt.cm.RdBu, extent=[0, bounds, 0, bounds])
    plt.scatter(points[:, 0], points[:, 1], color='red', marker='o')

    # Show plot
    plt.show()

def dynamic_affichage(points, steps=30, bounds=5):
    """Animates points moving along the gradient field over time."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, bounds)
    ax.set_ylim(0, bounds)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

    absc = np.linspace(0, bounds, 25)
    ordo = np.linspace(0, bounds, 25)
    X, Y = np.meshgrid(absc, ordo)

    dens = densite()
    grad = get_gradient_function()

    points_array = np.column_stack([X.ravel(), Y.ravel()])
    Z = np.array([dens(pt) for pt in points_array]).reshape(25, 25)
    U = np.array([grad(pt)[0] for pt in points_array]).reshape(25, 25)
    V = np.array([grad(pt)[1] for pt in points_array]).reshape(25, 25)

    ax.contour(X, Y, Z, cmap=plt.cm.RdBu, extent=[0, bounds, 0, bounds])
    ax.quiver(X, Y, U, V, color="black", alpha=0.6)

    scatter = ax.scatter(points[:, 0, 0], points[:, 0, 1], color='red')

    def update(frame):
        scatter.set_offsets(points[:, frame, :])  # Update positions
        return (scatter,)

    ani = animation.FuncAnimation(fig, update, frames=steps, interval=50, blit=True)
    return HTML(ani.to_jshtml())
