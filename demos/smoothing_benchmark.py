import os
import sys
from pathlib import Path

# Base libraries
import time
import igl
import numpy as np
from scipy import sparse



# Multigrid Solver
from gravomg import MultigridSolver, Hierarchy, Sampling
from gravomg.util import neighbors_from_stiffness, normalize_area, knn_undirected, neighbors_from_faces

from robust_laplacian import mesh_laplacian, point_cloud_laplacian
# Experiment util
from util import read_mesh

# Read mesh
if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} <filename>")
    sys.exit(1)
    
file_name = sys.argv[1]

V, F = read_mesh(Path(file_name).resolve())

num_v = V.shape[0]

print(f'Mesh loaded, {V.shape[0]} vertices')
 
# Normalize area and center around mean
V = normalize_area(V, F)

# Compute operators
M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_VORONOI)
S = -igl.cotmatrix(V, F)

Minv = sparse.diags(1 / M.diagonal())
neigh = neighbors_from_stiffness(S)

# Create reusable solver
t = time.perf_counter()
solver = MultigridSolver(V, 
                         neigh, 
                         M, 
                         verbose=True, 
                         ratio=8, 
                         lower_bound=1000, 
                         tolerance=1e-6, 
                         max_iter=100, 
                         sampling_strategy=Sampling.FASTDISK)
print(f'Our construction time: {time.perf_counter() - t}')


ui_tau = 0.0001

def smoothing(tau):
    lhs = M + tau * S
    lhs_csr = lhs.tocsr()
    
    rhs = M @ V

    t = time.perf_counter()
    mg_V = solver.solve(lhs_csr, rhs)
    print(f'Our time: {time.perf_counter() - t}')
    print(f'Our residual: {solver.residual(lhs, rhs, mg_V)}')   


smoothing(ui_tau)

solver.write_convergence(f"benchmark\{Path(file_name).stem}_v_{num_v}.txt")