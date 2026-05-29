import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def _tri_areas(V, F):
    v0, v1, v2 = V[F[:,0]], V[F[:,1]], V[F[:,2]]
    return 0.5*np.linalg.norm(np.cross(v1-v0, v2-v0), axis=1)

def mass_matrix_lumped(V, F):
    A = _tri_areas(V, F)
    I = F.reshape(-1)
    W = np.repeat(A/3.0, 3)
    n = V.shape[0]
    return sp.coo_matrix((W, (I, I)), shape=(n, n)).tocsr()

def laplacian_cotan(V, F, eps=1e-12):
    v0, v1, v2 = V[F[:,0]], V[F[:,1]], V[F[:,2]]
    e0, e1, e2 = v2-v1, v0-v2, v1-v0                      # edges opposite v0,v1,v2
    def cot(a, b):
        dot = (a*b).sum(-1)
        nrm = np.linalg.norm(np.cross(a,b), axis=1) + eps
        return dot / nrm
    c0, c1, c2 = cot(e1,e2), cot(e2,e0), cot(e0,e1)

    n = V.shape[0]
    i0,i1,i2 = F[:,0],F[:,1],F[:,2]
    I = np.concatenate([i0,i1,i2, i1,i2,i0, i2,i0,i1])
    J = np.concatenate([i1,i2,i0, i0,i1,i2, i0,i1,i2])
    W = 0.5*np.concatenate([c2,c0,c1, c2,c0,c1, c2,c0,c1])
    L = sp.coo_matrix((-W, (I, J)), shape=(n, n)).tocsr()
    L.setdiag(-np.asarray(L.sum(axis=1)).ravel())
    return L

def fem_heat_smooth(V, F, t=0.1):
    """Solve (M + t L) X = M X0."""
    V0 = np.asarray(V, float)
    M  = mass_matrix_lumped(V0, F)
    L  = laplacian_cotan(V0, F)
    A  = (M + t*L).tocsc()
    B  = M @ V0
    solver = spla.factorized(A)
    X = np.column_stack([solver(B[:,k]) for k in range(3)])
    return X.astype(np.float32)

def fem_biharmonic_smooth(V, F, lam=1e-3):
    """Solve (L^T M^{-1} L + lam M) X = lam M X0."""
    V0 = np.asarray(V, float)
    M  = mass_matrix_lumped(V0, F).tocsr()
    L  = laplacian_cotan(V0, F).tocsr()
    Minv = sp.diags(1.0 / (M.diagonal() + 1e-18))
    A  = (L.T @ Minv @ L) + lam * M
    B  = lam * (M @ V0)
    solver = spla.factorized(A.tocsc())
    X = np.column_stack([solver(B[:,k]) for k in range(3)])
    return X.astype(np.float32)
