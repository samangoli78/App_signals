import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import spsolve, splu


def laplace_interpolate_from_sparse(V, F, known_idx, known_val, lam=0.0):
    """
    Interpolate scalar from sparse constraints via harmonic extension on a triangle mesh.
    (L + lam*M) f = 0 on unknowns, with Dirichlet f = known_val on known_idx.
    """
    V = np.asarray(V, np.float64); F = np.asarray(F, np.int32)
    known_idx = np.asarray(known_idx, np.int64)
    known_val = np.asarray(known_val, np.float64)

    # Build operators using your function
    L, M = build_cotan_laplacian_and_mass(V, F)
    A = L + (lam * M if lam > 0 else 0)
    n = V.shape[0]

    # Deduplicate constraints (mean if multiple map to same vertex)
    order = np.argsort(known_idx)
    ki = known_idx[order]; kv = known_val[order]
    uniq, start = np.unique(ki, return_index=True)
    sums = np.add.reduceat(kv, start)
    counts = np.diff(np.append(start, len(kv)))
    fk = sums / counts

    mask_known = np.zeros(n, dtype=bool)
    mask_known[uniq] = True
    uk = uniq
    uu = np.where(~mask_known)[0]

    # Partition system
    Auu = A[uu][:, uu]
    Auk = A[uu][:, uk]

    rhs = -Auk @ fk
    fu = spsolve(Auu, rhs)

    f = np.zeros(n, dtype=np.float64)
    f[uk] = fk
    f[uu] = fu
    return f

def cotangent(a, b, c):
    # cot at a in triangle (a,b,c)
    u = b - a; v = c - a
    num = np.dot(u, v)
    den = np.linalg.norm(np.cross(u, v)) + 1e-15
    return num / den

def build_cotan_laplacian_and_mass(V, F):
    n = V.shape[0]
    I = []; J = []; W = []
    diag = np.zeros(n, dtype=np.float64)

    # per-vertex lumped area (Voronoi/mixed simplified by 1/3 area per incident tri)
    area = np.zeros(n, dtype=np.float64)

    for tri in F:
        i, j, k = tri
        vi, vj, vk = V[i], V[j], V[k]

        # triangle area
        A = 0.5 * np.linalg.norm(np.cross(vj - vi, vk - vi))
        a_i = A / 3.0
        area[i] += a_i; area[j] += a_i; area[k] += a_i

        # cotan weights
        ci = cotangent(vi, vj, vk)  # opposite edge (j,k)
        cj = cotangent(vj, vk, vi)  # opposite (k,i)
        ck = cotangent(vk, vi, vj)  # opposite (i,j)

        # edges accumulate symmetric weights w = (cot_alpha + cot_beta)/2
        for (a, b, w) in [(j, k, ci), (k, i, cj), (i, j, ck)]:
            I += [a, b]; J += [b, a]; W += [-w/2.0, -w/2.0]
            diag[a] += w/2.0; diag[b] += w/2.0

    L = coo_matrix((W + list(diag), (I + list(range(n)), J + list(range(n)))), shape=(n, n)).tocsr()

    # mass matrix (lumped)
    M = diags(area.clip(min=1e-15))
    return L, M

def heat_kernel_smooth_scalar(V, F, s, radius, nsteps=1):
    """
    Smooth scalar 's' on mesh (V,F) by heat-kernel with geodesic radius ~ 'radius'.
    Uses implicit Euler: (M + t L) u = M u0, repeated nsteps times (optional).
    """
    V = np.asarray(V, np.float64)
    F = np.asarray(F, np.int32)
    s = np.asarray(s, np.float64)

    L, M = build_cotan_laplacian_and_mass(V, F)

    # pick t from desired radius; sigma^2 ≈ 2t, kernel ~ exp(-d^2/(4t))
    t = (radius**2) / 4.0

    A = (M + t * L).tocsr()
    lu = splu(A.tocsc())  # factorize once; reuse if doing multiple steps

    u = s.copy()
    for _ in range(nsteps):
        rhs = M @ u
        u = lu.solve(rhs)

    return u



