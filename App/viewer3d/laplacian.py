"""Cotangent Laplacian + harmonic (Dirichlet) interpolation on a triangle mesh.

We solve  L f = 0  with  f[anchors] = anchor_values  to spread sparse electrode
measurements smoothly across the mesh. The cotangent Laplacian is the standard
discrete Laplace-Beltrami operator and gives angle-aware, geometrically
faithful interpolation (much nicer than nearest-neighbour or graph Laplacian).

By default, **negative cotangent contributions are clamped to zero** when
assembling edge weights. On obtuse triangles the raw cotan weights can be
negative, which breaks the discrete maximum principle and can introduce
spurious interior extrema while still formally minimizing a sign-indefinite
quadratic form. The clamped operator is a symmetric M-matrix (weighted graph
Laplacian): Dirichlet data at anchors are enforced exactly, and harmonic
values in the interior minimize a **positive** Dirichlet energy with
better-behaved discrete solutions.

This module is pure numpy / scipy.sparse, so it works wherever the rest of the
app does.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _solve_neumann_patch(
    L: sp.spmatrix,
    free: np.ndarray,
    anchor_idx: np.ndarray,
    anchor_val: np.ndarray,
) -> np.ndarray:
    """Solve the harmonic Dirichlet/Neumann problem on a connected sub-patch.

    The sub-mesh consists of ``free`` (unknowns) ∪ ``anchor_idx`` (Dirichlet).
    Vertices NOT in this patch are excluded; their would-be contributions are
    treated as Neumann (zero normal flux) — the discrete analogue is to
    rebuild the diagonal of the sub-Laplacian from in-patch off-diagonal sums
    only, not from the full mesh's row sum.

    Convention: ``L`` follows ``L[i,j] = -w_ij`` for ``j != i`` and
    ``L[i,i] = sum_j w_ij``. The Neumann sub-Laplacian therefore has

        L_neu[i, j]    = L[i, j]                            (in-patch off-diag)
        L_neu[i, i]    = -sum_{j in patch, j != i} L[i, j]  (rebuilt diagonal)

    so vertices on the patch boundary only "see" their in-patch neighbours.

    Returns ``f_free`` (length ``len(free)``).
    """
    free = np.asarray(free, dtype=np.int64).ravel()
    if free.size == 0:
        return np.zeros(0, dtype=np.float64)
    anchor_idx = np.asarray(anchor_idx, dtype=np.int64).ravel()
    anchor_val = np.asarray(anchor_val, dtype=np.float64).ravel()

    L_csc = L.tocsc()
    A_FF = L_csc[free, :][:, free].tolil()
    A_FA = L_csc[free, :][:, anchor_idx]

    # Off-diagonal row sums on the patch (free + anchor columns).
    ff_off_row_sum = np.asarray(A_FF.sum(axis=1)).ravel() - A_FF.diagonal()
    fa_row_sum = np.asarray(A_FA.sum(axis=1)).ravel()
    patch_off_row_sum = ff_off_row_sum + fa_row_sum

    # Replace the diagonal: D_neu = -off-diag-row-sum (so each row sums to 0).
    A_FF.setdiag(-patch_off_row_sum)
    A_FF = A_FF.tocsc()

    rhs = -np.asarray(A_FA @ anchor_val).ravel()
    try:
        f_F = spla.spsolve(A_FF, rhs)
    except Exception:
        # Singular for isolated free verts → fall back to least-squares.
        f_F, *_ = spla.lsqr(A_FF, rhs)
    return np.asarray(f_F, dtype=np.float64)


def cot_laplacian(
    V: np.ndarray,
    F: np.ndarray,
    eps: float = 1e-12,
    *,
    clamp_negative_cot_weights: bool = True,
) -> sp.csr_matrix:
    """Cotangent Laplacian ``L = D - W`` of a triangle mesh.

    Parameters
    ----------
    V : (n, 3) array of vertex positions.
    F : (m, 3) array of triangle indices.
    clamp_negative_cot_weights :
        If True (default), per-triangle cotangent weights that would make an
        edge contribution negative are set to zero before assembly. On obtuse
        triangles raw cot weights can be negative, which breaks the discrete
        maximum principle; clamping yields a symmetric M-matrix (weighted
        graph Laplacian) so harmonic extensions minimize a positive Dirichlet
        energy. Anchor vertices are still enforced as hard Dirichlet data in
        :func:`harmonic_interpolate` and the bounded / Neumann-patch solves.

    Returns
    -------
    L : ``(n, n)`` CSR matrix.  ``L`` is symmetric.  Row sums are zero.
    """
    V = np.asarray(V, dtype=np.float64).reshape(-1, 3)
    F = np.asarray(F, dtype=np.int64).reshape(-1, 3)
    n = V.shape[0]
    if F.size == 0:
        return sp.csr_matrix((n, n), dtype=np.float64)

    I, J, K = F[:, 0], F[:, 1], F[:, 2]
    vI, vJ, vK = V[I], V[J], V[K]

    # cotangent at A in triangle (A,B,C) = (AB . AC) / |AB x AC|.
    AB = vJ - vI
    AC = vK - vI
    nABxAC = np.linalg.norm(np.cross(AB, AC), axis=1)
    nABxAC = np.where(nABxAC > eps, nABxAC, eps)
    cot_I = np.einsum("ij,ij->i", AB, AC) / nABxAC

    BA = vI - vJ
    BC = vK - vJ
    nBAxBC = np.linalg.norm(np.cross(BA, BC), axis=1)
    nBAxBC = np.where(nBAxBC > eps, nBAxBC, eps)
    cot_J = np.einsum("ij,ij->i", BA, BC) / nBAxBC

    CA = vI - vK
    CB = vJ - vK
    nCAxCB = np.linalg.norm(np.cross(CA, CB), axis=1)
    nCAxCB = np.where(nCAxCB > eps, nCAxCB, eps)
    cot_K = np.einsum("ij,ij->i", CA, CB) / nCAxCB

    # Each triangle contributes 0.5 * cot(angle_opposite_to_edge) to that edge.
    w_JK = 0.5 * cot_I
    w_KI = 0.5 * cot_J
    w_IJ = 0.5 * cot_K
    if clamp_negative_cot_weights:
        w_JK = np.maximum(w_JK, 0.0)
        w_KI = np.maximum(w_KI, 0.0)
        w_IJ = np.maximum(w_IJ, 0.0)

    rows = np.concatenate([J, K, K, I, I, J])
    cols = np.concatenate([K, J, I, K, J, I])
    vals = np.concatenate([w_JK, w_JK, w_KI, w_KI, w_IJ, w_IJ])

    W = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    # Force symmetry to fight off floating-point drift.
    W = (W + W.T) * 0.5

    diag = np.array(W.sum(axis=1)).ravel()
    L = sp.diags(diag) - W
    return L.tocsr()


def harmonic_interpolate(
    L: sp.spmatrix,
    anchor_idx: np.ndarray,
    anchor_val: np.ndarray,
    free_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Solve ``L f = 0`` with Dirichlet boundary ``f[anchor_idx] = anchor_val``.

    Parameters
    ----------
    L : sparse Laplacian (n, n).
    anchor_idx, anchor_val : aligned 1D arrays.
    free_mask : optional bool array of length ``n``. If supplied, only
        vertices where ``free_mask`` is True (and that are not anchors) are
        solved for. Vertices outside the mask are returned as NaN — this is
        how the Dijkstra-bounded mode achieves a "local" effect.

    Returns
    -------
    ``f`` of length ``n``. Vertices outside the effective domain are NaN.
    """
    n = L.shape[0]
    anchor_idx = np.asarray(anchor_idx, dtype=np.int64).ravel()
    anchor_val = np.asarray(anchor_val, dtype=np.float64).ravel()
    if anchor_idx.size == 0:
        return np.full(n, np.nan, dtype=np.float64)

    # Multiple electrodes anchored to the same mesh vertex: average them.
    if anchor_idx.size > 1:
        unique, inverse = np.unique(anchor_idx, return_inverse=True)
        if unique.size != anchor_idx.size:
            sums = np.zeros(unique.size, dtype=np.float64)
            counts = np.zeros(unique.size, dtype=np.int64)
            np.add.at(sums, inverse, anchor_val)
            np.add.at(counts, inverse, 1)
            anchor_idx = unique
            anchor_val = sums / np.maximum(counts, 1)

    is_anchor = np.zeros(n, dtype=bool)
    is_anchor[anchor_idx] = True

    if free_mask is None:
        free = np.where(~is_anchor)[0]
    else:
        free_mask = np.asarray(free_mask, dtype=bool).reshape(-1)
        if free_mask.size != n:
            raise ValueError("free_mask must have length n")
        free = np.where(free_mask & ~is_anchor)[0]

    # The result starts as NaN: anything we don't actively solve stays NaN.
    f = np.full(n, np.nan, dtype=np.float64)
    f[anchor_idx] = anchor_val
    if free.size == 0:
        return f

    L_csc = L.tocsc()
    L_UU = L_csc[free, :][:, free]
    L_UM = L_csc[free, :][:, anchor_idx]

    rhs = -np.asarray(L_UM @ anchor_val).ravel()
    try:
        f_U = spla.spsolve(L_UU.tocsc(), rhs)
    except Exception:
        # Numerically singular block: fall back to LSQR.
        f_U, *_ = spla.lsqr(L_UU, rhs)

    f[free] = f_U
    return f


def _harmonic_min_dirichlet_on_patch(
    L: sp.spmatrix,
    anchor_idx: np.ndarray,
    anchor_val: np.ndarray,
    in_patch: np.ndarray,
) -> np.ndarray:
    """Harmonic extension on ``in_patch`` that minimises discrete Dirichlet energy.

    Known values are enforced only at ``anchor_idx`` (electrode measurements).
    Every other vertex inside the accepted patch is a free unknown; the patch
    rim uses the natural Neumann BC from restricting the cotan Laplacian to
    in-patch neighbours (see :func:`_solve_neumann_patch`).
    """
    n = int(L.shape[0])
    anchor_idx = np.asarray(anchor_idx, dtype=np.int64).ravel()
    anchor_val = np.asarray(anchor_val, dtype=np.float64).ravel()
    in_patch = np.asarray(in_patch, dtype=bool).reshape(-1)
    if in_patch.size != n:
        raise ValueError("in_patch must have length n")

    f = np.full(n, np.nan, dtype=np.float64)
    if anchor_idx.size == 0:
        return f

    if anchor_idx.size > 1:
        unique, inverse = np.unique(anchor_idx, return_inverse=True)
        if unique.size != anchor_idx.size:
            sums = np.zeros(unique.size, dtype=np.float64)
            counts = np.zeros(unique.size, dtype=np.int64)
            np.add.at(sums, inverse, anchor_val)
            np.add.at(counts, inverse, 1)
            anchor_idx = unique
            anchor_val = sums / np.maximum(counts, 1)

    is_anchor = np.zeros(n, dtype=bool)
    is_anchor[anchor_idx] = True
    f[anchor_idx] = anchor_val

    free = np.where(in_patch & ~is_anchor)[0]
    if free.size:
        f[free] = _solve_neumann_patch(L, free, anchor_idx, anchor_val)
    return f


def build_mesh_graph(V: np.ndarray, F: np.ndarray) -> sp.csr_matrix:
    """Symmetric weighted graph of mesh edges (weight = Euclidean edge length).

    Returned as an undirected ``(n, n)`` CSR matrix suitable for
    ``scipy.sparse.csgraph.dijkstra``. Each mesh edge contributes a single
    pair of (i, j) / (j, i) entries regardless of how many triangles share it.
    """
    V = np.asarray(V, dtype=np.float64).reshape(-1, 3)
    F = np.asarray(F, dtype=np.int64).reshape(-1, 3)
    n = V.shape[0]
    if F.size == 0:
        return sp.csr_matrix((n, n), dtype=np.float64)

    edges = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=0)
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    d = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)

    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    vals = np.concatenate([d, d])
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def dijkstra_radius_mask(
    graph: sp.csr_matrix,
    anchor_vidx: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Bool mask of vertices within ``radius`` (geodesic, along edges) of any anchor.

    Anchors themselves are always inside.
    """
    from scipy.sparse.csgraph import dijkstra

    n = graph.shape[0]
    anchors = np.asarray(anchor_vidx, dtype=np.int64).ravel()
    if anchors.size == 0:
        return np.zeros(n, dtype=bool)
    r = float(radius)
    if not np.isfinite(r) or r <= 0:
        return np.ones(n, dtype=bool)
    dist = dijkstra(graph, directed=False, indices=anchors, limit=r)
    min_dist = dist.min(axis=0)
    return min_dist <= r


def mean_edge_length(V: np.ndarray, F: np.ndarray) -> float:
    """Average length of unique mesh edges (Euclidean)."""
    V = np.asarray(V, dtype=np.float64).reshape(-1, 3)
    F = np.asarray(F, dtype=np.int64).reshape(-1, 3)
    if F.size == 0:
        return 0.0
    edges = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=0)
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    d = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)
    if d.size == 0:
        return 0.0
    return float(np.mean(d))


def _vertex_areas_barycentric(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Barycentric vertex area: one-third of each incident triangle's area.

    Used as the diagonal mass matrix in the heat method. On smooth meshes
    barycentric areas are accurate enough; the more rigorous "mixed area"
    (Voronoi when triangles are acute, barycentric when obtuse) is overkill
    here because the heat method already smooths small per-vertex variations.
    """
    V = np.asarray(V, dtype=np.float64).reshape(-1, 3)
    F = np.asarray(F, dtype=np.int64).reshape(-1, 3)
    n = int(V.shape[0])
    if F.size == 0 or n == 0:
        return np.zeros(n, dtype=np.float64)
    p_i = V[F[:, 0]]
    p_j = V[F[:, 1]]
    p_k = V[F[:, 2]]
    A_tri = 0.5 * np.linalg.norm(np.cross(p_j - p_i, p_k - p_i), axis=1)
    areas = np.zeros(n, dtype=np.float64)
    np.add.at(areas, F[:, 0], A_tri / 3.0)
    np.add.at(areas, F[:, 1], A_tri / 3.0)
    np.add.at(areas, F[:, 2], A_tri / 3.0)
    return areas


def heat_geodesic_distances_from_anchors(
    V: np.ndarray,
    F: np.ndarray,
    anchor_vidx: np.ndarray,
    radius: float | None,
    t_factor: float = 5.0,
    *,
    tau_reference_radius: float | None = None,
) -> np.ndarray:
    """Geodesic distance per anchor via Crane et al. (2013) heat method.

    Returns ``(n_anchors, n_vertices)``. If ``radius`` is finite and positive,
    vertices with reconstructed distance ``> radius`` are set to ``np.inf``.
    If ``radius`` is ``None``, the full distance field is returned (no
    trimming) so callers can apply a time-varying cut as ``phi <= r`` later.

    Parameters
    ----------
    tau_reference_radius :
        Optional UI geodesic radius used only to pick the diffusion time
        ``tau`` when it should track a *small* cut: ``tau`` is capped relative
        to this value so short-range iso-contours stay round on coarse meshes.

    Why heat method (rather than FMM / Dijkstra)?
        FMM has an O(h) discretization error from single-triangle Eikonal
        updates, which makes geodesic balls look polygonal at the rim even
        on moderately fine meshes. The heat method instead solves two
        sparse-Laplacian systems — short-time heat diffusion followed by a
        Poisson reconstruction of the unit-gradient field — and the result
        is a near-isotropic, smooth distance field whose iso-contours
        approximate true geodesic circles much more faithfully. The cost
        is a few sparse back-substitutions (cheap because we prefactor the
        operators once and reuse them across all anchors).

    Notes on convention
        ``cot_laplacian`` returns ``L = D - W`` with ``L[i,i] = sum_j w_ij``
        and ``L[i,j] = -w_ij`` (positive semi-definite). The discrete heat
        equation is then ``(M + tau L) u = M @ source``. We pin one mesh
        vertex when solving the Poisson reconstruction so the otherwise
        rank-1 nullspace (constants) doesn't blow up — ``φ`` is then
        per-anchor shifted so the anchor itself has distance zero.
    """
    V = np.asarray(V, dtype=np.float64).reshape(-1, 3)
    F = np.asarray(F, dtype=np.int64).reshape(-1, 3)
    anchors = np.asarray(anchor_vidx, dtype=np.int64).ravel()
    n = int(V.shape[0])
    n_anc = int(anchors.size)
    if n_anc == 0:
        return np.zeros((0, n), dtype=np.float64)
    if F.size == 0 or n == 0:
        return np.full((n_anc, n), np.inf, dtype=np.float64)

    L = cot_laplacian(V, F)
    M_diag = _vertex_areas_barycentric(V, F)
    if not np.isfinite(M_diag).all() or float(M_diag.sum()) <= 0:
        return np.full((n_anc, n), np.inf, dtype=np.float64)
    M = sp.diags(M_diag, format="csr")

    h = mean_edge_length(V, F)
    if h <= 0:
        return np.full((n_anc, n), np.inf, dtype=np.float64)
    # Diffusion time ``tau`` controls how peaked the initial heat ``u`` is.
    # For *small* geodesic radii ``r`` the default ``~5 h^2`` can over-smooth
    # the field so iso-contours at distance ``r`` look faceted like Dijkstra.
    # When ``tau_reference_radius`` is set (typically the UI cut radius), cap
    # ``tau`` so the diffusion length scale stays well below ``r``.
    base_tau = float(t_factor) * h * h
    tau = base_tau
    if (
        tau_reference_radius is not None
        and np.isfinite(tau_reference_radius)
        and float(tau_reference_radius) > 0.0
    ):
        # Never drive ``tau`` from a value far below the edge length: the heat
        # field must stay well-conditioned on the *whole* mesh when ``radius``
        # is ``None`` (uncached field reused at many UI radii).
        rref = max(float(tau_reference_radius), 6.0 * h)
        tau_cap = max((rref / 7.0) ** 2, 0.25 * h * h)
        tau = min(base_tau, tau_cap)

    # Pre-factor the heat operator (M + tau L). Positive definite on a
    # well-conditioned mesh because L is PSD and M is positive diagonal.
    from scipy.sparse.linalg import splu

    H = (M + tau * L).tocsc()
    try:
        H_solver = splu(H)
    except Exception:
        return np.full((n_anc, n), np.inf, dtype=np.float64)

    # For the Poisson solve we pin a single vertex (the one with the largest
    # incident area — typically the most numerically stable choice). This
    # vertex's row/column is removed and the remaining block is positive
    # definite. We pre-factor this block once and reuse it for all anchors;
    # per-anchor shifting brings the anchor's distance to 0.
    pinned = int(np.argmax(M_diag))
    free_mask = np.ones(n, dtype=bool)
    free_mask[pinned] = False
    free_idx = np.where(free_mask)[0]
    L_csc = L.tocsc()
    L_FF = L_csc[free_idx, :][:, free_idx].tocsc()
    try:
        L_solver = splu(L_FF)
    except Exception:
        return np.full((n_anc, n), np.inf, dtype=np.float64)

    # Per-face geometry needed for gradient + divergence (vectorised).
    p_i = V[F[:, 0]]
    p_j = V[F[:, 1]]
    p_k = V[F[:, 2]]
    e_ij = p_j - p_i
    e_ik = p_k - p_i
    e_jk = p_k - p_j
    cross_jk = np.cross(e_ij, e_ik)
    two_A = np.linalg.norm(cross_jk, axis=1)
    two_A_safe = np.where(two_A > 1e-12, two_A, 1.0)
    n_hat = cross_jk / two_A_safe[:, None]
    inv_two_A = 1.0 / two_A_safe

    # cot at each vertex of each face. cot(angle at v) = (e1 . e2) / (2 A)
    # where e1, e2 are the two edges incident to v in the face.
    cot_i = np.einsum("ij,ij->i", e_ij, e_ik) * inv_two_A
    cot_j = np.einsum("ij,ij->i", -e_ij, e_jk) * inv_two_A
    cot_k = np.einsum("ij,ij->i", -e_ik, -e_jk) * inv_two_A

    # The triangle-Eikonal vectors n_hat × e_x are recomputed per anchor
    # because they couple with u_i, u_j, u_k. Pre-cache them once.
    nxe_i = np.cross(n_hat, p_k - p_j)  # rotated edge opposite vertex i
    nxe_j = np.cross(n_hat, p_i - p_k)
    nxe_k = np.cross(n_hat, p_j - p_i)

    out = np.full((n_anc, n), np.inf, dtype=np.float64)
    rhs_heat = np.zeros(n, dtype=np.float64)

    for ai, a_v in enumerate(anchors.tolist()):
        a_v = int(a_v)
        if not (0 <= a_v < n):
            continue

        # ---- Step 1: short-time heat diffusion ----
        rhs_heat[:] = 0.0
        rhs_heat[a_v] = 1.0
        u = H_solver.solve(rhs_heat)

        # ---- Step 2: per-face unit gradient X = -∇u / |∇u| ----
        u_i = u[F[:, 0]]
        u_j = u[F[:, 1]]
        u_k = u[F[:, 2]]
        grad = (
            u_i[:, None] * nxe_i
            + u_j[:, None] * nxe_j
            + u_k[:, None] * nxe_k
        ) * inv_two_A[:, None]
        grad_mag = np.linalg.norm(grad, axis=1)
        safe_mag = np.where(grad_mag > 1e-12, grad_mag, 1.0)
        X = -grad / safe_mag[:, None]

        # ---- Step 3: per-vertex divergence ∇·X ----
        # For face (i, j, k) and vertex v in face, contribute
        # (1/2) [ cot(at opposite of edge1) X·edge1 + cot(at opposite of edge2) X·edge2 ]
        # where edge1, edge2 are the two edges from v inside this face.
        # Concretely for vertex i (edges e_ij and e_ik, opposite-vertex cotans cot_k and cot_j):
        X_dot_ij = np.einsum("ij,ij->i", X, e_ij)
        X_dot_ik = np.einsum("ij,ij->i", X, e_ik)
        X_dot_jk = np.einsum("ij,ij->i", X, e_jk)

        div_X = np.zeros(n, dtype=np.float64)
        np.add.at(div_X, F[:, 0], 0.5 * (cot_k * X_dot_ij + cot_j * X_dot_ik))
        # For vertex j the edges are -e_ij (= j→i) and e_jk; cotans at k and i.
        np.add.at(div_X, F[:, 1], 0.5 * (cot_k * (-X_dot_ij) + cot_i * X_dot_jk))
        # For vertex k the edges are -e_ik (= k→i) and -e_jk (= k→j); cotans at j and i.
        np.add.at(div_X, F[:, 2], 0.5 * (cot_j * (-X_dot_ik) + cot_i * (-X_dot_jk)))

        # ---- Step 4: Poisson reconstruction with one pin ----
        # Our cot-Laplacian uses the convention ``L = D - W`` which is the
        # *negative* of the continuous Laplace operator. Crane's recipe asks
        # for ``Δφ = ∇·X``, i.e. ``-L φ = ∇·X`` here, so the right-hand side
        # carries a leading minus sign. (Forgetting it produces a "distance"
        # of zero everywhere — the bug that made the geodesic ball
        # explode to cover the entire mesh.)
        rhs_poisson = -div_X[free_idx]
        phi_free = L_solver.solve(rhs_poisson)
        phi = np.zeros(n, dtype=np.float64)
        phi[free_idx] = phi_free
        # phi[pinned] stays 0 by construction (Dirichlet pin).

        # ---- Step 5: shift so the anchor has distance 0, clamp negatives ----
        phi -= phi[a_v]
        phi = np.maximum(phi, 0.0)

        if radius is not None and np.isfinite(radius) and radius > 0:
            phi = np.where(phi <= radius, phi, np.inf)
        out[ai] = phi

    return out


def dijkstra_distances_from_anchors(
    graph: sp.csr_matrix,
    anchor_vidx: np.ndarray,
    radius: float | None,
) -> np.ndarray:
    """Geodesic distance matrix ``(n_anchors, n_vertices)``.

    Unreachable / outside-radius entries are ``np.inf``. Edge-based Dijkstra
    is kept as the **fallback** for meshes where the FMM triangle table is
    unavailable; :func:`fmm_distances_from_anchors` is the preferred path
    because edge-Dijkstra over-estimates geodesic distance and produces
    jagged "diamond-shaped" cut boundaries on regular meshes.
    """
    from scipy.sparse.csgraph import dijkstra

    anchors = np.asarray(anchor_vidx, dtype=np.int64).ravel()
    n = graph.shape[0]
    if anchors.size == 0:
        return np.full((0, n), np.inf, dtype=np.float64)
    lim = float(radius) if (radius is not None and np.isfinite(radius) and radius > 0) else np.inf
    dist = dijkstra(graph, directed=False, indices=anchors, limit=lim)
    return np.asarray(dist, dtype=np.float64)


def _build_vertex_triangle_adjacency(F: np.ndarray, n_verts: int) -> tuple[np.ndarray, np.ndarray]:
    """CSR-style adjacency: triangles incident to each vertex.

    Returns ``(offsets, tri_indices)`` so that ``tri_indices[offsets[v]:offsets[v+1]]``
    is the list of triangle indices that include vertex ``v``.
    """
    F = np.asarray(F, dtype=np.int64).reshape(-1, 3)
    rep = np.repeat(np.arange(F.shape[0], dtype=np.int64), 3)
    flat = F.ravel()
    order = np.argsort(flat, kind="stable")
    sorted_v = flat[order]
    sorted_t = rep[order]
    counts = np.bincount(sorted_v, minlength=int(n_verts)).astype(np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return offsets, sorted_t


def fmm_distances_from_anchors(
    V: np.ndarray,
    F: np.ndarray,
    anchor_vidx: np.ndarray,
    radius: float | None,
) -> np.ndarray:
    """Fast Marching Method geodesic distance on a triangle mesh.

    Returns ``(n_anchors, n_vertices)``; vertices outside the ``radius`` ball
    of an anchor (or unreachable in the same connected component) are
    ``np.inf``.

    Implements the standard 2D upwind Eikonal triangle update from
    Kimmel & Sethian (1998): for an unknown vertex ``C`` on triangle
    ``(A, B, C)`` with known arrival times ``u_A, u_B``, the local arrival
    time at ``C`` is computed in the 2D triangle plane, falling back to the
    edge update when the local update is geometrically infeasible (obtuse
    case). The result respects the mesh metric, so the geodesic ball is a
    near-circular patch with sharp boundaries rather than the jagged
    edge-Dijkstra diamond.
    """
    import heapq

    V = np.asarray(V, dtype=np.float64).reshape(-1, 3)
    F = np.asarray(F, dtype=np.int64).reshape(-1, 3)
    anchors = np.asarray(anchor_vidx, dtype=np.int64).ravel()
    n = int(V.shape[0])
    if anchors.size == 0:
        return np.full((0, n), np.inf, dtype=np.float64)
    if F.size == 0 or n == 0:
        return np.full((anchors.size, n), np.inf, dtype=np.float64)

    lim = (
        float(radius)
        if (radius is not None and np.isfinite(radius) and radius > 0)
        else np.inf
    )

    offsets, tri_of = _build_vertex_triangle_adjacency(F, n)

    out = np.full((anchors.size, n), np.inf, dtype=np.float64)

    # Precompute Euclidean edge lengths for edge-update fallback.
    def edge_len(i: int, j: int) -> float:
        return float(np.linalg.norm(V[i] - V[j]))

    def eikonal_update(c: int, a: int, b: int, u_a: float, u_b: float) -> float:
        # Triangle (a, b, c). u_a, u_b are known arrival times. Solve for u_c.
        # Project c into the 2D local frame with a at origin, b on +x axis.
        Pa = V[a]
        Pb = V[b]
        Pc = V[c]
        AB = Pb - Pa
        L_ab = float(np.linalg.norm(AB))
        if L_ab < 1e-12:
            return min(u_a + edge_len(c, a), u_b + edge_len(c, b))
        e_x = AB / L_ab
        AC = Pc - Pa
        x_c = float(np.dot(AC, e_x))
        # perpendicular component
        perp = AC - x_c * e_x
        h = float(np.linalg.norm(perp))
        if h < 1e-12:
            return min(u_a + edge_len(c, a), u_b + edge_len(c, b))
        # Look for the front intersection on segment AB with linear u-interp.
        g = (u_b - u_a) / L_ab
        if abs(g) >= 1.0:
            return min(u_a + edge_len(c, a), u_b + edge_len(c, b))
        # Stationary point of T(x) = u_a + g*x + sqrt((x_c - x)^2 + h^2).
        y = g * h / np.sqrt(1.0 - g * g)
        x_star = x_c - y
        # Front intersection must lie on segment [0, L_ab] AND the ray from
        # the front to c must point into the unknown half-plane (causality
        # check: x_c - x_star and g must agree in sign with h's side).
        if 0.0 <= x_star <= L_ab:
            t_front = u_a + g * x_star
            return t_front + np.sqrt((x_c - x_star) ** 2 + h * h)
        return min(u_a + edge_len(c, a), u_b + edge_len(c, b))

    for ai, anc in enumerate(anchors.tolist()):
        if not (0 <= int(anc) < n):
            continue
        d = out[ai]
        d[int(anc)] = 0.0
        known = np.zeros(n, dtype=bool)
        heap: list[tuple[float, int]] = [(0.0, int(anc))]
        while heap:
            t_v, v = heapq.heappop(heap)
            if known[v]:
                continue
            if t_v > lim:
                break
            known[v] = True
            # Relax through all incident triangles.
            for ti in tri_of[offsets[v] : offsets[v + 1]]:
                a, b, c = F[int(ti)]
                a, b, c = int(a), int(b), int(c)
                # For each other vertex x in this triangle, attempt update.
                for x in (a, b, c):
                    if x == v or known[x]:
                        continue
                    # The "other other" vertex y might already be known →
                    # 2D Eikonal update; otherwise just edge update from v.
                    y = a + b + c - v - x
                    new_d = t_v + edge_len(v, x)
                    if known[y] and 0 <= y < n and np.isfinite(d[y]):
                        cand = eikonal_update(x, v, y, t_v, d[y])
                        if cand < new_d:
                            new_d = cand
                    if new_d < d[x] and new_d <= lim:
                        d[x] = new_d
                        heapq.heappush(heap, (new_d, x))

    return out


def harmonic_interpolate_bounded_cached(
    L: sp.spmatrix,
    dist: np.ndarray,
    anchor_idx: np.ndarray,
    anchor_val: np.ndarray,
    anchor_mask: np.ndarray | None,
    radius: float,
    mode: str = "local",
) -> np.ndarray:
    """Bounded harmonic on the geodesic cut patch.

    Parameters
    ----------
    L : sparse Laplacian (n, n)
    dist : (n_all_anchors, n) precomputed geodesic distances. ``np.inf`` for
        unreached entries.
    anchor_idx, anchor_val : the *active* anchors for this metric (subset of
        the cached anchor list).
    anchor_mask : bool array of length ``n_all_anchors`` selecting which rows of
        ``dist`` line up with ``anchor_idx``. If ``None``, ``dist`` is already
        the matching slice.
    radius : geodesic ball radius (must be > 0 and finite).
    mode :
        Accepted for API compatibility. Both ``"local"`` and ``"global"`` now
        solve the same problem: on the union of geodesic balls (the accepted
        patch), find the harmonic field with fixed anchor values that
        minimises discrete Dirichlet energy (Neumann rim on the patch cut).
    """
    n = L.shape[0]
    anchor_idx = np.asarray(anchor_idx, dtype=np.int64).ravel()
    anchor_val = np.asarray(anchor_val, dtype=np.float64).ravel()
    if anchor_idx.size == 0:
        return np.full(n, np.nan, dtype=np.float64)

    if anchor_mask is not None:
        d = dist[np.asarray(anchor_mask, dtype=bool)]
    else:
        d = np.asarray(dist, dtype=np.float64)
    if d.shape[0] != anchor_idx.size:
        raise ValueError(
            f"dist rows ({d.shape[0]}) must match anchor_idx size ({anchor_idx.size})"
        )

    r = float(radius)
    reach = d <= r
    n_reach = reach.sum(axis=0)
    in_patch = n_reach >= 1
    return _harmonic_min_dirichlet_on_patch(L, anchor_idx, anchor_val, in_patch)


def harmonic_interpolate_bounded(
    L: sp.spmatrix,
    graph: sp.csr_matrix,
    anchor_idx: np.ndarray,
    anchor_val: np.ndarray,
    radius: float,
    mode: str = "local",
) -> np.ndarray:
    """Per-anchor radius harmonic on the accepted geodesic patch.

    Vertices outside every anchor's geodesic ball stay NaN. Inside the union
    of balls, solves for the harmonic field with anchor Dirichlet data that
    minimises discrete Dirichlet energy (natural Neumann rim on the patch cut).

    ``mode`` is kept for API compatibility; ``"local"`` and ``"global"`` are
    equivalent.
    """
    from scipy.sparse.csgraph import dijkstra

    n = L.shape[0]
    anchor_idx = np.asarray(anchor_idx, dtype=np.int64).ravel()
    anchor_val = np.asarray(anchor_val, dtype=np.float64).ravel()
    if anchor_idx.size == 0:
        return np.full(n, np.nan, dtype=np.float64)

    # Dedup anchors that map to the same mesh vertex.
    if anchor_idx.size > 1:
        unique, inverse = np.unique(anchor_idx, return_inverse=True)
        if unique.size != anchor_idx.size:
            sums = np.zeros(unique.size, dtype=np.float64)
            counts = np.zeros(unique.size, dtype=np.int64)
            np.add.at(sums, inverse, anchor_val)
            np.add.at(counts, inverse, 1)
            anchor_idx = unique
            anchor_val = sums / np.maximum(counts, 1)

    r = float(radius)
    if not np.isfinite(r) or r <= 0:
        return harmonic_interpolate(L, anchor_idx, anchor_val)

    dist = dijkstra(graph, directed=False, indices=anchor_idx, limit=r)
    dist = np.asarray(dist)  # shape (n_anchors, n_vertices)
    reach = dist <= r
    n_reach = reach.sum(axis=0)
    in_patch = n_reach >= 1
    return _harmonic_min_dirichlet_on_patch(L, anchor_idx, anchor_val, in_patch)


def adaptive_k_per_anchor(
    n_vertices: int,
    n_anchors: int,
    *,
    has_geodesic_radius: bool,
    budget_frac: float = 0.22,
    k_min: int = 48,
    k_max: int = 12000,
) -> int:
    """How many nearest mesh vertices each anchor contributes to the local patch.

    Fewer electrodes → larger ``k`` (wider support). More electrodes → smaller
    ``k`` each so total work stays bounded and patches stay localized ("higher
    spatial resolution" near dense sampling).

    The union over all anchors is additionally capped by ``budget_frac`` of the
    full vertex count so extraction + sparse solve stay interactive.
    """
    n_vertices = int(max(1, n_vertices))
    n_anchors = int(max(1, n_anchors))
    cap = float(budget_frac) if has_geodesic_radius else min(0.45, float(budget_frac) + 0.18)
    total_budget = int(max(k_min * n_anchors, min(n_vertices, int(n_vertices * cap))))
    k = total_budget // n_anchors
    return int(max(k_min, min(k_max, min(n_vertices, k))))


def knn_union_for_anchors(
    V: np.ndarray,
    anchor_idx: np.ndarray,
    k_per_anchor: int,
) -> np.ndarray:
    """Union of ``k_per_anchor`` Euclidean-nearest mesh vertices per anchor."""
    V = np.asarray(V, dtype=np.float64).reshape(-1, 3)
    anchor_idx = np.asarray(anchor_idx, dtype=np.int64).ravel()
    n = V.shape[0]
    k = int(max(1, min(n, k_per_anchor)))
    if anchor_idx.size == 0:
        return np.zeros(0, dtype=np.int64)
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(V)
        out_sets: list[np.ndarray] = []
        for vi in anchor_idx:
            vi = int(vi)
            if not (0 <= vi < n):
                continue
            _, nn = tree.query(V[vi], k=k)
            nn = np.atleast_1d(nn)
            out_sets.append(nn.astype(np.int64, copy=False))
        if not out_sets:
            return np.zeros(0, dtype=np.int64)
        return np.unique(np.concatenate(out_sets))
    except Exception:
        acc: list[int] = []
        for vi in anchor_idx:
            vi = int(vi)
            if not (0 <= vi < n):
                continue
            d2 = np.sum((V - V[vi]) ** 2, axis=1)
            part = np.argpartition(d2, min(k, n) - 1)[:k]
            acc.extend(int(x) for x in part)
        return np.unique(np.asarray(acc, dtype=np.int64))


def harmonic_interpolate_on_indices(
    L: sp.spmatrix,
    graph: sp.csr_matrix | None,
    vertex_idx: np.ndarray,
    anchor_idx: np.ndarray,
    anchor_val: np.ndarray,
    radius: float | None,
) -> np.ndarray:
    """Harmonic field on full mesh; only ``vertex_idx`` rows may be non-NaN.

    Restricts the operator to the vertex subset (``L[U][:, U]``) and, when
    ``radius`` is finite, runs :func:`harmonic_interpolate_bounded` on that
    induced subgraph so geodesic balls are cheap. Outside ``vertex_idx`` the
    result is NaN.
    """
    n = L.shape[0]
    U = np.asarray(vertex_idx, dtype=np.int64).ravel()
    U = np.unique(U[(U >= 0) & (U < n)])
    if U.size == 0:
        return np.full(n, np.nan, dtype=np.float64)

    remap = np.full(n, -1, dtype=np.int64)
    remap[U] = np.arange(U.size, dtype=np.int64)

    anchor_idx = np.asarray(anchor_idx, dtype=np.int64).ravel()
    anchor_val = np.asarray(anchor_val, dtype=np.float64).ravel()
    ai = remap[anchor_idx]
    if np.any(ai < 0):
        # Should not happen if anchors came from the same knn union.
        miss = anchor_idx[ai < 0]
        U = np.unique(np.concatenate([U, miss]))
        remap = np.full(n, -1, dtype=np.int64)
        remap[U] = np.arange(U.size, dtype=np.int64)
        ai = remap[anchor_idx]

    Ls = L[U][:, U].tocsr()
    if graph is None:
        f_sub = harmonic_interpolate(Ls, ai, anchor_val)
    else:
        Gs = graph[U][:, U].tocsr()
        r = float(radius) if radius is not None else 0.0
        if np.isfinite(r) and r > 0:
            f_sub = harmonic_interpolate_bounded(Ls, Gs, ai, anchor_val, r)
        else:
            f_sub = harmonic_interpolate(Ls, ai, anchor_val)

    out = np.full(n, np.nan, dtype=np.float64)
    out[U] = f_sub
    return out


def map_points_to_vertices(V: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Closest mesh vertex for each input point (``(N,)`` int array)."""
    V = np.asarray(V, dtype=np.float64).reshape(-1, 3)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if V.size == 0 or pts.size == 0:
        return np.zeros(pts.shape[0], dtype=np.int64)
    try:
        from scipy.spatial import cKDTree

        _, idx = cKDTree(V).query(pts, k=1)
        return np.asarray(idx, dtype=np.int64).ravel()
    except Exception:
        out = np.empty(pts.shape[0], dtype=np.int64)
        for i, p in enumerate(pts):
            d = np.einsum("ij,ij->i", V - p, V - p)
            out[i] = int(np.argmin(d))
        return out


def max_path_dvds_per_anchor(
    V: np.ndarray,
    graph: sp.csr_matrix,
    anchor_idx: np.ndarray,
    phi: np.ndarray,
    *,
    local_radius: float,
    patch_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Strongest |dV/ds| along shortest-path tree edges from each anchor.

    For each anchor, run Dijkstra up to ``local_radius`` and scan tree edges
    (parent → child). The winning edge defines origin (midpoint), unit
    direction (increasing φ), and magnitude |Δφ|/edge_length (mV per mesh unit).

    Returns
    -------
    origins : (n_anchors, 3)
    directions : (n_anchors, 3) unit vectors
    magnitudes : (n_anchors,) — NaN when undefined
    """
    from scipy.sparse.csgraph import dijkstra

    V = np.asarray(V, dtype=np.float64).reshape(-1, 3)
    phi = np.asarray(phi, dtype=np.float64).reshape(-1)
    anchors = np.asarray(anchor_idx, dtype=np.int64).ravel()
    n_anc = int(anchors.size)
    origins = np.full((n_anc, 3), np.nan, dtype=np.float64)
    directions = np.zeros((n_anc, 3), dtype=np.float64)
    magnitudes = np.full(n_anc, np.nan, dtype=np.float64)
    if n_anc == 0:
        return origins, directions, magnitudes

    r_lim = float(local_radius)
    if not np.isfinite(r_lim) or r_lim <= 0:
        r_lim = np.inf
    pmask = None
    if patch_mask is not None:
        pmask = np.asarray(patch_mask, dtype=bool).reshape(-1)

    for ai, a in enumerate(anchors):
        a = int(a)
        if a < 0 or a >= V.shape[0] or not np.isfinite(phi[a]):
            continue
        dist, pred = dijkstra(
            graph, directed=False, indices=a, return_predecessors=True, limit=r_lim
        )
        dist = np.asarray(dist, dtype=np.float64).ravel()
        pred = np.asarray(pred, dtype=np.int64).ravel()
        in_ball = np.isfinite(dist) & (dist <= r_lim if np.isfinite(r_lim) else True)
        if pmask is not None and pmask.size == dist.size:
            in_ball &= pmask

        best_mag = -1.0
        best_origin = V[a]
        best_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        for v in np.where(in_ball)[0]:
            v = int(v)
            if v == a:
                continue
            u = int(pred[v])
            if u < 0:
                continue
            edge_vec = V[v] - V[u]
            edge_len = float(np.linalg.norm(edge_vec))
            if edge_len < 1e-12:
                continue
            dphi = float(phi[v]) - float(phi[u])
            mag = abs(dphi) / edge_len
            if mag > best_mag:
                best_mag = mag
                best_origin = 0.5 * (V[u] + V[v])
                best_dir = edge_vec / edge_len
                if dphi < 0:
                    best_dir = -best_dir

        if best_mag >= 0.0:
            magnitudes[ai] = best_mag
            origins[ai] = best_origin
            directions[ai] = best_dir

    return origins, directions, magnitudes
