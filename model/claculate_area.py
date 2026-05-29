import numpy as np

# ------------------------------
# Helpers
# ------------------------------

def _triangle_basis(p0, p1, p2):
    """Orthonormal basis (t1, t2, n) on triangle plane + triangle area."""
    e1 = p1 - p0
    e2 = p2 - p0
    n_raw = np.cross(e1, e2)
    A = 0.5 * np.linalg.norm(n_raw)
    if A == 0.0:
        # Degenerate; return some basis to avoid crashes
        t1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        n  = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        t2 = np.cross(n, t1)
        return p0, t1, t2, n, 0.0
    n = n_raw / (2.0 * A)  # unit normal
    t1 = e1 / (np.linalg.norm(e1) + 1e-20)
    t2 = np.cross(n, t1)
    return p0, t1, t2, n, A

def _project_to_plane(o, t1, t2, pts):
    """Project 3D points to 2D coordinates in triangle plane basis."""
    Q = np.asarray(pts, dtype=np.float64) - o
    return np.column_stack((Q @ t1, Q @ t2))  # (N,2)

def _solve_plane_s(xy, si):
    """Solve s(x,y) = alpha + beta x + gamma y for a single triangle."""
    # xy: (3,2), si: (3,)
    X = np.array([
        [1.0, xy[0,0], xy[0,1]],
        [1.0, xy[1,0], xy[1,1]],
        [1.0, xy[2,0], xy[2,1]],
    ], dtype=np.float64)
    y = np.asarray(si, dtype=np.float64).reshape(3)
    alpha, beta, gamma = np.linalg.solve(X, y)
    return alpha, beta, gamma

def _clip_poly_by_slab(q, a, b, eps=1e-12):
    """Clip polygon q (points in R^3 with s=q[:,2]) by a <= s <= b."""
    if len(q) == 0:
        return q
    def clip_once(poly, c, keep_ge):
        if len(poly) == 0:
            return poly
        out = []
        m = len(poly)
        def inside(sv):
            return (sv >= c - eps) if keep_ge else (sv <= c + eps)
        for i in range(m):
            p0 = poly[i]
            p1 = poly[(i+1) % m]
            s0 = p0[2]; s1 = p1[2]
            i0 = inside(s0); i1 = inside(s1)
            if i0 and i1:
                out.append(p1)
            elif i0 and not i1:
                # leaving: add intersection only
                ds = s1 - s0
                if abs(ds) > eps:
                    t = (c - s0) / ds
                    out.append(p0 + t * (p1 - p0))
            elif (not i0) and i1:
                # entering: add intersection then p1
                ds = s1 - s0
                if abs(ds) > eps:
                    t = (c - s0) / ds
                    out.append(p0 + t * (p1 - p0))
                out.append(p1)
            # out->out: add nothing
        return out

    out = clip_once(q, a, keep_ge=True)
    if not out: return []
    out = clip_once(out, b, keep_ge=False)
    return out

def _polygon_area_on_plane(pts, n_hat):
    """Area of a planar polygon in 3D given unit normal n_hat (shoelace in 3D)."""
    if len(pts) < 3:
        return 0.0
    acc = 0.0
    for i in range(len(pts)):
        p = pts[i]
        q = pts[(i+1) % len(pts)]
        acc += np.dot(np.cross(p, q), n_hat)
    return 0.5 * abs(acc)

# ------------------------------
# Main: band area via graph method
# ------------------------------

def band_area_graph(verts, faces, s, a, b, eps=1e-12, return_polys=False):
    """
    Exact area where scalar s (piecewise-linear) lies in [a,b] on a triangle mesh.

    verts: (N,3) float
    faces: (M,3) int
    s    : (N,)  float (NaN allowed -> face skipped)
    a,b  : thresholds with a <= b
    return_polys: if True, returns list of per-face band polygons in 3D (world coords)
    """
    a = float(np.asarray(a).reshape(()))
    b = float(np.asarray(b).reshape(()))
    if not (a <= b):
        raise ValueError(f"a ({a}) must be <= b ({b})")
    V = np.asarray(verts, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    S = np.asarray(s, dtype=np.float64).reshape(-1)

    total = 0.0
    polys = [] if return_polys else None

    for (i, j, k) in F:
        si = np.array([S[i], S[j], S[k]], dtype=np.float64)
        if np.any(~np.isfinite(si)):
            if return_polys: polys.append([])
            continue

        P = np.array([V[i], V[j], V[k]], dtype=np.float64)
        o, t1, t2, n_tri, A_tri = _triangle_basis(P[0], P[1], P[2])
        if A_tri == 0.0:
            if return_polys: polys.append([])
            continue

        smin = float(np.min(si)); smax = float(np.max(si))
        if smax < a - eps or smin > b + eps:
            if return_polys: polys.append([])
            continue
        if (a - eps) <= smin and smax <= (b + eps):
            # fully inside band
            total += A_tri
            if return_polys: polys.append(P.tolist())
            continue

        # Project to (x,y)
        XY = _project_to_plane(o, t1, t2, P)  # (3,2)
        alpha, beta, gamma = _solve_plane_s(XY, si)

        # Graph plane unit normal and stretch factor
        kappa = np.sqrt(1.0 + beta*beta + gamma*gamma)
        n_graph = np.array([-beta, -gamma, 1.0], dtype=np.float64) / kappa

        # Lift triangle to graph: q_i = (x_i, y_i, s_i)
        q = np.column_stack((XY, si))

        # Clip by slab a<=s<=b in graph space
        q_band = _clip_poly_by_slab([q[0], q[1], q[2]], a, b, eps=eps)
        if not q_band:
            if return_polys: polys.append([])
            continue

        # Area on graph, then divide by kappa to get base (surface) area
        A_graph = _polygon_area_on_plane(q_band, n_graph)
        A_band  = A_graph / kappa
        total  += A_band

        if return_polys:
            # Map polygon back to 3D world coords on the triangle plane:
            # q = (x,y,s) -> world = o + x*t1 + y*t2
            q_arr = np.asarray(q_band)
            world_poly = (o + np.outer(q_arr[:,0], t1) + np.outer(q_arr[:,1], t2))
            polys.append(world_poly.tolist())

    return (total, polys) if return_polys else total


def mesh_area_valid(verts, faces, s, eps=1e-12):
    """
    Total surface area over triangles whose vertex scalars are all finite.
    Faces with any non-finite s are excluded from the sum.
    """
    V = np.asarray(verts, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    S = np.asarray(s, dtype=np.float64).reshape(-1)

    area_total = 0.0
    for (i, j, k) in F:
        si = np.array([S[i], S[j], S[k]], dtype=np.float64)
        if np.any(~np.isfinite(si)):
            continue
        P0, P1, P2 = V[i], V[j], V[k]
        A = 0.5 * np.linalg.norm(np.cross(P1 - P0, P2 - P0))
        if A > eps:
            area_total += A
    return area_total


def band_integral(verts, faces, s, a, b, eps=1e-12):
    """
    Exact integral of s over the band {a <= s <= b}, ignoring faces that touch NaNs/Infs.

    Requires the same helpers you already use:
      _triangle_basis, _project_to_plane, _solve_plane_s, _clip_poly_by_slab
    """
    a = float(np.asarray(a).reshape(()))
    b = float(np.asarray(b).reshape(()))
    if not (a <= b):
        raise ValueError(f"a ({a}) must be <= b ({b})")
    V = np.asarray(verts, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    S = np.asarray(s, dtype=np.float64).reshape(-1)

    integral = 0.0

    for (i, j, k) in F:
        si = np.array([S[i], S[j], S[k]], dtype=np.float64)
        if np.any(~np.isfinite(si)):
            continue

        P = np.array([V[i], V[j], V[k]], dtype=np.float64)
        o, t1, t2, n_tri, A_tri = _triangle_basis(P[0], P[1], P[2])
        if A_tri <= eps:
            continue

        smin = float(np.min(si)); smax = float(np.max(si))
        # No overlap with [a,b]
        if smax < a - eps or smin > b + eps:
            continue

        # Fully inside band
        if (a - eps) <= smin and smax <= (b + eps):
            tri_mean = (si[0] + si[1] + si[2]) / 3.0
            integral += tri_mean * A_tri
            continue

        # Partial: work in triangle 2D coords
        XY = _project_to_plane(o, t1, t2, P)             # (3,2)
        alpha, beta, gamma = _solve_plane_s(XY, si)      # s(x,y) = alpha + beta x + gamma y

        q = np.column_stack((XY, si))                    # (x,y,s) per vertex
        q_band = _clip_poly_by_slab([q[0], q[1], q[2]], a, b, eps=eps)
        if not q_band:
            continue

        qb = np.asarray(q_band, dtype=np.float64)        # (m,3)
        poly_xy = qb[:, :2]                              # (m,2)

        A_poly, (xc, yc) = _poly_area_centroid_2d(poly_xy)
        if A_poly <= eps:
            continue

        # ∫_poly s dA = (alpha + beta*xc + gamma*yc) * A_poly
        integral += (alpha + beta * xc + gamma * yc) * A_poly

    return integral


# --- small helper for centroid-based integration (pure 2D, internal to this file) ---

def _poly_area_centroid_2d(poly_xy):
    """
    Signed area and centroid of a simple 2D polygon (x,y) with vertices in order.
    Returns (A>=0, (cx, cy)).
    """
    x = poly_xy[:, 0]; y = poly_xy[:, 1]
    s = x * np.roll(y, -1) - y * np.roll(x, -1)
    A2 = 0.5 * np.sum(s)
    A = float(abs(A2))
    if A == 0.0:
        return 0.0, (0.0, 0.0)
    cx = (1.0 / (6.0 * A2)) * np.sum((x + np.roll(x, -1)) * s)
    cy = (1.0 / (6.0 * A2)) * np.sum((y + np.roll(y, -1)) * s)
    return A, (cx, cy)
