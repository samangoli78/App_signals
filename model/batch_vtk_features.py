import os, numpy as np
from scipy.sparse.linalg import splu

import matplotlib.pyplot as plt
from claculate_area import band_area_graph,mesh_area_valid,band_integral
import heapq
from area_hist import compute_area_histograms_all_scalars, plot_area_histograms

from smooth import *

#needs build_cotan_laplacian_and_mass(V,F) from your smooth.py
def _edge_graph_lengths(V, F):
    n = V.shape[0]
    nbr = [[] for _ in range(n)]
    F = F.astype(np.int32, copy=False)
    for a,b,c in F:
        for u,v in ((a,b),(b,c),(c,a)):
            w = float(np.linalg.norm(V[u] - V[v]))
            nbr[u].append((v, w))
            nbr[v].append((u, w))
    return nbr

def _dijkstra_multi(nbr, sources, max_dist=np.inf):
    n = len(nbr)
    dist = np.full(n, np.inf, np.float64)
    h = []
    for s in np.asarray(sources, np.int32):
        if dist[s] > 0.0:
            dist[s] = 0.0
            heapq.heappush(h, (0.0, int(s)))
    while h:
        d,u = heapq.heappop(h)
        if d != dist[u] or d > max_dist:
            continue
        for v,w in nbr[u]:
            nd = d + w
            if nd < dist[v] and nd <= max_dist:
                dist[v] = nd
                heapq.heappush(h, (nd, v))
    return dist
def _mean_edge_length(V, F):
    E = set()
    for a, b, c in F:
        E.add((min(a,b), max(a,b)))
        E.add((min(b,c), max(c,a)))
        E.add((min(a,c), max(a,c)))
    if not E:
        return 1.0
    E = np.array(list(E), dtype=np.int32)
    le = np.linalg.norm(V[E[:,0]] - V[E[:,1]], axis=1)
    m = float(le.mean()) if le.size else 1.0
    return m if np.isfinite(m) and m > 0 else 1.0
def _interp_many_nanaware(
    verts, faces, known_idx, values_dict,
    lam=1e-6,
    radius_factor=10.0,
    add_diag=1e-12,
):
    import numpy as np
    from scipy.sparse import diags
    from scipy.sparse.linalg import splu

    V = np.asarray(verts, np.float64)
    F = np.asarray(faces,  np.int32)
    n = V.shape[0]

    # --- operators ---
    L, M = build_cotan_laplacian_and_mass(V, F)
    A = (L + lam * M).tocsr()

    # --- mean edge length (scale for geodesic radius) ---


    Lmean  = _mean_edge_length(V, F)
    radius = float(radius_factor * Lmean)

    # --- adjacency for geodesic ---
    nbr = _edge_graph_lengths(V, F)

    # --- shared bookkeeping on known indices ---
    ki_all = np.asarray(known_idx, np.int64)
    order  = np.argsort(ki_all)
    ki     = ki_all[order]
    uk_all, start = np.unique(ki, return_index=True)

    # --- GLOBAL DIJKSTRA (one geodesic mask for everyone) ---
    # seeds: all known vertices used anywhere (ignore NaNs here)
    seeds = uk_all
    dist = _dijkstra_multi(nbr, seeds, max_dist=radius)
    global_keep_vertices = np.isfinite(dist) & (dist <= radius)

    out = {}

    for name, vals in values_dict.items():
        kv = np.asarray(vals, np.float64)[order]

        # nan-mean across duplicates at each unique known vertex (THIS field only)
        sums   = np.add.reduceat(np.nan_to_num(kv, nan=0.0), start)
        counts = np.add.reduceat(~np.isnan(kv), start).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            fk = sums / np.maximum(counts, 1.0)
        fk[counts == 0] = np.nan

        # usable constraints for THIS field
        keep = ~np.isnan(fk)
        if not np.any(keep):
            # nothing to interpolate for this field
            f = np.full(n, np.nan, dtype=np.float64)
            out[name] = f
            continue

        K  = uk_all[keep]   # known vertex ids for this field
        fK = fk[keep]

        # ---- GLOBAL solve FOR THIS FIELD (unknowns = all verts except K) ----
        mask_known = np.zeros(n, dtype=bool)
        mask_known[K] = True
        uu = np.where(~mask_known)[0]

        Auu = A[uu][:, uu].tocsc()
        Auk = A[uu][:, K]
        rhs = -Auk @ fK

        if add_diag > 0.0:
            Auu = (Auu + add_diag * diags(np.maximum(M.diagonal()[uu], 1.0))).tocsc()

        fu = splu(Auu).solve(rhs)

        f = np.empty(n, dtype=np.float64)
        f[K]  = fK
        f[uu] = fu

        # ---- shared geodesic mask ----
        f[~global_keep_vertices] = np.nan
        out[name] = f

    return out


import numpy as np, vtk

def write_mesh_multi_scalar_vtp_vtk(verts, faces, scalars_dict, fname):
    V = np.asarray(verts, float)
    F = np.asarray(faces, np.int64)

    pts = vtk.vtkPoints()
    pts.SetDataTypeToFloat()
    pts.SetNumberOfPoints(V.shape[0])
    for i, p in enumerate(V):
        pts.SetPoint(i, float(p[0]), float(p[1]), float(p[2]))

    polys = vtk.vtkCellArray()
    for tri in F:
        cell = vtk.vtkTriangle()
        cell.GetPointIds().SetId(0, int(tri[0]))
        cell.GetPointIds().SetId(1, int(tri[1]))
        cell.GetPointIds().SetId(2, int(tri[2]))
        polys.InsertNextCell(cell)

    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetPolys(polys)

    pd = poly.GetPointData()
    n = V.shape[0]
    first_set = False
    for name, arr in scalars_dict.items():
        a = np.asarray(arr)
        if a.ndim == 1:
            if a.size != n:
                raise ValueError(f"Scalar '{name}' has length {a.size}, expected {n}")
            va = vtk.vtkFloatArray()
            va.SetName(str(name))
            va.SetNumberOfComponents(1)
            va.SetNumberOfTuples(n)
            for i, v in enumerate(a):
                va.SetValue(i, float(v))
            pd.AddArray(va)
            if not first_set:
                pd.SetScalars(va); first_set = True
        elif a.ndim == 2 and a.shape[0] == n and a.shape[1] in (2,3):
            comps = a.shape[1]
            va = vtk.vtkFloatArray()
            va.SetName(str(name))
            va.SetNumberOfComponents(comps)
            va.SetNumberOfTuples(n)
            for i in range(n):
                tup = [float(x) for x in a[i]]
                if comps == 2: tup = tup + [0.0]
                va.SetTuple(i, tup[:3])
            pd.AddArray(va)
        else:
            raise ValueError(f"Array '{name}' has shape {a.shape}, expected (N,) or (N,3)")

    if not fname.endswith(".vtp"):
        fname = fname.rsplit(".", 1)[0] + ".vtp"
    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(fname)
    w.SetInputData(poly)
    w.SetDataModeToAppended()
    w.EncodeAppendedDataOff()
    w.Write()
from differential_ops import (
    per_face_gradient_scalar, gradient_vertices_from_faces, cv_from_grad,
    laplacian_scalar_cotan, divergence_cotan, curl_normal, hessian_quadratic_fit,
    vertex_normals_area_weighted,fit_vertex_gradients_from_face_gradients,cv_from_face_grad_to_vertices
)
def _pick_lat_key(S):
    """
    Choose which scalar to treat as LAT for derivatives.
    Stim-only version.
    """
    if "Stim" in S:
        return "Stim"
    # fallback: whatever is there, just to avoid crashes
    return next(iter(S.keys()))


def _append_derivatives_to_S(S, V, F, L, M, lat_key=None):
    if lat_key is None:
        lat_key = _pick_lat_key(S)
    t = np.asarray(S[lat_key], float)

    # grad (NaN-aware)
    gF = per_face_gradient_scalar(V, F, t)
    gV=fit_vertex_gradients_from_face_gradients(V,F,gF)
    #gV = gradient_vertices_from_faces(V, F, gF)  # tangent-projected

    # CV
    #slowness, CV_mag, CV_vec = cv_from_grad(gV)
    slowness, CV_mag, CV_vec, slowness_f, CV_mag_f, CV_vec_f = cv_from_face_grad_to_vertices(
        V, F, gF, eps=1e-4, max_speed=1)
    # Laplacian
    #lap_t = laplacian_scalar_cotan(L, M, t)

    # Divergence & curl of CV vector
    # project CV_vec to tangent again for safety
    nrm = vertex_normals_area_weighted(V, F)
    vdotn = np.sum(CV_vec*nrm, axis=1)
    CV_vec_t = CV_vec - vdotn[:,None]*nrm
    """
    div_v = divergence_cotan(V, F, CV_vec_t)
    curln = curl_normal(V, F, CV_vec_t)

    # Hessian (2x2 in tangent frame → store components)
    H11, H12, H22 = hessian_quadratic_fit(V, F, t, rho_scale=1.5)"""

    # Add to dict: vectors as (N,3), scalars as (N,)
    S[f"{lat_key}_grad"]   = gV
    S[f"{lat_key}_slowness"] = slowness
    S[f"{lat_key}_CV_mag"] = CV_mag
    S[f"{lat_key}_CV_vec"] = CV_vec_t
    """S[f"{lat_key}_laplace"] = lap_t
    S[f"{lat_key}_divCV"]   = div_v
    S[f"{lat_key}_curlnCV"] = curln
    S[f"{lat_key}_H11"]     = H11
    S[f"{lat_key}_H12"]     = H12
    S[f"{lat_key}_H22"]     = H22"""
    return S


def _xml_header(): return '<?xml version="1.0"?>\n<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n'
def _xml_footer(): return '</VTKFile>\n'
def _dtype_str(a): return "Int32" if np.issubdtype(np.asarray(a).dtype, np.integer) else "Float32"

def _write_points_multi_vtp(points, arrays_numeric, arrays_string, fname):
    P = np.asarray(points, float); K = P.shape[0]
    with open(fname, "w", encoding="utf-8") as f:
        f.write(_xml_header())
        f.write(f'  <PolyData>\n    <Piece NumberOfPoints="{K}" NumberOfVerts="{K}" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">\n')
        # Points
        f.write('      <Points>\n')
        f.write('        <DataArray type="Float32" NumberOfComponents="3" Name="Points" format="ascii">\n')
        for p in P: f.write(f"          {p[0]:.9g} {p[1]:.9g} {p[2]:.9g}\n")
        f.write('        </DataArray>\n')
        f.write('      </Points>\n')
        # Verts (one vertex per point)
        f.write('      <Verts>\n')
        f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        for i in range(K): f.write(f"          {i}\n")
        f.write('        </DataArray>\n')
        f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        for i in range(1, K+1): f.write(f"          {i}\n")
        f.write('        </DataArray>\n')
        f.write('      </Verts>\n')
        # PointData
        f.write('      <PointData>\n')
        for name, arr in arrays_numeric.items():
            a = np.asarray(arr)
            f.write(f'        <DataArray type="{_dtype_str(a)}" Name="{name}" format="ascii">\n')
            for v in a: f.write(f"          {v}\n")
            f.write('        </DataArray>\n')
        for name, arr in arrays_string.items():
            f.write(f'        <DataArray type="String" Name="{name}" format="ascii">\n')
            for s in arr: f.write(f"          {str(s)}\n")
            f.write('        </DataArray>\n')
        f.write('      </PointData>\n')
        f.write('    </Piece>\n  </PolyData>\n')
        f.write(_xml_footer())

def export_lat_to_vtk(
    lat,
    verts, faces,
    coords, mesh_index,         # from cp.get_projection(...)
    cp_pnums,                   # cp.p_number (list/array of CARTO point numbers on the mesh side)
    out_dir,
    lam=1e-6
):
    """
    Stim-only export:
      - LAT = lat.Stim
      - Voltage = lat.Stim_Voltage
      - Duration = lat.Stim_dur (if present)
      - Deflection = lat.Stim_deflection (if present)
      - min_Voltage (if present)

    Produces:
      out_dir/mesh_multi.vtp  -> surface with ALL interpolated arrays
      out_dir/electrodes.vtp  -> projected points with raw per-point arrays + PointNumber + Label
    """
    os.makedirs(out_dir, exist_ok=True)

    # ----- align by point_number -----
    a = np.asarray(lat.p_numbers, dtype=int)      # LAT side
    b = np.asarray(cp_pnums,    dtype=int)  
    print("Lat.pnumber",a,"cp_pnumber",b)      # CP side (same order as coords/mesh_index)
    a_index = {v:i for i,v in enumerate(a)}
    b_index = {v:i for i,v in enumerate(b)}
    common = sorted(set(a_index)&set(b_index), key=lambda x:(a_index[x], b_index[x]))
    if not common:
        raise RuntimeError("No common point numbers between LAT and CP sets.")
    ai = np.array([a_index[v] for v in common], dtype=int)  # indices into LAT arrays
    bi = np.array([b_index[v] for v in common], dtype=int)  # indices into CP arrays

    # ----- gather per-point arrays (LAT order -> aligned to ai) -----
    def A(x, dtype=float):
        return np.asarray(x, dtype=dtype)[ai]

    labels   = np.asarray(lat.labels, dtype=str)[ai]
    pnums    = np.asarray(lat.p_numbers, dtype=int)[ai]
    pt_coords = np.asarray(coords, float)[bi]
    known_idx = np.asarray(mesh_index, int)[bi]

    # numeric fields (add/remove as needed)
    First        = A(lat.First,        float)
    Second       = A(lat.Second,       float)
    Third        = A(lat.Third,        float)
    SR           = A(lat.SR,           float) if getattr(lat, "SR", None) is not None else np.full_like(First, np.nan)

    Sinus_dur    = A(lat.Sinus_dur,    float) if getattr(lat, "Sinus_dur", None) is not None else np.full_like(First, np.nan)

    First_V      = A(lat.First_Voltage,  float)
    Second_V     = A(lat.Second_Voltage, float)
    Third_V      = A(lat.Third_Voltage,  float)
    SR_V        = A(lat.Voltage_sinus,  float)
    min_V       = A(lat.min_Voltage, float)

    First_dur    = A(lat.First_dur,    float)
    Second_dur   = A(lat.Second_dur,   float)
    Third_dur    = A(lat.Third_dur,    float)

    First_delta  = A(lat.First_Delta,  float)
    Second_delta = A(lat.Second_Delta, float)
    Third_delta  = A(lat.Third_Delta,  float)

    First_defl   = A(lat.First_deflection,  float) if getattr(lat, "First_deflection", None)  is not None else np.full_like(First, np.nan)
    Second_defl  = A(lat.Second_deflection, float) if getattr(lat, "Second_deflection", None) is not None else np.full_like(First, np.nan)
    Third_defl   = A(lat.Third_deflection,  float) if getattr(lat, "Third_deflection", None)  is not None else np.full_like(First, np.nan)

    # per-point numeric dict (aligned to known_idx)
    values_point = {
        # LATs / timings
        "First": First, "Second": Second, "Third": Third, "SR": SR,
        "Sinus_dur": Sinus_dur,
        # Voltages
        "First_Voltage": First_V, "Second_Voltage": Second_V, "Third_Voltage": Third_V,
        "SR_Voltage": SR_V,
        "min_Voltage": min_V,
        # Durations
        "First_dur": First_dur, "Second_dur": Second_dur, "Third_dur": Third_dur,
        # Deltas
        "First_Delta": First_delta, "Second_Delta": Second_delta, "Third_Delta": Third_delta,
        # Deflections
        "First_deflection": First_defl, "Second_deflection": Second_defl, "Third_deflection": Third_defl,
    }

    
    print("interpolating")
    # ----- interpolate ALL to mesh -----
    S = _interp_many_nanaware(verts, faces, known_idx, values_point, lam=lam)

    # ---- build L, M once ----
    V = np.asarray(verts, float); F = np.asarray(faces, int)
    L, M = build_cotan_laplacian_and_mass(V, F)

    # ---- append derivative fields for a chosen LAT (First/SR/Second/Third) ----
    try:
        if "First"  in S: S = _append_derivatives_to_S(S, V, F, L, M, lat_key="First")
        if "Second" in S: S = _append_derivatives_to_S(S, V, F, L, M, lat_key="Second")
        if "Third"  in S: S = _append_derivatives_to_S(S, V, F, L, M, lat_key="Third")
        if "SR"     in S: S = _append_derivatives_to_S(S, V, F, L, M, lat_key="SR")
    except Exception as e:
        print("[DERIV]", e)
    
    # ---------- AREA / INTEGRAL LOGIC (Stim_Voltage only) ----------
    voltage_keys = ["First_Voltage","Second_Voltage","Third_Voltage","min_Voltage","SR_Voltage"]
    CV_Keys=["First_CV_mag","Second_CV_mag","Third_CV_mag","SR_CV_mag"] 
    areas_5   = {}
    areas_15  = {}
    integrals = {}
    slow_CV={}
    normal_CV={}
    integrals_CV={}

    if CV_Keys:
        for key in CV_Keys:
            area_total = mesh_area_valid(verts, faces, S[key])
            try:
                A3 = band_area_graph(verts, faces, S[key], 0.0, 0.3)
                int_band = band_integral(verts, faces, S[key],
                                         np.nanmin(S[key]), np.nanmax(S[key]))
                slow_CV[key]   = float(A3)/area_total
                integrals_CV[key] = float(int_band)/area_total
            except Exception as e:
                print(f"[AREA] Skipped {key} (0–0.5) due to error: {e}")
            try:
                A1 = band_area_graph(verts, faces, S[key], 0.3, 1)
                int_band = band_integral(verts, faces, S[key],
                                         np.nanmin(S[key]), np.nanmax(S[key]))
                normal_CV[key]   = float(A1)/area_total
            except Exception as e:
                print(f"[AREA] Skipped {key} (0–0.5) due to error: {e}")

    if slow_CV:
        labels_ = list(slow_CV.keys())
        vals   = np.array([slow_CV[k] for k in labels_], dtype=float)
        total  = float(np.sum(vals))
        perc   = (vals / total * 100.0) if total > 0 else np.zeros_like(vals)

        fig, ax = plt.subplots(figsize=(6, 3.2))
        x = np.arange(len(labels_))
        ax.bar(x, vals, align="center")

        ax.set_xticks(x)
        ax.set_xticklabels(labels_, rotation=0)
        ax.set_ylabel("Area/ mesh Area")
        ax.set_title("Band area comparison: 0.0–0.3, CV")

        for xi, v, p in zip(x, vals, perc):
            ax.text(xi, v, f"{v:.3g}\n({p:.1f}%)",
                    ha="center", va="bottom", fontsize=9)

        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()

        out_png = os.path.join(out_dir, "CV_band_area_0_0p3.png")
        fig.savefig(out_png, dpi=300, bbox_inches="tight", transparent=True)
        plt.close(fig)
        print("[AREA] Saved comparison plot:", out_png)
    else:
        print("[AREA] No Stim_CV area (0–0.3) plotted.")

    if normal_CV:
        labels_ = list(normal_CV.keys())
        vals   = np.array([normal_CV[k] for k in labels_], dtype=float)
        total  = float(np.sum(vals))
        perc   = (vals / total * 100.0) if total > 0 else np.zeros_like(vals)

        fig, ax = plt.subplots(figsize=(6, 3.2))
        x = np.arange(len(labels_))
        ax.bar(x, vals, align="center")

        ax.set_xticks(x)
        ax.set_xticklabels(labels_, rotation=0)
        ax.set_ylabel("Area/ mesh Area")
        ax.set_title("Band area comparison: 0.0–0.3, CV")

        for xi, v, p in zip(x, vals, perc):
            ax.text(xi, v, f"{v:.3g}\n({p:.1f}%)",
                    ha="center", va="bottom", fontsize=9)

        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()

        out_png = os.path.join(out_dir, "CV_band_area_0p3_1.png")
        fig.savefig(out_png, dpi=300, bbox_inches="tight", transparent=True)
        plt.close(fig)
        print("[AREA] Saved comparison plot:", out_png)
    else:
        print("[AREA] No Stim_CV area (0.3_1) plotted.")

    if voltage_keys:
        # 0.0–0.5 band
        for key in voltage_keys:
            area_total = mesh_area_valid(verts, faces, S[key])
            try:
                A5 = band_area_graph(verts, faces, S[key], 0.0, 0.5)
                int_band = band_integral(verts, faces, S[key],
                                         np.nanmin(S[key]), np.nanmax(S[key]))
                areas_5[key]   = float(A5)/area_total
                integrals[key] = float(int_band)/area_total
            except Exception as e:
                print(f"[AREA] Skipped {key} (0–0.5) due to error: {e}")

        # 0.0–1.5 band
            try:
                A15 = band_area_graph(verts, faces, S[key], 0.0, 1.5)
                areas_15[key] = float(A15)/area_total
            except Exception as e:
                print(f"[AREA] Skipped {key} (0–1.5) due to error: {e}")
    else:
        area_total = None
        print("[AREA] Stim_Voltage not in S; skipping area metrics.")

    # ---- 0.0–0.5 plot ----
    if areas_5:
        labels_ = list(areas_5.keys())
        vals   = np.array([areas_5[k] for k in labels_], dtype=float)
        total  = float(np.sum(vals))
        perc   = (vals / total * 100.0) if total > 0 else np.zeros_like(vals)

        fig, ax = plt.subplots(figsize=(6, 3.2))
        x = np.arange(len(labels_))
        ax.bar(x, vals, align="center")

        ax.set_xticks(x)
        ax.set_xticklabels(labels_, rotation=0)
        ax.set_ylabel("Area/ mesh Area")
        ax.set_title("Band area comparison: 0.0–0.5")

        for xi, v, p in zip(x, vals, perc):
            ax.text(xi, v, f"{v:.3g}\n({p:.1f}%)",
                    ha="center", va="bottom", fontsize=9)

        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()

        out_png = os.path.join(out_dir, "voltage_band_area_0_0p5.png")
        fig.savefig(out_png, dpi=300, bbox_inches="tight", transparent=True)
        plt.close(fig)
        print("[AREA] Saved comparison plot:", out_png)
    else:
        print("[AREA] No Stim_Voltage area (0–0.5) plotted.")

    # ---- 0.0–1.5 plot ----
    if areas_15:
        labels_ = list(areas_15.keys())
        vals   = np.array([areas_15[k] for k in labels_], dtype=float)
        total  = float(np.sum(vals))
        perc   = (vals / total * 100.0) if total > 0 else np.zeros_like(vals)

        fig, ax = plt.subplots(figsize=(6, 3.2))
        x = np.arange(len(labels_))
        ax.bar(x, vals, align="center")

        ax.set_xticks(x)
        ax.set_xticklabels(labels_, rotation=0)
        ax.set_ylabel("Area/mesh Area")
        ax.set_title("Band area comparison: 0.0–1.5")

        for xi, v, p in zip(x, vals, perc):
            ax.text(xi, v, f"{v:.3g}\n({p:.1f}%)",
                    ha="center", va="bottom", fontsize=9)

        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()

        out_png = os.path.join(out_dir, "voltage_band_area_0_1p5.png")
        fig.savefig(out_png, dpi=300, bbox_inches="tight", transparent=True)
        plt.close(fig)
        print("[AREA] Saved comparison plot:", out_png)
    else:
        print("[AREA] No Stim_Voltage area (0–1.5) plotted.")

    # ---- integrals plot ----
    if integrals:
        labels_ = list(integrals.keys())
        vals   = np.array([integrals[k] for k in labels_], dtype=float)
        total  = float(np.sum(vals))
        perc   = (vals / total * 100.0) if total > 0 else np.zeros_like(vals)

        fig, ax = plt.subplots(figsize=(6, 3.2))
        x = np.arange(len(labels_))
        ax.bar(x, vals, align="center")

        ax.set_xticks(x)
        ax.set_xticklabels(labels_, rotation=0)
        ax.set_ylabel("V(mv)")
        ax.set_title("mean_voltage")

        for xi, v, p in zip(x, vals, perc):
            ax.text(xi, v, f"{v:.3g}\n({p:.1f}%)",
                    ha="center", va="bottom", fontsize=9)

        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()

        out_png = os.path.join(out_dir, "mean_integral.png")
        fig.savefig(out_png, dpi=300, bbox_inches="tight", transparent=True)
        plt.close(fig)
        print("[AREA] Saved comparison plot:", out_png)
    else:
        print("[AREA] No Stim_Voltage integrals plotted.")
    
    if integrals_CV:
        labels_ = list(integrals_CV.keys())
        vals   = np.array([integrals_CV[k] for k in labels_], dtype=float)
        total  = float(np.sum(vals))
        perc   = (vals / total * 100.0) if total > 0 else np.zeros_like(vals)

        fig, ax = plt.subplots(figsize=(6, 3.2))
        x = np.arange(len(labels_))
        ax.bar(x, vals, align="center")

        ax.set_xticks(x)
        ax.set_xticklabels(labels_, rotation=0)
        ax.set_ylabel("V(mv)")
        ax.set_title("mean_voltage")

        for xi, v, p in zip(x, vals, perc):
            ax.text(xi, v, f"{v:.3g}\n({p:.1f}%)",
                    ha="center", va="bottom", fontsize=9)

        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()

        out_png = os.path.join(out_dir, "mean_integral_CV.png")
        fig.savefig(out_png, dpi=300, bbox_inches="tight", transparent=True)
        plt.close(fig)
        print("[AREA] Saved comparison plot:", out_png)
    else:
        print("[AREA] No Stim_CV integrals plotted.")

    # ----- write mesh (multi scalar) -----
    mesh_vtk = os.path.join(out_dir, "mesh_multi.vtp")
    write_mesh_multi_scalar_vtp_vtk(verts, faces, S, mesh_vtk)

    # ----- write points (numeric + string) -----
    labels_arr = np.asarray(labels)
    labels_num = (labels_arr == "POS").astype(np.int8)
    pt_arrays_num = dict(values_point)
    pt_arrays_num["PointNumber"] = pnums
    pt_arrays_num["Label"] = labels_num
    pt_arrays_str = {}

    pts_vtp = os.path.join(out_dir, "electrodes.vtp")
    _write_points_multi_vtp(pt_coords, pt_arrays_num, pt_arrays_str, pts_vtp)

    print("[VTK] wrote:")
    print("  ", mesh_vtk)
    print("  ", pts_vtp)
    return mesh_vtk, pts_vtp
