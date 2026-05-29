# area_hist.py
import os, math, numpy as np
import matplotlib.pyplot as plt
from claculate_area import band_area_graph   # <- uses your function

def _nice_bins_for_scalar(s_array, step=0.5):
    s = np.asarray(s_array, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return np.array([0.0, 1.0])
    smin, smax = float(np.min(s)), float(np.max(s))
    lo = step * math.floor(smin / step)
    hi = step * math.ceil (smax / step)
    if hi <= lo: hi = lo + step
    return np.arange(lo, hi + step*1.0000001, step, dtype=np.float64)

def compute_area_histograms_all_scalars(verts, faces, S_dict, step=0.5, max_bins_per_scalar=300, only_keys=None):
    names = list(S_dict.keys()) if only_keys is None else [k for k in only_keys if k in S_dict]
    out = {}
    for name in names:
        arr = np.asarray(S_dict[name], dtype=np.float64)
        edges = _nice_bins_for_scalar(arr, step=step)
        if edges.size - 1 > max_bins_per_scalar:
            idx = np.linspace(0, edges.size-1, max_bins_per_scalar+1, dtype=int)
            edges = edges[idx]
        areas = np.zeros(edges.size - 1, dtype=np.float64)
        for i in range(edges.size - 1):
            a, b = float(edges[i]), float(edges[i+1])
            areas[i] = band_area_graph(verts, faces, arr, a, b)
        out[name] = (edges, areas)
    return out

def plot_area_histograms(results_dict, out_dir, fname="scalar_band_areas_0p5.png", cols=3):
    names = list(results_dict.keys())
    if not names:
        return None
    rows = int(math.ceil(len(names)/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0*cols, 2.8*rows), squeeze=False)
    for idx, name in enumerate(names):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        edges, areas = results_dict[name]
        centers = 0.5 * (edges[:-1] + edges[1:])
        width   = (edges[1:] - edges[:-1])
        ax.bar(centers, areas, width=width, align="center")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Scalar bin")
        ax.set_ylabel("Area")
        ax.grid(True, alpha=0.25)
    # hide unused subplots
    for j in range(len(names), rows*cols):
        r, c = divmod(j, cols)
        axes[r][c].axis("off")
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return path
