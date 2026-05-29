from Json_aux.EXTRACT_FROM_JASON import LAT_points
import json,os,sys
import numpy as np
import tkinter as tk
from tkinter import filedialog
from batch_vtk_features import export_lat_to_vtk,_mean_edge_length
import pymeshlab as ml
from CARTO_Tool import Carto
from cartopoints import Carto_points
from smooth import *
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def remesh_isotropic_equal_triangles(
    verts,
    faces,
    target_edge=None,
    edge_factor=0.8,
    iterations=10,
):
    """
    Simple isotropic remeshing: equal-ish triangles, geometry respected.

    verts, faces : input mesh
    target_edge  : absolute target edge length. If None, use edge_factor * mean_edge.
    edge_factor  : factor * mean_edge when target_edge is None.
    iterations   : remeshing iterations.
    """
    V = np.asarray(verts, float)
    F = np.asarray(faces, int)

    if target_edge is None:
        target_edge = edge_factor * _mean_edge_length(V, F)

    try:
   
        targetlen_obj = ml.PureValue(float(target_edge))
    except AttributeError:
 
        targetlen_obj = ml.AbsoluteValue(float(target_edge))

    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(V, F), "mesh")

  
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_unreferenced_vertices()

  
    ms.meshing_isotropic_explicit_remeshing(
        targetlen=targetlen_obj,
        iterations=int(iterations),
    )

    mout = ms.current_mesh()
    V_new = mout.vertex_matrix()
    F_new = mout.face_matrix()
    return V_new, F_new

if __name__=="__main__":
    from datetime import datetime
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_dir = os.path.join(base_dir, f"VTK_exports {ts}")
    os.makedirs(out_dir, exist_ok=True)
    root = tk.Tk()
    root.withdraw()
    file_path=None
    # make sure dialog appears on top
    if not file_path:
        root.attributes("-topmost",True)
        file_path = filedialog.askopenfilename(
    parent=root,
    title="Open JSON file",
    filetypes=[("JSON files", "*.json")]
)

        root.destroy()
        print(file_path)
    with open(file_path,"r") as f:
        a=json.load(f)

    # use stim-only extractor
    lat = LAT_points(a)                           
    a   = np.array(lat.p_numbers, dtype="int")
    #V1  = np.array(lat.voltage, dtype="float64")       
    #print("..............",V1,lat.LAT)
    # no Second/Third in stim-only mode
    # V2,V3 not needed
    from File_handler import write_mesh_with_scalar_vtk,read_vtp_mesh
    carto=Carto()
    verts, faces, scals1, scals2,LAT = carto.pars_mesh_file_with_electrode()  # verts:(N,3), faces:(M,3)
    scalars = {}
    if scals1 is not None:
        scalars.update({"s1":scals1})
    if scals2 is not None:
        scalars.update({"s2":scals2})
    if LAT is not None:
        scalars["LAT"] = LAT

    out = write_mesh_with_scalar_vtk(verts, faces, scalars,path=out_dir,filename="carto_map")
    """root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    """
    """vtk_path = filedialog.askopenfilename(
        parent=root,
        title="Open custom VTK file to project features on",
        filetypes=[("VTK PolyData files", "*.vtp *.vtk")]
    )

    root.destroy()

    if not vtk_path:
        raise SystemExit("No VTK file selected.")

    poly, V, F = read_vtp_mesh(vtk_path)"""
    
    #verts,faces=remesh_isotropic_equal_triangles(verts,faces,edge_factor=1.5)
    
    cp=Carto_points(carto)
    cp.extract_all()
    projection=cp.get_projection(cp.points,verts)
    b=np.array(cp.p_number,dtype="int")
    print(cp.p_number,projection)
    coords,mesh_index=projection
    mesh_index=np.array(mesh_index,dtype="int")
    a_index = {v: i for i, v in enumerate(a)}
    b_index = {v: i for i, v in enumerate(b)}

    # find common point numbers
    common = sorted(set(a_index.keys()) & set(b_index.keys()), key=lambda x: (a_index[x], b_index[x]))

    # now extract index arrays
    ai = np.array([a_index[v] for v in common], dtype=int)
    bi = np.array([b_index[v] for v in common], dtype=int)
    assert np.all(a[ai] == b[bi])
    


    print("data sent")
    export_lat_to_vtk(
        lat=lat,
        verts=verts, faces=faces,
        coords=coords, mesh_index=mesh_index,
        cp_pnums=np.array(cp.p_number, dtype=int),
        out_dir=out_dir,
        lam=1e-6
    )
    # verify
    assert np.all(a[ai] == b[bi])
    print(f"{len(common)} common point numbers")
    print("ai[:10] =", a[ai[:]])
    print("bi[:10] =", b[bi[:]])
    print(a[ai[:]]==b[bi[:]])
    #mesh_index[bi],V1[ai]
    