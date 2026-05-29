import numpy as np
import os
from datetime import datetime
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def _xml_header():
    return '<?xml version="1.0"?>\n<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n'

def _xml_footer():
    return '</VTKFile>\n'

def _dtype_str(a: np.ndarray) -> str:
    # minimal mapping; extend if you need more
    if a.dtype == np.float32:
        return "Float32"
    if a.dtype == np.float64:
        return "Float64"
    if a.dtype == np.int32:
        return "Int32"
    if a.dtype == np.int64:
        return "Int64"
    # default: most numeric arrays will be ok as Float32
    return "Float32"


def _write_points_multi_vtp(points,
                           arrays_numeric: None | dict = None,
                           arrays_string: None | dict = None,
                           path: None | str = None,
                           fname: None | str = None):
    """
    :param points: the coords of the points
    :param arrays_numeric: dict name -> values or None
    :param arrays_string: dict name -> labels or None
    :param path: output directory or None
    :param fname: output file name or None
    """

    # -------- path logic --------
    if path is None:
        try:
            import __main__
            base = os.path.dirname(os.path.abspath(__main__.__file__))
        except Exception:
            base = os.getcwd()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(base, "results", ts)
        os.makedirs(path, exist_ok=True)

    if fname is None:
        fname = "points.vtp"
    elif not fname.lower().endswith(".vtp"):
        fname += ".vtp"

    out = os.path.join(path, fname)

    # -------- write VTP --------
    P = np.asarray(points, float); K = P.shape[0]
    with open(out, "w", encoding="utf-8") as f:
        f.write(_xml_header())
        f.write(f'  <PolyData>\n    <Piece NumberOfPoints="{K}" NumberOfVerts="{K}" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">\n')

        # Points
        f.write('      <Points>\n')
        f.write('        <DataArray type="Float32" NumberOfComponents="3" Name="Points" format="ascii">\n')
        for p in P:
            f.write(f"          {p[0]:.9g} {p[1]:.9g} {p[2]:.9g}\n")
        f.write('        </DataArray>\n')
        f.write('      </Points>\n')

        # Verts
        f.write('      <Verts>\n')
        f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        for i in range(K):
            f.write(f"          {i}\n")
        f.write('        </DataArray>\n')
        f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        for i in range(1, K + 1):
            f.write(f"          {i}\n")
        f.write('        </DataArray>\n')
        f.write('      </Verts>\n')

        # PointData (only if provided)
        if arrays_numeric is not None or arrays_string is not None:
            f.write('      <PointData>\n')

            if arrays_numeric is not None:
                for name, arr in arrays_numeric.items():
                    a = np.asarray(arr)
                    f.write(f'        <DataArray type="{_dtype_str(a)}" Name="{name}" format="ascii">\n')
                    for v in a:
                        f.write(f"          {v}\n")
                    f.write('        </DataArray>\n')

            if arrays_string is not None:
                for name, arr in arrays_string.items():
                    f.write(f'        <DataArray type="String" Name="{name}" format="ascii">\n')
                    for s in arr:
                        f.write(f"          {str(s)}\n")
                    f.write('        </DataArray>\n')

            f.write('      </PointData>\n')

        f.write('    </Piece>\n  </PolyData>\n')
        f.write(_xml_footer())

    return out



def write_vtp_mesh(V, F,
                   scalars_numeric: None | dict = None,
                   scalars_string: None | dict = None,
                   path: None | str = None,
                   filename: None | str = None):
    """
    V: (N,3) array of vertex positions
    F: (M,3) array of triangle vertex indices
    scalars_numeric: dict name -> array (numeric only), or None
    scalars_string: dict name -> array (string only), or None
    path: output directory or None
    filename: output file name or None"""
    # -------- path logic --------
    if path is None:
        try:
            import __main__
            # if __main__.__file__ is not available (e.g. in interactive environments), fallback to current working directory
            base = os.path.dirname(os.path.abspath(__main__.__file__))
        except Exception:
            # fallback to current working directory if __main__.__file__ is not available (e.g. in interactive environments)
            base = os.getcwd()
        # create a timestamped subdirectory to avoid overwriting previous exports
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(base, "results", ts)
        os.makedirs(path, exist_ok=False)
    # handle filename
    if filename is None:
        filename = "mesh.vtp"
    elif not filename.lower().endswith(".vtp"):
        filename += ".vtp"

    out = os.path.join(path, filename)

    # -------- build polydata from V, F --------
    V = np.asarray(V, float)
    F = np.asarray(F, int)

    pts = vtk.vtkPoints()
    pts.SetNumberOfPoints(V.shape[0])
    for i in range(V.shape[0]):
        x, y, z = V[i]
        pts.SetPoint(i, float(x), float(y), float(z))

    polys = vtk.vtkCellArray()
    for a, b, c in F:
        tri = vtk.vtkTriangle()
        tri.GetPointIds().SetId(0, int(a))
        tri.GetPointIds().SetId(1, int(b))
        tri.GetPointIds().SetId(2, int(c))
        polys.InsertNextCell(tri)

    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetPolys(polys)

    # -------- add PointData scalars (optional) --------
    pd = poly.GetPointData()

    if scalars_numeric is not None:
        for name, arr in scalars_numeric.items():
            a = np.asarray(arr)
            # allow (K,) or (K, nc)
            if a.ndim == 1:
                nc = 1
                a2 = a.reshape(-1, 1)
            else:
                nc = a.shape[1]
                a2 = a

            vtk_arr = vtk.vtkFloatArray()
            vtk_arr.SetName(str(name))
            vtk_arr.SetNumberOfComponents(int(nc))
            vtk_arr.SetNumberOfTuples(V.shape[0])

            for i in range(V.shape[0]):
                if nc == 1:
                    vtk_arr.SetTuple1(i, float(a2[i, 0]))
                else:
                    vtk_arr.SetTuple(i, [float(x) for x in a2[i]])
            pd.AddArray(vtk_arr)

    if scalars_string is not None:
        for name, arr in scalars_string.items():
            a = np.asarray(arr, dtype=object)

            vtk_arr = vtk.vtkStringArray()
            vtk_arr.SetName(str(name))
            vtk_arr.SetNumberOfTuples(V.shape[0])

            for i in range(V.shape[0]):
                vtk_arr.SetValue(i, str(a[i]))
            pd.AddArray(vtk_arr)

    # -------- write --------
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(out)
    writer.SetInputData(poly)
    writer.Write()

    return out






def read_vtp_mesh(filename) -> tuple[vtk.vtkPolyData, np.ndarray, np.ndarray]:
    """
    Reads a VTK PolyData file (.vtp or .vtk) and returns the polydata object, vertex array, and face array.
    returns:
        poly: vtk.vtkPolyData object containing the mesh
        V: (N,3) array of vertex positions
        F: (M,3) array of triangle vertex indices
        to access point data arrays, you can do: poly.GetPointData().GetArray("array_name") and then convert to numpy if needed. to get array names, you can iterate over poly.GetPointData().GetNumberOfArrays() and get each array's name.
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    elif ext == ".vtk":
        reader = vtk.vtkPolyDataReader()
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    reader.SetFileName(filename)
    reader.Update()
    poly = reader.GetOutput()

    pts_vtk = poly.GetPoints()
    if pts_vtk is None:
        raise ValueError("No points found in PolyData")

    V = vtk_to_numpy(pts_vtk.GetData()).astype(np.float64)
    # Note: vtkCellArray for polys is a bit tricky; it stores connectivity in a flat array where each cell is prefixed by the number of points in that cell (3 for triangles). So we need to reshape it accordingly.
    polys = poly.GetPolys()
    conn = vtk_to_numpy(polys.GetData())
    faces_raw = conn.reshape(-1, 4)   # [3, i, j, k]
    F = faces_raw[:, 1:4].astype(np.int64)
    # polys.GetData() returns a flat array like [3, i, j, k, 3, i2, j2, k2, ...] where the first number (3) indicates the number of points in the cell (triangle). We reshape it to (-1, 4) to get rows of [3, i, j, k], and then take columns 1:4 for the vertex indices.  
    # to access point data arrays, you can do: poly.GetPointData().GetArray("array_name") and then convert to numpy if needed. to get array names, you can iterate over poly.GetPointData().GetNumberOfArrays() and get each array's name.
    return poly, V, F



def write_mesh_with_scalar_vtk(verts, faces, scalars: None | dict = None,
                               path=None, filename=None):
    """
    verts : (N,3)
    faces : (M,3)
    scalars : dict name -> array (numeric only), or None
    """
    if scalars is not None and len(scalars) == 0:
        scalars = None

    return write_vtp_mesh(
        verts,
        faces,
        scalars_numeric=scalars,
        scalars_string=None,
        path=path,
        filename=filename
    )


