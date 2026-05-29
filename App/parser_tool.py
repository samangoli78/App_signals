from __future__ import annotations
import os,re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import vtk

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from CARTO_Tool import Carto
class Parser_carto:
    def __init__(self,carto:"Carto"):
        self.carto=carto
        self.bipolar=None
        self.unipolar=None
        self.vertices=None
        self.triangles=None
    def parse_mesh_file(self):
        vertices = []
        triangles = []
        main_meshes=[name for name in os.listdir(self.carto.path) if name.endswith(".mesh") and self.carto.maps[self.carto.i] in name]
        print(main_meshes)
        with open(os.path.join(self.carto.path,main_meshes[0]), 'r',errors="ignore") as file:
            lines = file.readlines()
            vertices_section = False
            triangles_section = False
            k=0
            for line in lines:
            
                line = line.strip()
                if line.startswith('[VerticesSection]'):
                    vertices_section = True
                    triangles_section = False
                if line.startswith('[TrianglesSection]'):
                    vertices_section = False
                    triangles_section = True
                if line.startswith("[VerticesColorsSection]"):
                    vertices_section = False
                    triangles_section = False
                if vertices_section and '=' in line:
                    parts = line.split('=')
                    coords = parts[1].strip().split()[:3]
                    vertices.append([float(coord) for coord in coords])
                if triangles_section and '=' in line:
                    if k==0: 
                        k=1
                        continue
                    parts = line.split('=')
                    indices = parts[1].strip().split()[:3]
                    triangles.append([int(index) for index in indices])

        return np.array(vertices), np.array(triangles)
        
    def mesh_build(self):
        vertices, triangles = self.parse_mesh_file()
        self.test=triangles
        faces = np.hstack([[3] + list(tri) for tri in triangles])
        return [vertices, faces]
    
    def pars_mesh_file_with_electrode(self):
        """
        Docstring for pars_mesh_file_with_electrode
        
        :param self: Description
        :type self: Type
        :return: np.array(vertices), np.array(triangles), np.array(unipolar_values), np.array(bipolar_values),np.array(LAT_values)
        :rtype: tuple of np.array
        """
        vertices = []
        triangles = []
        unipolar_values = []
        bipolar_values = []
        LAT_values = []
        main_meshes=[name for name in os.listdir(self.carto.path) if name.endswith(".mesh") and self.carto.maps[self.carto.i] in name]
        print(main_meshes)
        path=os.path.join(self.carto.path,main_meshes[0])
        with open(path, 'r',errors="ignore") as file:
            lines = file.readlines()
            vertices_section = False
            triangles_section = False
            vertices_colors_section = False
            f=0
            f1=0
            for line in lines:
                line = line.strip()
                if line.startswith('[VerticesSection]'):
                    vertices_section = True
                    triangles_section = False
                    vertices_colors_section = False
                if line.startswith('[TrianglesSection]'):
                    vertices_section = False
                    triangles_section = True
                    vertices_colors_section = False
                if line.strip()=="[VerticesColorsSection]":
                    vertices_section = False
                    triangles_section = False
                    vertices_colors_section = True
                if line.strip() == "[VerticesAttributesSection]":
                    vertices_colors_section = False
                if vertices_section and '=' in line:
                    parts = line.split('=')
                    coords = parts[1].strip().split()[:3]
                    vertices.append([float(coord) for coord in coords])
                if triangles_section and '=' in line:
                    if f==0:
                        f=1
                        continue
                    parts = line.split('=')
                    indices = parts[1].strip().split()[:3]
                    triangles.append([int(index) for index in indices])
                if vertices_colors_section and '=' in line:
                    if f1==0:
                        f1=1
                        continue
                    parts = line.split('=')
                    colors = parts[1].strip().split()
                    unipolar_values.append(float(colors[0]))
                    bipolar_values.append(float(colors[1]))
                    LAT_values.append(float(colors[2]))
        self.unipolar=np.array(unipolar_values)
        self.bipolar=np.array(bipolar_values)
        self.LAT=np.array(LAT_values)
        self.vertices=np.array(vertices)
        self.triangles=np.array(triangles)
        return np.array(vertices), np.array(triangles), np.array(unipolar_values), np.array(bipolar_values),np.array(LAT_values)
    

    def save_mesh_toVTK(self,fname="output.vtp",abs_path=None):
        verts,faces,uni,bi,lat=self.pars_mesh_file_with_electrode()
        #def write_mesh_multi_scalar_vtp_vtk(verts, faces, scalars_dict, fname):
        if abs_path is None:
            abs_path = os.getcwd()
        elif not os.path.isdir(abs_path):
            print(f"[Warning] Invalid path: {abs_path}. Using current directory instead.")
            abs_path = os.getcwd()
        else:
            abs_path = os.path.abspath(abs_path)
        if fname is None:
            fname = "output.vtp"
        elif not fname.lower().endswith(".vtp"):
            fname += ".vtp"

        # 3. Compose final output file path
        fullpath = os.path.join(abs_path, fname)

        V = np.asarray(verts, float)
        F = np.asarray(faces, np.int64)
        scalar_dict={"unipolar":uni,"bipolar":bi,"LAT":lat}
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
        for name, arr in scalar_dict.items():
            a = np.asarray(arr, float).ravel()
            if a.size != n:
                raise ValueError(f"Scalar '{name}' has length {a.size}, expected {n}")
            va = vtk.vtkFloatArray()
            va.SetName(str(name))
            va.SetNumberOfComponents(1)
            va.SetNumberOfTuples(n)
            for i, v in enumerate(a):
                va.SetValue(i, float(v))  # np.nan passes through
            pd.AddArray(va)
            # Make the first array the active scalars for coloring
            if pd.GetScalars() is None:
                pd.SetScalars(va)

        w = vtk.vtkXMLPolyDataWriter()
        w.SetFileName(fullpath)
        w.SetInputData(poly)
        w.SetDataModeToAppended()   # binary blocks; NaNs preserved
        w.EncodeAppendedDataOff()
        w.Write()

                    
                