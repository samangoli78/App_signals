import os,sys
import numpy as np
from scipy.spatial import KDTree

from CARTO_Tool import Carto
class Carto_points:
    def __init__(self,carto:Carto,triple=False):
        self.carto=carto
        self.positive=None
        self.negative=None
        self.points=None
        self.p_number=None
        carto.extracting_color_coding(triple)
        self.CAR=carto.car_extract()
    def extract_pos(self,identifier:list=["POS","VERDE","VERD","VER","GREEN","HSC+","HSC","POSITIVE"]):
        self.positive=np.array(self.CAR.loc[np.isin(self.CAR["label_color"].str.upper(),identifier),"x":"z"]).astype(float)

    def extract_neg(self,identifier:list=["NEG","NARANJA","NARANJ","NAR","ORANGE","HSC-","NEGATIVE"]):
        self.negative=np.array(self.CAR.loc[np.isin(self.CAR["label_color"].str.upper(),identifier),"x":"z"]).astype(float)
    def extract_all(self):
        self.carto.extracting_color_coding(triple=False)
        self.CAR=self.carto.car_extract()
        self.points=np.array(self.CAR.loc[:,"x":"z"]).astype(float)
        print(self.CAR.columns)
        self.p_number=np.array(self.CAR.loc[:,"point number"])
    def get_projection(self,points,mesh_points:list):
        if points is not np.array:
            points=np.array(points).astype(float)

        kdtree = KDTree(mesh_points)
        distances, indices = kdtree.query(points)
        projected_points = mesh_points[indices]
        return projected_points,indices




if __name__=="__main__":
    carto=Carto(r"D:/data_Carto/PERUADE/New_Case_3Extra_01")
    verts, faces, scals1, scals2,LAT = carto.pars_mesh_file_with_electrode()  # verts:(N,3), faces:(M,3)

    cp=Carto_points(carto)
    cp.extract_all()
    print("original_points",cp.points[:10])
    projection=cp.get_projection(cp.points,verts)
    print("projected points",projection[:10])
    
