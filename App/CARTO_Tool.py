import os,re
import numpy as np
import pandas as pd
from collections import Counter
import os

from .PARSER_Tool import Parser_carto
import tkinter as tk
from tkinter import filedialog
import time
import xml.etree.ElementTree as ET
from lxml import etree
from pathlib import Path


def log(input:str,*args) -> str:
    if not hasattr(log,"initialized"):
        log.initialized=True
        mode="w"
    else:
        mode="a"
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"log.txt"),mode,encoding="utf-8") as f:
        out=""
        for inp in [*args,input]:
            if isinstance(inp,str):
                f.write(inp)
                out+=inp
            elif isinstance(inp,pd.DataFrame):
                f.write(inp.to_string(index=False))
                out+=inp.to_string()
            else:
                f.write(str(inp))
                out+=str(inp)

        f.write("\n")
    return out


class Carto(Parser_carto):
    cartos=[]
    def __init__(self,path=None) -> None:
        Carto.cartos.append(self)
        super().__init__(self)
        self.i=None
        self.cont=None
        if path and os.path.exists(path):
            self.path=path
            self.on_init()
        else:
            print(log("could not initialize \nchoose your file from the dialogue")) 
            root = tk.Tk()
            root.withdraw()
            
            # make sure dialog appears on top
            root.attributes("-topmost",True)
            file_path = filedialog.askdirectory(parent=root)
            log("path is "+file_path)
            root.destroy()
            print(file_path)
            self.path=file_path
            self.on_init()

        
    def on_init(self):
        main_xmls=[name for name in os.listdir(self.path) if name.endswith(".xml") and "Export" not in name]
        map_data=[]
        for xml in main_xmls:
            try:
                tree=ET.parse(os.path.join(self.path,xml))
                self.tree=tree.getroot()
                map_data=[{key:attr for key,attr in map.items()} for map in self.tree.find("Maps").findall("Map")]
                print(log(f"the map_data is a list and contains {map_data} "))
                if len(map_data)!=0:
                    break
            except:
                pass
        if len(map_data)==0:
            print(log("couldnt encrypt the data the map_data length was 0"))
            return False
        
        self.maps=[i["Name"] for i in map_data]

        print(log("options: ", {i:self.maps[i] for i in range(len(self.maps))}, "\nchoose your file with its index passing to set_cat_value method"))
        self.root=tk.Tk()
        
        self.frames=[]
        self.labels=[]
        for i in range(len(self.maps)):
            self.frames.append(tk.Frame(self.root))
            self.frames[i].pack(fill=tk.X)
            self.labels.append(tk.StringVar())
            self.labels[i]=self.maps[i]
            Button=tk.Button(self.frames[i],text=self.labels[i],command= lambda a=i : self.index(a), font=("timesnewroman", 10),)
            
            Button.pack(expand=True,fill=tk.X)
        self.root.attributes("-topmost",True)
        self.root.after(150, lambda: self.root.attributes('-topmost', False))
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.root.geometry(f"{int(self.root.winfo_width())+25}x{int(self.root.winfo_height())+150}")
        self.root.mainloop()
    def quit(self):
        self.root.quit()
        self.root.destroy() 
    def index(self,i):
        self.i=i
        print(log("this map has been selected",self.maps[i],self.i))
        self.quit()



    def Car_file(self):
        with open(os.path.join(self.path,self.maps[self.i]+"_car.txt")) as file:
            content=[m.split() for m in file.readlines()]
            content=np.array(content[1:])
            data=pd.DataFrame(data=content,columns=[chr(ord("A")+h) if h+ord("A")<=ord("Z") else "A"+chr(h-1+2*ord("A")-ord("Z")) for h in range(content.shape[1])])
        return data

    def extracting_color_coding(self,triple=False):
       
        #triple extra only
        self.color_dict={dict(ch.items())["ID"]:dict(ch.items())["Full_Name"] for ch in self.tree.find("Maps").find("TagsTable")}
        print(log(f"the color coding was extracted from main xml file in maps and then tags table and the ID:Full_Name is in {self.color_dict}"))
        new_color_dict={}
        if triple:
            for i,j in self.color_dict.items():
                if np.isin(j.lower(),["verde","green","orange","naranja","hsc+","hsc-","hsc","pos","positive","negative","neg","ver","nar"]):
                    new_color_dict.update({i:j})
            print(log(f"the triple extra is called so the new color coding is filtered and is :  {self.color_dict}"))
            self.color_dict=new_color_dict
        
        return self.color_dict
                
    
    def car_extract(self,triple=False):
        self.extracting_color_coding(triple)
        data= self.Car_file().values
        labels=data[:,15]
        new_labels=[]
        for i,m in enumerate(labels) :
            if np.isin(m,list(self.color_dict.keys())): 
                new_labels.append([self.color_dict[m], data[i,2],data[i,27],data[i,4],data[i,5],data[i,6]])
            else: 
                pass
        if len(new_labels)==0:
            return None
        return pd.DataFrame(
            np.array(new_labels),
            columns=["label_color","point number","sample","x","y","z"]
            )
    
    def electrodes(self,triple=False):
        file_number,electrode,refference=[],[],[]
        car=self.car_extract(triple)
        if car is not None:
            for i in car.loc[:,"point number"].values:
                parser = etree.XMLParser(resolve_entities=False, remove_blank_text=True)
                root = etree.parse(os.path.join(
                        self.path,
                        self.maps[self.i]+f"_P{i}"+"_Point_Export.xml" 
                        ), parser).getroot()
                
                
                ecg=root.find(".//ECG")
                file_number.append(ecg.get('FileName'))
                electrode.append([ecg.get('UnipolarMappingChannel'),ecg.get('BipolarMappingChannel')])
                refference.append(ecg.get('ReferenceChannel'))


            c2=pd.Series(file_number,name="file_number")
            c3=pd.DataFrame(np.array(electrode).reshape(-1,2),columns=["unipolar","bipolar"])
            c4=pd.Series(refference,name="refference_chanel")
            return pd.concat(
                [
                    car[["label_color","sample","point number"]],
                    c2,
                    c3,
                    car[["x","y","z"]],
                    c4
                    ],
                axis=1
                )
        else:
            print("could not process")
    def Signals(self,triple=False):
        data=self.electrodes(triple)
        
        #check to delete Nan values from the data frame
        for i in data.columns:
            data=data.drop(data[data[i].isnull()].index)
        content=[]
        t=0
        indexes=np.linspace(0,2.5, 2500)
        for i in np.unique(data["file_number"].values):
            t=time.time()
            with open(os.path.join(self.path,i), "r", encoding="utf-8") as f:
                _=f.readline()                                # line 0 (unused)
                gain_line = f.readline().strip()            # line 1
                header_line = f.readline().strip()          # line 2
            gain = float(gain_line.split("=")[-1])
            cols = [m.split("(")[0] for m in header_line.split()]  # strip "(...)" suffixes
            df = pd.read_csv(
                os.path.join(self.path,i),
                skiprows=3,
                sep=r" ",             # <- instead of delim_whitespace=True
                names=cols,
                dtype=np.float32,
                engine="c",
            )

            # scale in-place (vectorized)
            df *= gain

            # build index without Python loops

            df.index = indexes


            content.append([data.loc[data["file_number"] == i, :].reset_index(drop=True),i,df])
            print(time.time()-t)
        self.cont=content
        return content
    
    def  create_output(self,content:list):
        base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        out_root = base_dir / "output"
        out_root.mkdir(exist_ok=True)
        for file_number, meta_df, signals_df in content:
            sub = out_root / file_number[:-4]
            sub.mkdir(exist_ok=True)

            meta_path = sub / f"{file_number}_meta.csv"
            sig_path = sub / f"{file_number}_sig.csv"
            meta_df.to_csv(meta_path,sep="\t",            # tab-separated like CARTO
                            index=False,
                            float_format="%.3f" ) # only 3 decimals)
                # second sheet: signals (second sf)
            signals_df.to_csv(sig_path,sep="\t", 
                            index=False,
                            float_format="%.3f" )



  


