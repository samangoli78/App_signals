import os,re
import numpy as np
import pandas as pd
from collections import Counter
from PARSER_Tool import Parser_carto
import tkinter as tk
from tkinter import filedialog
import time



class Carto(Parser_carto):
    cartos=[]
    def __init__(self,path=None,i=None) -> None:
        self.cartos.append(self)
        self.i=None
        self.cont=None
        if path and os.path.exists(path):
            self.path=path
            if i==None:
                self.on_init()
        else:
            print("could not initialize \nchoose your file from the dialogue") 
            
            file_path = filedialog.askdirectory()
            
            print(file_path)
            self.path=file_path
            
            self.on_init()

        
    def on_init(self):
        self.new_name_for_files()
        self.sorted_file_types_with_number()
        self.geting_categories()
        print("options: ", {i:self.cats[i] for i in range(len(self.cats))}, "\nchoose your file with its index passing to set_cat_value method")
        self.root=tk.Tk()
        self.frames=[]
        self.labels=[]
        for i in range(len(self.cats)):
            self.frames.append(tk.Frame(self.root))
            self.frames[i].pack()
            self.labels.append(tk.StringVar())
            self.labels[i]=self.cats[i]
            Button=tk.Button(self.frames[i],text=self.labels[i],command= lambda a=i : self.index(a), font=("timesnewroman", 10))
            
            Button.pack()
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.root.mainloop()
    def quit(self):
        self.root.quit()
        self.root.destroy() 

    def index(self,i):
        print(i)
        self.set_cat_value(i)
        print(self.cats[i],self.i)
        self.quit()


    def new_name_for_files(self):
        file_names=os.listdir(self.path)
        pattern1=re.compile(r"(.*_)(\d+)(\.txt)")
        pattern2=re.compile(r"(.*P)(\d+)(.*\.xml)")
        new_file_names,newname=[],""
        for name in file_names:
            if pattern2.search(name):newname=pattern2.sub(r"\1\3",name)
            elif pattern1.search(name):newname=pattern1.sub(r"\1\3",name)
            else:newname=name
            new_file_names.append(newname)
        return new_file_names
    

    def sorted_file_types_with_number(self):
        a=dict(Counter(self.new_name_for_files()))
        return {list(a.keys())[i]:a[list(a.keys())[i]] for i in [x for x,y in sorted(enumerate(a.values()), key=lambda x : x[0])]}
    

    def geting_categories(self):
        
        final_sorted=self.sorted_file_types_with_number()
        pattern=re.compile(r"(.*)ECG(.*)",re.I)
        k=[m for m in final_sorted.keys() if pattern.search(m) ]
        self.cats = [pattern.sub(r"\1",m)[:-1] for m in k ]
        return self.cats
    
    def set_cat_value(self,i):
        self.geting_categories()
        self.i=i
        super().__init__(self)
        self.extracting_color_coding()
    
    def files_cat(self,exclusive=None):
        dic=self.sorted_file_types_with_number()
        cats=self.cats
        Ids=[[True if cats[j] in i else False for i in dic.keys()  ] for j in range(len(cats))]
        if not exclusive:
            return [list(dic.items())[i] for i,m in enumerate(Ids[self.i]) if m]
        exclusive=[]
        for i in range(len(Ids[0])):
            ind=True    
            for j in range(len(cats)):
                if Ids[j][i] == True:
                    ind=False
                
            exclusive.append(ind)
        return [list(dic.items())[i] for i,m in enumerate(exclusive) if m]

    def Car_file(self):
        with open(os.path.join(self.path,self.cats[self.i]+"_car.txt")) as file:
            content=[m.split() for m in file.readlines()]
            content=np.array(content[1:])
            data=pd.DataFrame(data=content,columns=[chr(ord("A")+h) if h+ord("A")<=ord("Z") else "A"+chr(h-1+2*ord("A")-ord("Z")) for h in range(content.shape[1])])
        return data

    def extracting_color_coding(self,triple=False):
        name=""
        for k in [m for m in [x for x,y in self.files_cat(True)] if (m.endswith(".xml") and re.search(r"(.*) (\d+-\d+-\d)",m))]:
            name+=k
        with open(os.path.join(self.path,name)) as file:
            a=file.readlines()
        indexes=[i for i,m in enumerate(a) if "tagstable" in m.lower()]
        data_xml=[i.strip(" ><\n/") for i in a[indexes[0]+1:indexes[1]]]
        pattern=re.compile(r"(.+\")( )(.+)")
        data=[]
        for i in data_xml:
            i_=i
            dat=[]
            while(True):
                if(pattern.search(i_)==None): 
                    dat.append(i_)
                    break
                dat.append(pattern.sub(r"\3",i_))
                i_=pattern.sub(r"\1",i_)
            
            data.append({i.split("=")[0]:i.split("=")[1].strip("\"") for i in dat})
        output=data
        
        #triple extra only
        if triple:
            output=[]
            for i in data:
                for j in i.keys():
                    if re.search("full_name",j,re.I):
                        if np.isin(i[j].lower(),["verde","green","orange","naranja","hsc+","hsc-","hsc","pos","positive","negative","neg","ver","nar"]):
                            output.append(i)
        
        color_coding=[{i[k]:i[j] for k in i.keys() if re.search("id",k,re.I) for j in i.keys() if re.search("full_name",j,re.I)  }for i in output]
        dict={}
        {dict.update(color_coding[i]) for i in range(len(color_coding))}
        self.color_dict=dict
        return output
                
    
    def car_extract(self,colors=None,triple=False):
        if colors and not triple:
            self.color_dict=colors
        elif triple:
            self.extracting_color_coding(triple)
        else:
            self.extracting_color_coding()
        self.color_dict
        data= self.Car_file().values
        labels=data[:,15]
        new_labels=[]
        for i,m in enumerate(labels) :
            if np.isin(m,list(self.color_dict.keys())): 
                new_labels.append([self.color_dict[m], data[i,2],data[i,27],data[i,4],data[i,5],data[i,6]])
            else: #print("not in defined colors") 
                pass
        if len(new_labels)==0:
            new_labels=[[None]*6]
        return pd.DataFrame(
            np.array(new_labels),
            columns=["label_color","point number","sample","x","y","z"]
            )
    
    def electrodes(self,triple=False):
        file_number,electrode,refference=[],[],[]
        car=self.car_extract(self.color_dict,triple)
        for i in car.loc[:,"point number"].values:
        
            with open(
                os.path.join(
                    self.path,
                    self.cats[self.i]+f"_P{i}"+"_Point_Export.xml" 
                    )
                ) as file:
                conts=file.readlines()
                pattern=re.compile(r"ECG FileName=\"([^\"]+)\"")
                pattern1=re.compile(r"UnipolarMappingChannel=\"([^\"]+)\" BipolarMappingChannel=\"([^\"]+)\" ReferenceChannel=\"([^\"]+)\"")

                for line_num,m in enumerate(conts):
                    if pattern.search(m) and pattern1.search(m):
                        file_number.append(pattern.search(m).group(1).strip())
                        electrode.append([pattern1.search(m).group(ii).strip() for ii in range(1,3)])
                        refference.append(pattern1.search(m).group(3))
            

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
    
    def Signals(self):
        data=self.electrodes()
        #check to delete Nan values from the data frame
        for i in data.columns:
            data=data.drop(data[data[i].isnull()].index)
        content=[]
        t=0
        for i in np.unique(data["file_number"].values):
            t=time.time()
            with open(os.path.join(self.path,i),"r") as file:
                content.append([
                data.loc[data["file_number"]==i,:].reset_index(drop=True),
                os.path.join(self.path,i),
                self.construct_dataframe_signals(file.readlines())
                ])
            print(time.time()-t)
        self.cont=content
        return content
    def construct_dataframe_signals(self,signals):
        gain=float(signals[1].split("=")[-1])
        l=[]
        for k in signals[3:]:
            row=k.strip().split()
            if len(row)!=0:
                l.append(row)
        data_np=np.vstack(l)
        data_np=data_np.astype("float32")
        data_np=data_np*gain
        index=list(np.linspace(0,2.5,len(data_np[:,0])))
        data_pd=pd.DataFrame(
            data_np,
            columns=[m.split("(")[0] for m in signals[2].split()],
            index=index
            )
        return data_pd
  
