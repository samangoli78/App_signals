# public libs
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as lines
from matplotlib.patches import Rectangle
from scipy.signal import find_peaks
from PIL import ImageGrab
import pandas as pd
import librosa
from tkinter import filedialog
import json
import traceback


# private libs
from Triple_Extra import Triple_Extra
from Signals import find_start,butter_bandpass_filter
from Table import Table
from CARTO_Tool import Carto
from Area import Area
from Ax import ax



class Params:
    def __init__(self) -> None:
        pass


class App(tk.Tk):
    apps=[]
    def __init__(self, name="Saman", carto:Carto=None):
        self.apps.append(self)
        self.is_running=True
        self.carto = carto
        self.cont = self.carto.cont
        self.params=Params()
        self.params.name=name
        self.triple_active=False
        self.check_boxes={}
        self.i, self.j = 0, 0
        self.Table=[]
        self.i_j_to_index()
        self.creating_delta()
        # going to the first item of the list, which is arbitrary and could be any other numbers, 
        # and accesss the first element that is a dataframe containing information of the points
        all_columns=self.carto.cont[0][0].columns
        self.Table=pd.DataFrame(self.Table,columns=all_columns)
        self.Table=pd.concat([self.Table,pd.DataFrame(np.zeros(len(self.to_i_j)),columns=["delta"])],axis=1)

    def creating_delta(self):
        #delta is a list of dictionaries that contains the information about the stimulations and sinus signals.
        #preallocation of the delta list
        self.delta=[0]*len(self.to_i_j)
        
    def i_j_to_index(self):
        self.to_index=[]
        self.to_i_j=[]
        self.labels_memory=[]
        carto=self.carto
        ind=0
        # structure of the cont: [section1,section2,section3,...]
        # structure of the section: [dataframe, name of the file, signals]
        for i,section in enumerate(carto.cont):
            self.to_index.append([])
            self.labels_memory.append([])
            section: tuple[pd.DataFrame, str, pd.DataFrame]
            for j,dat in enumerate(section[0].values):
                self.labels_memory[i].append(carto.cont[i][0].loc[j,"label_color"])
                #hiding the labels
                carto.cont[i][0].loc[j,"label_color"]=""
                # i,j to index
                self.to_index[i].append(ind)
                #indexs to i,j
                self.to_i_j.append([i,j])
                # updating the table matrix
                self.Table.append(dat)
                # increaing the index after each nested itteration
                ind+=1
                


    def start(self):

        super().__init__()
        print("start")
        self.start_x_y = []
        self.direction=1   
        self.geometry("600x600")
        self.title(f"My {self.params.name} APP")

        self.frame=tk.Frame(self,padx=5,pady=10,background="grey")
        self.frame.pack(fill="x",expand=False)
        self.button_dropdown=tk.Button(self.frame,text="Options",command=self.drop_down)
        self.button_dropdown.pack(side=tk.LEFT,padx=10)
        self.label = tk.Label(self.frame, text=f"point {self.cont[self.i][0]['point number'].values[self.j]}",bg="grey",fg="white")
        self.label.config(font=("timesnewroman", 10))
        self.label.pack(fill="x",side="left",padx=10)
        
        self.check_boxes={"Energy":None,"Only_Green":None}
        for key in self.check_boxes.keys():
            self.check_boxes[key]=tk.IntVar()
            check_box=tk.Checkbutton(self.frame,variable=self.check_boxes[key],command=self.checker,text=key,font=("timesnewroman",10))
            check_box.pack(side=tk.LEFT,fill="x", expand=False)
        self.button=tk.Button(self.frame,text="is triple extra",command=self.triple_protocol)
        self.button.pack(side=tk.LEFT,padx=10)
        self.button_screen=tk.Button(self.frame,text="screen shot",command=lambda name=None:self.capture_window(name))
        self.button_screen.pack(side=tk.LEFT,padx=10)
        self.panned_window=ttk.PanedWindow(self,orient="vertical")
        self.panned_window.pack(expand=True,fill=tk.BOTH)
        self.frame3=tk.Frame(self)
        self.panned_window.add(self.frame3,weight=1)
        #self.frame3.pack(expand=True,fill="both")
        self.table=Table(self.frame3,self.Table)
        self.set_figure()
        self.main()

    def capture_window(self,name=None):
            root=self
            if name is None:
                name=f"screenshot{root.cont[root.i][0].reset_index(drop=True)['point number'][root.j]}.png"
            x = root.winfo_rootx()
            y = root.winfo_rooty()
            width = root.winfo_width()
            height = root.winfo_height()    #get details about window
            takescreenshot = ImageGrab.grab(bbox=(x, y, x+width, y+height))
            takescreenshot.save(name,dpi=(1920,1080))

    def triple_protocol(self,event=None):

        self.triple_active=not(self.triple_active)
        if self.triple_active:
            self.button.config(text="not triple extra")
        else:
            self.button.config(text="is triple extra")
        self.update_plot()
        
    def set_figure(self,x=2,y=1):
        
        self.fig, self.axes = plt.subplots(x,y)
        plt.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.05)
        self.axes={m:ax(self.axes[i],m,self.fig) for i,m in enumerate(["top","bot"])}
        self.fig.clf()
        self.axes["top"]=ax(self.fig.add_subplot(2,1,1),"top",self.fig)
        self.axes["bot"]=ax(self.fig.add_subplot(2,1,2),"bot",self.fig)
        [x.update(xlim=[0,2.5],ylim=[-1,1]) for x in self.axes.values()]
        self.frame1=tk.Frame(self,pady=5,background="white")
        self.panned_window.add(self.frame1,weight=2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame1)
        self.canvas.get_tk_widget().grid(column=0,row=0,rowspan=2,sticky=tk.NSEW)

        self.ccs={}
        for r,i in enumerate(self.axes.keys()):
            self.cc=tk.Canvas(self.frame1,width=110,height=130,bg="white",highlightbackground="white")
            self.cc.grid(column=1,row=r,sticky="")
            self.ccs[i]=self.cc

        self.frame1.grid_rowconfigure((0,1), weight=1)
        self.frame1.grid_columnconfigure(0, weight=6)
        self.frame1.grid_columnconfigure(1, weight=1)
        self.plot()

    def main(self):

        #self.bind("<Right>",self.p_increase)
        #self.bind("<Left>",self.p_decrease)
        #self.bind("<Up>",self.s_increase)
        #self.bind("<Down>",self.s_decrease)
        # Bind right mouse button press and release events
        def mpl_arrow(event):
            if event.key=="right":
                self.p_increase(event)
            elif event.key=="left":
                self.p_decrease(event)
            elif event.key=="up":
                self.s_increase(event)
            elif event.key=="down":
                self.s_decrease(event)

        self.canvas.mpl_connect("key_press_event",mpl_arrow) 
        self.canvas.mpl_connect("button_press_event", self.on_right_click) 
        self.canvas.mpl_connect("button_release_event", self.on_right_release)
        self.table.table.bind("<Button-1>",self.select)
        self.table.table.bind("<Return>",self.on_enter)
        
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        super().protocol("WM_DELETE_WINDOW", self.quit)
        # Create the figure and axis objects


    def quit(self):
        self.is_running = False
        super().quit()
        super().destroy() 


    def checker(self):
        self.update_plot()


    def on_right_click(self, event):
        
        if event.button == 3:  # Right-click
            for ax in self.axes.values():
                if ax.ax==event.inaxes:
                    self.selected_ax=ax
                    self.start_x_y = [event.xdata,event.ydata]  # Store the x-coordinate
                    self.end_x_y = self.start_x_y
                    self.follower=self.canvas.mpl_connect("motion_notify_event", self.on_right_follow_mouse)
                    self.rect=None


    def on_right_follow_mouse(self,event):
      
        self.end_x_y=[event.xdata,event.ydata]
        if self.rect:
            self.rect.set_xy(self.start_x_y)
            self.rect.set_height(self.end_x_y[1]-self.start_x_y[1])
            self.rect.set_width(self.end_x_y[0]-self.start_x_y[0])
        else:
            rect=Rectangle(self.start_x_y,width=self.end_x_y[0]-self.start_x_y[0],height=self.end_x_y[1]-self.start_x_y[1],color="#00008525")
            self.selected_ax.ax.add_artist(rect)
            self.rect=rect
        self.canvas.draw_idle()
        self.update()
    

    def on_right_release(self, event):
        if event.button == 3:
            if self.rect:
                    self.rect.remove()
            self.canvas.mpl_disconnect(self.follower)
            if event.inaxes==self.selected_ax.ax:

                xlim=[self.start_x_y[0],self.end_x_y[0]]
                ylim=[self.start_x_y[1],self.end_x_y[1]]
                duration=lambda x:x[1]-x[0]
                if duration(xlim)< 0:
                    xlim=xlim[-1::-1]
                if duration(ylim)< 0:
                    ylim=ylim[-1::-1]
                if duration(xlim) <0.01 and duration(ylim) <0.01:
                    if self.selected_ax.key == "bot":
                        xlim=[0,2.5]
                        ylim=[-10,10]
                    else:
                        xlim=[0,2.5]
                        ylim=[-1,1]
            
                self.selected_ax.update(xlim=xlim,ylim=ylim)
                self.selected_ax.set()
                for a in self.Areas[self.selected_ax.key]:
                    a.configure_shade_attr(x=a.x,y=self.selected_ax.ylim)
                    for lin in a.lines:
                        lin.set_ydata(self.selected_ax.ylim)
            self.canvas.draw_idle()
            self.update()
 
    def on_scroll(self, event):
        for ax in self.axes.values():
            if ax.ax == event.inaxes:
                ylim = ax.ylim
                zoom_factor = 0.1  # Define how fast you want to zoom

                if event.button == 'up':  # Zoom in
                    scale_factor = 1 - zoom_factor
                elif event.button == 'down':  # Zoom out
                    scale_factor = 1 + zoom_factor
                else:
                    return

                # Update limits for both axes
                ax.update(ylim=[ylim[0] * scale_factor, ylim[1] * scale_factor])
                ax.set()
                if self.Areas[ax.key]: 
                    for a in self.Areas[ax.key]:
                        a.configure_shade_attr(x=a.x,y=ax.ylim)
                        for lin in a.lines:
                            lin.set_ydata(ax.ylim)
                    
        self.canvas.draw_idle()
        self.update()

    def plot_main(self,ax,x,y2,arg="top",reff=[]):
        c1={"stim":[]}
        c1["sinus"]=[]
        c1["refs_stim"]=[]
        c1["refs_sinus"]=[]
        c1["voltage_stim"]=[]
        c1["voltage_sinus"]=[]
        addition=[["#FFf5F5","High frequency"],            
                    ["#FFc5c5","Low frequency"]]
        ax_object=ax
        ax=ax.ax
        ax.plot(x,y2,alpha=0.5,linewidth=0.6)
        egm=Triple_Extra(t=x,EGM=y2)
        if len(reff)>0:
            egm.find_windows(self.axes["bot"].ax,stiulation=reff,refference=butter_bandpass_filter(data=self.cont[self.i][2]["V5"].values,cutoff=[5,180],fs=1000,order=2),margin=0)
        else:
            egm.find_windows(self.axes["bot"].ax,stiulation=self.cont[self.i][2]["CS1"].values,refference=self.cont[self.i][2]["V5"].values,margin=20)
            reff=self.cont[self.i][2]["CS1"].values
    
        for ii in range(len(egm.stim_start)):
            A=Area(self,ind=ii)
            start=egm.stim_start[ii]
            end=egm.stim_start[ii]+egm.stim_duration[ii]
            xx=x[start+8:end-10]
            yy=y2[start+8:end-10]
            yy1=yy
            x_Energy,y_low,y_high,y_total=self.Energy(ax,xx,yy,legends=addition)
            if self.delta[self.to_index[self.i][self.j]]==0:
                output=find_start(x_Energy,y_low,length=2,ax=ax,operation=None,Th=0.15)
                if output is not None:
                    start_n=int(start+8+output[0])
                    end_n=int(start+8+output[1])
            else:
                start_n,end_n=self.delta[self.to_index[self.i][self.j]][2]["stim"][ii]
            xx=x[start_n:end_n]
            yy=y2[start_n:end_n]
            if np.abs(yy1).max()>0.05:
                
                A.add_area(np.array([start_n,end_n]),ylim=ax_object.ylim,xlim=ax_object.xlim,color="green")
                A.plot_area(ax)
                self.Areas[arg].append(A)
                self.canvas.mpl_connect('pick_event', A.clickonline)
                ax.plot(xx,yy ,label=f"duration: {end_n-start_n} ms",linewidth=0.6)
            c1["refs_stim"].append(egm.stim_ref[ii])
            c1["stim"].append([start_n,end_n])
            c1["voltage_stim"].append(np.abs(yy).max())
            
        for ii in range(len(egm.sinus_start)):
            start=egm.sinus_start[ii]
            end=egm.sinus_start[ii]+egm.sinus_duration[ii]
            if egm.stim_start[0]<start<egm.stim_start[-1]+egm.stim_duration[-1]:
                start=egm.stim_start[-1]+egm.stim_duration[-1]
                if end-start<100:
                    continue
            A=Area(self,ind=ii)
            
            xx=x[start:end]
            yy=y2[start:end]
            yy1=yy
            x_Energy,y_low,y_high,y_total=self.Energy(ax,xx,yy,legends=addition)
            if self.delta[self.to_index[self.i][self.j]] == 0:
                output=find_start(x_Energy,y_low,length=2,ax=ax,operation=None,Th=0.15)
                if output is not None:
                    start_n=int(start+output[0])
                    end_n=int(start+output[1])
            else:
                start_n,end_n=self.delta[self.to_index[self.i][self.j]][2]["sinus"][ii]

            xx=x[start_n:end_n]
            yy=y2[start_n:end_n]
            if np.abs(yy1).max()>0.05:
                # argument value that is the duration to be shaded must be in numpy array format
                
                A.add_area(np.array([start_n,end_n]),ylim=ax_object.ylim,xlim=ax_object.xlim,color="#A776AD")
                A.plot_area(ax)
                self.Areas[arg].append(A)
                self.canvas.mpl_connect('pick_event', A.clickonline)
                
                ax.plot(xx,yy ,label=f"duration: {end_n-start_n} ms",linewidth=0.6)
            try:
                c1["refs_sinus"].append(egm.sinus_ref[ii])
            except Exception as e:
                traceback.print_exc()
            c1["sinus"].append([start_n,end_n])
            c1["voltage_sinus"].append(np.abs(yy).max())
        if self.delta[self.to_index[self.i][self.j]] == 0 or "refs_sinus" not in self.delta[self.to_index[self.i][self.j]][2].keys():
            self.delta[self.to_index[self.i][self.j]]=[self.cont[self.i][0]["point number"].values[self.j],self.cont[self.i][0]["label_color"].values[self.j],c1]
            
        else:
            print("already managed")
            
        self.table.table.add_data(", ".join([f"{key}: {', '.join([str(ii) for ii in value])}" for key,value in c1.items() if "voltage" not in key]),
                                               self.to_index[self.i][self.j],-1)
        
                

        #if len(y2[arg[0]:arg[1]])!=0:
        #    indexes=find_peaks(y2[self.arg[arg][0]:self.arg[arg][1]],prominence=y2[self.arg[arg][0]:self.arg[arg][1]].max()*0.15,height=y2[self.arg[arg][0]:self.arg[arg][1]].max()*0.2,distance=5)
        #    ax.scatter((indexes[0]+self.arg[arg][0])*0.001,y2[indexes[0]+self.arg[arg][0]],s=2,color="black")
    
        ax.set_title(self.cont[self.i][0]["label_color"].values[self.j])
        self.create_legend(leg=ax.get_legend_handles_labels(),canvas=self.ccs[ax_object.key],addition=addition)
        #ax.legend(fontsize=8)
        ax_object.set()
        

    def plot(self):
        print(self.i,self.j)
        #manual legend ? cleanup; cleaning all canvas that were created in init and then stored in self.ccs as canvases.
        [cc.delete("all") for cc in self.ccs.values()]

        # getting key value of my ax class which is up mid as middle or bot as bottom. the purpose of this line of code is to have a new object for shading areas each time we update.
        self.Areas={i.key:[] for i in self.axes.values()}
        
        # load the data
        data_pd=self.cont[self.i][2]
        x=data_pd.index
        car=self.cont[self.i][0]
        y2=data_pd.loc[:,car.loc[self.j,"bipolar"]].values
        V5=data_pd.loc[:,"V5"].values
        V5=butter_bandpass_filter(data=V5,cutoff=[5,180],fs=1000,order=2)
        try:
            M=[data_pd.loc[:,ii].values for ii in ["M4","M3"]]
            M=M[1]-M[0]
        except Exception as e:
            traceback.print_exc()
        try:
            M=data_pd.loc[:,"CS1"].values 
        except Exception as e:
            traceback.print_exc()

        

        # this loop is for when the Only green check box is activated, it will checks the labels untill one of the labels in the list will show up. otherwise it will continue searching. 
        if self.check_boxes["Only_Green"].get() and not np.isin(self.cont[self.i][0].values[self.j][0].upper(),["VERDE","VER","GREEN","POSITIVE","POS"]) :
            if self.direction>=0:
                self.p_increase()
                return
            else:
                self.p_decrease()
                return

        # make another pointer to the axes dictionary for easing of access
        axes=self.axes

        
        

        # check if the protocol is needed
        if self.triple_active:
            ax=axes["top"]
            self.plot_main(ax,x=x,y2=y2,arg=ax.key,reff=M)
            #ax=axes["mid"]
            #self.plot_main(ax,x=x,y2=y2,arg=ax.key,reff=M)

        # default mode for all kinds of signals.   
        else:
            
            ax=axes["top"]
            ax.ax.plot(x,y2,alpha=0.5,linewidth=0.6)
            ax.ax.set_title(car.loc[self.j,"label_color"])
            ax.set()
            
        # plotting refferences
        ax1_object=axes["bot"]
        ax1=ax1_object.ax

        
        Point_test=find_peaks(V5,0.5)
        ax1.plot(x,V5,label="V5",alpha=1)
        #ax1.plot(x,y2,label="y2",alpha=0.7)
        try:
            ax1.plot(x,M,label="M",alpha=1)
        except Exception as e:
            traceback.print_exc()
        ax1.grid(True)
        ax1.scatter(Point_test[0]*0.001,V5[Point_test[0]])

        #Point_test=find_peaks(M,2)
        #ax1.scatter(Point_test[0]*0.001,M[Point_test[0]])
        ax1_object.set()
        
        self.create_legend(leg=ax1.get_legend_handles_labels(),canvas=self.ccs[ax1_object.key])
        #plt.savefig(r"C:\Users\cardio\Desktop\Output case 1 26_7_2024\out_{i}_{x}_{j}.jpeg".format(j=j_,x=cont[i_][1][:-4],i=i_))
        self.canvas.draw_idle()
        self.update()
        
            
     
    
    def select(self,event):
        i,j=self.to_i_j[event[0]]
        self.i=i
        self.j=j
    
        self.update_plot()
    def on_enter(self,event):

        
        row,col,val=event
        print(event)
        self.i,self.j=self.to_i_j[row]
        self.update_plot()
        if col == 1:
            self.delta[self.to_index[self.i][self.j]][1]=val
            self.carto.cont[self.i][0].loc[self.j,'label_color'] = val
        try:
            i,j=self.i,self.j=self.to_i_j[row+1]
        except Exception as e:
            print(e, "reached end of the table")
            traceback.print_exc()
            i,j=0,0
        self.table.table.go_to(self.to_index[i][j])
        self.table.table.refill(row=self.to_index[i][j])
        self.i=i
        self.j=j
    
        self.update_plot()

    def update_plot(self):
        for i in self.axes.values():
            i.ax.clear()
        self.plot()
        self.label.config(text=f"point {self.cont[self.i][0].reset_index(drop=True)['point number'][self.j]}")
        self.canvas.draw()
        print(self.to_index[self.i][self.j])
        self.table.table.go_to(self.to_index[self.i][self.j])
        self.update()

    def p_increase(self,event=None):
        self.direction=1
        if self.j >= len(self.cont[self.i][0])-1:
            
            self.s_increase(None)
            self.j=0
            self.update_plot()
        else:
            self.j += 1
            self.update_plot()

    def p_decrease(self,event=None):
        self.direction=-1
        if self.j <= 0:
            self.s_decrease(None)
            self.j=len(self.cont[self.i][0])-1
            self.update_plot()
        else:
            self.j -= 1            
            self.update_plot()

    def s_increase(self,event=None):
        self.direction=1
        
        if self.i >= len(self.cont)-1:
            self.i=0
            
        else:
            self.i += 1
            

        if event:
            self.j=0
            self.update_plot()


    def s_decrease(self,event=None):
        self.direction=-1
        
        if self.i <= 0:
            self.i=len(self.cont)-1
        
        else:
            self.i -= 1
            
        if event:
            self.j=0
            self.update_plot()
    def drop_down(self):
        menu=tk.Menu(master=self.frame1,tearoff=False)
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
        def save():
            file=filedialog.asksaveasfilename(confirmoverwrite=True,defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
            if file is not None:
                with open(file,"w") as f:
                    json.dump(self.delta,f,default=convert)
                print(f"File is saved in {file}")
        def load():
            file_path = filedialog.askopenfilename(
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
            if file_path:
                with open(file_path, "r") as f:
                    data = json.load(f)
                print(f"Data loaded from {file_path}")
                self.delta=data
            in_table=[]
            labels=[]
            for index,delt in enumerate(self.delta):
                try:
                    i,j=self.to_i_j[index]
                    print(delt[0],delt[1],self.carto.cont[i][0].loc[j,'point number'])
                    if self.carto.cont[i][0].loc[j,'point number']==delt[0]:    
                        self.carto.cont[i][0].loc[j,'label_color']=delt[1]
                        labels.append(delt[1])
                    else:
                        print(f"mismatch{self.carto.cont[i][0].loc[j,'point number']} {delt[0]}")
                        labels.append("mismatch")
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    labels.append("")
                try:
                    in_table.append([", ".join([f"{key}: {', '.join([str(ii) for ii in value])}" for key,value in delt[2].items() if "voltage" not in key])])
                except Exception as e:
                    traceback.print_exc()
                    in_table.append("")
            print(in_table)
            print(labels)
            self.Table["delta"]=pd.Series(in_table)
            self.Table["label_color"]=pd.Series(labels)

            del self.table
            self.table=Table(self.frame3,self.Table)
            self.table.table.bind("<Button-1>",self.select)
            self.table.table.bind("<Return>",self.on_enter)
        menu.add_command(label="Save_Delta",command=save)
        menu.add_command(label="Load_Delta",command=load)
        menu.tk_popup(self.button_dropdown.winfo_rootx(),self.button_dropdown.winfo_rooty()+self.button_dropdown.winfo_height())
    def create_legend(self,leg,canvas,addition=None):
        if isinstance(addition,type(None)):
            leg=np.vstack([[ii.get_color() for ii in leg[0]],leg[1]]).T
            
        else:
            leg=np.vstack([[ii.get_color() for ii in leg[0]],leg[1]]).T
            leg=np.concat([leg,np.vstack(addition)],0)
            #print(leg)
        for xx,m in enumerate(leg):
            
            canvas.create_text(50,20+20*xx,text=m[1])
            canvas.create_line(100,20+20*xx,150,20+20*xx,fill=m[0])
            canvas.create_line(0,10+20*xx,150,10+20*xx,fill="black")
        canvas.create_line(0,10+20*len(leg),150,10+20*len(leg),fill="black")
    def Energy(self,ax,x,y,legends=None):   
        stft_=librosa.stft(y,n_fft=100,hop_length=5,win_length=35,window="hann",center=True)
        Xdb_=np.abs(stft_)
        

        freq=Xdb_.shape[0]
        y__=np.sum(Xdb_[freq//5:,:],0)/Xdb_.shape[0]
        #y__=minmax_scale(y__, feature_range=(0, 1), axis=0, copy=True)
        y_high=np.convolve(y__,np.hanning(5))/np.sum(np.hanning(5))
        x=np.linspace(x.min(),x.max(),len(y_high))

        y__=np.sum(Xdb_[1:freq//2,:],0)/Xdb_.shape[0]
        #y__=minmax_scale(y__, feature_range=(0, 1), axis=0, copy=True)
                
        y_low=np.convolve(y__,np.hanning(5))/np.sum(np.hanning(5))
        y__=np.sum(Xdb_[:,:],0)/Xdb_.shape[0]
        #y__=minmax_scale(y__, feature_range=(0, 1), axis=0, copy=True)
        y_total=np.convolve(y__,np.hanning(5))/np.sum(np.hanning(5))
        for key,state in self.check_boxes.items():
            if (state and key.lower() == "energy") :
                #first is freqs second is time
                
                if legends is None:
                    ax.plot(x,y_high,color="#FFf5F5")
                else:
                    ax.plot(x,y_high,color=legends[0][0])
                
                
                if legends is None:
                    ax.plot(x,y_low,color="#FFc5c5")
                else:
                    ax.plot(x,y_low,color=legends[1][0])

        return [x,y_low,y_high,y_total]


 


    


