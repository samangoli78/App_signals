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
from .Triple_Extra import Triple_Extra
from .Signals import find_start,butter_bandpass_filter
from .Table import TableWidget
from .CARTO_Tool import Carto
from .Area import Area
from .Ax import ax
from .Toplevel import Toplevel as tp



class App(tk.Tk):
    #meta parameters
    apps=[]
    n_fft=[100]
    hop_length=[5]
    win_length=[35]
    high_b0=[40]
    high_b1=[200]
    low_b0=[3]
    low_b1=[150]
    len_hann=[5]
    max_pooling_length=[1]
    TH=[0.45]

    def __init__(self, name="Saman", carto:Carto=None):
        # make *new copies* so each instance has independent lists
        self.n_fft = App.n_fft.copy()
        self.hop_length = App.hop_length.copy()
        self.win_length = App.win_length.copy()
        self.high_b0 = App.high_b0.copy()
        self.high_b1 = App.high_b1.copy()
        self.low_b0 = App.low_b0.copy()
        self.low_b1 = App.low_b1.copy()
        self.len_hann = App.len_hann.copy()
        self.max_pooling_length = App.max_pooling_length.copy()
        self.TH = App.TH.copy()
        App.apps.append(self)


        self.forcefull=False
        self.is_running=True
        self.carto = carto
        self.cont = self.carto.cont
        self.triple_active=False
        self.name=name
        self.VT_active=False
        self.check_boxes={}
        self.i, self.j = 0, 0
        self.Table=[]
        #self.i_j_to_index(labels="hide")
        self.i_j_to_index(labels="unhide")
        self.creating_delta()
        all_columns = self.carto.cont[0][0].columns
        self.Table = pd.DataFrame(self.Table, columns=all_columns)

        # sanity check: lengths must match
        n = len(self.to_i_j)
        if len(self.Table) != n:
            raise ValueError(f"Row count mismatch: Table={len(self.Table)} vs index map={n}")
        # going to the first item of the list, which is arbitrary and could be any other numbers, 
        # and accesss the first element that is a dataframe containing information of the points

        self.Table = self.Table.assign(Comment="", delta=pd.Series(self.delta, dtype=object))

    def creating_delta(self):
        #delta is a list of dictionaries that contains the information about the stimulations and sinus signals.
        #preallocation of the delta list
        self.delta=[None]*len(self.to_i_j)
        
    def i_j_to_index(self,labels="unhide"):
        #"takes i , j gives index"
        self.to_index=[]
        #"takes index give i_j"
        self.to_i_j=[]
        self.labels_memory=[]
        carto=self.carto
        ind=0
        # structure of the cont: [section1,section2,section3,...]
        # structure of the section: [dataframe, name of the file, signals]
        for i,section in enumerate(carto.cont):
            df,_,_=section
            self.to_index.append([])
            self.labels_memory.append([])
            section: tuple[pd.DataFrame, str, pd.DataFrame]
            col = "label_color"
            col_idx = df.columns.get_loc(col)  # for fast iat use
            for j in range(len(df)):
                self.labels_memory[i].append(df.iat[j, col_idx])
                #hiding the labels
                if labels=="hide":
                    df.iat[j, col_idx] = ""
                # i,j to index
                self.to_index[i].append(ind)
                #indexs to i,j
                self.to_i_j.append([i,j])
                # updating the table matrix
                self.Table.append(df.iloc[j].values)
                # increaing the index after each nested itteration
                ind+=1
                
    def start(self):

        super().__init__()
        print("start")
        self.start_x_y = []
        self.direction=1   
        self.geometry("600x600")
        self.title(f"My {self.name} APP")

        self.frame=tk.Frame(self,padx=5,pady=10,background="grey")
        self.frame.pack(fill="x",expand=False)
        self.button_dropdown=tk.Button(self.frame,text="Options",command=self.drop_down)
        self.button_dropdown.pack(side=tk.LEFT,padx=10)
        df_temp:pd.DataFrame=self.cont[self.i][0]
        self.label = tk.Label(self.frame, text=f"point {df_temp.iat[self.j,df_temp.columns.get_loc('point number')]}",bg="grey",fg="white")
        self.label.config(font=("timesnewroman", 10))
        self.label.pack(fill="x",side="left",padx=10)
        
        self.check_boxes={"Energy":None}
        for key in self.check_boxes.keys():
            self.check_boxes[key]=tk.IntVar()
            check_box=tk.Checkbutton(self.frame,variable=self.check_boxes[key],text=key,font=("timesnewroman",10))
            check_box.pack(side=tk.LEFT,fill="x", expand=False)
        self.button_trip=tk.Button(self.frame,text="Switch to Triple Extra Protocol",command=self.triple_protocol)
        self.button_trip.pack(side=tk.LEFT,padx=10)
        self.button_VT=tk.Button(self.frame,text="Switch to VT Protocol",command=self.VT_protocol)
        self.button_VT.pack(side=tk.LEFT,padx=10)
        self.button_VT=tk.Button(self.frame,text="compute_all",command=self.compute_all)
        self.button_VT.pack(side=tk.LEFT,padx=10)
        self.button_screen=tk.Button(self.frame,text="screen shot",command=lambda name=None:self.capture_window(name))
        self.button_screen.pack(side=tk.LEFT,padx=10)
        self.panned_window=ttk.PanedWindow(self,orient="vertical")
        self.panned_window.pack(expand=True,fill=tk.BOTH)
        self.frame3=tk.Frame(self)
        self.panned_window.add(self.frame3,weight=1)
        #self.frame3.pack(expand=True,fill="both")
        self.table=TableWidget(self.frame3,self.Table)
        self.table.pack(fill="both", expand=True)           # IMPORTANT: it's a Frame now
        tv = self.table.tree                               # the EditableTree instance

        tv.defaults = {1: ["Reject", "POS", "NEG"]}         # column index for label_color in YOUR table
        self.frame1=tk.Frame(self,pady=5,background="white")
        self.panned_window.add(self.frame1,weight=2)
        self.set_figure()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame1)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.frame1.grid_rowconfigure((0,1), weight=1)
        self.frame1.grid_columnconfigure(0, weight=6)
        self.frame1.grid_columnconfigure(1, weight=1)
        self.main_control()
        self.plot()      

    def capture_window(self,name=None):
            root=self
            if name is None:
                name=f"screenshot{root.cont[root.i][0].reset_index(drop=True)['point number'][root.j]}.png"
            x = root.winfo_rootx()
            y = root.winfo_rooty()
            width = root.winfo_width()
            height = root.winfo_height()    #get details about window
            takescreenshot = ImageGrab.grab(bbox=(x, y, x+width, y+height))
            takescreenshot.save(name,dpi=500)

    def triple_protocol(self,event=None):
        self.triple_active=not(self.triple_active)
        if self.triple_active:
            self.button_trip.config(text="Turn off Triple Extra Protocol")
        else:
            self.button_trip.config(text="Switch to Triple Extra Protocol")
        self.update_plot()

    def VT_protocol(self,event=None):

        self.VT_active=not(self.VT_active)
        if self.VT_active:
            self.button_VT.config(text="Turn off VT Protocol")
        else:
            self.button_VT.config(text="Switch to VT Protocol")
        self.update_plot()
        
    def set_figure(self,x=2,y=1,mod=False):
        self.fig = plt.figure()
        self.axes = {}
        ax_top=self.fig.add_subplot(2, 1, 1)
        ax_top.set_xlim([0,2.5])
        ax_top.set_ylim([-3,3])
        ax_top.set_autoscale_on(False)   # master autoscale switch off
        ax_top.autoscale(False)
        ax_top.set_autoscalex_on(False)
        ax_top.set_autoscaley_on(False)

        ax_bot=self.fig.add_subplot(2, 1, 2, sharex=ax_top)
        ax_bot.set_ylim([-20,20])
        ax_bot.set_autoscale_on(False)   # master autoscale switch off
        ax_bot.autoscale(False)
        ax_bot.set_autoscalex_on(False)
        ax_bot.set_autoscaley_on(False)

        self.fig.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.05)
        self.axes["top"]=ax_top
        self.axes["bot"]=ax_bot

    def main_control(self):

        #self.bind("<Right>",self.p_increase)
        #self.bind("<Left>",self.p_decrease)
        #self.bind("<Up>",self.s_increase)
        #self.bind("<Down>",self.s_decrease)
        # Bind right mouse button press and release events
        def mpl_arrow(event):
            if event.key=="right":
                self.s_increase(event)
            elif event.key=="left":
                self.s_decrease(event)
            elif event.key=="up":
                self.p_decrease(event)
            elif event.key=="down":
                self.p_increase(event)

        self.canvas.mpl_connect("key_press_event",mpl_arrow) 
        self.canvas.mpl_connect("button_press_event", self.on_right_click) 
        
        tv = self.table.tree

        # row selection (mouse click / Up/Down changes selection)
        def _table_select_ctx(ctx):
            # ctx["row"] is the table row index
            if ctx["row"] is None:
                return
            i,j=self.to_i_j[ctx["row"]]
            print("selected row with select binding function",ctx["row"])
            self.i=i
            self.j=j
            self.update_plot()

        tv.on_select_row = _table_select_ctx


        def _table_move_ctx(ctx):
            # You want Up/Down to navigate points in your plot:
            if ctx["key"] == "Up":
                self.p_decrease()   # Up should go to previous (your original mapping was inverted)
            elif ctx["key"] == "Down":
                self.p_increase()

        # cell move (Up/Down/Left/Right/Enter triggers this hook)
        tv.on_cell_move = _table_move_ctx

        def _table_commit_ctx(ctx):
            # ctx has: row, col, value
            row = ctx.get("row")
            col = ctx.get("col")
            val = ctx.get("value")
            if row is None:
                return
            # --- A) Save committed value to delta (your requirement) ---
            i, j = self.to_i_j[row]
            idx = self.to_index[i][j]
            if col == 1:  # your label_color column rule
                if self.triple_active:
                    print(self.delta[idx])
                    self.delta[idx][1] = val
                df_carto = self.carto.cont[i][0]
                df_carto.iat[j, df_carto.columns.get_loc("label_color")] = val
            # --- B) After commit, the table may auto-go-next (dropdown does this).
            # Sync app to the table's FINAL current row (once).
            tree = self.table.tree
            def follow_table_final_row():
                cur_iid = tree.cur_iid or (tree.selection()[0] if tree.selection() else None)
                if not cur_iid:
                    return
                new_row = tree._row_index_from_iid(cur_iid)
                if new_row is None:
                    return
                self.i, self.j = self.to_i_j[new_row]
                self.update_plot()
            tree.after_idle(follow_table_final_row)

        # edit committed (Return, dropdown pick, focus-out if you set it to commit)
        tv.on_edit_commit = _table_commit_ctx
        
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        super().protocol("WM_DELETE_WINDOW", self.quit)
        # Create the figure and axis objects

    def quit(self):
        self.is_running = False
        super().quit()
        super().destroy() 

    def on_right_click(self, event):
        if event.button == 3:  # Right-click
            ax = event.inaxes
            if ax is None or event.xdata is None or event.ydata is None:
                return
            hold_ms = 250          # threshold: tweak
            move_px = 4         # drag threshold in pixels
            down=True
            hold=False
            start_x_y = (event.xdata, event.ydata)
            old_x_y = (event.xdata, event.ydata)
            cid_motion=None
            cid_release=None
            def on_motion(event):
                nonlocal hold, down, cid_motion, cid_release,old_x_y
                if not down:
                    return
                dx = abs(event.xdata - old_x_y[0])
                dy = abs(event.ydata - old_x_y[1])
                old_x_y = (event.xdata, event.ydata)
                if dx>move_px or dy>move_px:
                    hold=True
                else:
                    hold=False
                dist = (dx ** 2 + dy ** 2) ** 0.5
                if dist > move_px:
                    down = False
                return
        
            cid_motion  = self.canvas.mpl_connect("motion_notify_event", on_motion)
            cid_release = self.canvas.mpl_connect("button_release_event", on_release)

        start_x_y = None
        end_x_y = None
        rect = None
        selected_ax = None
        follower = None
        release_cid = None

        def on_right_follow_mouse(event):
            nonlocal end_x_y, rect, selected_ax, start_x_y

            if event.inaxes != selected_ax or event.xdata is None or event.ydata is None:
                return

            end_x_y = [event.xdata, event.ydata]

            if rect:
                rect.set_xy(start_x_y)
                rect.set_height(end_x_y[1] - start_x_y[1])
                rect.set_width(end_x_y[0] - start_x_y[0])
            else:
                rect = Rectangle(
                    start_x_y,
                    width=end_x_y[0] - start_x_y[0],
                    height=end_x_y[1] - start_x_y[1],
                    color="#00008525",
                )
                selected_ax.add_artist(rect)

            self.canvas.draw_idle()
            self.update()

        def on_right_release(event):
            nonlocal rect, follower, release_cid, selected_ax, start_x_y, end_x_y

            if event.button == 3:
                if rect:
                    rect.remove()
                    rect = None

                if follower is not None:
                    self.canvas.mpl_disconnect(follower)
                    follower = None

                if release_cid is not None:
                    self.canvas.mpl_disconnect(release_cid)
                    release_cid = None

                if event.inaxes == selected_ax and start_x_y is not None and end_x_y is not None:
                    xlim = [start_x_y[0], end_x_y[0]]
                    ylim = [start_x_y[1], end_x_y[1]]

                    duration = lambda x: x[1] - x[0]
                    if duration(xlim) < 0:
                        xlim = xlim[::-1]
                    if duration(ylim) < 0:
                        ylim = ylim[::-1]

                    if duration(xlim) < 0.01 and duration(ylim) < 0.01:
                        for key, ax in self.axes.items():
                            if selected_ax == ax and key == "bot":
                                xlim = [0, 2.5]
                                ylim = [-10, 10]
                            elif selected_ax == ax and key == "top":
                                xlim = [0, 2.5]
                                ylim = [-1, 1]

                    selected_ax.set_xlim(xlim)
                    selected_ax.set_ylim(ylim)

                    for key, ax in self.axes.items():
                        if selected_ax == ax and key == "top":
                            for a in self.Areas[key]:
                                try:
                                    a.configure_shade_attr(x=a.x, y=selected_ax.get_ylim())
                                    for lin in a.lines:
                                        lin.set_ydata(selected_ax.get_ylim())
                                except Exception:
                                    pass

                self.canvas.draw_idle()
                self.update()


            selected_ax = ax
            start_x_y = [event.xdata, event.ydata]
            end_x_y = start_x_y[:]  # copy

            follower = self.canvas.mpl_connect("motion_notify_event", on_right_follow_mouse)
            release_cid = self.canvas.mpl_connect("button_release_event", on_right_release)


            selected_key = None
            for key, a in self.axes.items():
                if a is selected_ax:
                    selected_key = key
                    break  
            options = ["stim", "sinus"]

            var = tk.StringVar(value=options[0])   # selected value
            parent=self.canvas.get_tk_widget()
            combo = ttk.Combobox(
                parent,
                textvariable=var,
                values=options,
                state="readonly"    # user must choose from list
            )
            x = event.guiEvent.x_root - parent.winfo_rootx()
            y = event.guiEvent.y_root - parent.winfo_rooty()
            combo.place(x=x, y=y)
            on_plot_x=event.xdata
            def on_selected(event):
                nonlocal selected_key,on_plot_x
                value = var.get()
                print("Selected:", value, selected_key)
                combo.destroy()
                start_x=on_plot_x
                end_x=on_plot_x+50/1000 # 50 ms window
                if selected_key is not None:
                    A = Area(
                        app=self,
                        index=len(self.Areas[selected_key]),          # <- same meaning as your old ind
                        key=value,       # <- where it lives / your label
                        kind=value,      # <- delta dict key to update on release
                        t0_s=start_x,
                        t1_s=end_x,
                        color="#7DAD76" if value=="stim" else "#A776AD",
                        fs_hz=1000.0
                    )
                    A.attach(selected_ax)           # <- creates artists + connects pick/follow/release + right-click menu on THIS ax canvas
                    self.Areas[selected_key].append(A)        # <- keep reference so you can update ylim later if needed
                    self.canvas.draw_idle()
                    selected_key=None
                else:
                    combo.destroy() 
                    self.canvas.draw_idle()

            combo.bind("<<ComboboxSelected>>", on_selected)

            
    def on_scroll(self, event):
        ax = event.inaxes
        if ax is None:
            return
        y0, y1 = ax.get_ylim()
        yc = 0.5 * (y0 + y1)
        half_range = 0.5 * (y1 - y0)

        zoom_factor = 0.1

        if event.button == 'up':        # zoom in
            scale = 1 - zoom_factor
        elif event.button == 'down':    # zoom out
            scale = 1 + zoom_factor
        else:
            return

        new_half = half_range * scale
        ax.set_ylim(yc - new_half, yc + new_half)
        selected_ax=ax
        for key, ax in self.axes.items():
            if ax is selected_ax and key == "top":
                for area in self.Areas[key]:
                    area.configure_shade_attr(x=area.t, y=ax.get_ylim())

                        
        ax.figure.canvas.draw_idle()
        self.update()

    def plot_main(self,ax,x,y2,ECG_reff=None,stim_reff=None,plot=True):
        c1={"stim":[]}
        c1["sinus"]=[]
        c1["refs_stim"]=[]
        c1["refs_sinus"]=[]
        c1["voltage_stim"]=[]
        c1["voltage_sinus"]=[]
        c1["deflection_stim"]=[]
        c1["deflection_sinus"]=[]
        addition=[["#FFf5F5","High frequency"],            
                    ["#FFc5c5","Low frequency"]]
        self.Areas["top"]=[]
        Areas=self.Areas["top"]
        if plot:
            ax.plot(x,y2,alpha=0.5,linewidth=0.6)

        car=self.cont[self.i][0]
        p_number=int(car.iat[self.j,car.columns.get_loc('point number')])
        label=car.iat[self.j,car.columns.get_loc("label_color")]
        memory=self.delta[self.to_index[self.i][self.j]] 
        try:
            try:
                p_number_mem=int(memory[0])
                label_mem=str(memory[1])
                dict=memory[2]
                if p_number==p_number_mem:
                    print("sanity check the pnumbers are aligned")
                else:
                    raise Exception("Pnumbers didnt coincide fall back to processing again")
                stims=dict.get("stim")
                sinus=dict.get("sinus")
                ii=0
                for s in stims:
                    if s is not None:
                        if plot:
                            
                            start_n,end_n=s
                            t0_s = start_n * 0.001
                            t1_s = end_n   * 0.001
                            xx=x[start_n:end_n]
                            yy1=y2[start_n:end_n]
                            A = Area(
                                app=self,
                                index=ii,          # <- same meaning as your old ind
                                key="stim",       # <- where it lives / your label
                                kind="stim",      # <- delta dict key to update on release
                                t0_s=t0_s,
                                t1_s=t1_s,
                                color="#7DAD76",
                                fs_hz=1000.0
                            )
                            ii+=1
                            A.attach(ax)           # <- creates artists + connects pick/follow/release + right-click menu on THIS ax canvas
                            Areas.append(A)        # <- keep reference so you can update ylim later if needed
                            ax.plot(xx,yy1 ,label=f"duration: {end_n-start_n} ms",linewidth=0.6)
                ii=0
                for s in sinus:
                    if s is not None:
                        if plot:
                            start_n,end_n=s
                            
                            t0_s = start_n * 0.001
                            t1_s = end_n   * 0.001
                            xx=x[start_n:end_n]
                            yy1=y2[start_n:end_n]
                            A = Area(
                                app=self,
                                index=ii,          # <- same meaning as your old ind
                                key="sinus",       # <- where it lives / your label
                                kind="sinus",      # <- delta dict key to update on release
                                t0_s=t0_s,
                                t1_s=t1_s,
                                color="#A776AD",
                                fs_hz=1000.0
                            )
                            ii+=1
                            A.attach(ax)           # <- creates artists + connects pick/follow/release + right-click menu on THIS ax canvas
                            Areas.append(A)        # <- keep reference so you can update ylim later if needed
                            ax.plot(xx,yy1 ,label=f"duration: {end_n-start_n} ms",linewidth=0.6)
                            

            except Exception as e:
                print(e)
                raise Exception("memory failed")

        except Exception as e:
            print(e)
            print("data not loaded proceed to analyze the data")

            egm=Triple_Extra(t=x,EGM=y2)
            try:
                egm.find_windows(self.axes["bot"],stiulation=stim_reff,refference=ECG_reff,margin=0)
            except Exception as e:
                raise e
            ii=0
            for i_temp,_ in enumerate(egm.stim_start):
                detected=False
                start=egm.stim_start[i_temp]+5
                end=egm.stim_start[i_temp] +egm.stim_duration[i_temp]-10
                xx=x[start:end]
                yy=y2[start:end]
                yy1=yy.copy()
                checkbox:tk.IntVar=self.check_boxes.get("Energy")
                if checkbox.get():
                    output=find_start(app=self,x_woi=xx,y_woi=yy,length=2,ax=ax,operation=None,Th=0.15,alpha=self.TH[0])
                else:
                    output=find_start(app=self,x_woi=xx,y_woi=yy,length=2,ax=None,operation=None,Th=0.15,alpha=self.TH[0])

                if output is not None:
                    start_n=int(start+output[0])
                    end_n=int(start+output[1])
                    detected=True
                else:
                    pass

                if detected:
                    
                    xx=x[start_n:end_n]
                    yy1=y2[start_n:end_n]
                    if plot:
                        t0_s = start_n * 0.001
                        t1_s = end_n   * 0.001

                        A = Area(
                            app=self,
                            index=ii,          # <- same meaning as your old ind
                            key="stim",       # <- where it lives / your label
                            kind="stim",      # <- delta dict key to update on release
                            t0_s=t0_s,
                            t1_s=t1_s,
                            color="#76AD85",
                            fs_hz=1000.0
                        )
                        ii+=1
                        A.attach(ax)           # <- creates artists + connects pick/follow/release + right-click menu on THIS ax canvas
                        Areas.append(A)        # <- keep reference so you can update ylim later if needed

                        ax.plot(xx,yy1 ,label=f"duration: {end_n-start_n} ms",linewidth=0.6)
                        defl=deflection(butter_bandpass_filter(y2,(2,250),order=2),ax,start_n,end_n)
                        defl+=deflection(-butter_bandpass_filter(y2,(2,250),order=2),ax,start_n,end_n,inverse=True)
                    else:
                        defl=deflection(butter_bandpass_filter(y2,(2,250),order=2),None,start_n,end_n)
                        defl+=deflection(-butter_bandpass_filter(y2,(2,250),order=2),None,start_n,end_n,inverse=True)  
                    c1["refs_stim"].append(egm.stim_ref[i_temp])
                    c1["stim"].append([start_n,end_n])
                    c1["voltage_stim"].append(yy1.max()-yy1.min())
                    c1["deflection_stim"].append(defl)
                else:
                    c1["refs_stim"].append(egm.stim_ref[i_temp])
                    c1["stim"].append(None)
                    c1["voltage_stim"].append(None)
                    c1["deflection_stim"].append(None)
                    

            ii=0    
            for i_temp,_ in enumerate(egm.sinus_start):
                detected=False
                start=egm.sinus_start[i_temp]
                end=egm.sinus_start[i_temp]+egm.sinus_duration[i_temp]
                xx=x[start:end]
                yy=y2[start:end]
                yy1=yy.copy()
                checkbox:tk.IntVar=self.check_boxes.get("Energy")
                if checkbox.get():
                    output=find_start(app=self,x_woi=xx,y_woi=yy,length=2,ax=ax,operation=None,Th=0.15,alpha=self.TH[0])
                else:
                    output=find_start(app=self,x_woi=xx,y_woi=yy,length=2,ax=None,operation=None,Th=0.15,alpha=self.TH[0])
                if output is not None:
                    start_n=int(start+output[0])
                    end_n=int(start+output[1])
                    detected=True
                else:
                    pass

                if detected:
                    
                    xx=x[start_n:end_n]
                    yy1=y2[start_n:end_n]
                    if plot:
                        # start_n/end_n are samples -> convert to seconds (fs=1000 like your code)
                        t0_s = start_n * 0.001
                        t1_s = end_n   * 0.001

                        A = Area(
                            app=self,
                            index=ii,          # <- same meaning as your old ind
                            key="sinus",       # <- where it lives / your label
                            kind="sinus",      # <- delta dict key to update on release
                            t0_s=t0_s,
                            t1_s=t1_s,
                            color="#A776AD",
                            fs_hz=1000.0
                        )
                        ii+=1
                        A.attach(ax)           # <- creates artists + connects pick/follow/release + right-click menu on THIS ax canvas
                        Areas.append(A)        # <- keep reference so you can update ylim later if needed
                        ax.plot(xx,yy1 ,label=f"duration: {end_n-start_n} ms",linewidth=0.6)
                        defl=deflection(butter_bandpass_filter(y2,(2,250),order=2),ax,start_n,end_n)
                        defl+=deflection(-butter_bandpass_filter(y2,(2,250),order=2),ax,start_n,end_n,inverse=True)
                    else:
                        defl=deflection(butter_bandpass_filter(y2,(2,250),order=2),None,start_n,end_n)
                        defl+=deflection(-butter_bandpass_filter(y2,(2,250),order=2),None,start_n,end_n,inverse=True)  
                    c1["refs_sinus"].append(egm.sinus_ref[i_temp])
                    c1["sinus"].append([start_n,end_n])
                    c1["voltage_sinus"].append(yy1.max()-yy1.min())
                    c1["deflection_sinus"].append(defl)
                else:
                    c1["refs_sinus"].append(egm.sinus_ref[i_temp])
                    c1["sinus"].append(None)
                    c1["voltage_sinus"].append(None)
                    c1["deflection_sinus"].append(None)
                    
            idx=self.to_index[self.i][self.j]
            self.delta[idx] = [str(p_number), label, c1]       
            tv = self.table.tree  # EditableTree
            row = self.to_index[self.i][self.j]
            iid = f"row{row}"

            vals = list(tv.item(iid)["values"])
            vals[-1] = ", ".join(
                f"{k}: {', '.join(map(str, v))}"
                for k, v in c1.items()
                if "voltage" not in k
            )
            tv.item(iid, values=vals)

        ax.set_title(label)

    def plot(self,plot=True):
        print(self.i,self.j)

        # getting key value of my ax class which is up mid as middle or bot as bottom. the purpose of this line of code is to have a new object for shading areas each time we update.
        self.Areas={key:{} for key in ["top","down"]}
        
        # load the data
        car:pd.DataFrame=self.cont[self.i][0]
        tag=self.cont[self.i][1]
        print("tag is",tag)
        data_pd=self.cont[self.i][2]
        x=data_pd.index
        y2=data_pd.loc[:,car.iat[self.j, car.columns.get_loc("bipolar")]].values
        V5=data_pd.loc[:,"V5"].values
        #V5=butter_bandpass_filter(data=V5,cutoff=[5,180],fs=1000,order=2)
        try:
            M=[data_pd.loc[:,ii].values for ii in ["M1","M2"]]
            M=M[1]-M[0]
            #M=M[1]
        except Exception as e:
            pass


        # check if the protocol is needed
        if self.triple_active:
            ax=self.axes["top"]
            self.plot_main(ax,x=x,y2=y2,ECG_reff=V5,stim_reff=M,plot=plot)


        # default mode for all kinds of signals.   
        else:
            
            ax=self.axes["top"]
            ax.plot(x,y2,alpha=0.5,linewidth=0.6)
            ax.set_title(car.iat[self.j,car.columns.get_loc("label_color")])

        if plot:
            # plotting refferences
            ax1=self.axes["bot"]
            Point_test=find_peaks(V5,0.5)
            ax1.plot(x,V5,label="V5",alpha=1)
            try:
                ax1.plot(x,M,label="M",alpha=1)
            except Exception as e:
                traceback.print_exc()
            ax1.grid(True)
            ax1.scatter(Point_test[0]*0.001,V5[Point_test[0]])          
            self.update()
        return True
        
    def update_plot(self,forcefull=False,plot=True):
        if forcefull:
            self.forcefull=True
        else:
            self.forcefull=False
        if plot:
            for ax in self.axes.values():
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()

                ax.cla()

                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_autoscale_on(False)
                ax.set_autoscalex_on(False)
                ax.set_autoscaley_on(False)

        print("current index",self.to_index[self.i][self.j])
        self.plot(plot=plot)

        car:pd.DataFrame  
        car=self.cont[self.i][0]
        self.label.config(text=f"point {car.iat[self.j,car.columns.get_loc('point number')]}")
        self.table.tree.see(f"row{self.to_index[self.i][self.j]}")
        self.table.tree.selection_set(f"row{self.to_index[self.i][self.j]}")
        self.table.tree.focus(f"row{self.to_index[self.i][self.j]}")
        self.table.tree.cur_iid = f"row{self.to_index[self.i][self.j]}"
        self.update()
        self.canvas.draw_idle()

    def compute_all(self):
        for i in range(len(self.to_i_j)):
            self.i,self.j=self.to_i_j[i]
            try:    
                self.update_plot(forcefull=True,plot=False)
                self.forcefull=False
            except Exception as e:
                print(e)
                continue

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


            self.Table["delta"]=pd.Series(in_table)
            self.Table["label_color"]=pd.Series(labels)

            self.table.tree.init_from_df(self.Table)

        def _toplevel():
            top_window=tk.Toplevel()
            top_window.attributes("-topmost", True)
            tp(top_window,self)
        menu.add_command(label="Save_Delta",command=save)
        menu.add_command(label="Load_Delta",command=load)
        menu.add_command(label="meta_parameters",command=_toplevel)
        menu.tk_popup(self.button_dropdown.winfo_rootx(),self.button_dropdown.winfo_rooty()+self.button_dropdown.winfo_height())
    
    def Energy(self,ax,x,y,legends=None):   
        #stft_=librosa.stft(y,n_fft=100,hop_length=1,win_length=35,window="hann",center=True)
        #Xdb_=np.abs(stft_)
        freqs,time,mags=librosa.reassigned_spectrogram(y,n_fft=int(self.win_length[0]),hop_length=int(self.hop_length[0]),win_length=int(self.win_length[0]),window="hann",center=True,sr=1000)
        #print(freqs,time)
        time=time.flatten()+x.min()
        freqs=freqs.flatten()
        from scipy.stats import binned_statistic_2d
        #print(x.min(),x.max())
        time_bins = np.linspace(x.min(), x.max(), int((x.max()-x.min())/0.001) + 1)

        freq_bins = np.linspace(0, 500, mags.shape[0])
        Xdb_=mags
        #print(time_bins,time)
        magnitude=np.abs(mags.flatten())
        H_count, _, _, _ = binned_statistic_2d(
            time, freqs, magnitude,
            statistic='count', bins=[time_bins, freq_bins]
        )
        mask=H_count == 0
        H_count[mask]=1
        H_sum, _, _, _ = binned_statistic_2d(
            time, freqs, magnitude,
            statistic='sum', bins=[time_bins, freq_bins]
        )
        H_sum=H_sum/H_count


        Xdb_=H_sum.T
        #print(Xdb_.shape,Xdb_)
        freq=Xdb_.shape[0]
        len_han=int(self.len_hann[0])
        y__=np.mean(Xdb_[freq//5:,:],0)
        
        window=np.ones(len_han)
        #y__=minmax_scale(y__, feature_range=(0, 1), axis=0, copy=True)
        y_high=np.convolve(y__,window,mode="same")/np.sum(window)
        x=np.linspace(x.min(),x.max(),len(y_high))

        #y__=np.sum(Xdb_[1:freq//2,:],0)/Xdb_.shape[0]
        y__=np.mean(Xdb_[int(freq*self.low_b0[0]/500):int(freq*self.low_b1[0]/500),:],0)
        #y__=minmax_scale(y__, feature_range=(0, 1), axis=0, copy=True)
                
        y_low=np.convolve(y__,window,mode="same")/np.sum(window)
        y__=np.mean(Xdb_[:,:],0)
        #y__=minmax_scale(y__, feature_range=(0, 1), axis=0, copy=True)
        y_total=np.convolve(y__,window,mode="same")/np.sum(window)
        for key,state in self.check_boxes.items():
            print(key,state)
            if (state.get() and key.lower() == "energy") :
                #first is freqs second is time
                
                if legends is None:
                    ax.plot(x,y_high,color="#FFf5F5")
                else:
                    ax.plot(x,y_high,color=legends[0][0])
                
                
                if legends is None:
                    ax.plot(x,y_total,color="#FFc5c5")
                else:
                    ax.plot(x,y_total,color=legends[1][0])
        x_E=x
        return x_E,y_low,y_high,y_total








def deflection(y,ax,start,end,inverse=False):
    signal=y[start:end]
    if len(signal)!=0:
        indexes=find_peaks(y[start:end],prominence=signal.max()*0.15,height=signal.max()*0.2,distance=5)
        if inverse:
            if ax:
                ax.scatter((indexes[0]+start)*0.001,-signal[indexes[0]],s=2,color="black")
        else:
            if ax:
                ax.scatter((indexes[0]+start)*0.001,signal[indexes[0]],s=2,color="black")
    return len(indexes[0])

    


