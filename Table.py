import os
from CARTO_Tool import Carto
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import pandas as pd


class TreeView_Edit(ttk.Treeview):

    

    def __init__(self,master,**kwargs):
        super().__init__(master,**kwargs)
        self.params={}
        self.cmds={}
        self.default=None
        master:tk.Tk
        self.params["Enter_func"]=lambda x:print(x)
        self.params["select_func"]=lambda x:print(x)
        self.params["refill_func"]=lambda x:print(x)
        def_func=lambda event:print(event)
        for f in ["Focus_out_func","Focus_in_func","Enter_func","right_click_func","left_click_func"]:
            if f not in self.params.keys():
                self.params.update({f:def_func})
        super().bind("<Button-3>",self.pop)
        super().bind("<Button-1>",self.on_select)
        super().bind("<FocusOut>",self.params["Focus_out_func"])
        super().bind("<Double-Button-1>",self.double_click)
        super().bind("<FocusIn>",self.params["Focus_in_func"])
        self.temp=None
        def cmd(event):
            if self.temp is None:
                return
            txt=f"{event},child:,{self.get_children()},focus_displayof:{self.focus_displayof()},focus_force:{self.focus_force()}"
            txt+=f"identify_row:{self.identify_row(event.y)},identify_column:{self.identify_column(event.x)},identify_region:{self.identify_region(event.x,event.y)},identify_element:{self.identify_element(event.x,event.y)}"
            try:
                self.lab.configure(text=txt)
            except:
                self.lab=tk.Label(self.temp,text=txt,wraplength=self.temp.winfo_width())
                self.lab.pack()
        self.cmds["motion"]=cmd

        super().bind("<Motion>",self.cmds["motion"])
        #super().bind("<Double-Button-1>",self.double_click)
        self.master=master
         
    def double_click(self,event):
        self.params["refill_func"]([self.get_children().index(self.identify_row(event.y))])
        self.refill(event)
        """if self.identify_region(event.x,event.y)=="separator":
            [self.column(column=col,width=40) for col in self["columns"]]"""
    def pop(self,event):
        menu=tk.Menu(self.master,tearoff=False)
        menu.add_command( label="Change",command=lambda event=event:self.refill(event))
        menu.add_command(label="Save",command=self.save)
        def cmd():
            self.temp=tk.Toplevel()
            self.temp.geometry("200x200")
        self.cmds["top_level"]=cmd
        menu.add_command(label="info",command=self.cmds["top_level"])
        menu.add_separator()
        menu.add_command(label="Exit",command=menu.destroy)
        menu.tk_popup(event.x_root,event.y_root)
      

    def on_enter(self,event,col:str,ID:str,func):

        # get the data inside the entry
        if isinstance(event, tk.Event):
            wiget=event.widget
            
        else:
            wiget=event

        new_text=wiget.get()
        # using ID to access the current value in the selected row
        
        selected_values=self.item(ID).get("values")
        
        # using col to access the desired cell in the selected row
        
        selected_values[col]=new_text
        
        # you can assign new value to the table with table.item method identifying the ID and value, optionally if you want to change the tag you need to specify "changed color" place holder first.
        
        self.item(ID,values=selected_values,tags='changed_tag')
        
        # then you can assign the desired color to the " changed color" place holder that you just assigned.
        
        self.tag_configure("changed_tag",background="#A5A5A5")
        
        # dont forget to destroy the entry widget.
        
        wiget.destroy()
        func([self.get_children().index(ID),col,new_text])
    
    def add_data(self,data,row,col):
        ID=self.get_children()[row]
        data_T=self.item(ID).get("values")
        data_T[col]=data
        self.item(ID,values=data_T,tags="new")
        
    def set_default(self,dif:list[list]):

        self.default=dif
    


    def bind(self,identifier:str,func):
        
        # change the binding function of "select" event to a custom function, designed for better customizing the module for external usrsers
        if identifier=="<Button-1>":
            self.params["select_func"]=func
        elif identifier=="<Return>":
            self.params["Enter_func"]=func
        elif identifier=="<Double-Button-1>":
            self.params["refill_func"]=func
    def on_select(self,event):
        
        # on calling "select" the data of the selected cell will be generated and returned
        
        out=self.find(event)
        self.params["select_func"]([self.get_children().index(self.identify_row(event.y))])
        return out
        
    

    def find(self,event):
        
        # to see if it is in cell or heading or text
        
        selected_region=self.identify_region(event.x,event.y)
        
        # finding column in string = "#$$"
        
        selected_column=self.identify_column(event.x)
        
        # finding the ID where the mouse is on_row
        
        selected_ID=self.identify_row(event.y)
        
        # to check if they are all valid
        
        if selected_column and selected_ID and selected_region=="cell":
        
            #geting the value of a row based on its ID
        
            data=self.item(selected_ID).get("values")
        
            #except the first char "#" others identify the column and it starts from 1
        
            col=int(selected_column[1:])-1
        
            #the first element indicates the index as we identified a certain column for indexes at the begining on the input table
        
            row=data[0]
        
            return row,col,data 
        
        else:
        
            return None


    def refill(self,event=None,row=None):
        
        if event is not None:
            selected_ID=self.identify_row(event.y)
            print(selected_ID)
            out=self.find(event)
            print(out,self.get_children())
        elif row is not None:
            try:
                
                selected_ID=f"row{row}"
                out=[row,self.col,self.item(selected_ID).get("values")]
            except Exception as e:
                print(e)
                return
        else:
            return
        

        

        if out is not None:
            # select
      
            self.selection_set(selected_ID)
      
            # focus
      
            self.focus(selected_ID)
            # accessing only column and data outputs of the self.find() method

            col,data=out[1:]
            self.col=col
            # getting the coordinate of the cell in the frame 

            coord=self.bbox(selected_ID,self["columns"][col])

            entry_edit=tk.Entry(self.master)

            # putting an entry widget right on the coordinate with the same size
            print(col)
            entry_edit.place(x=coord[0],y=coord[1],width=coord[2],height=coord[3])
            # selecting all in entry widget

            entry_edit.select_range(0,tk.END)

            # insert the same cell data to the entry widget

            entry_edit.insert(0,data[col])

            # focus on the entry widget 

            entry_edit.focus_force()
            menu=tk.Menu(self.master,tearoff=False)

            def cmd(item):
                entry_edit.delete(0,tk.END)
                entry_edit.insert(0,item)
                self.on_enter(entry_edit,col,selected_ID,func=self.params["Enter_func"])

            if self.default:
                for r in self.default:
                    if col==r[0]:
                        for item in r[1:]:
                            menu.add_command(label=item,command=lambda item=item:cmd(item))
                        tree_x = self.winfo_rootx()
                        tree_y = self.winfo_rooty()
                        x = tree_x + coord[0]
                        y = tree_y + coord[1]+coord[3]
                        menu.tk_popup(x=x,y=y)
                        menu.grab_release()  # Release the grab so the entry can still receive input.
                        entry_edit.focus_force()


            # bind <focus out> to be able to destroy entry widget while the mouse is not on the entry anymore

            entry_edit.bind("<FocusOut>",lambda event: event.widget.destroy())

            # bind enter, to be able to save entry data to the selected cell and then destroy the entry widget

            entry_edit.bind("<Return>",lambda event,selected_column=col,selected_ID=selected_ID,
                            :self.on_enter(event,selected_column,selected_ID,func=self.params["Enter_func"]))
            
    def go_to(self,row):
      
        # when you have the row, you can easilly find the ID by geting chidren method and choosing the right index
      
        ID=self.get_children()[row]
      
        # see
      
        self.see(ID)
      
        # select
      
        self.selection_set(ID)
      
        # focus
      
        self.focus(ID)

    def extract_data(self):
        row_ids=self.get_children()
        out=[self["columns"]]
        for id in row_ids:
            out.append(self.item(id).get("values"))
        return out
    
    def save(self):
        file=filedialog.asksaveasfilename(confirmoverwrite=True,filetypes=(
        ("Text files", "*.txt"),
        ("excel", ("*.csv", "*.xml")),
        ("All Files", "*.*")))
        out=self.extract_data()
        if file is not None:
  
            format=file.split("/")[-1].split(".")[-1]
  
            if format=="txt":
                with open(file,"w") as file:
                    file.writelines([";".join([str(item) for item in line]+["\n"]) for line in out])
            elif format=="csv":
                pd.DataFrame(out[1:],columns=out[0]).to_csv(file,index=False)
            else:
                pd.DataFrame(out[1:],columns=out[0]).to_csv(file,index=False)
                print("format not indicated and the table was saved as csv file")
            

            
    





class Table:
    def __init__(self,master,table:pd.DataFrame):
   
        # gettimg a root for the table it could be a tk.Tk or tk.Frame
   
        self.master=master
        self.master: tk.Tk

   
        # make a frame for the treeviewedit object
   
        self.tree_frame = tk.Frame(self.master, width=500, height=200)
   
        # making the desired scroll bars vertically and horizontally
   
        self.h_scroll = ttk.Scrollbar(self.master, orient="horizontal")
        self.v_scroll = ttk.Scrollbar(self.master, orient="vertical")
        
        # listing the columns of the input table
        
        cols=table.columns.tolist()
        
        # adding an index column
        
        cols.insert(0,"index")
        self.table=TreeView_Edit(self.tree_frame,columns=cols,selectmode=tk.BROWSE,
                                show="headings",xscrollcommand=self.h_scroll.set,
                                yscrollcommand=self.v_scroll.set)
        
        # binding scroll bars with their commands
        
        self.h_scroll.config(command=self.table.xview)
        self.v_scroll.config(command=self.table.yview)
        
        # assign heading to the columns
        
        [self.table.heading(val,text=val) for i,val in enumerate(cols)]
        
        # insert values of the table in addition to the first column that involve indexes
        
        [self.table.insert(parent="",index=tk.END,iid=f"row{i}",values=np.insert(value,0,i).tolist()) for i,value in enumerate(table.values)]
        
        # configuring columns format 
        
        [self.table.column(column=val,stretch=tk.YES,minwidth=30,width=60,anchor="center") for i,val in enumerate(cols)]
        
        # ?
        
        self.master.grid_rowconfigure(1, weight=1)
        self.master.grid_columnconfigure(0, weight=1)
        
        # packing table in tree frame
        
        self.table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # placing tree frame
        
        self.tree_frame.grid(row=1,column=0,sticky=tk.NSEW)
        self.tree_frame.grid_propagate(False)
        
        #placing scroll bars
        
        self.v_scroll.grid(row=1,column=1,sticky=tk.NS)
        self.h_scroll.grid(row=2,column=0,sticky=tk.EW )
        #tk.Button(self.master,command=self.table.save).grid(column=0,row=3)




if __name__=="__main__":
    #carto=Carto(r"F:/New_Case_3Extra")
    #print(carto.path)
    table:pd.DataFrame = None
    #table=carto.car_extract()
    #print(table)
    root=tk.Tk()
    table=pd.DataFrame(np.zeros((4,4)),columns=[i for i in range(4)])
    table=Table(root,table)
    table.table.set_default([[2,"kdfjlvhsa","lsdfvhdls","slsdfhsdlhj","zfhsalh"]])
    #print(table.h_scroll)
    table.table.go_to(2 )
    root.mainloop()
