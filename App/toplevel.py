import tkinter as tk
from App import App

class Toplevel:
    created=False
    def __init__(self,root:tk.Toplevel,app:App=None):
        self.root=root
        if Toplevel.created:
            self.prot()
            return
        self.app=app
        self.root.protocol("WM_DELETE_WINDOW", self.prot)
        self.root.geometry("400x300")

        Title=tk.Label(self.root,text="STFT")
        Title.grid(column=0,row=0,sticky=tk.NS,columnspan=3)

        self.create_generic(self.app.win_length,name="Window length",row=1,from__=2,to__=200,length__=100)
        

        label_H=tk.Label(self.root,text="Hop length")
        label_H.grid(column=0,row=2,sticky=tk.NS)
        self.slider_H=tk.Scale(self.root,from_      = 0,       # MVC-Model-Part value-min-limit
                to         =  self.app.win_length[0],       # MVC-Model-Part value-max-limit
                length     = 100,         # MVC-Visual-Part layout geometry [px]
                resolution =   1,     # MVC-Controller-Part stepping
                tickinterval=self.app.win_length[0]//4,
                orient=tk.HORIZONTAL)
        self.slider_H.grid(row=2,column=1,sticky=tk.EW,ipady=10,)
        self.entry_H=tk.Entry(self.root)
        def slider_decorator(func,widget:tk.Scale,var:list,*args,**kwargs):
            func(var=var,widget=widget,*args,**kwargs)
            widget.config(from_=0,to=int(2*self.app.win_length[0]),tickinterval=int(2*self.app.win_length[0])//4)


        self.slider_H.config(command=lambda val,var=self.app.hop_length,widget=self.slider_H,widget_entry=self.entry_H:
                             slider_decorator(self.Slider_function,var=var,val=val,widget=widget,widget_entry=widget_entry))
        self.entry_H.grid(column=2,row=2,sticky=tk.W)
        entry_H_content=[""]
        self.entry_H.insert(0,str(self.app.hop_length[0]))
        self.entry_H.bind("<Return>",lambda event,var=self.app.hop_length,prev=entry_H_content,widget=self.entry_H,widget_slider=self.slider_H:
                          self.Entry_check(event,var,prev,widget=widget,widget_slider=widget_slider))



        self.create_generic(self.app.high_b0,name="lower threshold of high frequency band pass filter",row=3,from__=10,to__=500)
        self.create_generic(self.app.high_b1,name="higher threshold of high frequency band pass filter",row=4,from__=10,to__=500)

        self.create_generic(self.app.low_b0,name="lower threshold of low frequency band pass filter",row=5,from__=1,to__=200)
        self.create_generic(self.app.low_b1,name="higher threshold of low frequency band pass filter",row=6,from__=1,to__=200)

        self.create_generic(self.app.len_hann,name="hanning window length applied on Energy",row=7,from__=1,to__=10)
        self.create_generic(self.app.TH,name="Threshold",row=8,from__=0,to__=1,resolution__=0.01,digit=3)

        self.root.grid_columnconfigure((0,1,2),weight=1)
        self.root.grid_rowconfigure((1,2,3,4,5,6,7,8),weight=4)
        self.root.grid_rowconfigure(0,weight=1)
        Toplevel.created=True

        
    def Entry_check(self,event,var:list,prev:list,widget:tk.Entry,widget_slider:tk.Scale):
        entery=widget.get()
        if prev[0]!=entery:
            try:
                var[0]=float(entery)
                widget_slider.set(int(entery))
                self.app.update_plot(forcefull=True)
            except Exception as e:
                print("the format incorect",e)
            prev[0]=entery
    def Slider_function(self,var:list,val,widget:tk.Scale,widget_entry:tk.Entry):
        print(val)
        try:
            var[0]=float(val)
            widget_entry.delete(0,tk.END)
            widget_entry.insert(0,str(val))
            self.app.after(10,lambda :self.app.update_plot(forcefull=True))
        except Exception as e:
            print("smth went wrong",e)

    def create_generic(self,var,name="Window length",row=1,from__=2,to__=200,length__=100,resolution__=1,digit=None):
        label_w=tk.Label(self.root,text=name)
        label_w.grid(row=row,column=0)
        self.slider_w=tk.Scale(self.root,from_      = from__,       # MVC-Model-Part value-min-limit
                to         =  to__,       # MVC-Model-Part value-max-limit
                length     = length__,         # MVC-Visual-Part layout geometry [px]
                resolution =   resolution__,     # MVC-Controller-Part stepping
                tickinterval=(to__-from__)/4,
                orient=tk.HORIZONTAL)
        if digit:
            self.slider_w.config(digits=digit)
        self.slider_w.grid(row=row,column=1,sticky=tk.EW,ipady=10)
        self.slider_w.set(var[0])
        self.entry_w=tk.Entry(self.root)
        self.slider_w.config(command=lambda val,var=var,widget=self.slider_w,widget_entry=self.entry_w:
                             self.Slider_function(var=var,val=val,widget=widget,widget_entry=widget_entry))
        self.entry_w.grid(column=2,row=row,sticky=tk.W)
        entry_w_content=[""]
        self.entry_w.insert(0,str(var[0]))
        self.entry_w.bind("<Return>",lambda event,var=var,prev=entry_w_content,widget=self.entry_w,widget_slider=self.slider_w:
                          self.Entry_check(event,var,prev,widget=widget,widget_slider=widget_slider))


    def prot(self):
        Toplevel.created=False
        self.root.destroy()




if __name__=="__main__":
    master=tk.Tk()
    t=Toplevel(tk.Toplevel(master))
    master.mainloop()

    
