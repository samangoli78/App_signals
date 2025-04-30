import tkinter as tk


class checkbox_combo_dropdown:
    def __init__(self, master, title, options):
        self.master = master
        self.title = title
        self.options = options
        mb=  tk.Button ( master, text="CheckComboBox",relief=tk.SUNKEN, borderwidth=1,
                        command=lambda root=master,title=title,options=options:self.create_dropdown(root,title,options))	 
        mb.pack()
        menu  =  tk.Menu ( root)
        tearoff_window = menu.winfo_toplevel()

        for opt in options:
            variable=tk.IntVar()
            menu.add_checkbutton ( label=opt,variable=variable , command=lambda x=opt,y=variable:self.callback(x,y))
        self.menu=menu
        
    def create_dropdown(self,root, title, options):
        
       self.menu.tk_popup(root.winfo_rootx(),root.winfo_rooty()+root.winfo_height())
    def callback(self,label,value):
        print(f"{label}={value.get()}")


if __name__ == "__main__":
    root = tk.Tk()
    options = ["one", "two", "three"]
    ccd = checkbox_combo_dropdown(root, "title", options)
    root.mainloop()