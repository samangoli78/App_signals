
# This class is not necessary as you already initialized the figure in the App class.
# Remove the Fig class entirely if it's not needed.

class ax:
    Axes=[]
    def __init__(self,ax,keyword=None,fig=None):
        self.Axes.append(self)
        self.ax=ax
        self.ylim=[]
        self.xlim=[]
        self.key=keyword
        self.figure=fig
    def update(self,**kwargs):
        if kwargs["ylim"]:
            self.ylim=kwargs.get('ylim')
        else:
            self.ylim=self.ax.get_ylim()
        if kwargs.get("xlim"):
            self.xlim=kwargs.get("xlim")
        else:
            self.xlim=self.ax.get_xlim()
    def set(self):
        self.ax.set_ylim(self.ylim)
        self.ax.set_xlim(self.xlim)
    def delete(self):
        for i,obj in enumerate(self.Axes):
            if obj == self:
                del self.Axes[i]
