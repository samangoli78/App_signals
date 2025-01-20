

from matplotlib.patches import Rectangle

class Shades:
    def __init__(self,app) -> None:
        self.app=app
        self.rectangle=0

    def add_shade(self,x=(0,1),y=(-1,1),config=False,**kwargs):
        self.ylim=y
        for key,value in kwargs.items():
            if key=="color":
                self.color_init=value
        if (x[1]-x[0])< 0:
            self.color="red"
            if config:
                self.rectangle.set_xy([x[1],y[0]])
                self.rectangle.set_height(y[1]-y[0])
                self.rectangle.set_width(x[0]-x[1])
                self.rectangle.set(color=self.color)
            else:
                rectangle=Rectangle([x[1],y[0]],width=x[0]-x[1],height=y[1]-y[0],color=self.color,alpha=0.2)
                self.rectangle=rectangle
        else:
            self.color=self.color_init
            if config:
                self.rectangle.set_xy([x[0],y[0]])
                self.rectangle.set_height(y[1]-y[0])
                self.rectangle.set_width(x[1]-x[0])
                self.rectangle.set(color=self.color)
            else:
                rectangle=Rectangle([x[0],y[0]],width=x[1]-x[0],height=y[1]-y[0],color=self.color,alpha=0.2)
                self.rectangle=rectangle


    def plot(self,ax):
        ax.add_artist(self.rectangle)
    