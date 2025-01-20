
from Shades import Shades
import matplotlib.lines as lines

class Area(Shades):
    def __init__(self,app,ind) -> None:
        super().__init__(app)
        self.index=ind
        self.app=app

    def add_area(self,arg,ylim:tuple[2],xlim:tuple[2],color):
        lines=[]
        self.ylim=ylim
        self.xlim=xlim
        self.arg_sample=arg
        #changing index to time by multiplication with sampling time
        arg=arg*0.001
        self.x=arg
        lines.append(self.define_line(x0=self.x[0],x1=self.x[0],y0=ylim[0],y1=ylim[1]))
        lines.append(self.define_line(x0=self.x[1],x1=self.x[1],y0=ylim[0],y1=ylim[1]))
        self.add_shade(x=self.x,y=ylim,color=color)
        self.color=color
        self.lines=lines
    
    def plot_area(self,ax):

        self.ax=ax
        ax.add_line(self.lines[0])
        ax.add_line(self.lines[1])
        super().plot(ax)      

    def define_line(self,x0,x1,y0,y1):
       
        x = [x0,x1]
        y = [y0,y1]
        line=lines.Line2D(x, y, picker=3,linewidth=0.1,color="black",alpha=0.3)
        return line


    def configure_shade_attr(self,x,y,step_x=0.1,step_y=0.1,**kwargs):

        for key,value in kwargs.items():
            if key=="color":
                self.color_init_=value
       
        self.add_shade(x=x,y=y,config=True)


    def clickonline(self, event=None):
        if event.mouseevent.button != 1:
            return
        for iter,(line) in enumerate(self.lines):
            if event.artist == line :
                print("line selected ", event.artist)
                self.selected_line=line
                self.iter=iter
                self.follower = self.app.canvas.mpl_connect("motion_notify_event", self.followmouse)
                self.releaser = self.app.canvas.mpl_connect("button_release_event", self.releaseonclick)
            

    def followmouse(self, event=None):

        self.selected_line.set_xdata([event.xdata, event.xdata])
        self.x[self.iter]=event.xdata
        self.configure_shade_attr(x=self.x,y=self.ylim)
        self.app.canvas.draw_idle()

    def releaseonclick(self, event=None):

        self.app:App
        self.app.canvas.mpl_disconnect(self.releaser)
        self.app.canvas.mpl_disconnect(self.follower)
        self.arg_second=self.x
        self.arg_sample=self.arg_second*1000
        if self.color=="green":
            key="stim"
        else:
            key="sinus"
        update=[int(arg) for arg in self.arg_sample]
        if update[1]>=update[0]:
            self.app.delta[self.app.to_index[self.app.i][self.app.j]][2][key][self.index]=update
            self.app.update_plot()
        



