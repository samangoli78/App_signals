import sys
print(sys.prefix)

from App.CARTO_Tool import Carto
#carto=Carto(path='F:/Export_Study-1-05_02_2024-10-58-39')
carto=Carto()
carto.extracting_color_coding(triple=False)

carto.Signals()
from App.App import App
app = App(name="Test", carto=carto)  
app.start()
app.mainloop() 