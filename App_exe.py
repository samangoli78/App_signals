import sys
print(sys.prefix)

from CARTO_Tool import Carto
#carto=Carto(path='F:/Export_Study-1-05_02_2024-10-58-39')
carto=Carto("F:/data_Carto/PERUADE/New_Case_3Extra_01")
carto.extracting_color_coding(triple=False)

carto.Signals()
from App import App
app = App(name="Test", carto=carto)  
app.start()
app.mainloop() 