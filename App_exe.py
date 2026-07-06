#%%

import sys
print(sys.prefix)

from App.carto import Carto
#carto=Carto(path='F:/Export_Study-1-05_02_2024-10-58-39')
carto=Carto()
carto.extracting_color_coding(triple=False)

carto.Signals()


from App.main_app import App
app = App(carto=carto)
app.start(name="Test")
app.mainloop() 
# %%
