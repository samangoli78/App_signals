#%%
import os
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import time
import xml.etree.ElementTree as ET
from lxml import etree
from pathlib import Path
import sys

from sympy import flatten

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from parser_tool import MapSection, Parser_carto
else:
    from .parser_tool import MapSection, Parser_carto


def log(input: str, *args) -> str:
    if not hasattr(log, "initialized"):
        log.initialized = True
        mode = "w"
    else:
        mode = "a"
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "log.txt"), mode, encoding="utf-8") as f:
        out = ""
        for inp in [*args, input]:
            if isinstance(inp, str):
                f.write(inp)
                out += inp
            elif isinstance(inp, pd.DataFrame):
                f.write(inp.to_string(index=False))
                out += inp.to_string()
            else:
                f.write(str(inp))
                out += str(inp)
        f.write("\n")
    return out


class Carto(Parser_carto):
    cartos = []

    def __init__(self, path=None, i=None) -> None:
        Carto.cartos.append(self)
        self.i = i
        self.cont = None
        Parser_carto.__init__(self, self)
        if path and os.path.exists(path):
            self.path = path
            self.on_init()
        else:
            print(log("could not initialize \nchoose your file from the dialogue"))
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            file_path = filedialog.askdirectory(parent=root)
            log("path is " + file_path)
            root.destroy()
            self.path = file_path
            self.on_init()

    def on_init(self):
        main_xmls = [name for name in os.listdir(self.path) if name.endswith(".xml") and "Export" not in name]
        map_data = []
        for xml in main_xmls:
            try:
                self.xml_path=os.path.join(self.path, xml)
                tree = ET.parse(self.xml_path)
                self.tree = tree.getroot()
                map_data = [{key: attr for key, attr in map_.items()} for map_ in self.tree.find("Maps").findall("Map")]
                if len(map_data) != 0:
                    break
            except Exception:
                pass
        try:
            self.registration_matrix=np.array(self.tree.find("Meshes").find("RegistrationMatrix").text.split(),dtype=np.float32).reshape(4,4)
        except Exception:
            print(Exception)
            self.registration_matrix=np.eye(4)
        if len(map_data) == 0:
            print("couldnt encrypt the data the map_data length was 0")
            print(log("couldnt encrypt the data the map_data length was 0"))
            return False
        self.maps = [i["Name"] for i in map_data]
        if self.i is None:
            self.root = tk.Tk()
            self.root.title("Select map")

            container = tk.Frame(self.root)
            container.pack(fill=tk.BOTH, expand=True)

            canvas = tk.Canvas(container, highlightthickness=0)
            scrollbar = tk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
            scrollable = tk.Frame(canvas)

            def _update_scroll_region(_event=None):
                canvas.configure(scrollregion=canvas.bbox("all"))

            scrollable.bind("<Configure>", _update_scroll_region)
            canvas_window = canvas.create_window((0, 0), window=scrollable, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            def _on_canvas_configure(event):
                canvas.itemconfigure(canvas_window, width=event.width)

            canvas.bind("<Configure>", _on_canvas_configure)

            def _on_mousewheel(event):
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

            self._mousewheel_handler = _on_mousewheel
            self.root.bind_all("<MouseWheel>", _on_mousewheel)

            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            self.frames = []
            self.labels = []
            for i in range(len(self.maps)):
                self.frames.append(tk.Frame(scrollable))
                self.frames[i].pack(fill=tk.X)
                self.labels.append(tk.StringVar())
                self.labels[i] = self.maps[i]
                button = tk.Button(
                    self.frames[i],
                    text=self.labels[i],
                    command=lambda a=i: self.index(a),
                    font=("timesnewroman", 10),
                )
                button.pack(expand=True, fill=tk.X)

            self.root.update_idletasks()
            max_height = 480
            content_height = scrollable.winfo_reqheight()
            width = max(scrollable.winfo_reqwidth() + scrollbar.winfo_reqwidth() + 25, 280)
            height = min(content_height + 10, max_height)
            self.root.geometry(f"{width}x{height}")

            self.root.attributes("-topmost", True)
            self.root.after(150, lambda: self.root.attributes("-topmost", False))
            self.root.protocol("WM_DELETE_WINDOW", self.quit)
            self.root.mainloop()

    def quit(self):
        try:
            self.root.unbind_all("<MouseWheel>")
        except Exception:
            pass
        self.root.quit()
        self.root.destroy()

    def index(self, i):
        self.i = i
        self.quit()

    def Car_file(self):
        self.car_path=os.path.join(self.path, self.maps[self.i] + "_car.txt")
        with open(self.car_path) as file:
            self.version=file.readline().split()[0].split("_")[1]
            content = [m.split() for m in file.readlines()]
            content = np.array(content)
            data = pd.DataFrame(
                data=content,
                columns=[chr(ord("A") + h) if h + ord("A") <= ord("Z") else "A" + chr(h - 1 + 2 * ord("A") - ord("Z")) for h in range(content.shape[1])],
            )
        return data

    def extracting_color_coding(self, triple=False):
        self.color_dict = {dict(ch.items())["ID"]: dict(ch.items())["Full_Name"] for ch in self.tree.find("Maps").find("TagsTable")}
        new_color_dict = {}
        if triple:
            for i, j in self.color_dict.items():
                if np.isin(j.lower(), ["verde", "green", "orange", "naranja", "hsc+", "hsc-", "hsc", "pos", "positive", "negative", "neg", "ver", "nar"]):
                    new_color_dict.update({i: j})
            self.color_dict = new_color_dict
        return self.color_dict

    def car_extract(self, triple=False):
        self.extracting_color_coding(triple)
        data = self.Car_file().values
        labels = data[:, 15]
        new_labels = []
        for i, m in enumerate(labels):
            if np.isin(m, list(self.color_dict.keys())):
                new_labels.append([self.color_dict[m], data[i, 2], data[i, 4], data[i, 5], data[i, 6]])
        if len(new_labels) == 0:
            return None
        return pd.DataFrame(np.array(new_labels), columns=["label_color", "point number", "x", "y", "z"])

    def electrodes(self, triple=False):
        file_number, electrode, refference = [], [], []
        car = self.car_extract(triple)
        if car is not None:
            for i in car.loc[:, "point number"].values:
                parser = etree.XMLParser(resolve_entities=False, remove_blank_text=True)
                root = etree.parse(os.path.join(self.path, self.maps[self.i] + f"_P{i}" + "_Point_Export.xml"), parser).getroot()
                ecg = root.find(".//ECG")
                file_number.append(ecg.get("FileName"))
                electrode.append([ecg.get("UnipolarMappingChannel"), ecg.get("BipolarMappingChannel")])
                refference.append(ecg.get("ReferenceChannel"))
            c2 = pd.Series(file_number, name="file_number")
            c3 = pd.DataFrame(np.array(electrode).reshape(-1, 2), columns=["unipolar", "bipolar"])
            c4 = pd.Series(refference, name="refference_chanel")
            return pd.concat([car[["label_color", "point number"]], c2, c3, car[["x", "y", "z"]], c4], axis=1)
        return None

    

    def Signals(self, triple=False):
        data = self.electrodes(triple)
        content = []
        total_duration_s = 2.5
        # Multiple electrode points often share the same exported ECG file
        # (one underlying recording, several mapping channels). Iterating
        # ``data.iterrows()`` and re-selecting all matching rows per row would
        # emit one ``MapSection`` per point AND each section would also
        # contain every other point that shares the file — multiplying the
        # table row count (N points sharing a file → N² table entries).
        # Iterate unique file names so each recording becomes a single
        # section whose ``points_df`` lists every point that uses it once.
        #seen_files: set[str] = set()
        for fname in np.unique(data["file_number"].values):
            
            t = time.time()
            with open(os.path.join(self.path, fname), "r", encoding="utf-8") as f:
                _ = f.readline()
                gain_line = f.readline().strip()
                if self.version == "6":
                    header_line = f.readline().strip()
                else:
                    port_data = f.readline().split()
                    vals = [port_data[2].split("=")[-1], port_data[5].split("=")[-1], port_data[7].split("=")[-1]]
                    mask = data["file_number"] == fname
                    data.loc[mask, "unipolar"] = vals[0]
                    data.loc[mask, "bipolar"] = vals[1]
                    data.loc[mask, "refference_chanel"] = vals[2]
                    header_line = f.readline().strip()
            gain = float(gain_line.split("=")[-1])
            cols = [m.split("(")[0] for m in header_line.split()]
            skiprows = 3 if self.version == "6" else 4
            df = pd.read_csv(
                os.path.join(self.path, fname),
                skiprows=skiprows,
                sep=r"\s+",
                names=cols,
                dtype=np.float32,
                engine="c",
                skip_blank_lines=True,
            )
            df *= gain
            df.index = np.linspace(0, total_duration_s, len(df), endpoint=False)
            content.append(
                MapSection(
                    points_df=data.loc[data["file_number"] == fname, :].reset_index(drop=True),
                    file_name=fname,
                    signals_df=df,
                )
            )
            print(f"Time taken for {fname}: {round(time.time() - t, 2)} seconds")
        self.cont = content
        return content

    def create_output(self, content: list):
        base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        out_root = base_dir / "output"
        out_root.mkdir(exist_ok=True)
        for section in content:
            meta_df = section.points_df
            file_number = section.file_name
            signals_df = section.signals_df
            sub = out_root / file_number[:-4]
            sub.mkdir(exist_ok=True)
            meta_path = sub / f"{file_number}_meta.csv"
            sig_path = sub / f"{file_number}_sig.csv"
            meta_df.to_csv(meta_path, sep="\t", index=False, float_format="%.3f")
            signals_df.to_csv(sig_path, sep="\t", index=False, float_format="%.3f")
#%%

if __name__ == "__main__":
    #%%    
    carto=Carto('C:/Users/saman/Downloads/Export_FA-3-EXT-BIATRI-06_17_2026-10-44-34')


    # %%
    carto_old=Carto('C:/Users/saman/Downloads/SCUB_P007/P_007/ExportData11_06_25 14_39_24/Patient 2025_05_13/PERSUADE/Export_PERSUADE-06_11_2025-14-35-41')
    carto_old.Car_file()

    # %%
    carto.Signals()
    # %%
    carto_old.Signals()
    # %%
    carto_old.registration_matrix

    # %%
    carto.registration_matrix
    # %%