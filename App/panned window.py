import tkinter as tk
from tkinter import ttk


# Create the main window
root = tk.Tk()
root.title("Resizable Complex Window")
root.geometry("600x400")

# Create a PanedWindow (horizontal layout)
paned_window = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
paned_window.pack(fill=tk.BOTH, expand=True)

# Add frames to the PanedWindow
left_frame = tk.Frame(paned_window, bg="lightblue", width=200, height=400)
right_frame = tk.Frame(paned_window, bg="lightgreen", width=400, height=400)
def but_press():
    m=tk.Menu(right_frame,tearoff=False)
    m.add_command(label="save",command=save)
    m.add_command(label="exit",command=m.destroy)
    m.tk_popup(button.winfo_rootx(),button.winfo_rooty()+button.winfo_height())
def save():
    print("save")
button=tk.Button(right_frame,text="Menu",command=but_press)
button.pack()


paned_window.add(left_frame, weight=1)  # The left frame is resizable
paned_window.add(right_frame, weight=3)  # The right frame is larger and resizable

# Add content to the left frame
left_label = tk.Label(left_frame, text="Left Frame", bg="lightblue")
left_label.pack(pady=10)

# Add content to the right frame
right_label = tk.Label(right_frame, text="Right Frame", bg="lightgreen")
right_label.pack(pady=10)

# Run the application
root.mainloop()
