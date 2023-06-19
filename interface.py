import tkinter as tk
window = tk.Tk()
frame = tk.Frame(window, width=400, height=400)
frame.pack()
label = tk.Label(frame, text="file to compress")
label.place(x=0,y=0)
entry = tk.Entry(frame, fg="yellow", bg="blue", width=20)
entry.place(x=0, y=20)
label = tk.Label(frame, text="username")
label.place(x=0,y=40)
entry = tk.Entry(frame, fg="yellow", bg="blue", width=20)
entry.place(x=0, y=60)

button = tk.Button(
    text="compress",
    width=25,
    height=5,
    bg="blue",
    fg="yellow",
    command=lambda x:x
)

window.mainloop()