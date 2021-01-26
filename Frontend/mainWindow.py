import tkinter.filedialog
import tkinter as tk
from tkinter import *
from Frontend.Frames import *
from Backend.main import run_simulation



OL_OPTIONS = [
    'K-NN',
    'NN',
    'SVM',
    'Naive-Bayes'
]

OFS_OPTIONS = [
    'Alpha Investing'
]

class Window:


    def __init__(self,root):
        self.root=root
        self.start_frame=tk.Frame(root)
        self.ol_param_frame=OL_Param(root)


        self.set_frame_position()

        self.create_param_textbox()
        self.create_file_chooser()
        self.create_selected_algo()

        self.run_btn=tk.Button(self.start_frame, text="Run",command=self.run, relief=GROOVE)

        self.set_item_position()

    def create_param_textbox(self):
        self.b_size_label=tk.Label(self.start_frame,text="Enter bach size:")
        self.b_size_input=tk.Entry(self.start_frame)

        self.t_index_lable=tk.Label(self.start_frame,text="Enter target index:")
        self.t_index_input = tk.Entry(self.start_frame)

        self.t_name_lable=tk.Label(self.start_frame,text="Enter target name:")
        self.t_name_input = tk.Entry(self.start_frame)

    def create_file_chooser(self):
        self.fc_lable=tk.Label(self.start_frame,text="Choose dataset:")
        self.btn_browse=tk.Button(self.start_frame, text="Browse" ,command=self.browse_button_press, relief=GROOVE)
        self.path = StringVar()
        self.file_path=tk.Label(self.start_frame,textvariable=self.path,relief=GROOVE, width=50,height=1)

    def select_OL_algorithm(self,event):
        self.ol_param_frame.grid_forget()
        if event==OL_OPTIONS[0]:
            self.ol_param_frame = KNN_Param(self.root)
        elif event == OL_OPTIONS[1]:
            self.ol_param_frame = NN_Param(self.root)
        elif event==OL_OPTIONS[2]:
            self.ol_param_frame = SVM_Param(self.root)
        elif event==OL_OPTIONS[3]:
            self.ol_param_frame = NB_Param(self.root)

        self.set_frame_position()

    def create_selected_algo(self):
        self.ol_lable = tk.Label(self.start_frame, text="Choose OL Algorithm:")
        self.ol_var=StringVar(self.start_frame)
        self.ol_var.set("")
        self.ol_menu = OptionMenu(self.start_frame, self.ol_var, *OL_OPTIONS,command=self.select_OL_algorithm)

        self.ofs_lable = tk.Label(self.start_frame, text="Choose OFS Algorithm:")
        self.ofs_var=StringVar(self.start_frame)
        self.ofs_var.set("")
        self.ofs_menu = OptionMenu(self.start_frame, self.ofs_var, *OFS_OPTIONS)

    def set_frame_position(self):
        self.start_frame.grid(row=0, column=0)
        self.ol_param_frame.grid(row=1, column=0)

    def set_item_position(self):
        self.fc_lable.grid(row=0, column=0)
        self.btn_browse.grid(row=1, column=0)
        self.file_path.grid(row=2, column=0)

        self.b_size_label.grid(row=0, column=1)
        self.b_size_input.grid(row=0, column=2)
        self.t_name_lable.grid(row=1, column=1)
        self.t_name_input.grid(row=1, column=2)
        self.t_index_lable.grid(row=2, column=1)
        self.t_index_input.grid(row=2, column=2)

        self.ol_lable.grid(row=0, column=3)
        self.ol_menu.grid(row=0, column=4)

        self.ofs_lable.grid(row=1, column=3)
        self.ofs_menu.grid(row=1, column=4)

        self.run_btn.grid(row=3, column=2)

    def browse_button_press(self):
        file = tk.filedialog.askopenfile(parent=self.start_frame, title='Choose a file')
        if file:
            self.path.set(file.name)
            file.close()

    def run(self):
        print(self.ol_param_frame.get_params())

        if self.path.get() == "":
            self.popup("Please choose a dataset","Error")
            return
        if self.t_name_input.get() == "":
            self.popup("Please Enter target name","Error")
            return
        if self.t_index_input.get() == "":
            self.popup("Please Enter target index","Error")
            return
        if not self.t_index_input.get().isnumeric() or int(self.t_index_input.get())<0:
            self.popup("Please Enter a valid target index", "Error")
            return
        if self.b_size_input.get()=="":
            self.popup("Please Enter bach size","Error")
            return
        if not self.b_size_input.get().isnumeric() or int(self.b_size_input.get())<=0:
            self.popup("Please Enter a valid bach size", "Error")
            return
        if self.ol_var.get()=="":
            self.popup("Please choose online learning algorithm", "Error")
            return
        if self.ofs_var.get()=="":
            self.popup("Please choose online feature selection algorithm", "Error")
            return
        if not self.ol_param_frame.validation():
            self.popup("Please Enter params to online feature selection algorithm", "Error")
            return


        # run_simulation(self.path.get(),self.t_name_input.get(),self.t_index_input.get())

    def popup(self,message,title):
        popup = tk.Toplevel()
        popup.geometry('300x100')
        popup.wm_title(title)
        tk.Label(popup,text=str(message)).pack()






