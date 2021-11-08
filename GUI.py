#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MIT License

Copyright (c) 2021 by the authors listed in the LICENSE file

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import tkinter as tk
from tkinter import ttk
from tkinter import font as tkFont
from tkinter import filedialog as fd
from tkinter import messagebox
from tkinter import simpledialog

import pandas as pd
import numpy as np
import json
import multiprocessing
import threading
from datetime import datetime
import os, time

import lntt


        
def is_float(val):
    try:
        val = float(val)
        return not np.isnan(val)
    except ValueError:
        return False
isnumeric = np.vectorize(is_float, otypes = [bool])
        
def is_float_or_nan(val):
    try:
        val = float(val)
        return True
    except ValueError:
        return False
isnumeric_or_nan = np.vectorize(is_float_or_nan, otypes = [bool])


class TextScrollCombo(ttk.Frame):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid_propagate(False)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.txt = tk.Text(self, bg = "white")
        self.txt.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        scrollb = ttk.Scrollbar(self, command=self.txt.yview)
        scrollb.grid(row=0, column=1, sticky='nsew')
        self.txt['yscrollcommand'] = scrollb.set
        

    def insert(self, pos, text):
        enabling = self.txt["state"] == "disabled"
            
        if enabling: self.txt["state"] = "normal"
        self.txt.insert(pos, text)
        self.txt.yview_moveto(1)
        if enabling: self.txt["state"] = "disabled"
        self.update()
        
        
    def delete(self, pos, ending):
        enabling = self.txt["state"] == "disabled"
            
        if enabling: self.txt["state"] = "normal"
        self.txt.delete(pos, ending)
        if enabling: self.txt["state"] = "disabled"
        self.update()
        
        
    def get(self, s, e):
        return self.txt.get(s, e)
    
    
    
class CreateToolTip(object):
    '''
    create a tooltip for a given widget
    '''
    def __init__(self, widget, text, width):
        self.widget = widget
        self.text = text
        self.width = width
        self.widget.bind("<Motion>", self.enter)
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.close)
        self.tw = None
        

    def enter(self, event = None):
        if event == None: return
    
        if event.x > self.width or event.y > 15:
            self.close()
            return        
        x = event.x + self.widget.winfo_rootx() + 20
        y = event.y + self.widget.winfo_rooty() + 20

        if self.tw != None:
            self.tw.wm_geometry("+%d+%d" % (x, y))
            return
        
        self.tw = tk.Toplevel(self.widget)
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(self.tw, text=self.text, justify='left',
                    background='#ffffe0', relief='solid', borderwidth=1,
                    font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def close(self, event = None):
        if self.tw:
            self.tw.destroy()
            self.tw = None
            
    
class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command = self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion = self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand = True)
        scrollbar.pack(side="right", fill="y")
    
    
    def bind_all_event(self):
        for child in self.scrollable_frame.winfo_children():
            child.bind("<Button-4>", lambda event: self.mouse_wheel(-1))
            child.bind("<Button-5>", lambda event: self.mouse_wheel(1))
            child.bind_all("<MouseWheel>", lambda event: self.mouse_wheel(-int(event.delta) / 120))
    
    
    def mouse_wheel(self, delta):
        self.canvas.yview_scroll(int(delta), "units")
        
        
        

class ProgressBar(tk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.root = tk.Toplevel(root)
        self.pack()
        
        self.colors = ["#dd0000", "#dd3333", "#dd6666", "#dd9999", "#ddbbbb"]
        self.scanner = [10] * 8
        self.w = 640
        self.h = 100
        self.dw = 6
        self.bw = 10
        self.bh = 40
        self.l = 70
        self.current, self.direction = 0, 1
        self.root.title('File is loading, please wait ...')
        self.root.geometry("%ix%i" % (self.w, self.h))
        self.root.resizable(0, 0) #Don't allow resizing in the x or y direction
        self.create_widgets()
        self.draw_progress()
        
        
    def create_widgets(self):
        self.canvas = tk.Canvas(self.root, width = self.w, height = self.h)
        self.canvas.pack()
        
        
    def draw_progress(self):
        self.scanner = [s + 1 for s in self.scanner]
        self.scanner[self.current] = 0
        self.current += self.direction
        if self.current == 7 and self.direction == 1: self.direction = -1
        if self.current == 0 and self.direction == -1: self.direction = 1
        
        for i in range (4):
            s = ((self.w - self.dw) >> 1) - i * (self.l + self.dw)
            points = [s, self.bh, s, self.h - self.bh, s - self.l, self.h - self.bh, s - self.l, self.bh]
            color = self.colors[self.scanner[3 - i]] if self.scanner[3 - i] < len(self.colors) else "#dddddd"
            self.canvas.create_polygon(points, fill = color)
        for i in range (4):
            s = ((self.w + self.dw) >> 1) + i * (self.l + self.dw)
            points = [s, self.bh, s, self.h - self.bh, s + self.l, self.h - self.bh, s + self.l, self.bh]
            color = self.colors[self.scanner[i + 4]] if self.scanner[i + 4] < len(self.colors) else "#dddddd"
            self.canvas.create_polygon(points, fill = color)
            
        self.root.after(80, self.draw_progress)



class Welcome(tk.Frame):
    def __init__(self, root, queues):
        super().__init__(root)
        self.root = root
        self.pack()
        self.queues = queues
        
        self.width = 630
        self.height = 180
        self.root.title('Lightweight Normalization & Testing Tool')
        self.root.geometry("%ix%i" % (self.width, self.height))
        self.root.resizable(0, 0) #Don't allow resizing in the x or y direction
        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.close_window)
        
    
    def create_widgets(self):
        helv20 = tkFont.Font(family='Arial', size=20, weight=tkFont.BOLD)
        button_open_sheet = tk.Button(self.root, text = "Load Excel sheet", width = 17, height = 3)
        button_open_sheet.place(x=25, y=25)
        button_open_sheet['font'] = helv20
        button_open_sheet["command"] = self.open_Excel_loading
        
        
        button_open_parameters = tk.Button(self.root, text = "Load project file", width = 17, height = 3)
        button_open_parameters.place(x=325, y=25)
        button_open_parameters['font'] = helv20
        button_open_parameters["command"] = self.open_parmeters_loading
    
    
    def open_Excel_loading(self):
        self.root.destroy()
        app = Excel_loading(tk.Tk(), self.queues)
        app.mainloop()
        
        
        
    def open_parmeters_loading(self):
        json_name = fd.askopenfilename(title = "Select project file", filetypes = (("LNTT project files","*.lntt"),("all files","*.*")))
        # TODO go on main screen
    
        if len(json_name) != 0:
            try:
                parameters = json.loads(open(json_name, "rt").read())
            except:
                messagebox.showerror("Error", "Your parameter file '%s' is not readable or seems to be corrupted." % json_name)
                return
            
            self.root.destroy()
            lntt_gui = LNTT_GUI(tk.Tk(), parameters, self.queues)
            lntt_gui.mainloop()
                

        
    def close_window(self):
        self.terminate_signal = True
        self.queues[0].put(None)
        self.queues[1].put(None)
        self.root.destroy()
    
    
    
    
    

class Excel_loading(tk.Frame):
    def __init__(self, root, queues):
        super().__init__(root)
        self.root = root
        self.pack()
        self.excel_file = None
        self.excel_name = ""
        self.queues = queues
        self.parameters = parameters = {"output_folder": "",
                                        "data_file": "",
                                        "data_sheet": "",
                                        "skip_rows": [],
                                        "na_values": [],
                                        "delimiter": ",",
                                        "with_cv_threshold": False,
                                        "cv_threshold": 0.15,
                                        "with_data_imputation": False,
                                        "data_imputation_value": 3,
                                        "with_gene_name_request": False,
                                        "accession_column": "",
                                        "with_normalization": False,
                                        "norm_ref_condition": "",
                                        "pairwise_log_fc": False,
                                        "with_normalization_plotting": False,
                                        "conditions": {},
                                        "statistical_testing": []
                                       }
        
        self.width = 565
        self.height = 300
        self.root.title('Lightweight Normalization & Testing Tool')
        self.root.geometry("%ix%i" % (self.width, self.height))
        self.root.resizable(0, 0) #Don't allow resizing in the x or y direction
        self.root.protocol("WM_DELETE_WINDOW", self.close_window)
        self.progressbar = None
        self.loaded = False
        self.create_widgets()
        
        
    def close_window(self):
        self.terminate_signal = True
        self.queues[0].put(None)
        self.queues[1].put(None)
        try: self.progressbar.destroy()
        except: pass
        self.root.destroy()
        
        
    
    def create_widgets(self):
        self.delimiters = {",": ",", ".": ".", "<space>": " ", "<tab>": "\t"}
        self.loaded_csv = False
        self.button_open_sheet = tk.Button(self.root, text = "Open Excel sheet", width = 19)
        self.button_open_sheet.place(x = 25, y = 22)
        self.button_open_sheet["command"] = self.open_Excel_sheet
        
        self.label_excel_sheets = tk.Label(self.root, text = "Available sheets")
        self.label_excel_sheets.place(x = 200, y = 8)
        
        self.combo_excel_sheets = ttk.Combobox(self.root, state='readonly', width = 23)
        self.combo_excel_sheets.pack()
        self.combo_excel_sheets.place(x = 200, y = 27)
        self.combo_excel_sheets["state"] = "disabled"
        self.combo_excel_sheets.bind("<<ComboboxSelected>>", self.combo_sheet_change)
    
        
        self.label_excel_columns = tk.Label(self.root, text = "Excel columns found")
        self.label_excel_columns.place(x = 25, y = 80)
    
        self.listbox_columns = tk.Listbox(self.root, width = 50, height = 10, bg = "white")
        self.listbox_columns.place(x = 25, y = 100)
        
        self.label_skip_rows = tk.Label(self.root, text = "Skip n first rows")
        self.label_skip_rows.place(x = 400, y = 8)
        
        self.skip_rows = tk.IntVar()
        self.skip_rows.trace('w', lambda name, index, mode, skip_rows = self.skip_rows: self.spinbox_skip_rows_change())
    
        self.spinbox_skip_rows = tk.Spinbox(self.root, from_ = 0, to = 50, width = 15, bg = "white")#, textvariable = self.skip_rows)
        self.spinbox_skip_rows.place(x = 400, y = 25)
        self.spinbox_skip_rows["state"] = "disabled"
        self.spinbox_skip_rows["textvariable"] = self.skip_rows
        
        self.label_na = tk.Label(self.root, text = "Values representing N/A")
        self.label_na.place(x = 400, y = 80)
        
        self.text_na_values = tk.Text(self.root, bg = "white", width = 18, height = 8)
        self.text_na_values.place(x = 400, y = 100)
        
        self.button_continue = tk.Button(self.root, text = "Continue", width = 10)
        self.button_continue.place(x = 452, y = 244)
        self.button_continue["command"] = self.proceed
        self.button_continue["state"] = "disabled"
        

        
        
    def open_Excel_sheet(self):
        self.excel_name = fd.askopenfilename(title = "Select Excel file",filetypes = (("Excel file", "*.xlsx"), ("Excel 97 file", "*.xls"), ("Comma separated values file", "*.csv"), ("Tab separated values file", "*.tsv"), ("all files", "*.*")))
        if len(self.excel_name) == 0: return
    
        self.parameters["data_file"] = self.excel_name
        self.open_progress_bar()
        self.loaded = False
        threading.Thread(target = self.load_file).start()
        
        
        
    def load_file(self):
        try:
            excel_file = pd.ExcelFile(self.excel_name)
            self.combo_excel_sheets["values"] = list(excel_file.sheet_names)
            self.button_continue["state"] = "normal"
            self.parameters["data_sheet"] = self.combo_excel_sheets.get()
            self.label_excel_sheets["text"] = "Available sheets"
            
        except Exception:
            self.loaded_csv = True
            self.label_excel_sheets["text"] = "CSV delimiter"
            self.combo_excel_sheets["values"] = list(self.delimiters.keys())
            
        
        self.combo_excel_sheets["state"] = "enabled"
        self.combo_excel_sheets.current(0)
        self.button_continue["state"] = "normal"
        self.spinbox_skip_rows["state"] = "readonly"
        self.load_excel_sheet()
    
    
    def combo_sheet_change(self, event = None):
        if self.loaded_csv: self.parameters["delimiter"] = self.delimiters[self.combo_excel_sheets.get()]
        self.open_progress_bar()
        threading.Thread(target = self.load_excel_sheet).start()
        
        
    def spinbox_skip_rows_change(self, event = None):
        self.open_progress_bar()
        threading.Thread(target = self.load_excel_sheet).start()
        
        
    def open_progress_bar(self):
        self.loaded = False
        self.progressbar = ProgressBar(self.root)
        self.close_progress_bar()
        
        
        
    def close_progress_bar(self):
        if self.loaded:
            try: self.progressbar.root.destroy()
            except: pass
            self.progressbar = None
        else:
            self.root.after(100, self.close_progress_bar)
    
    def load_excel_sheet(self):
        if self.excel_name == "": return
    
        self.loaded = False
        self.text_na_values.delete('1.0', tk.END)
        self.listbox_columns.delete(0, tk.END)
        self.parameters["skip_rows"] = list(range(self.skip_rows.get()))
        if not self.loaded_csv: self.parameters["data_sheet"] = self.combo_excel_sheets.get()
        try:
            self.data_frame = pd.read_excel(self.excel_name, self.combo_excel_sheets.get(), skiprows = self.parameters["skip_rows"])
        except Exception:
            try:
                self.data_frame = pd.read_csv(self.excel_name, delimiter = self.parameters["delimiter"], skiprows = self.parameters["skip_rows"])
            except Exception:
                return
            
        # go through columns and check if they are mainly number columns and if there are as "N/A" value denoted cells
        columns = [c for c in self.data_frame]
        num_col = self.data_frame.shape[0]
        potential_na = set()
        for column in columns[::-1]:
            self.listbox_columns.insert(0, column)
            nums = isnumeric(self.data_frame[column])
            if np.sum(nums) / num_col > 0.5 and np.sum(~nums) > 0:
                nums = isnumeric_or_nan(self.data_frame[column])
                potential_na |= set(self.data_frame[column][~nums])
        for na_value in potential_na: self.text_na_values.insert(tk.END, "%s\n" % na_value)
        
        self.loaded = True
    

    def proceed(self):
        self.parameters["na_values"] = [f for f in self.text_na_values.get("1.0", tk.END).split("\n") if len(f) > 0]
        self.root.destroy()
        lntt_gui = LNTT_GUI(tk.Tk(), self.parameters, self.queues, data_frame = self.data_frame)
        lntt_gui.mainloop()
    
    
    
    
    
    
    
    
    
class LNTT_GUI(tk.Frame):
    def __init__(self, root, parameters, queues, data_frame = None):
        super().__init__(root)
        self.root = root
        self.pack()
        self.parameters = parameters
        self.height = 700
        self.width = int(self.height * 1.618033)
        self.root.title('Lightweight Normalization & Testing Tool')
        self.root.geometry("%ix%i" % (self.width, self.height))
        self.root.resizable(0, 0) #Don't allow resizing in the x or y direction
        self.progressbar = None
        self.loaded = False
        self.lntt_queue, self.main_queue = queues
        self.lntt_runs = False
        self.log_content = []
        self.root.protocol("WM_DELETE_WINDOW", self.close_window)
        self.terminate_signal = False
        self.combos_label_column = []
        self.combos_custom_labeling = []
        self.texts_custom_labels = []
        self.combos_treated = []
        self.combos_wt = []
        self.listboxes_anova = []
        self.test_widgets = []
        self.open_progress_bar()
        self.root.withdraw()
        threading.Thread(target = self.load_file, args = (data_frame,)).start()
        
        
        
    def load_file(self, data_frame):
        if type(data_frame) == pd.DataFrame:
            self.data_frame = data_frame
        else:
            try:
                self.data_frame = pd.read_excel(self.parameters["data_file"], self.parameters["data_sheet"], skiprows = self.parameters["skip_rows"])
            except Exception:
                try:
                    self.data_frame = pd.read_csv(self.parameters["data_file"], delimiter = self.parameters["delimiter"], skiprows = self.parameters["skip_rows"])
                except Exception:
                    messagebox.showerror("Error", "The data sheet '%s' could not be found or loaded properly." % self.parameters["data_file"])
                    self.close_window()
                    
                    
        self.loaded = True
        self.data_frame_columns = [column for column in self.data_frame]
        self.root.deiconify()
        self.create_widgets()
        
        
        
    def open_progress_bar(self):
        self.loaded = False
        self.progressbar = ProgressBar(self.root)
        self.close_progress_bar()
        
        
        
    def close_progress_bar(self):
        if self.loaded:
            try: self.progressbar.root.destroy()
            except: pass
            self.progressbar = None
        else:
            self.root.after(100, self.close_progress_bar)  
        
        
    def create_widgets(self):
        
        #######################################################################
        ## widgets for CV module
        #######################################################################
        self.labelframe_cv = tk.LabelFrame(self.root, text = "CV filtering (?)", width = int(self.width * 0.3), height = int(self.height * 0.15))
        self.labelframe_cv.place(x = 25, y = 25)
        CreateToolTip(self.labelframe_cv,
                      'When applying the CV filtering, all entries are\n'
                      'discarded that have a relative variance, i.e., the\n'
                      'coefficient of variation (CV) over all conditions\n'
                      'less than the defined threshold.', 85)

        self.cv = tk.IntVar()
        self.cv_perc = tk.IntVar()
        self.cv_perc.set(int(self.parameters["cv_threshold"] * 100))
        with_cv = False
        if "with_cv_threshold" in self.parameters and self.parameters["with_cv_threshold"]:
            self.cv.set(1)
            with_cv = True
        
        self.checkbutton_cv = tk.Checkbutton(self.labelframe_cv, text = "Enable CV filtering", variable = self.cv, command = self.enabling_cv)
        self.checkbutton_cv.place(x = 5, y = 5)
        
        self.root.update()
        self.label_cv_perc = tk.Label(self.labelframe_cv, text = "CV percentage")
        self.label_cv_perc.place(x = self.checkbutton_cv.winfo_x() + 5, y = 30)
        
        self.root.update()
        self.cv_perc.trace('w', lambda name, index, mode, cv_perc = self.cv_perc: self.change_cv())
        self.spinbox_cv = tk.Spinbox(self.labelframe_cv, from_ = 0, to = 1000, increment = 1, bg = "white", textvariable = self.cv_perc, state = "disabled", width = 11)
        self.spinbox_cv.place(x = self.checkbutton_cv.winfo_x() + 5, y = 50)
        if with_cv: self.spinbox_cv["state"] = "readonly"
        
        self.root.update()
        self.label_cv_perc_sign = tk.Label(self.labelframe_cv, text = "%")
        self.label_cv_perc_sign.place(x = self.spinbox_cv.winfo_x() + self.spinbox_cv.winfo_width() + 5, y = 50)
        
        
        
    
        
        #######################################################################
        ## widgets for data imputation module
        #######################################################################
        self.labelframe_imputation = tk.LabelFrame(self.root, text = "Data imputation (?)", width = (self.width * 0.3), height = int(self.height * 0.15))
        self.labelframe_imputation.place(x = int(self.width * 0.3) + 50, y = 25)
        CreateToolTip(self.labelframe_imputation, 
                      'Data imputation is being applyed to all entries that\n'
                      'have at most [threshold] missing values. For each\n'
                      'missing value, the abundance distribution within the\n'
                      'according measurement is taken and a random number is\n'
                      'picked within the 5% quantile.', 115)
        
        self.imputation = tk.IntVar()
        self.imputation_number = tk.IntVar()
        self.imputation_number.set(self.parameters["data_imputation_value"])
        set_imputation = False
        if "with_data_imputation" in self.parameters and self.parameters["with_data_imputation"]:
            self.imputation.set(1)
            set_imputation = True
        
        self.checkbutton_imputation = tk.Checkbutton(self.labelframe_imputation, text = "Enable data imputation", variable = self.imputation, command = self.enabling_imputation)
        self.checkbutton_imputation.place(x = 5, y = 5)
        
        
        self.label_imputation = tk.Label(self.labelframe_imputation, text = "Max. imputation number")
        self.label_imputation.place(x = 10, y = 30)
        
        self.imputation_number.trace('w', lambda name, index, mode, imputation_number = self.imputation_number: self.change_imputation())
        self.spinbox_imputation = tk.Spinbox(self.labelframe_imputation, from_ = 0, to = 1000, increment = 1, bg = "white", textvariable = self.imputation_number, state = "disabled", width = 11)
        self.spinbox_imputation.place(x = 10, y = 50)
        if set_imputation: self.spinbox_imputation["state"] = "readonly"
        
        
        
        
        
        
        
        #######################################################################
        ## widgets for gene name request module
        #######################################################################
        self.labelframe_genes = tk.LabelFrame(self.root, text = "Gene name request (?)", width = (self.width * 0.3), height = int(self.height * 0.15))
        self.labelframe_genes.place(x = 2 * int(self.width * 0.3) + 75, y = 25)
        CreateToolTip(self.labelframe_genes, 
                      'This module is dedicated to request gene names\n'
                      'from Uniprot when providing accession numbers.', 135)
    
        self.genes = tk.IntVar()
        if "with_gene_name_request" in self.parameters and self.parameters["with_gene_name_request"]:
            self.genes.set(1)
        
        self.checkbutton_genes = tk.Checkbutton(self.labelframe_genes, text = "Enable gene name request", variable = self.genes, command = self.enabling_genes)
        self.checkbutton_genes.place(x = 5, y = 5)
        
        self.label_genes = tk.Label(self.labelframe_genes, text = "Select accession column")
        self.label_genes.place(x = 10, y = 30)
        
        self.combo_genes = ttk.Combobox(self.labelframe_genes, width = 23, values = self.data_frame_columns)
        self.combo_genes.place(x = 10, y = 50)
        self.combo_genes["state"] = "readonly" if self.genes.get() != 0 else "disabled"
        self.combo_genes.current(0)
        found_gene_col = -1
        if "accession_column" in self.parameters:
            for i, column in enumerate(self.data_frame_columns):
                if self.parameters["accession_column"] == column:
                    self.combo_genes.current(i)
                    found_gene_col = i
                    break
        self.combo_genes.bind("<<ComboboxSelected>>", self.genes_select_column)
        if found_gene_col == -1: self.genes_select_column()
            
        
        
        #######################################################################
        ## widgets for condition module
        #######################################################################
        self.root.update()
        h_condition = self.labelframe_genes.winfo_height() + self.labelframe_genes.winfo_y() + 10
        self.labelframe_conditions = tk.LabelFrame(self.root, text = "Experiment conditions", width = self.labelframe_genes.winfo_width() + self.labelframe_genes.winfo_x() - 25, height = int(self.height * 0.25))
        self.labelframe_conditions.place(x = 25, y = h_condition)
        
        self.button_add_condition = tk.Button(self.labelframe_conditions, text = "Add condition", width = 12)
        self.button_add_condition.place(x = 10, y = 20)
        self.button_add_condition["command"] = self.add_condition
        
        self.button_delete_condition = tk.Button(self.labelframe_conditions, text = "Delete condition", width = 12)
        self.button_delete_condition.place(x = 10, y = 60)
        self.button_delete_condition["command"] = self.delete_condition
        
        self.root.update()
        x_lc = self.button_delete_condition.winfo_x() + self.button_delete_condition.winfo_width() + 10
        
        self.label_conditions = tk.Label(self.labelframe_conditions, text = "Registered conditions")
        self.label_conditions.place(x = x_lc, y = 0)
        
        self.listbox_conditions = tk.Listbox(self.labelframe_conditions, width = 20, height = 7, selectmode = "single", bg = "white")
        self.listbox_conditions.place(x = x_lc, y = 20)
        if "conditions" in self.parameters:
            for condition in self.parameters["conditions"]:
                self.listbox_conditions.insert("end", condition)
            
        else:
            self.parameters["conditions"] = {}
        self.listbox_conditions.bind("<<ListboxSelect>>", self.show_condition_columns)
        self.listbox_conditions.configure(exportselection = False)
        
        self.root.update()
        x_lcol = self.listbox_conditions.winfo_x() + self.listbox_conditions.winfo_width() + 10
        self.label_condition_columns = tk.Label(self.labelframe_conditions, text = "Columns in condition")
        self.label_condition_columns.place(x = x_lcol, y = 0)
        
        self.listbox_condition_columns = tk.Listbox(self.labelframe_conditions, width = 48, height = 7, selectmode = "extended", bg = "white")
        self.listbox_condition_columns.place(x = x_lcol, y = 20)
        self.listbox_condition_columns.configure(exportselection = False)
        
        self.root.update()
        x_lcolbuttons = self.listbox_condition_columns.winfo_x() + self.listbox_condition_columns.winfo_width() + 10
        
        self.button_add_column = tk.Button(self.labelframe_conditions, text = "<-", font = ("Curier", 16), width = 2)
        self.button_add_column.place(x = x_lcolbuttons, y = 30)
        self.button_add_column["command"] = self.add_columns_to_condition
        
        self.button_withdraw_column = tk.Button(self.labelframe_conditions, text = "->", font = ("Curier", 16), width = 2)
        self.button_withdraw_column.place(x = x_lcolbuttons, y = 70)
        self.button_withdraw_column["command"] = self.withdraw_columns_from_condition
        
        
        self.root.update()
        x_lallcol = self.button_withdraw_column.winfo_x() + self.button_withdraw_column.winfo_width() + 10
        self.label_all_columns = tk.Label(self.labelframe_conditions, text = "Columns in data set")
        self.label_all_columns.place(x = x_lallcol, y = 0)
        
        self.listbox_all_columns = tk.Listbox(self.labelframe_conditions, width = 48, height = 7, selectmode = "extended", bg = "white")
        self.listbox_all_columns.place(x = x_lallcol, y = 20)
        for column in self.data_frame_columns: self.listbox_all_columns.insert("end", column)
        self.listbox_all_columns.configure(exportselection = False)
        
        self.pairwise_log_fc = tk.IntVar()
        if "pairwise_log_fc" in self.parameters and self.parameters["pairwise_log_fc"]:
            self.pairwise_log_fc.set(1)
        
        self.checkbutton_genes = tk.Checkbutton(self.labelframe_conditions, text = "Store pairwise\nlog FC over\nall conditions", variable = self.pairwise_log_fc, command = self.change_pairwise_log_fc)
        self.checkbutton_genes.place(x = 5, y = 95)
        
        
        
        #######################################################################
        ## widgets for statistical testing module
        #######################################################################
        self.root.update()
        y_test = self.labelframe_conditions.winfo_height() + self.labelframe_conditions.winfo_y() + 10
        self.labelframe_test = tk.LabelFrame(self.root, text = "Statistical testing", width = 400, height = int(self.height * 0.52))
        self.labelframe_test.place(x = 25, y = y_test)
        
        self.button_add_test = tk.Button(self.labelframe_test, text = "Add test", width = 10, height = 1)
        self.button_add_test.place(x = 5, y = 5)
        self.button_add_test["command"] = self.add_test
        
        self.button_delete_test = tk.Button(self.labelframe_test, text = "Delete test", width = 10)
        self.button_delete_test.place(x = 120, y = 5)
        self.button_delete_test["command"] = self.delete_test
        
        self.notebook = ttk.Notebook(self.labelframe_test, width = 380, height = 260)
        self.notebook.place(x = 5, y = 50)
        
        if "statistical_testing" in self.parameters:
            for i, testing in enumerate(self.parameters["statistical_testing"]):
                self.add_test(False)
                   
        
        
        #######################################################################
        ## widgets for normalization module
        #######################################################################
        self.root.update()
        h_norm = self.labelframe_conditions.winfo_height() + self.labelframe_conditions.winfo_y() + 10
        x_norm = self.labelframe_test.winfo_width() + self.labelframe_test.winfo_x() + 10
        w_norm = self.labelframe_genes.winfo_width() + self.labelframe_genes.winfo_x() - x_norm
        self.labelframe_normalization = tk.LabelFrame(self.root, text = "Normalization (?)", width = w_norm, height = int(self.height * 0.1))
        self.labelframe_normalization.place(x = x_norm, y = h_norm)
        CreateToolTip(self.labelframe_normalization, 
                      'The normalization is based on the rank invariant\n'
                      'set normalization approach and was designed to cope\n'
                      'with missing values. The module picks a measurement\n'
                      'as reference within the selected reference condition\n'
                      'that has the least missing values.', 100)
        
        self.normalization = tk.IntVar()
        if "with_normalization" in self.parameters and self.parameters["with_normalization"]: self.normalization.set(1)
        self.checkbutton_normalization = tk.Checkbutton(self.labelframe_normalization, text = "Enable normalization", variable = self.normalization, command = self.enabling_normalization)
        self.checkbutton_normalization.place(x = 5, y = 17)
        
        
        self.label_reference_condition = tk.Label(self.labelframe_normalization, text = "Select reference condition")
        self.label_reference_condition.place(x = 200, y = 0)
        
        
        self.combo_reference_conditions = ttk.Combobox(self.labelframe_normalization, width = 25)
        self.combo_reference_conditions.place(x = 200, y = 20)
        self.combo_reference_conditions["state"] = "disabled"
        
        parameter_condition = self.parameters["norm_ref_condition"] if "norm_ref_condition" in self.parameters else ""
        self.update_conditions_for_normalization()
        self.combo_reference_conditions.bind("<<ComboboxSelected>>", self.change_norm_condition)
        if "norm_ref_condition" in self.parameters:
            for i, condition in enumerate(self.combo_reference_conditions["values"]):
                if condition == parameter_condition:
                    self.combo_reference_conditions.current(i)
                    break
        
        self.normalization_plots = tk.IntVar()
        if "with_normalization_plotting" in self.parameters and self.parameters["with_normalization_plotting"]: self.normalization_plots.set(1)
        self.checkbutton_normalization_plots = tk.Checkbutton(self.labelframe_normalization, text = "Create normalization plots", variable = self.normalization_plots, command = self.enabling_normalization_plots)
        self.checkbutton_normalization_plots.place(x = 450, y = 17)
        
        
        
        
        
        #######################################################################
        ## widgets for logging module
        #######################################################################
        self.root.update()
        y_log = self.labelframe_normalization.winfo_y() + self.labelframe_normalization.winfo_height() + 10
        self.labelframe_log = tk.LabelFrame(self.root, text = "Logging", width = w_norm, height = int(self.height * 0.30))
        self.labelframe_log.place(x = x_norm, y = y_log)
        
        
        self.text_log = TextScrollCombo(self.labelframe_log, width = 640, height = 175)
        self.text_log.place(x = 5, y = 5)
        self.text_log.txt["state"] = "disabled"
        
        
        
        
        
        #######################################################################
        ## widgets for operation module
        #######################################################################
        self.root.update()
        y_ops = self.labelframe_log.winfo_height() + self.labelframe_log.winfo_y() + 10
        h_ops = self.labelframe_test.winfo_y() + self.labelframe_test.winfo_height() - y_ops
        self.labelframe_ops = tk.LabelFrame(self.root, text = "Operations", width = w_norm, height = h_ops)
        self.labelframe_ops.place(x = x_norm, y = y_ops)
        
        self.button_select_output = tk.Button(self.labelframe_ops, text = "Select output folder", width = 15)
        self.button_select_output.place(x = 5, y = 5)
        self.button_select_output["command"] = self.select_output_folder
        
        self.button_start_lntt = tk.Button(self.labelframe_ops, text = "Start process", width = 15)
        self.button_start_lntt.place(x = 250, y = 5)
        self.button_start_lntt["command"] = self.start_lntt
        
        self.button_store_parameters = tk.Button(self.labelframe_ops, text = "Save project", width = 15)
        self.button_store_parameters.place(x = 500, y = 5)
        self.button_store_parameters["command"] = self.store_parameters
        
        
        
    def close_window(self):
        self.terminate_signal = True
        self.main_queue.put(None)
        self.lntt_queue.put(None)
        try: self.progressbar.root.destroy()
        except: pass
        self.root.destroy()
        
        
    def update_conditions_for_normalization(self):
        curr_condition = None
        self.combo_reference_conditions["state"] = "normal"
        if len(self.combo_reference_conditions["values"]) > 0:
            curr_condition = self.combo_reference_conditions.get()
        self.combo_reference_conditions.delete(0, tk.END)
        self.combo_reference_conditions["values"] = [condition for condition in self.parameters["conditions"]]
        if len(self.combo_reference_conditions["values"]) > 0:
            self.combo_reference_conditions.current(0)
            for i, condition in enumerate(self.combo_reference_conditions["values"]):
                if curr_condition == condition:
                    self.combo_reference_conditions.current(i)
                    break
        self.combo_reference_conditions["state"] = "normal" if self.normalization.get() and len(self.combo_reference_conditions["values"]) > 0 else "disabled"
        self.change_norm_condition()
        
        
    def enabling_cv(self):
        self.spinbox_cv["state"] = "readonly" if self.cv.get() else "disabled"
        self.parameters["with_cv_threshold"] = self.cv.get() == True
        
        
    def change_cv(self):
        self.parameters["cv_threshold"] = self.cv_perc.get() / 100
    
    
    def enabling_imputation(self):
        self.spinbox_imputation["state"] = "readonly" if self.imputation.get() else "disabled"
        self.parameters["with_data_imputation"] = self.imputation.get() != 0
        
    def change_pairwise_log_fc(self):
        self.parameters["pairwise_log_fc"] = self.pairwise_log_fc.get() != 0
        if self.pairwise_log_fc.get() == 1:
            cols = 0
            for c in self.parameters["conditions"]:
                cols += len(self.parameters["conditions"][c])
            if cols > 11:
                num_pairwise = (cols * (cols - 1)) >> 1
                MsgBox = tk.messagebox.askquestion ('Warning', 'Currently, %i columns would cause %i additional columns. Do you want to continue?' % (cols, num_pairwise), icon = 'warning')
                if MsgBox != 'Yes':
                    self.pairwise_log_fc.set(0)
        
        
        
    def change_imputation(self):
        self.parameters["data_imputation_value"] = self.imputation_number.get()
    
    
    
    def enabling_genes(self):
        self.combo_genes["state"] = "readonly" if self.genes.get() else "disabled"
        self.parameters["with_gene_name_request"] = self.genes.get() != 0
        
        label_columns = ["no labeling"]
        if self.genes.get() and "Genes" not in self.data_frame: label_columns.append("Genes")
        label_columns += [c for c in self.data_frame]
            
        for page_id, combo in enumerate(self.combos_label_column):
            value = combo.get()
            combo["values"] = label_columns
            combo.current(0)
            for i, column in enumerate(combo["values"]):
                if column == value:
                    combo.current(i)
                    break
            self.change_combo(page_id, "labeling_column", combo)
            
            
        
        custom_labeling = ["all significant"]
        if self.genes.get() and "Genes" not in self.data_frame: custom_labeling.append("Custom on Genes")
        custom_labeling += ["Custom on %s" % c for c in self.data_frame]
            
        for page_id, combo in enumerate(self.combos_custom_labeling):
            value = combo.get()
            combo["values"] = custom_labeling
            combo.current(0)
            for i, column in enumerate(combo["values"]):
                if column == value:
                    combo.current(i)
                    break
            self.change_combo(page_id, "custom_labeling", combo)
            
            
            
        
    def genes_select_column(self, event = None):
        self.parameters["accession_column"] = self.combo_genes.get()
        
        
    def add_condition(self):
        w = tk.simpledialog.askstring(title = "Add condition", prompt="Please name a new condition:")
        if w != None:
            self.parameters["conditions"][w] = []
            self.listbox_conditions.insert("end", w)
            self.update_conditions_for_normalization()
        
            conditions = [c for c in self.parameters["conditions"]]
            for combo in self.combos_treated:
                value = combo.get()
                combo["values"] = conditions
                combo.current(0)
                for i, condition in enumerate(combo["values"]):
                    if condition == value:
                        combo.current(i)
                        break
                    
            for combo in self.combos_wt:
                value = combo.get()
                combo["values"] = conditions
                combo.current(0)
                for i, condition in enumerate(combo["values"]):
                    if condition == value:
                        combo.current(i)
                        break
                    
            # anova
            for anova in self.listboxes_anova:
                selections = set([anova.get(pos) for pos in anova.curselection()])
                anova.delete(0, tk.END)
                for i, condition in enumerate(conditions):
                    anova.insert(tk.END, condition)
                    if condition in selections:
                        anova.select_set(first = i)
                        
                    
    
    def delete_condition(self):
        if len(self.listbox_conditions.curselection()) > 0:
            if self.listbox_conditions.size() == 1 and len(self.notebook.tabs()) > 0:
                MsgBox = tk.messagebox.askquestion ('Delete all conditions', 'You are going to delete the last condition. Therefore, all statistical tests will be deleted. Do you want to proceed?', icon = 'warning')
                if MsgBox == 'yes':
                    pos = self.listbox_conditions.curselection()[0]
                    condition = self.listbox_conditions.get(pos)
                    del self.parameters["conditions"][condition]
                    self.listbox_conditions.delete(pos, pos)
                    self.listbox_condition_columns.delete(0, tk.END)
                    while len(self.notebook.tabs()) > 0:
                        self.delete_test()
            else:
                pos = self.listbox_conditions.curselection()[0]
                condition = self.listbox_conditions.get(pos)
                del self.parameters["conditions"][condition]
                self.listbox_conditions.delete(pos, pos)
                self.listbox_condition_columns.delete(0, tk.END)
                
            if self.listbox_conditions.size() > 0:
                conditions = [c for c in self.parameters["conditions"]]
                for test_id, combo in enumerate(self.combos_treated):
                    value = combo.get()
                    combo["values"] = conditions
                    combo.current(0)

                    for i, condition in enumerate(combo["values"]):
                        if condition == value:
                            combo.current(i)
                            break
                    self.change_combo(test_id, "condition_treated", combo)
                        
                for test_id, combo in enumerate(self.combos_wt):
                    value = combo.get()
                    combo["values"] = conditions
                    combo.current(0)
                    for i, condition in enumerate(combo["values"]):
                        if condition == value:
                            combo.current(i)
                            break
                    self.change_combo(test_id, "condition_wt", combo)
                    
                # anova
                for test_id, anova in enumerate(self.listboxes_anova):
                    selections = set([anova.get(pos) for pos in anova.curselection()])
                    anova.delete(0, tk.END)
                    for i, condition in enumerate(conditions):
                        anova.insert(tk.END, condition)
                        if condition in selections:
                            anova.select_set(first = i)
                            
                    self.conditions_anova_change(test_id)
                        
        self.update_conditions_for_normalization()
        
        
        
    def show_condition_columns(self, event = None):
        self.listbox_condition_columns.delete(0, tk.END)
        for column in self.parameters["conditions"][self.listbox_conditions.get(self.listbox_conditions.curselection()[0])]:
            self.listbox_condition_columns.insert("end", column)
        
        
        
    def add_columns_to_condition(self):
        if len(self.listbox_conditions.curselection()) > 0 and len(self.listbox_all_columns.curselection()) > 0:
            condition = self.listbox_conditions.get(self.listbox_conditions.curselection()[0])
            added_cols = set(self.parameters["conditions"][condition])
            for pos in self.listbox_all_columns.curselection():
                column = self.listbox_all_columns.get(pos)
                if column not in added_cols:
                    self.parameters["conditions"][condition].append(column)
                    self.listbox_condition_columns.insert("end", column)
                    
                    
                    
    def withdraw_columns_from_condition(self):
        if len(self.listbox_conditions.curselection()) > 0 and len(self.listbox_condition_columns.curselection()) > 0:
            condition = self.listbox_conditions.get(self.listbox_conditions.curselection()[0])
            for pos in sorted(list(self.listbox_condition_columns.curselection()), reverse = True):
                column = self.listbox_condition_columns.get(pos)
                self.listbox_condition_columns.delete(pos, pos)
                self.parameters["conditions"][condition].remove(column)
         
         
         
    def enabling_normalization(self):
        self.parameters["with_normalization"] = self.normalization.get() != 0
        self.combo_reference_conditions["state"] = "normal" if self.normalization.get() and len(self.combo_reference_conditions["values"]) > 0 else "disabled"
        
        
        
    def enabling_normalization_plots(self):
        self.parameters["with_normalization_plotting"] = self.normalization.get() != 0
        
        
        
    def change_norm_condition(self, event = None):
        if self.combo_reference_conditions["state"] != "disabled":
            self.parameters["norm_ref_condition"] = self.combo_reference_conditions.get()
        
        
        
    def logging_process(self):
        interrput = False
        while not interrput:
            response = self.main_queue.get()
            if type(response) == str:
                date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                self.log_content.append("%s: %s" % (date_time, response))
                self.text_log.delete("1.0", "end")
                self.text_log.insert("1.0", "\n".join(self.log_content))
            else:
                self.lntt_runs = False
                if not self.terminate_signal: self.button_start_lntt["text"] = "Start process"
                break
        
        
        
    def start_lntt(self):
        if self.lntt_runs:
            self.lntt_queue.put(1)
        else:
            if "output_folder" not in self.parameters or len(self.parameters["output_folder"]) == 0 or not os.path.exists(self.parameters["output_folder"]):
                messagebox.showerror("Error", "No valid output directory selected.")
                return
            
            for i, stat in enumerate(self.parameters["statistical_testing"]):
                if stat["test"] == "anova":
                    if len(self.listboxes_anova[i].curselection()) == 0:
                        messagebox.showerror("Error", "In test %i, the ANOVA has no selected conditions." % (i + 1))
                        return
            
            self.lntt_runs = True
            self.log_content = []
            self.text_log.delete("1.0", "end")
            self.button_start_lntt["text"] = "Stop process"
            self.lntt_queue.put(self.parameters)
            threading.Thread(target = self.logging_process).start()
        
        
        
    def select_output_folder(self):
        dir_name = fd.askdirectory(title = "Select output directory")
        if dir_name != None:
            self.parameters["output_folder"] = dir_name
            
            
            
    def store_parameters(self):
        json_name = fd.asksaveasfilename(title = "Save parameter file", filetypes = (("LNTT project files","*.lntt"),))
        if json_name != None:
            json_name = str(json_name)
            if json_name.lower()[-5:] != ".lntt": json_name += ".lntt"
            with open(json_name, "wt") as output_file:
                output_file.write(json.dumps(self.parameters))
               
               
               
    def add_test(self, new_entry = True):
        if self.listbox_conditions.size() == 0:
            messagebox.showerror("Error", "No conditions available. Please set up at least one condition first.")
            return
        page = ttk.Frame(self.notebook)
        page_id = len(self.notebook.tabs())
        self.notebook.add(page, text='Test %i' % (page_id + 1))
        if new_entry:
            condition = self.listbox_conditions.get(0)
            self.parameters["statistical_testing"].append({"test": "ttest",
                                                           "significance_level": 0.05,
                                                           "condition_treated": condition,
                                                           "condition_wt": condition,
                                                           "anova_conditions": [],
                                                           "ttest_paired": False,
                                                           "ttest_type": "two-sided",
                                                           "ttest_equal_variance": False,
                                                           "pval_adjust": "fdr_bh",
                                                           "log_values": False,
                                                           "with_stat_test_plotting": True,
                                                           "labeling_column": "",
                                                           "custom_labeling": "",
                                                           "custom_labels": [],
                                                           "log_fc_thresholds": [-1, 1],
                                                           })
            stat_data = self.parameters["statistical_testing"][-1]
        
        else: stat_data = self.parameters["statistical_testing"][page_id]
        
        frame = ScrollableFrame(page)
        frame.canvas["width"] = 355
        frame.canvas["height"] = 250
        frame.place(x = 5, y = 5)

        inner_frame = tk.Frame(frame.scrollable_frame)
        inner_frame.pack()
        inner_frame["width"] = 350
        self.test_widgets.append({"ttest": [], "anova": []})
        
        x_widgets, w_labels, h_distance = 170, 20, 7
        
        label_test_type = tk.Label(frame.scrollable_frame, text = "Test type", justify = "right", anchor = "e")
        label_test_type.place(x = 5, y = 10)
        label_test_type["width"] = w_labels
        
        current_test = "ttest"
        combo_test_type = ttk.Combobox(frame.scrollable_frame, values = ["ttest", "anova"], width = 20)
        combo_test_type.pack()
        combo_test_type["state"] = "readonly"
        combo_test_type.place(x = 170, y = 10)
        combo_test_type.current(0)
        if not new_entry:
            for i, condition in enumerate(combo_test_type["values"]):
                if condition == stat_data["test"]:
                    combo_test_type.current(i)
                    break
        combo_test_type.bind("<<ComboboxSelected>>", lambda x: self.change_combo(page_id, "test", combo_test_type))
        current_test = combo_test_type["values"][combo_test_type.current()]
        
        
        
        self.root.update()
        label_significance_level = tk.Label(frame.scrollable_frame, text = "Significance level", justify = "right", anchor = "e")
        label_significance_level.place(x = 5, y = combo_test_type.winfo_height() + combo_test_type.winfo_y() + h_distance)
        label_significance_level["width"] = w_labels
        label_significance_level.origin = {"ttest": [5, combo_test_type.winfo_height() + combo_test_type.winfo_y() + h_distance], "anova": [5, combo_test_type.winfo_height() + combo_test_type.winfo_y() + h_distance]}
        significance_level_number = tk.DoubleVar()
        significance_level_number.trace('w', lambda name, index, mode, significance_level_number = significance_level_number: self.change_significance_level(page_id, significance_level_number))
        parameter_val = stat_data["significance_level"] if "significance_level" in stat_data and not new_entry else 0.05
        spinbox_significance_level = tk.Spinbox(frame.scrollable_frame, from_ = 0, to = 1, width = 20, increment = 0.01, bg = "white", textvariable = significance_level_number)
        spinbox_significance_level["state"] = "readonly"
        spinbox_significance_level.place(x = x_widgets, y = combo_test_type.winfo_height() + combo_test_type.winfo_y() + h_distance)
        spinbox_significance_level.origin = {"ttest": [x_widgets, combo_test_type.winfo_height() + combo_test_type.winfo_y() + h_distance], "anova": [x_widgets, combo_test_type.winfo_height() + combo_test_type.winfo_y() + h_distance]}
        significance_level_number.set(parameter_val)
        self.test_widgets[-1]["ttest"].append(label_significance_level)
        self.test_widgets[-1]["ttest"].append(spinbox_significance_level)
        self.test_widgets[-1]["anova"].append(label_significance_level)
        self.test_widgets[-1]["anova"].append(spinbox_significance_level)
        
        
        
        self.root.update()
        label_pval_adjust = tk.Label(frame.scrollable_frame, text = "Pvalue adjustment", justify = "right", anchor = "e")
        label_pval_adjust.place(x = 5, y = spinbox_significance_level.winfo_height() + spinbox_significance_level.winfo_y() + h_distance)
        label_pval_adjust["width"] = w_labels
        label_pval_adjust.origin = {"ttest": [5, spinbox_significance_level.winfo_height() + spinbox_significance_level.winfo_y() + h_distance], "anova": [5, spinbox_significance_level.winfo_height() + spinbox_significance_level.winfo_y() + h_distance]}
        combo_pval_adjust = ttk.Combobox(frame.scrollable_frame, values = [c for c in lntt.pval_adjust_methods], width = 20)
        combo_pval_adjust.place(x = x_widgets, y = spinbox_significance_level.winfo_height() + spinbox_significance_level.winfo_y() + h_distance)
        combo_pval_adjust.current(0)
        combo_pval_adjust["state"] = "readonly"
        combo_pval_adjust.origin = {"ttest": [x_widgets, spinbox_significance_level.winfo_height() + spinbox_significance_level.winfo_y() + h_distance], "anova": [x_widgets, spinbox_significance_level.winfo_height() + spinbox_significance_level.winfo_y() + h_distance]}
        for i, condition in enumerate(combo_pval_adjust["values"]):
            if condition == stat_data["pval_adjust"]:
                combo_pval_adjust.current(i)
                break
        combo_pval_adjust.bind("<<ComboboxSelected>>", lambda x: self.change_combo(page_id, "pval_adjust", combo_pval_adjust))
        self.test_widgets[-1]["ttest"].append(label_pval_adjust)
        self.test_widgets[-1]["ttest"].append(combo_pval_adjust)
        self.test_widgets[-1]["anova"].append(label_pval_adjust)
        self.test_widgets[-1]["anova"].append(combo_pval_adjust)
        
        
        self.root.update()
        label_conditions_anova = tk.Label(frame.scrollable_frame, text = "Conditions for anova", justify = "right", anchor = "e")
        label_conditions_anova.place(x = 5, y = combo_pval_adjust.winfo_height() + combo_pval_adjust.winfo_y() + h_distance)
        label_conditions_anova["width"] = w_labels
        label_conditions_anova.origin = {"anova": [5, combo_pval_adjust.winfo_height() + combo_pval_adjust.winfo_y() + h_distance]}
        listbox_conditions_anova = tk.Listbox(frame.scrollable_frame, width = 20, height = 7, selectmode = "multiple", bg = "white")
        listbox_conditions_anova.place(x = x_widgets, y = combo_pval_adjust.winfo_height() + combo_pval_adjust.winfo_y() + h_distance)
        listbox_conditions_anova.origin = {"anova": [x_widgets, combo_pval_adjust.winfo_height() + combo_pval_adjust.winfo_y() + h_distance]}
        anova_set = set(stat_data["anova_conditions"] if "anova_conditions" in stat_data else [])
        if "conditions" in self.parameters:
            for i, condition in enumerate([self.listbox_conditions.get(i) for i in range(self.listbox_conditions.size())]):
                listbox_conditions_anova.insert("end", condition)
                if condition in anova_set: listbox_conditions_anova.select_set(first = i)
        listbox_conditions_anova.configure(exportselection = False)
        listbox_conditions_anova.bind("<<ListboxSelect>>", lambda event: self.conditions_anova_change(page_id, event))
        self.listboxes_anova.append(listbox_conditions_anova)
        self.test_widgets[-1]["anova"].append(label_conditions_anova)
        self.test_widgets[-1]["anova"].append(listbox_conditions_anova)
        
        self.root.update()
        label_condition_treated = tk.Label(frame.scrollable_frame, text = "Condition 'treated'", justify = "right", anchor = "e")
        label_condition_treated.place(x = 5, y = combo_pval_adjust.winfo_height() + combo_pval_adjust.winfo_y() + h_distance)
        label_condition_treated["width"] = w_labels
        label_condition_treated.origin = {"ttest": [5, combo_pval_adjust.winfo_height() + combo_pval_adjust.winfo_y() + h_distance]}
        combo_condition_treated = ttk.Combobox(frame.scrollable_frame, values = [self.listbox_conditions.get(i) for i in range(self.listbox_conditions.size())], width = 20)
        combo_condition_treated["state"] = "readonly"
        combo_condition_treated.place(x = x_widgets, y = combo_pval_adjust.winfo_height() + combo_pval_adjust.winfo_y() + h_distance)
        combo_condition_treated.origin = {"ttest": [x_widgets, combo_pval_adjust.winfo_height() + combo_pval_adjust.winfo_y() + h_distance]}
        if self.listbox_conditions.size() > 0: combo_condition_treated.current(0)
        if not new_entry:
            for i, condition in enumerate(combo_condition_treated["values"]):
                if condition == stat_data["condition_treated"]:
                    combo_condition_treated.current(i)
                    break
        combo_condition_treated.bind("<<ComboboxSelected>>", lambda x: self.change_combo(page_id, "condition_treated", combo_condition_treated))
        self.combos_treated.append(combo_condition_treated)
        self.test_widgets[-1]["ttest"].append(label_condition_treated)
        self.test_widgets[-1]["ttest"].append(combo_condition_treated)
        
        
        self.root.update()
        label_condition_wt = tk.Label(frame.scrollable_frame, text = "Condition 'control'", justify = "right", anchor = "e")
        label_condition_wt.place(x = 5, y = combo_condition_treated.winfo_height() + combo_condition_treated.winfo_y() + h_distance)
        label_condition_wt["width"] = w_labels
        label_condition_wt.origin = {"ttest": [5, combo_condition_treated.winfo_height() + combo_condition_treated.winfo_y() + h_distance]}
        combo_condition_wt = ttk.Combobox(frame.scrollable_frame, values = [self.listbox_conditions.get(i) for i in range(self.listbox_conditions.size())], width = 20)
        combo_condition_wt["state"] = "readonly"
        combo_condition_wt.place(x = x_widgets, y = combo_condition_treated.winfo_height() + combo_condition_treated.winfo_y() + h_distance)
        combo_condition_wt.origin = {"ttest": [x_widgets, combo_condition_treated.winfo_height() + combo_condition_treated.winfo_y() + h_distance]}
        if self.listbox_conditions.size() > 0: combo_condition_wt.current(0)
        if not new_entry:
            for i, condition in enumerate([self.listbox_conditions.get(i) for i in range(self.listbox_conditions.size())]):
                if condition == stat_data["condition_wt"]:
                    combo_condition_wt.current(i)
                    break
        combo_condition_wt.bind("<<ComboboxSelected>>", lambda x: self.change_combo(page_id, "condition_wt", combo_condition_wt))
        self.combos_wt.append(combo_condition_wt)
        self.test_widgets[-1]["ttest"].append(label_condition_wt)
        self.test_widgets[-1]["ttest"].append(combo_condition_wt)
        
        
        self.root.update()
        label_paired = tk.Label(frame.scrollable_frame, text = "Paired test", justify = "right", anchor = "e")
        label_paired.place(x = 5, y = combo_condition_wt.winfo_height() + combo_condition_wt.winfo_y() + h_distance)
        label_paired["width"] = w_labels
        label_paired.origin = {"ttest": [5, combo_condition_wt.winfo_height() + combo_condition_wt.winfo_y() + h_distance]}
        paired = tk.IntVar()
        checkbutton_paired = tk.Checkbutton(frame.scrollable_frame, variable = paired, command = lambda: self.change_checkbox(page_id, "ttest_paired", paired))
        checkbutton_paired.place(x = x_widgets, y = combo_condition_wt.winfo_height() + combo_condition_wt.winfo_y() + h_distance)
        checkbutton_paired.origin = {"ttest": [x_widgets, combo_condition_wt.winfo_height() + combo_condition_wt.winfo_y() + h_distance]}
        self.test_widgets[-1]["ttest"].append(label_paired)
        self.test_widgets[-1]["ttest"].append(checkbutton_paired)
        
        self.root.update()
        label_ttest_type = tk.Label(frame.scrollable_frame, text = "Ttest type", justify = "right", anchor = "e")
        label_ttest_type.place(x = 5, y = checkbutton_paired.winfo_height() + checkbutton_paired.winfo_y() + h_distance)
        label_ttest_type["width"] = w_labels
        label_ttest_type.origin = {"ttest": [5, checkbutton_paired.winfo_height() + checkbutton_paired.winfo_y() + h_distance]}
        combo_ttest_type = ttk.Combobox(frame.scrollable_frame, values = [t for t in lntt.ttest_types], width = 20)
        combo_ttest_type.place(x = x_widgets, y = checkbutton_paired.winfo_height() + checkbutton_paired.winfo_y() + h_distance)
        combo_ttest_type.current(0)
        combo_ttest_type["state"] = "readonly"
        combo_ttest_type.origin = {"ttest": [x_widgets, checkbutton_paired.winfo_height() + checkbutton_paired.winfo_y() + h_distance]}
        if not new_entry:
            for i, condition in enumerate(combo_ttest_type["values"]):
                if condition == stat_data["ttest_type"]:
                    combo_ttest_type.current(i)
                    break
        combo_ttest_type.bind("<<ComboboxSelected>>", lambda x: self.change_combo(page_id, "ttest_type", combo_ttest_type))
        self.test_widgets[-1]["ttest"].append(label_ttest_type)
        self.test_widgets[-1]["ttest"].append(combo_ttest_type)
        
        
        
        self.root.update()
        label_equal_varience = tk.Label(frame.scrollable_frame, text = "Equal variances", justify = "right", anchor = "e")
        label_equal_varience.place(x = 5, y = combo_ttest_type.winfo_height() + combo_ttest_type.winfo_y() + h_distance)
        label_equal_varience["width"] = w_labels
        label_equal_varience.origin = {"ttest": [5, combo_ttest_type.winfo_height() + combo_ttest_type.winfo_y() + h_distance]}
        equal_variance = tk.IntVar()
        if not new_entry and "ttest_equal_variance" in stat_data: equal_variance.set(stat_data["ttest_equal_variance"])
        checkbutton_equal_varience = tk.Checkbutton(frame.scrollable_frame, variable = equal_variance, command = lambda: self.change_checkbox(page_id, "ttest_equal_variance", equal_variance))
        checkbutton_equal_varience.place(x = x_widgets, y = combo_ttest_type.winfo_height() + combo_ttest_type.winfo_y() + h_distance)
        checkbutton_equal_varience.origin = {"ttest": [x_widgets, combo_ttest_type.winfo_height() + combo_ttest_type.winfo_y() + h_distance]} 
        self.test_widgets[-1]["ttest"].append(label_equal_varience)
        self.test_widgets[-1]["ttest"].append(checkbutton_equal_varience)
        
        
        
        self.root.update()
        label_log_values = tk.Label(frame.scrollable_frame, text = "Test on log values", justify = "right", anchor = "e")
        label_log_values.place(x = 5, y = checkbutton_equal_varience.winfo_height() + checkbutton_equal_varience.winfo_y() + h_distance)
        label_log_values["width"] = w_labels
        label_log_values.origin = {"ttest": [5, checkbutton_equal_varience.winfo_height() + checkbutton_equal_varience.winfo_y() + h_distance], "anova": [5, listbox_conditions_anova.winfo_height() + listbox_conditions_anova.winfo_y() + h_distance]}
        log_values = tk.IntVar()
        if not new_entry and "log_values" in stat_data: log_values.set(stat_data["log_values"])
        checkbutton_log_values = tk.Checkbutton(frame.scrollable_frame, variable = log_values, command = lambda: self.change_checkbox(page_id, "log_values", log_values))
        checkbutton_log_values.place(x = x_widgets, y = checkbutton_equal_varience.winfo_height() + checkbutton_equal_varience.winfo_y() + h_distance)
        checkbutton_log_values.origin = {"ttest": [x_widgets, checkbutton_equal_varience.winfo_height() + checkbutton_equal_varience.winfo_y() + h_distance], "anova": [x_widgets, listbox_conditions_anova.winfo_height() + listbox_conditions_anova.winfo_y() + h_distance]}
        self.test_widgets[-1]["ttest"].append(label_log_values)
        self.test_widgets[-1]["ttest"].append(checkbutton_log_values)
        self.test_widgets[-1]["anova"].append(label_log_values)
        self.test_widgets[-1]["anova"].append(checkbutton_log_values)
        

        
        
        self.root.update()
        label_with_test_plot = tk.Label(frame.scrollable_frame, text = "With plotting", justify = "right", anchor = "e")
        label_with_test_plot.place(x = 5, y = checkbutton_log_values.winfo_height() + checkbutton_log_values.winfo_y() + h_distance)
        label_with_test_plot["width"] = w_labels
        label_with_test_plot.origin = {"ttest": [5, checkbutton_log_values.winfo_height() + checkbutton_log_values.winfo_y() + h_distance], "anova": [5, checkbutton_log_values.winfo_height() + checkbutton_log_values.winfo_y() + h_distance]}
        with_test_plot = tk.IntVar()
        if not new_entry and "with_stat_test_plotting" in stat_data: with_test_plot.set(stat_data["with_stat_test_plotting"])
        checkbutton_with_test_plot = tk.Checkbutton(frame.scrollable_frame, variable = with_test_plot, command = lambda: self.change_checkbox(page_id, "with_stat_test_plotting", with_test_plot))
        checkbutton_with_test_plot.place(x = x_widgets, y = checkbutton_log_values.winfo_height() + checkbutton_log_values.winfo_y() + h_distance)
        checkbutton_with_test_plot.origin = {"ttest": [x_widgets, checkbutton_log_values.winfo_height() + checkbutton_log_values.winfo_y() + h_distance], "anova": [x_widgets, checkbutton_log_values.winfo_height() + checkbutton_log_values.winfo_y() + h_distance]}
        self.test_widgets[-1]["ttest"].append(label_with_test_plot)
        self.test_widgets[-1]["ttest"].append(checkbutton_with_test_plot)
        self.test_widgets[-1]["anova"].append(label_with_test_plot)
        self.test_widgets[-1]["anova"].append(checkbutton_with_test_plot)
        
        
        
        
        self.root.update()
        label_label_column = tk.Label(frame.scrollable_frame, text = "Column to be labeled", justify = "right", anchor = "e")
        label_label_column.place(x = 5, y = checkbutton_with_test_plot.winfo_height() + checkbutton_with_test_plot.winfo_y() + h_distance)
        label_label_column["width"] = w_labels
        label_label_column.origin = {"ttest": [5, checkbutton_with_test_plot.winfo_height() + checkbutton_with_test_plot.winfo_y() + h_distance]}
        label_columns = ["no labeling"]
        if self.genes.get() and "Genes" not in self.data_frame: label_columns.append("Genes")
        label_columns += [c for c in self.data_frame]
        combo_label_column = ttk.Combobox(frame.scrollable_frame, values = label_columns, width = 20)
        combo_label_column["state"] = "readonly"
        combo_label_column.place(x = x_widgets, y = checkbutton_with_test_plot.winfo_height() + checkbutton_with_test_plot.winfo_y() + h_distance)
        combo_label_column.origin = {"ttest": [x_widgets, checkbutton_with_test_plot.winfo_height() + checkbutton_with_test_plot.winfo_y() + h_distance]}
        combo_label_column.current(0)
        if not new_entry:
            if len(stat_data["labeling_column"]) > 1 and stat_data["labeling_column"] != "no labeling": 
                for i, condition in enumerate(combo_label_column["values"]):
                    if condition == stat_data["labeling_column"]:
                        combo_label_column.current(i)
                        break
        combo_label_column.bind("<<ComboboxSelected>>", lambda x: self.change_combo(page_id, "labeling_column", combo_label_column))
        self.combos_label_column.append(combo_label_column)
        self.test_widgets[-1]["ttest"].append(label_label_column)
        self.test_widgets[-1]["ttest"].append(combo_label_column)
        
        
        
        self.root.update()
        label_custom_labeling = tk.Label(frame.scrollable_frame, text = "Custom labeling", justify = "right", anchor = "e")
        label_custom_labeling.place(x = 5, y = combo_label_column.winfo_height() + combo_label_column.winfo_y() + h_distance)
        label_custom_labeling["width"] = w_labels
        label_custom_labeling.origin = {"ttest": [5, combo_label_column.winfo_height() + combo_label_column.winfo_y() + h_distance]}
        custom_labeling_list = ["all significant"]
        if self.genes.get() and "Genes" not in self.data_frame: custom_labeling_list.append("Custom on Genes")
        custom_labeling_list += ["Custom on %s" % c for c in self.data_frame]
        combo_custom_labeling = ttk.Combobox(frame.scrollable_frame, values = custom_labeling_list, width = 20)
        combo_custom_labeling["state"] = "readonly"
        combo_custom_labeling.place(x = x_widgets, y = combo_label_column.winfo_height() + combo_label_column.winfo_y() + h_distance)
        combo_custom_labeling.origin = {"ttest": [x_widgets, combo_label_column.winfo_height() + combo_label_column.winfo_y() + h_distance]}
        combo_custom_labeling.current(0)
        if not new_entry:
            if len(stat_data["custom_labeling"]) > 1 and stat_data["custom_labeling"] != "no labeling": 
                for i, condition in enumerate(combo_custom_labeling["values"]):
                    if condition == "Custom on %s" % stat_data["custom_labeling"]:
                        combo_custom_labeling.current(i)
                        break
        combo_custom_labeling.bind("<<ComboboxSelected>>", lambda x: self.change_combo(page_id, "custom_labeling", combo_custom_labeling))
        self.combos_custom_labeling.append(combo_custom_labeling)
        self.test_widgets[-1]["ttest"].append(label_custom_labeling)
        self.test_widgets[-1]["ttest"].append(combo_custom_labeling)
        
        
        
        self.root.update()
        label_custom_labels = tk.Label(frame.scrollable_frame, text = "Custom labels", justify = "right", anchor = "e")
        label_custom_labels.place(x = 5, y = combo_custom_labeling.winfo_height() + combo_custom_labeling.winfo_y() + h_distance)
        label_custom_labels["width"] = w_labels
        label_custom_labels.origin = {"ttest": [5, combo_custom_labeling.winfo_height() + combo_custom_labeling.winfo_y() + h_distance]}
        text_custom_labels = TextScrollCombo(frame.scrollable_frame, width = 156, height = 120)
        text_custom_labels.place(x = x_widgets, y = combo_custom_labeling.winfo_height() + combo_custom_labeling.winfo_y() + h_distance)
        text_custom_labels.origin = {"ttest": [x_widgets, combo_custom_labeling.winfo_height() + combo_custom_labeling.winfo_y() + h_distance]}
        self.texts_custom_labels.append(text_custom_labels)
        if not new_entry:
            l = len(stat_data["custom_labels"])
            for line, custom_label in enumerate(stat_data["custom_labels"]): 
                text_custom_labels.insert(tk.END, "%s%s" % (custom_label, "\n" if line + 1 < l else ""))
                
        self.enable_custom_labels(page_id, combo_custom_labeling)
        text_custom_labels.txt.bind("<KeyRelease>", lambda x: self.change_text(page_id, "custom_labels", text_custom_labels))
        self.test_widgets[-1]["ttest"].append(label_custom_labels)
        self.test_widgets[-1]["ttest"].append(text_custom_labels)
        
        
        
        
        
        self.root.update()
        label_log_fc = tk.Label(frame.scrollable_frame, text = "Log_2 fold change thr.", justify = "right", anchor = "e")
        label_log_fc.place(x = 5, y = text_custom_labels.winfo_height() + text_custom_labels.winfo_y() + h_distance)
        label_log_fc["width"] = w_labels
        label_log_fc.origin = {"ttest": [5, text_custom_labels.winfo_height() + text_custom_labels.winfo_y() + h_distance]}
        log_fc = tk.DoubleVar()
        log_fc.trace('w', lambda name, index, mode, log_fc = log_fc: self.change_log_fc(page_id, log_fc))
        parameter_fc = stat_data["log_fc_thresholds"][1] if "log_fc_thresholds" in stat_data and not new_entry else 1
        spinbox_log_fc = tk.Spinbox(frame.scrollable_frame, from_ = 0, to = 10, width = 20, bg = "white", increment = 0.1, textvariable = log_fc)
        spinbox_log_fc["state"] = "readonly"
        spinbox_log_fc.place(x = x_widgets, y = text_custom_labels.winfo_height() + text_custom_labels.winfo_y() + h_distance)
        spinbox_log_fc.origin = {"ttest": [x_widgets, text_custom_labels.winfo_height() + text_custom_labels.winfo_y() + h_distance]}
        log_fc.set(parameter_fc)
        self.root.update()
        label_plus_minus = tk.Label(frame.scrollable_frame, text = "±")
        label_plus_minus.place(x = spinbox_log_fc.winfo_x() - 10, y = spinbox_log_fc.winfo_y())
        label_plus_minus.origin = {"ttest": [spinbox_log_fc.winfo_x() - 10, spinbox_log_fc.winfo_y()]}
        self.test_widgets[-1]["ttest"].append(label_log_fc)
        self.test_widgets[-1]["ttest"].append(spinbox_log_fc)
        self.test_widgets[-1]["ttest"].append(label_plus_minus)
        
        
        frame.bind_all_event()
        self.show_current_test(page_id, current_test)
        
        
        self.root.update()
        max_height = 100
        for testing_type in self.test_widgets[-1]:
            for widget in self.test_widgets[-1][testing_type]:
                max_height = max(max_height, widget.winfo_height() + widget.winfo_y())
        inner_frame["height"] = max_height + 3 * h_distance
        
        
        
    def conditions_anova_change(self, test_id, event = None):
        self.parameters['statistical_testing'][test_id]["anova_conditions"] = []
        for i in self.listboxes_anova[test_id].curselection():
            self.parameters['statistical_testing'][test_id]["anova_conditions"].append(self.listboxes_anova[test_id].get(i))
            
      
        
    def change_log_fc(self, test_id, number):
        self.parameters["statistical_testing"][test_id]["log_fc_thresholds"] = [-number.get(), number.get()]
        
        
        
    def change_significance_level(self, test_id, number):
        self.parameters["statistical_testing"][test_id]["significance_level"] = number.get()
        
        
        
    def change_checkbox(self, test_id, par, var):
        self.parameters["statistical_testing"][test_id][par] = var.get()
        
        
        
    def change_combo(self, test_id, par, combo):
        value = combo["values"][combo.current()]
        if par in {"custom_labeling", "labeling_column"} and combo.current() == 0: value = ""
        if par == "custom_labeling": value = value.replace("Custom on ", "")
        self.parameters["statistical_testing"][test_id][par] = value
        
        if par == "test": self.show_current_test(test_id, combo["values"][combo.current()])
        elif par == "custom_labeling":
            self.enable_custom_labels(test_id, combo)
            
            
            
    def change_text(self, test_id, par, text_field):
        self.parameters["statistical_testing"][test_id][par] = text_field.txt.get("1.0", tk.END).split("\n")[:-1]
        
        
    
    def enable_custom_labels(self, test_id, combo):
        if combo.current() == 0:
            self.texts_custom_labels[test_id].txt["bg"] = "#dddddd"
            self.texts_custom_labels[test_id].txt["state"] = "disabled"
            
        else:
            self.texts_custom_labels[test_id].txt["bg"] = "#ffffff"
            self.texts_custom_labels[test_id].txt["state"] = "normal"
        
        
        
    def show_current_test(self, test_id, test_type):
        for testing_type in self.test_widgets[test_id]:
            for widget in self.test_widgets[test_id][testing_type]: widget.place_forget()
        
        for widget in self.test_widgets[test_id][test_type]:
            x, y = widget.origin[test_type]
            widget.place(x = x, y = y)
            
    
    
    def delete_test(self):
        if len(self.notebook.tabs()) > 0:
            tab_id = self.notebook.select()
            tab_index = self.notebook.index(tab_id)
            self.notebook.forget(tab_id)
            self.parameters["statistical_testing"].pop(tab_index)
            self.combos_label_column.pop(tab_index)
            self.combos_custom_labeling.pop(tab_index)
            self.texts_custom_labels.pop(tab_index)
            self.combos_treated.pop(tab_index)
            self.combos_wt.pop(tab_index)
            self.test_widgets.pop(tab_index)
            self.listboxes_anova.pop(tab_index)
            for i, tab in enumerate(self.notebook.tabs()):
                self.notebook.tab(tab, text = "Test %i" % (i + 1))
          
    
    
    
if __name__ == "__main__":
    lntt_queue = multiprocessing.Queue()
    main_queue = multiprocessing.Queue()
    lntt_instance = lntt.LNTT(lntt_queue, main_queue)
    lntt_instance.start()

    
    
    welcome = Welcome(tk.Tk(), [lntt_queue, main_queue])
    welcome.mainloop()
    
    """
    
    parameters = json.loads(open("test.lntt", "rt").read())
    lntt_gui = LNTT_GUI(tk.Tk(), parameters, [lntt_queue, main_queue])
    lntt_gui.mainloop()

    """
