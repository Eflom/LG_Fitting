# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 19:27:20 2017

@author: EFlom
"""
import easygui
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()