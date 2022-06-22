# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 09:13:46 2022

@author: AOG
"""

import tkinter as tk
import tkinter.filedialog
import tkinter.scrolledtext
import numpy as np
import time
import cv2 as cv
import math

from PIL import ImageTk, Image  

from pybundle import PyBundle


class pybundle_gui(tk.Frame):
    
    rawPreviewImage = None
    rawPreviewSize = 300
    
    procPreviewImage = None
    procPreviewSize = 300
    
    calibPreviewImage = None
    calibPreviewSize = 300
    
    rawImage = None
    calibImage = None
    reconImage = None
    
    coreSize = 3
    gridSize = 400
    
    def load_raw(self):
        
        filename = tk.filedialog.askopenfilename()
        self.statusText.insert(tk.INSERT, "Loading Raw Image: " + filename + "\n")
        loadImage = Image.open(filename)
        self.rawImage = np.array(loadImage)
        
        maxDim = max(np.shape(self.rawImage))
        newW = math.floor(np.shape(self.rawImage)[1] / maxDim * self.calibPreviewSize)
        newH = math.floor(np.shape(self.rawImage)[0] / maxDim * self.calibPreviewSize)
        
        loadImage = loadImage.resize((newW, newH))

        self.rawPreviewImage = ImageTk.PhotoImage(loadImage)
        self.rawImageDisp.configure(image = self.rawPreviewImage) 
     
    def load_calib(self):
        filename = tk.filedialog.askopenfilename()
        self.statusText.insert(tk.INSERT, "Loading Calibration Image: " + filename + "\n")

        loadImage = Image.open(filename)
      
        self.calibImage = np.array(loadImage)
        
        maxDim = max(np.shape(self.calibImage))
        newW = math.floor(np.shape(self.calibImage)[1] / maxDim * self.calibPreviewSize)
        newH = math.floor(np.shape(self.calibImage)[0] / maxDim * self.calibPreviewSize)
        
        loadImage = loadImage.resize((newW, newH))
        #self.calibImage = cv.imread(filename)[:,:,0]
  
        self.calibPreviewImage = ImageTk.PhotoImage(loadImage)
        self.calibImageDisp.configure(image = self.calibPreviewImage) 
        
        
    def calibrate(self):
        #calibTriInterp(img, coreSize, gridSize, **kwargs):
               
        #centreX = kwargs.get('centreX', -1)
        #centreY = kwargs.get('centreY', -1)
        #radius = kwargs.get('radius', -1)
        #filterSize = kwargs.get('filterSize', 0)
        #normalise = kwargs.get('normalise', None)
        #autoMask = kwargs.get('autoMask', False)
        #background = kwargs.get('background', None)
        self.statusText.insert(tk.INSERT, "Starting Calibration... \n")
        time.sleep(0.5)
        if self.useBackground.get() == 1:
            background = self.calibImage
        else:
            background = None
        self.calibration = PyBundle.calibTriInterp(self.calibImage, self.coreSize, self.gridSize, background=background, autoMask=True)
        self.statusText.insert(tk.INSERT, "Found " + str(len(self.calibration[0])) + " cores. \n")

    def reconstruct(self):
        self.reconImage = PyBundle.reconTriInterp(self.rawImage, self.calibration)
        self.statusText.insert(tk.INSERT, "Performing reconstrucion. \n")

        displayImage = Image.fromarray(PyBundle.to8bit(self.reconImage))
        displayImage = displayImage.resize((self.procPreviewSize,self.procPreviewSize))
        self.procPreviewImage = ImageTk.PhotoImage(displayImage)
        self.procImageDisp.configure(image = self.procPreviewImage) 

    def save_recon(self):
        filename = tk.filedialog.asksaveasfilename(filetypes=[("Tiff", ".tif")], defaultextension=".tif")
        self.statusText.insert(tk.INSERT, "Saving reconstrucion to: " + filename + " \n")

        saveImage = Image.fromarray(PyBundle.to8bit(self.reconImage))
        saveImage.save(filename)
        
        

    def save_background(self):
        
        minV = np.min(self.reconImage)
        maxV = np.max(self.reconImage)
        
        filename = tk.filedialog.asksaveasfilename(filetypes=[("Tiff", ".tif")], defaultextension=".tif")
        calibration2  = self.calibration
        calibration2 = list(calibration2)
        calibration2[9] = None
        calibration2 = tuple(calibration2)
        self.calibImageRecon = PyBundle.reconTriInterp(self.calibImage, calibration2)
        saveImage = Image.fromarray(PyBundle.to8bit(self.calibImageRecon, minVal = minV, maxVal = maxV))
        self.statusText.insert(tk.INSERT, "Saving backgriund to: " + filename + " using min = " + str(minV) + ", min = " + str(maxV) + "\n")

        saveImage.save(filename)

        
        
        
        
    def __init__(self, root):
        tk.Frame.__init__(self, root)
        root.geometry("1400x500")
        
        #image1 = Image.open("<path/image_name>")
        blankImage = Image.new('L', (self.rawPreviewSize, self.rawPreviewSize))
        self.rawPreviewImage = ImageTk.PhotoImage(blankImage)
        
        blankImage = Image.new('L', (self.calibPreviewSize, self.calibPreviewSize))
        self.calibPreviewImage = ImageTk.PhotoImage(blankImage)
        
        blankImage = Image.new('L', (self.procPreviewSize, self.procPreviewSize))
        self.procPreviewImage = ImageTk.PhotoImage(blankImage)
        
        
        
        self.rawImageTitle = tk.Label(text="Raw Image")
        self.rawImageTitle.grid(row = 1, column = 1)
        
        self.rawImageTitle = tk.Label(text="Calibration Image")
        self.rawImageTitle.grid(row = 1, column = 2)
        
        self.rawImageTitle = tk.Label(text="Processed Image")
        self.rawImageTitle.grid(row = 1, column = 3)

        
        self.rawImageDisp = tk.Label(image = self.rawPreviewImage)
        self.rawImageDisp.grid(row = 2, column = 1, rowspan = 6)
        self.calibImageDisp = tk.Label(image = self.calibPreviewImage)
        self.calibImageDisp.grid(row = 2, column = 2, rowspan = 6)
        self.procImageDisp = tk.Label(image = self.procPreviewImage)
        self.procImageDisp.grid(row = 2, column = 3, rowspan = 6)

        
        
        loadRawButton = tk.Button(
            text="Load Raw Image(s)",
            width=25,
            height=1,
            command = self.load_raw) 
        
        
        loadBackgroundButton = tk.Button(
            text="Load Background Image",
            width=25,
            height=1,
            command = self.load_calib) 
        
        
        calibrateButton = tk.Button(
            text="Calibrate",
            width=25,
            height=1,
            command = self.calibrate)  
        
        processButton = tk.Button(
            text="Process",
            width=25,
            height=1,
            command = self.reconstruct)  
        
        saveProcessedButton = tk.Button(   
            command = self.save_recon,
            text="Saved Processed Image(s)",
            width=25,
            height=1) 
        
        saveBackgroundButton = tk.Button(   
            command = self.save_background,
            text="Saved Reference Background",
            width=25,
            height=1) 
        self.useBackground = tk.IntVar()
        useBackground = tk.Checkbutton(text = "Subtrct Background", variable = self.useBackground)
        
        
        loadRawButton.grid(row = 2, column = 4, padx = 20)
        loadBackgroundButton.grid(row = 3, column = 4, padx = 20)
        calibrateButton.grid(row = 4, column = 4, padx = 20)
        processButton.grid(row = 5, column = 4, padx = 20)
        saveProcessedButton.grid(row = 6, column = 4, padx = 20)
        saveBackgroundButton.grid(row = 7, column = 4, padx = 20)
        useBackground.grid(row = 1, column = 5, padx = 20)
        
        
        self.statusText = tk.scrolledtext.ScrolledText(wrap = tk.WORD, 
                                      width = 120, 
                                      height = 8, 
                                      font = ("Verdana",
                                              10))
        self.statusText.grid(row = 8, column = 1, columnspan = 3, sticky="W", padx = 20, pady = 20)
         
        
        
        
        
        
if __name__=='__main__':
    root = tk.Tk()
    view = pybundle_gui(root)
    #view.pack(side="top", fill="both", expand=True)
    root.mainloop()