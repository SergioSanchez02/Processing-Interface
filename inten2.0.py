import sys
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from PyQt4 import uic, QtGui, QtCore
import scipy.ndimage as sp  
import scipy.misc as sv
from scipy.ndimage.filters import gaussian_filter as gaussian
from tkinter.filedialog import askopenfilename
from math import *
from tkinter import ttk
from tkinter import *

global img
img=[]
global img3
img3=[]
global img2
img2=[]

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)
################################ Diffusion Anisotropic Filter ##########################################
def forwardDifferenceGradient(imgF):  
  diffY = np.zeros_like(imgF)  
  diffX = np.zeros_like(imgF)  
  diffY[:-1, :] = np.diff(imgF, axis = 0)  
  diffX[:, :-1] = np.diff(imgF, axis = 1)  
  return diffY, diffX

def sigmoidFormula(gradientY, gradientX, k):  
  cGradientY = np.exp(-(gradientY/k) **2.)  
  cGradientX = np.exp(-(gradientX/k) **2.)  
  YVal = cGradientY * gradientY  
  XVal = cGradientX * gradientX  
  return YVal, XVal  

def tanhFormula(gradientY, gradientX, k):  
  cGradientY = 1./(1. + ((gradientY/k)**2))  
  cGradientX = 1./(1. + ((gradientX/k)**2))  
  YVal = cGradientY * gradientY  
  XVal = cGradientX * gradientX  
  return YVal, XVal 
##########################################################################################################

class ImageProcessing2(QtGui.QDialog):
    
    def __init__(self):
        QtGui.QDialog.__init__(self)
        # Set up the user interface from Designer.
        self.ui = uic.loadUi('./ImageProcessing2.0.ui')
        self.ui.show()        
        self.connect(self.ui.actionAnisotropic_Filter, QtCore.SIGNAL("triggered()"),lambda:self.DiffAnisFilt())
        self.connect(self.ui.actionLoad, QtCore.SIGNAL("triggered()"),lambda:self.Load())
        self.connect(self.ui.actionGaussian_Filter, QtCore.SIGNAL("triggered()"),lambda:self.Convolution())
        self.connect(self.ui.actionAveraging, QtCore.SIGNAL("triggered()"),lambda:self.Averaging())
        self.connect(self.ui.actionGaussian_Filter_2, QtCore.SIGNAL("triggered()"),lambda:self.Gaussian())
        self.connect(self.ui.actionMedian_Filter, QtCore.SIGNAL("triggered()"),lambda:self.MedianFilter())
        self.connect(self.ui.actionBilateral_Filter, QtCore.SIGNAL("triggered()"),lambda:self.Bilateral())
        self.connect(self.ui.actionErosion, QtCore.SIGNAL("triggered()"),lambda:self.Erosion())
        self.connect(self.ui.actionDilation, QtCore.SIGNAL("triggered()"),lambda:self.Dilation())
        self.connect(self.ui.actionOpening, QtCore.SIGNAL("triggered()"),lambda:self.Opening())
        self.connect(self.ui.actionClosing, QtCore.SIGNAL("triggered()"),lambda:self.Closing())
        self.connect(self.ui.actionGradient, QtCore.SIGNAL("triggered()"),lambda:self.Gradient())
        self.connect(self.ui.actionCanny_Edge_Detection, QtCore.SIGNAL("triggered()"),lambda:self.Canny())
        #self.connect(self.ui., QtCore.SIGNAL("triggered()"),lambda:self.Load())
        self.connect(self.ui.actionSobel_X, QtCore.SIGNAL("triggered()"),lambda:self.SobelX())
        self.connect(self.ui.actionSobel_Y, QtCore.SIGNAL("triggered()"),lambda:self.SobelY())
        
        self.connect(self.ui.pushActImage, QtCore.SIGNAL("clicked()"),lambda:self.ActImage())    
        self.connect(self.ui.pushHistogram, QtCore.SIGNAL("clicked()"),lambda:self.Histogram())
        self.connect(self.ui.pushSegmentation, QtCore.SIGNAL("clicked()"),lambda:self.Segmentation())
        self.connect(self.ui.pushUndo, QtCore.SIGNAL("clicked()"),lambda:self.Undo())
        self.connect(self.ui.pushPrevImage, QtCore.SIGNAL("clicked()"),lambda:self.PrevImage())

    def Undo(self):
        global img2
        global img3
        img3=img2
    def Segmentation(self):
        global img3
        global img2
        global img
        thresh4  =cv2.adaptiveThreshold(img3,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        blur = cv2.GaussianBlur(thresh4,(25,25),0)
        ret3,img3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)    
        plt.imshow(img3,'gray') 
        plt.xticks([]), plt.yticks([])
        plt.show()
        
    def Histogram(self):
        global img3
        global img2
        global img
        #plt.grid(True)
        plt.hist(img3.ravel(),256)
        plt.show()
        
    def SobelY(self):
        global img3
        global img2
        global img
        if img2==[]:
            img2=img
        else:
            img2=img3
        transformacion = cv2.Sobel(img2,cv2.CV_64F,0,1,ksize=5)
        img3=transformacion        
    def SobelX(self):
        global img3
        global img2
        global img
        if img2==[]:
            img2=img
        else:
            img2=img3
        transformacion = cv2.Sobel(img2,cv2.CV_64F,1,0,ksize=5)
        img3=transformacion
        
    def Canny(self):
        global img3
        global img2
        global img
        if img2==[]:
            img2=img
        else:
            img2=img3
        kernel = np.ones((5,5),np.uint8)
        transformacion = cv2.Canny(img2,20,40)
        img3=transformacion
        
    def Gradient(self):
        global img3
        global img2
        global img
        if img2==[]:
            img2=img
        else:
            img2=img3
        kernel = np.ones((5,5),np.uint8)
        transformacion = cv2.morphologyEx(img2, cv2.MORPH_GRADIENT, kernel)
        img3=transformacion        
    def Closing(self):
        global img3
        global img2
        global img
        if img2==[]:
            img2=img
        else:
            img2=img3
        kernel = np.ones((5,5),np.uint8)
        transformacion = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel)
        img3=transformacion
    def Opening(self):
        global img3
        global img2
        global img
        if img2==[]:
            img2=img
        else:
            img2=img3
        kernel = np.ones((5,5),np.uint8)
        transformacion = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel)
        img3=transformacion
        
    def Erosion(self):
        global img3
        global img2
        global img
        if img2==[]:
            img2=img
        else:
            img2=img3
        kernel = np.ones((5,5),np.uint8)
        transformacion = cv2.erode(img2,kernel,2) 
        img3=transformacion
    
    def Bilateral(self):
        global img3
        global img2
        global img
        if img2==[]:
            img2=img
        else:
            img2=img3
        transformacion = cv2.bilateralFilter(img2,9,75,75)
        img3=transformacion
        
    def MedianFilter(self):
        global img3
        global img2
        global img
        if img2==[]:
            img2=img
        else:
            img2=img3
        transformacion = cv2.medianBlur(img2,5)
        img3=transformacion
        
    def Gaussian(self):
        global img3
        global img2
        global img
        if img2==[]:
            img2=img
        else:
            img2=img3
        transformacion = cv2.GaussianBlur(img2,(5,5),0)
        img3=transformacion
        
    def Averaging(self):
        global img3
        global img2
        global img
        if img2==[]:
            img2=img
        else:
            img2=img3
        transformacion = cv2.blur(img2,(5,5))
        img3=transformacion
        
    def Convolution(self):
        global img3
        global img2
        global img
        if img2==[]:
            img2=img
        else:
            img2=img3
        kernel = np.ones((5,5),np.float32)/25
        transformacion = cv2.filter2D(img2,-1,kernel)
        img3=transformacion
        
    def Load(self):
        global archivo
        ventana=Tk()
        archivo=askopenfilename()
        ventana.destroy()
        global img
        global img2
        global img3
        img2=[]
        img3=[]
        img = cv2.imread(archivo,0)
        #return archivo  

    def DiffAnisFilt(self):
        global archivo
        global img3
        global img2
        global img
        if img2==[]:
            img2=img
        else:
            img2=img3

        #sv.imsave('anisotropic.jpg', img2)    
        img = sp.imread(archivo, mode = "L")
        img = img.astype("float32")  
        shiftedY = np.zeros_like(img)  
        shiftedX = np.zeros_like(img)  
        for i in range(10):  
            dY, dX = forwardDifferenceGradient(img)  
            cY, cX = sigmoidFormula(dY, dX, 20)  
            shiftedY[:] = cY  
            shiftedX[:] = cX  
            shiftedY[1:,:] -= cY[:-1,:]  
            shiftedX[:,1:] -= cX[:,:-1]  
            img += 0.25*(shiftedY+shiftedX)
        img3=img
    
    def ActImage(self):
          global img3
          global img2
          global img
          if (img3==[]):
              plt.imshow(img,'gray')
          else:
              plt.imshow(img3,'gray')     
          plt.xticks([]), plt.yticks([])
          plt.show()

    def PrevImage(self):
          global img3
          global img2
          global img
          plt.imshow(img2,'gray')     
          plt.xticks([]), plt.yticks([])
          plt.show()

    def Dilation(self):
        global img3
        global img2
        global img
        if img2==[]:
            img2=img
        else:
            img2=img3
        #ventana=input('ingrese el tamaño de la ventana ')
        #iteraciones=input('ingrese el numero de iteraciones ')
        kernel = np.ones((5,5),np.uint8)
        transformacion = cv2.dilate(img2,kernel,2) 
        img3=transformacion
        
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    ImageProcessing2()
    sys.exit(app.exec_())
