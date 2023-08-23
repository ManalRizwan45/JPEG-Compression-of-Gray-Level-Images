import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
import tkinter as tk
from tkinter import filedialog
import time
import random
import scipy.fftpack
from scipy.fftpack import idct
from scipy.fftpack import dct
from PIL import Image,ImageTk
from tkinter import font

luminanceTable=np.array([
    [16,11,10,16,24,15,1,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,81,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
])


def load_greyimage():
    file_path = filedialog.askopenfilename()
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return img


img = load_greyimage()
h = img.shape[0]
w = img.shape[1]

if (h!=w):
    if h>w:
        newRes=h
    else:
        newRes=w   
    paddedimg=np.zeros((newRes,newRes))
    for i in range(h):
        for j in range(w):
            paddedimg[i,j]=img[i,j]
    img=np.copy(paddedimg)
    h=newRes
    w=newRes



if(h%8!=0):
    newRes=h+(abs((h%8)-8))
   
    paddedimg=np.zeros((newRes,newRes))
    for i in range(h):
        for j in range(w):
            paddedimg[i,j]=img[i,j]
    img=np.copy(paddedimg)
    h=newRes
    w=newRes
    print(f"h{h} newRes{newRes}")

def levelshift(): 
      #centered at 0 [-128 to +127]  

   levelShiftedImg = np.zeros((h, w))

   for i in range(h):
      for j in range(w):
       levelShiftedImg[i,j] = img[i,j] - 128
    
   return levelShiftedImg


def DCT(levelShiftedImg):
   
    DCTApplied = np.zeros((h, w))

    
   
    for i in range (0,h,8):   #8*8 window
        for j in range (0,w,8):
           DCTApplied[i:i+8,j:j+8]=scipy.fftpack.dct(levelShiftedImg[i:i+8,j:j+8],type=2)
           
        
    
    return DCTApplied

def inverseDCT(dequantized_img):
    
    InverseDCTApplied = np.zeros((h, w))

    for i in range (0,h,8):   #8*8 window
        for j in range (0,w,8):
            InverseDCTApplied[i:i+8,j:j+8]=scipy.fftpack.idct(dequantized_img[i:i+8,j:j+8],type=2)

    

    return InverseDCTApplied

def myQuantizeImage(img):
    quantizedIMG=np.zeros((h,w))
    for i in range (0,h,8):   #8*8 window
        for j in range (0,w,8):
            quantizedIMG[i:i+8,j:j+8]=np.divide(img[i:i+8,j:j+8],luminanceTable).astype(int)
    
    return quantizedIMG

def myDeQuantize(img):
    dequantizedIMG=np.zeros((h,w))
    for i in range (0,h,8):   #8*8 window
        for j in range (0,w,8):
            try:
               dequantizedIMG[i:i+8,j:j+8]=np.multiply(img[i:i+8,j:j+8],luminanceTable).astype(int)
            except:
                print(f"problem at{i},{j}\n")
    return dequantizedIMG


def find_q(img, windowsize, mean):
    count = 0
    for i in range(0, windowsize, 1):
        for j in range(0, windowsize, 1):
            try:
                if img[i][j] > mean:
                    count = count + 1
            except IndexError as e:
                print(f'{e} i={i}, j={j}')
    return count

def encode(quantized_img):
    windowsize = 4

    # intializing the bit plane
    m = np.zeros((h, w), dtype=bool)

    valuesofHL = []

    for i in range(0, h, windowsize):
        for j in range(0, w, windowsize):

            # local window containing 4*4 non-overlapping pixels
            localmean = quantized_img[i:i + windowsize, j:j + windowsize].mean()
            localstd = quantized_img[i:i + windowsize, j:j + windowsize].std()

            q = find_q(quantized_img[i:i + windowsize, j:j + windowsize], windowsize, localmean)
            if q != 0:
                H = localmean + (localstd * ((((windowsize ** 2) - q) / q) ** (1 / 2)))
                L = localmean - (localstd * ((q / ((windowsize ** 2) - q)) ** (1 / 2)))
            else:
                H = localmean
                L = localmean

            valuesofHL.append((H, L))
            
            for k in range(i, i + windowsize, 1):  # assiging values of 0 and 1
                for n in range(j, j + windowsize, 1):
                    if quantized_img[k][n] >= localmean:
                        m[k][n] = True
                    else:
                        m[k][n] = False


    compressed_m = np.packbits(m)
    return compressed_m, valuesofHL

def decode(compressed_m, windowsize, valuesofHL):
    # decompressing the binary array using numpy.unpackbits()
    m = np.unpackbits(compressed_m)
    m = m.reshape((h, w))

    decompressed_img = np.zeros([h, w], dtype=np.uint8)
    counter = 0

    for i in range(0, h, windowsize):
        for j in range(0, w, windowsize):

            local_H = valuesofHL[counter][0]
            local_L = valuesofHL[counter][1]

            for k in range(i, i + windowsize, 1):
                for n in range(j, j + windowsize, 1):
                    if m[k][n] == True:
                     decompressed_img[k][n] = local_H
                    else:
                     decompressed_img[k][n] = local_L

            counter = counter + 1

    return decompressed_img

def myDelevelshift(img):
    range=255
    min=np.amin(img)
    max=np.amax(img)

    normalized=((img-min)/range).astype(int)
    scaled=normalized*255

    return normalized

def histogramEqualize(matrix):
    row,col=matrix.shape
    newMatrix=np.zeros((row,col))
    histogram=[]  #indexes as gray levels and values as frequencey
    for i in range(256):
        histogram.append(0)

    for i in range(row):
        for j in range(col):
            histogram[int(matrix[i,j])]+=1
           
    summ=sum(histogram)
    cumSum=[]
    for i in range(len(histogram)):
        cumSum.append(0)
    cumSum[0]=histogram[0]
    for i in range(1,256):
        cumSum[i]=cumSum[i-1]+histogram[i]
    for i in range(len(cumSum)):
        cumSum[i]=math.ceil((cumSum[i]/summ)*(len(histogram)-1))

    for i in range(row):
        for j in range(col):
            newMatrix[i,j]=cumSum[int(matrix[i,j])]
    return newMatrix

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0  # Assuming the pixel range is 0-255
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr



level_shifted_img=levelshift()


DCT_img=DCT(level_shifted_img)


quantized_img=myQuantizeImage(DCT_img)

encoded_img,valuesofHL=encode(quantized_img)


decoded_img=decode(encoded_img,4,valuesofHL)


dequantized_img=myDeQuantize(decoded_img)



inverse_DCT=inverseDCT(DCT_img)


de_level_shifted=myDelevelshift(inverse_DCT)


equalized_img=histogramEqualize(de_level_shifted)



class mygui:
    imageLabel=None
    def __init__(self,root):
        self.root=root
        
        self.imageLabel1=tk.Label(self.root)
        self.imageLabel2=tk.Label(self.root)
        self.imageLabel3=tk.Label(self.root)
        self.imageLabel4=tk.Label(self.root)
        self.imageLabel5=tk.Label(self.root)
        self.imageLabel6=tk.Label(self.root)
        self.imageLabel7=tk.Label(self.root)
        self.imageLabel8=tk.Label(self.root)
        self.imageLabel9=tk.Label(self.root)
        self.imageLabel10=tk.Label(self.root)
        
        self.inputInfo1=tk.Label(root)
        self.inputInfo2=tk.Label(root)
        self.inputInfo3=tk.Label(root)
        self.inputInfo4=tk.Label(root)
        self.inputInfo5=tk.Label(root)
        self.inputInfo6=tk.Label(root)
        self.inputInfo7=tk.Label(root)
        self.inputInfo8=tk.Label(root)
        self.inputInfo9=tk.Label(root)
        self.inputInfo10=tk.Label(root)

        self.Info1=tk.Label(root)
        self.Info2=tk.Label(root)
        self.Info3=tk.Label(root)
        self.Info4=tk.Label(root)


        self.imageLabel1.grid(row=1,column=0)
        self.imageLabel2.grid(row=1,column=1)
        self.imageLabel3.grid(row=1,column=2)
        self.imageLabel4.grid(row=3,column=0)
        self.imageLabel5.grid(row=3,column=1)
        self.imageLabel6.grid(row=3,column=2)
        self.imageLabel7.grid(row=5,column=0)
        self.imageLabel8.grid(row=5,column=1)
        self.imageLabel9.grid(row=5,column=2)
        self.imageLabel10.grid(row=7,column=1)
        
        self.inputInfo1.grid(row=2,column=0)
        self.inputInfo1.configure(text="Original img")
        self.inputInfo2.grid(row=2,column=1)
        self.inputInfo2.configure(text="Level shifted img")
        self.inputInfo3.grid(row=2,column=2)
        self.inputInfo3.configure(text="Quantized img")
        self.inputInfo4.grid(row=4,column=0)
        self.inputInfo4.configure(text="DCT img")
        self.inputInfo5.grid(row=4,column=1)
        self.inputInfo5.configure(text="Encoded img")
        self.inputInfo6.grid(row=4,column=2)
        self.inputInfo6.configure(text="Decoded img")
        self.inputInfo7.grid(row=6,column=0)
        self.inputInfo7.configure(text="Dequantized img")
        self.inputInfo8.grid(row=6,column=1)
        self.inputInfo8.configure(text="Inverse DCT img")
        self.inputInfo9.grid(row=6,column=2)
        self.inputInfo9.configure(text="Delevel Shifted img")
        self.inputInfo10.grid(row=8,column=1)
        self.inputInfo10.configure(text="Histogram Equalized img")
        

        self.Info1.grid(row=9,column=1)
        self.Info2.grid(row=10,column=1)
        self.Info3.grid(row=11,column=1)
        self.Info4.grid(row=12,column=1)


    def showImage(self,image1,image2,image3,image4,image5,image6,image7,image8,image9,image10):
        img1=Image.fromarray(image1)
        self.photo1=ImageTk.PhotoImage(image=img1)
        self.imageLabel1.configure(image=self.photo1)

        img2=Image.fromarray(image2)
        self.photo2=ImageTk.PhotoImage(image=img2)
        self.imageLabel2.configure(image=self.photo2)

        img3=Image.fromarray(image3)
        self.photo3=ImageTk.PhotoImage(image=img3)
        self.imageLabel3.configure(image=self.photo3)

        img4=Image.fromarray(image4)
        self.photo4=ImageTk.PhotoImage(image=img4)
        self.imageLabel4.configure(image=self.photo4)

        img6=Image.fromarray(image6)
        self.photo6=ImageTk.PhotoImage(image=img6)
        self.imageLabel6.configure(image=self.photo6)

        img7=Image.fromarray(image7)
        self.photo7=ImageTk.PhotoImage(image=img7)
        self.imageLabel7.configure(image=self.photo7)

        img8=Image.fromarray(image8)
        self.photo8=ImageTk.PhotoImage(image=img8)
        self.imageLabel8.configure(image=self.photo8)

        img9=Image.fromarray(image9)
        self.photo9=ImageTk.PhotoImage(image=img9)
        self.imageLabel9.configure(image=self.photo9)

        img10=Image.fromarray(image10)
        self.photo10=ImageTk.PhotoImage(image=img10)
        self.imageLabel10.configure(image=self.photo10)

    def showData(self,originalimg,encodedimg,lastimg):
         original_size=str(originalimg.nbytes)
        
         compressed_size=str(encodedimg.nbytes)

         decompressed_size=str(lastimg.nbytes)

         psnrval=str(calculate_psnr(originalimg,lastimg))

         self.Info1.configure(text=f'Original Size: {original_size} Bytes',font=font.Font(size=16),background='yellow')
         self.Info2.configure(text=f'Compressed Size: {compressed_size} Bytes',font=font.Font(size=16),background='yellow')
         self.Info3.configure(text=f'Decompressed Size: {decompressed_size} Bytes',font=font.Font(size=16),background='yellow')
         self.Info4.configure(text=f'PSNR: {psnrval}',font=font.Font(size=16),background='yellow')
    



root=tk.Tk()

main_frame=tk.Frame(root)
main_frame.pack(fill=tk.BOTH,expand=1)

my_canvas=tk.Canvas(main_frame)
my_canvas.pack(side=tk.LEFT,fill=tk.BOTH,expand=1)

my_scrollbar=tk.Scrollbar(main_frame,orient=tk.VERTICAL,command=my_canvas.yview)
my_scrollbar.pack(side=tk.RIGHT,fill=tk.Y)

my_canvas.configure(yscrollcommand=my_scrollbar.set)

my_canvas.bind_all("<MouseWheel>", lambda event: my_canvas.yview_scroll(-1 * (event.delta // 120), tk.UNITS))

second_frame = tk.Frame(my_canvas, width = 1000, height = 100)

my_canvas.create_window((0, 0), window=second_frame, anchor="nw")
g=mygui(second_frame)
g.showImage(img,level_shifted_img,quantized_img,DCT_img,encoded_img,decoded_img,dequantized_img,inverse_DCT,de_level_shifted,equalized_img)
g.showData(img,encoded_img,equalized_img)

root.mainloop()

