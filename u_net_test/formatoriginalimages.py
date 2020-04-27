import os
import random
import math
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

image_size =256
main_dir = os.getcwd()                                                          # main folder directory (do not change!)

testdata_dir = os.path.join(main_dir,'_data_test')				# directory for test data  
image_dir =os.path.join(testdata_dir, '_data_wholemouse_test_256_256')


label_path = '/home/psridharan/Unet-Berkan/annotation_correlation/TS'
x_original = os.path.join(label_path, 'inputs')                                      # path to origainal images (300 *300*4)
y_original = os.path.join(label_path, 'targets')                                     # path to annotated labels (300*300*4)



list_of_images = [f for f in os.listdir(x_original) if f.endswith('.tif')]
list_of_labels = [f for f in os.listdir(y_original) if f.endswith('.png')]
inputimage_dir =os.path.join(label_path,'Formated_inputs_pngform')
targetimage_dir =os.path.join(label_path,'Formated_targets_pngform')

for i in range(len(list_of_images)):
      imagepath= os.path.join(x_original, list_of_images[i])
      image = Image.open(imagepath)    
      #image = image.convert('RGB')     
      image = image.convert('L')
      width, height = image.size           
      area = (17, 17, 273, 273) 
      cropped_image = image.crop(area)  
      cropped_image.save(os.path.join(inputimage_dir, os.path.splitext(list_of_images[i])[0])+ '.png') 

      targetpath= os.path.join(y_original, list_of_labels[i])
      trimage = Image.open(targetpath)    
      width, height = trimage.size           
      trarea = (17, 17, 273, 273) 
      trcropped_image = trimage.crop(trarea)  
      trcropped_image.save(os.path.join(targetimage_dir, os.path.splitext(list_of_labels[i])[0])+ '.png') 




