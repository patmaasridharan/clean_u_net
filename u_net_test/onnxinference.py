
import classifier
import unet_origin as unet_origin
import torch.optim as optim

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.autograd import Variable

import dataset
from test_callbacks import PredictionsSaverCallback
import imresizepython

import os
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
from PIL import Image
from dataset import TestImageDataset
import multiprocessing
import shutil
import time
import cv2
import imageio
import pylab as plt
import onnx
import onnxruntime
import scipy.misc
import time
import numpy as np
from matplotlib import pylab as plt
import matplotlib.image as mpimg

image_size = 256                                                                # image size should be power of 2 and at least 32
image_height = image_size                                                       # input image should have same height and width
image_width = image_size                                                        # input image should have same height and width
image_channel = 1                                                               # only tested with channel size 1
input_img_resize = (image_size, image_size)
origin_img_size = (image_size, image_size)
start = time.process_time()
model_dir = '/home/psridharan/Unet-Berkan/u_net_test/saved_model/Unetresized200epoch.onnx'   # path to saved model

data_dir = '/home/psridharan/Unet-Berkan/annotation_correlation/ReconstructionData/Scan_106/recon715/raw'       
output_dir = '/home/psridharan/Unet-Berkan/annotation_correlation/ReconstructionData/Scan_106/recon715'  

list_of_images = [f for f in os.listdir(data_dir) if f.endswith('.bin')]

sess = onnxruntime.InferenceSession(model_dir)

input_name = sess.get_inputs()[0].name
print("input name", input_name)
input_shape = sess.get_inputs()[0].shape
print("input shape", input_shape)
input_type = sess.get_inputs()[0].type
print("input type", input_type)

output_name = sess.get_outputs()[0].name
print("output name", output_name)
output_shape = sess.get_outputs()[0].shape
print("output shape", output_shape)
output_type = sess.get_outputs()[0].type
print("output type", output_type)

for i in range(np.size(list_of_images)):
    
    #read the labels for dice calculation
    imagename = list_of_images[i]
    img_path = os.path.join(data_dir, imagename) 
    img = np.fromfile(img_path, dtype='float64', sep="")
    img = img.reshape([332, 332])
    image= img.copy()
    out=   image-image.min()
    out1= out/(image.max()- image.min())
    p=out1*255
    Img_out =imresizepython.imresize(p, output_shape=(256, 256))
    

    plt.imsave('/home/psridharan/Unet-Berkan/annotation_correlation/ReconstructionData/Scan_106/recon715/matout.png' ,Img_out)

   #  imageorg= mpimg.imread('/home/psridharan/Unet-Berkan/annotation_correlation/ReconstructionData/Scan_106/recon715/imgorg.jpg')
   #  plt.imsave('/home/psridharan/Unet-Berkan/annotation_correlation/ReconstructionData/Scan_106/recon715/imgorgout.png' , imageorg)
   #  #img_resize=(256, 256)
   #  #img = img.resize(img_resize, Image.ANTIALIAS)
   #  #img = np.asarray(img.convert("L"), dtype=np.float32)
    imagenew = Img_out.astype(np.float32)
   #  image1 = imageorg.astype(np.float32)
    #image = (image - 0) / 1
    imageinput= np.random.random((1,1,256,256))
    imageinput[0,0,:,:]=imagenew
    imageinput = imageinput.astype(np.float32)
    res = sess.run([output_name], {input_name: imageinput})
    tensor = torch.from_numpy(res[0]).float()
    probs = torch.sigmoid(tensor)
    probs = probs.data.cpu().numpy()
    outimage = Image.fromarray((probs[0]* 255).astype(np.uint8))
    path_to_save = os.path.join(output_dir, "out.jpg") 
    outimage.save(path_to_save)
    print(time.process_time() - start)



