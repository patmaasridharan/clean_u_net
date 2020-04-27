
import classifier
import unet_origin as unet_origin
import torch.optim as optim

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

import dataset
from test_callbacks import PredictionsSaverCallback

import os
import random
import math
import numpy as np
from multiprocessing import cpu_count

from dataset import TestImageDataset
import multiprocessing
import shutil
import time

def main():

    #oooooooooo user inputs (!!!change only this part!!!) oooooooooo#

    batch_size = 5                                                                  # how many images are included in one training step 
    threshold = 200																	# treshold for creating binary mask from the target images that are in the range of 0-255

    image_size = 256                                                                # image size should be power of 2 and at least 32
    image_height = image_size                                                       # input image should have same height and width
    image_width = image_size                                                        # input image should have same height and width
    image_channel = 1                                                               # only tested with channel size 1
    input_img_resize = (image_size, image_size)
    origin_img_size = (image_size, image_size)
    
    main_dir = os.getcwd()                                                          # main folder directory (do not change!)

    test_dir = os.path.join(main_dir,'_data_test')									# directory for test data

    model_dir = 'saved_model/model_kidney_liver_2018-10-02_13h07'					# path to saved model

    data_dir = os.path.join(test_dir, \
    '_data_kidney_test_{}_{}'.format(image_size, image_size))                     # data folder name: "_name_{img_size}_{img_size}"

    x_train = os.path.join(data_dir, 'inputs')                                      # path to training images
    y_train = os.path.join(data_dir, 'targets')                                     # path to ground truth data

    #oooooooooo !!! do not make any change after this !!! oooooooooo#
    
    # -- Optional parameters
    threads = cpu_count()
    use_cuda = torch.cuda.is_available()

    ##### prepare the data for testing #####

    list_of_images = [f for f in os.listdir(x_train) if f.endswith('.jpg')]
    list_of_labels = [f for f in os.listdir(y_train) if f.endswith('.jpg')]

    X_test = []; y_test = []

    for i in range(len(list_of_images)):
      X_test.append(os.path.join(x_train, list_of_images[i]))
      y_test.append(os.path.join(y_train, list_of_labels[i]))

    # Testing callbacks
    pred_saver_cb = PredictionsSaverCallback(origin_img_size, threshold)

    # -- Define our neural net architecture
    net = unet_origin.UNetOriginal(image_channel, image_height, image_width)    # (channel, image_height, image_width)
    classifier_seg = classifier.segment(net)

    test_ds = TestImageDataset(X_test, input_img_resize)
    test_loader = DataLoader(test_ds, batch_size,
                             sampler=SequentialSampler(test_ds),
                             num_workers=threads,
                             pin_memory=use_cuda)

    print("Testing on {} samples".format(len(test_loader.dataset)))

    if not os.path.exists('predictions'):
      os.makedirs('predictions')

    if os.path.exists('predictions/prediction'):
      shutil.rmtree('predictions/prediction')

    # Predict & save

    net.load_state_dict(torch.load(model_dir))

    start_pred = time.time()

    classifier_seg.predict(test_loader, callbacks=[pred_saver_cb])
    
    end_pred = time.time()

    print("prediction time: {}".format(end_pred - start_pred))


if __name__ == "__main__":
    # Workaround for a deadlock issue on Pytorch 0.2.0: https://github.com/pytorch/pytorch/issues/1838
    multiprocessing.set_start_method('spawn', force=True)
    main()