import classifier
import unet_origin
import torch.optim as optim
import helpers

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler

import augmentation as aug
from train_callbacks import ModelSaverCallback

import os
import random
import math
from multiprocessing import cpu_count

from dataset import TrainImageDataset
import multiprocessing

def main():

    #oooooooooo user inputs (!!!change only this part!!!) oooooooooo#

    batch_size = 5                                                                  # how many images are included in one training step
    epochs = 500                                                                    # how many times network iterates through all data
    validation_ratio = 0.15                                                         # split ratio between training and test data

    image_size = 256                                                                # image size should be power of 2 and at least 32
    image_height = image_size                                                       # input image should have same height and width
    image_width = image_size                                                        # input image should have same height and width
    image_channel = 1                                                               # only tested with channel size 1

    learning_rate = 0.01                                                            # learning rate
    momentum = 0.99                                                                 # how much you change the learning rate during training
    
    main_dir = os.getcwd()                                                          # main folder directory (do not change this line!)

    region = 'kidney_liver'

    data_dir = os.path.join(main_dir, 
      '_data_kidney_liver_train_{}_{}'.format(image_size, image_size))              # data folder name: "_name_{img_size}_{img_size}"

    x_train = os.path.join(data_dir, 'inputs')                                      # path to training images
    y_train = os.path.join(data_dir, 'targets')                                     # path to ground truth data

    model_saver_cb = ModelSaverCallback(os.path.join(main_dir, 
            'output/model_' + region + '_' + helpers.get_model_timestamp()), \
            verbose=True)                                                           # path to save model

    #oooooooooo !!! do not make any change after this !!! oooooooooo#

    #oooooooooo system dependent parameters oooooooooo#

    threads = cpu_count()                                                           # number of available cpu cores (4 in our case)
    use_cuda = torch.cuda.is_available()                                            # return "True" is cuda device available
    
    #oooooooooo prepare the data for training oooooooooo#

    list_of_images = [f for f in os.listdir(x_train) if f.endswith('.jpg')]
    list_of_labels = [f for f in os.listdir(y_train) if f.endswith('.jpg')]

    X_data = []; y_data = []; X_valid = []; y_valid = []

    valid_size = math.ceil(len(list_of_images)*validation_ratio)
    train_size = len(list_of_images) - (valid_size)

    mix_list = random.sample(range(len(list_of_images)), len(list_of_images))
    
    for i in range(train_size):
      X_data.append(os.path.join(x_train, list_of_images[mix_list[i]]))
      y_data.append(os.path.join(y_train, list_of_labels[mix_list[i]]))

    for i in range(valid_size):
      X_valid.append(os.path.join(x_train, list_of_images[mix_list[train_size + i]]))
      y_valid.append(os.path.join(y_train, list_of_labels[mix_list[train_size + i]]))

    train_ds = TrainImageDataset(X_data, y_data, X_transform=aug.augment_img)

    train_loader = DataLoader(train_ds, batch_size,
                            sampler=SequentialSampler(train_ds),
                            num_workers=threads, pin_memory=use_cuda)

    valid_ds = TrainImageDataset(X_valid, y_valid)

    valid_loader = DataLoader(valid_ds, batch_size,
                              sampler=SequentialSampler(valid_ds),
                              num_workers=threads, pin_memory=use_cuda)

    net = unet_origin.UNetOriginal(image_channel, image_height, image_width)         # (channel, image_height, image_width)
    classifier_seg = classifier.segment(net, epochs)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    
    print("Training on {} samples and validating on {} samples "
          .format(len(train_loader.dataset), len(valid_loader.dataset)))
    
    # Train the classifier
    classifier_seg.train(train_loader, valid_loader, optimizer,
                     epochs, callbacks=[model_saver_cb])

if __name__ == "__main__":
    # Workaround for a deadlock issue on Pytorch 0.2.0: https://github.com/pytorch/pytorch/issues/1838
    multiprocessing.set_start_method('spawn', force=True)
    main()