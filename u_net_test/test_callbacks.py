import gzip
import csv
import cv2
import numpy as np
import scipy.misc
import os


class Callback:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class PredictionsSaverCallback(Callback):
    def __init__(self, origin_img_size, threshold):
        self.threshold = threshold
        self.origin_img_size = origin_img_size

    def get_mask_rle(self, prediction, name, files_name):

        mask = cv2.resize(prediction, self.origin_img_size)

        # save the predictions
        path_to_save = 'predictions/prediction'

        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)

        scipy.misc.imsave(os.path.join(path_to_save, os.path.basename(name)), mask)

    
    def __call__(self, *args, **kwargs):
        if kwargs['step_name'] != "predict":
            return

        probs = kwargs['probs']
        files_name = kwargs['files_name']

        for (pred, name) in zip(probs, files_name):

            self.get_mask_rle(pred, name, files_name)