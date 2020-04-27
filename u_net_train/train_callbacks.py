import cv2
import torch
import numpy as np
import scipy.misc as scipy
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt


class Callback:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

class ModelSaverCallback(Callback):
    def __init__(self, path_to_model, verbose=False):
        self.verbose = verbose
        self.path_to_model = path_to_model

    def __call__(self, *args, **kwargs):
        if kwargs['step_name'] != "train":
            return

        pth = self.path_to_model
        net = kwargs['net']
        torch.save(net.state_dict(), pth)

        if self.verbose:
            print("Model saved in {}".format(pth))
