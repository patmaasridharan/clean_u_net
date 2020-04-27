import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.onnx
from onnx_tf.backend import prepare

import unet_origin as unet_origin



batch_size = 5                                                                  # how many images are included in one training step 
threshold = 200																	# treshold for creating binary mask from the target images that are in the range of 0-255

image_size = 256                                                                # image size should be power of 2 and at least 32
image_height = image_size                                                       # input image should have same height and width
image_width = image_size                                                        # input image should have same height and width
image_channel = 1   

model_dir = 'saved_model/model_kidney_liver_2018-10-02_13h07'

# -- Define our neural net architecture
net = unet_origin.UNetOriginal(image_channel, image_height, image_width)    # (channel, image_height, image_width)

net.load_state_dict(torch.load(model_dir))

dummy_input = torch.randn(batch_size, image_channel, image_height, image_width)
dummy_output = net(dummy_input)
print(dummy_output)

# Export to ONNX format
torch.onnx.export(net, dummy_input, 'saved_model/seg.onnx', export_params=True, input_names=['test_input'], output_names=['test_output'] ,dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})