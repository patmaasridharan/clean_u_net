import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.onnx
import time
import unet_origin as unet_origin

import onnx
import onnxruntime

batch_size = 5                                                                  # how many images are included in one training step 
threshold = 200																	# treshold for creating binary mask from the target images that are in the range of 0-255

image_size = 256                                                                # image size should be power of 2 and at least 32
image_height = image_size                                                       # input image should have same height and width
image_width = image_size                                                        # input image should have same height and width
image_channel = 1   

model_dir = '/home/psridharan/Unet-Berkan/u_net_train/output/model200epoch_wholemouse_resizedimages_2020-02-06_18h18'

# -- Define our neural net architecture
net = unet_origin.UNetOriginal(image_channel, image_height, image_width)    # (channel, image_height, image_width)

net.load_state_dict(torch.load(model_dir, map_location=lambda storage, location: storage))
net.eval()

dummy_input = torch.randn(1, image_channel, image_height, image_width).type(torch.FloatTensor)
dummy_output = net(dummy_input)


# Export to ONNX format
torch.onnx.export(net, dummy_input, '/home/psridharan/Unet-Berkan/u_net_test/saved_model/Unetresized200epoch.onnx', export_params=True,  do_constant_folding=True, opset_version=11 , verbose = True)
start = time.process_time()
onnx_model = onnx.load("/home/psridharan/Unet-Berkan/u_net_test/saved_model/Unetresized200epoch.onnx")

onnx.helper.printable_graph(onnx_model.graph)

onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession("/home/psridharan/Unet-Berkan/u_net_test/saved_model/Unetresized200epoch.onnx")

