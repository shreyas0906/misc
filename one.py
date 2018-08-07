#import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe
#import cv2


# caffe.set_mode_cpu()
caffe.set_device(0)
caffe.set_mode_gpu()

net = caffe.Net('conv.prototxt', caffe.TEST)


im = cv2.imread('test-image.jpeg',0)
im_input = im[np.newaxis, np.newaxis, : ,: ]

net.blobs['data'].reshape(*im_input.shape)
net.blobs['data'].data[...] = im_input

print [(k, v.data.shape) for k, v in net.blobs.items()]

net.forward()

