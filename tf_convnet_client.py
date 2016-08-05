'''
Command line client for TensorFlow inference server
Author: Win Woo

Bare bones example code based on MNIST TensorFlow example
'''

from grpc.beta import implementations

import numpy
import sys
from PIL import Image
from numpy import array

import tf_convnet_inference_pb2

host = "127.0.0.1"
port = 9000
image_size = 150

def main(argv):
    filename = argv[0]
    img = Image.open(filename)
    width, height = img.size

    # crop out the center 300x300
    crop_h = (width - image_size)/2
    crop_v = (height - image_size)/2
    img = img.crop((crop_h, crop_v, width-crop_h, height-crop_v))

    # resize the resulting image to 150x150
    img = img.resize((image_size, image_size))

    # convert to grayscale
    img = img.convert('L')

    arr = array(img).reshape(image_size * image_size).astype(float)
    print(arr) #debug

    # build the request
    request = tf_convnet_inference_pb2.BoxImageRequest()
    for pixel in arr:
        request.image_data.append(pixel)

    # call the gRPC server
    channel = implementations.insecure_channel(host, port)
    stub = tf_convnet_inference_pb2.beta_create_BoxImageService_stub(channel)
    result = stub.Classify(request, 8.0)

    print(result.value)

if __name__ == '__main__':
    main(sys.argv[1:])
