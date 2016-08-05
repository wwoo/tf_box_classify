from flask import Flask, render_template, request
from io import BytesIO
from PIL import Image
from numpy import array
from grpc.beta import implementations

import base64
import numpy
import os
import time
import sys

import tf_convnet_inference_pb2

app = Flask(__name__)

host = "127.0.0.1"
port = 9000
timeout = 8.0
crop_image_size = 300
resize_image_size = 150

labels = ['open', 'tilted', 'upright', 'spilt']

@app.route('/cam', methods=['GET'])
def route_camera():
    return render_template('cam.html')

@app.route('/classify', methods=['POST'])
def classify_file():
    data_str = request.form['img']

    if data_str:
        # strip "data:image/jpeg;base64," from the data
        data_str = data_str[data_str.find(",")+1:]
        img = Image.open(BytesIO(base64.b64decode(data_str)))

        # crop out the center 300x300
        width, height = img.size
        crop_h = (width - crop_image_size)/2
        crop_v = (height - crop_image_size)/2
        img = img.crop((crop_h, crop_v, width-crop_h, height-crop_v))

        # resize the resulting image to 150x150
        img = img.resize((resize_image_size, resize_image_size))

        # convert to grayscale
        img = img.convert('L')

        # debug - save the image
        img.save("/tmp/debug_box.jpg")

        # convert the image to an array
        arr = array(img).reshape(resize_image_size * resize_image_size).astype(float)

        # build the request
        grpc_request = tf_convnet_inference_pb2.BoxImageRequest()
        for pixel in arr:
            grpc_request.image_data.append(pixel)

        # call the gRPC server
        channel = implementations.insecure_channel(host, port)
        stub = tf_convnet_inference_pb2.beta_create_BoxImageService_stub(channel)
        result = stub.Classify(grpc_request, timeout)

        # return the predicted label
        values = numpy.array(result.value)
        predicted_label = labels[numpy.argmax(values)]

        return '{ "predicted_label": \"' + predicted_label + '\"}'

    return "{}}"

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)
