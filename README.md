# Classifying boxes using TensorFlow

This is a simple example of how to build a convolutional neural network (CNN) with TensorFlow.  It extends existing MNIST convnet examples with TensorFlow input queues for reading training and validation data in common image formats like JPEG.  You should be able to easily change the sample code and text files to use your own training data.

[Link to YouTube Video](https://www.youtube.com/watch?v=40iJ0yS572E)

The sample training data and code classifies images of a toy box into four states:

1. Upright (box is upright)
2. Tilted (box is tilted on its side)
3. Open (box is upright and open)
4. Spilled (box is tilted on it's side, open with contents spilled)

115 images for each class is used to train the model.

The main code samples are:

`tf_convnet_test.py` reads the training and validation data from train.txt and valid.txt, then builds and executes the TensorFlow graph. Run it with no arguments:

```
python tf_convnet_test.py
```

`tf_convnet_export.py` does all the above, and also exports the trained model.  You'll need to install or build TensorFlow Serving to run this. Run it with no arguments:

```
python tf_convnet_export.py
```

`tf_convnet_inference.cc` serves the trained TensorFlow Serving.  You'll need to install or build TensorFlow Serving to compile it. The code is slightly modified from the MNIST inference sample included with the TensorFlow SDK, most notably changing the shape of the input data and number of output classes.  Run it by specifying the port and location of the exported model.

```
tf_convnet_inference --port=9000 location/to/model/dir/
```

`web_server.py` is a simple Flask application that takes images using a webcam and sends them to the inference server for classification. It has only been "tested" on Chrome. Run it with no arguments:

```
python web_server.py
```

I don't have a pattern for the box, but if you want to build your own, it is

* 50x50x50mm
* made of balsa wood
* has white electrical tape on the sides to hold it together
* has an articulating "lid"
* filled coloured pom poms on the inside
* marked on four sides with a "this side up" decal
