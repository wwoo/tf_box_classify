'''
Another Convolutional Network Example using TensorFlow Library
Author: Win Woo

Based off Aymeric Damien's example at:
https://github.com/aymericdamien/TensorFlow-Examples/

Shows how to use TensorFlow input queues and image decoding to
train a simple convolutional neural network.
'''

import tensorflow as tf
import numpy as np
import os
import time

# hyper parameters to use for training
TRAIN_BATCH_SIZE = 10
TRAIN_EPOCHS = 5
VALID_BATCH_SIZE = 40
VALID_EPOCHS = None
SHUFFLE_BATCHES = True
LEARNING_RATE = 0.01
NUM_CLASSES = 4
KEEP_PROB = 0.75

# image parameters
IMAGE_SIZE = 150
IMAGE_RESIZE_FACTOR = 1
IMAGE_CHANNELS = 3

def get_image_label_list(image_label_file):
    filenames = []
    labels = []
    for line in open(image_label_file, "r"):
        filename, label = line[:-1].split(' ')
        filenames.append(filename)
        labels.append(int(label))

    # == debug ==
    print "get_image_label_list: read " + str(len(filenames)) \
        + " items"
    # == /debug ==

    return filenames, labels

def read_image_from_disk(input_queue):
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    rgb_image = tf.image.decode_jpeg(file_contents, channels=IMAGE_CHANNELS,
        name="decode_jpeg")

    return rgb_image, label

def inputs(train_file, batch_size=TRAIN_BATCH_SIZE, num_epochs=TRAIN_EPOCHS):
    image_list, label_list = get_image_label_list(train_file)
    input_queue = tf.train.slice_input_producer([image_list, label_list],
        num_epochs=num_epochs, shuffle=SHUFFLE_BATCHES)
    image, label = read_image_from_disk(input_queue)
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS])
    image_batch, label_batch = tf.train.batch([image, label],
        batch_size=batch_size)

    return preprocess_images(image_batch), tf.one_hot(tf.to_int64(label_batch),
        NUM_CLASSES, on_value=1.0, off_value=0.0)

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1],
        padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2, layer=""):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1],
        strides=[1, k, k, 1], padding='SAME')

def conv_net(x, weights, biases, image_size, keep_prob=KEEP_PROB):
    # Reshape the images to image_size x image_size
    x = tf.reshape(x, shape=[-1, image_size, image_size, IMAGE_CHANNELS])

    # Convolution and max pooling layers
    # Each max pooling layer reduces dimensionality by 2

    with tf.name_scope('layer1'):
        # Convolution and max pooling layer 1
        conv1 = conv2d(x, weights['wc1'], biases['bc1'])
        conv1 = maxpool2d(conv1, k=2)

    with tf.name_scope('layer2'):
        # Convolution and max pooling layer 2
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        conv2 = maxpool2d(conv2, k=2)

    with tf.name_scope('layer3'):
        # Convolution and max pooling layer 3
        conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
        conv3 = maxpool2d(conv3, k=2)

    with tf.name_scope('layer4'):
        # Convolution and max pooling layer 4
        conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
        conv4 = maxpool2d(conv4, k=2)

    with tf.name_scope('fully_connected'):
        # Fully-connected layer
        fc1 = tf.reshape(conv4, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)

        # Apply dropout
        fc1 = tf.nn.dropout(fc1, keep_prob)

    with tf.name_scope('output'):
        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    # maxpool_slice = tf.slice(conv1, [0, 0, 0, 0], [TRAIN_BATCH_SIZE, 75, 75, 1])
    # maxpool_sample = tf.image_summary("maxpool_sample", maxpool_slice, max_images=10)

    return out

def generate_image_summary(x, weights, biases, step, image_size=IMAGE_SIZE):
    # Reshape the images to image_size x image_size
    x = tf.reshape(x, shape=[-1, image_size, image_size, IMAGE_CHANNELS])

    with tf.name_scope('generate_image_summary'):
        x = tf.nn.conv2d(x, weights['wc1'], strides=[1, 1, 1, 1],
            padding='SAME')
        x = tf.nn.bias_add(x, biases['bc1'])
        x_slice = tf.slice(x, [0, 0, 0, 0],
            [TRAIN_BATCH_SIZE, image_size, image_size, 1])
        conv_summary = tf.image_summary("img_conv_{:05d}".format(step),
            x_slice, max_images=1)
        relu_summary = tf.image_summary("img_relu_{:05d}".format(step),
            tf.nn.relu(x_slice), max_images=1)

        #conv = conv2d(x, weights['wc1'], biases['bc1'])
        #conv_slice = tf.slice(conv, [0, 0, 0, 0],
        #    [TRAIN_BATCH_SIZE, image_size, image_size, 1])
        #image_summary = tf.image_summary("conv_{:05d}".format(step),
        #    conv_slice, max_images=1)

    return conv_summary, relu_summary

def preprocess_images(image_batch, resize_factor=IMAGE_RESIZE_FACTOR):
    new_image_size = int(round(IMAGE_SIZE / resize_factor))
    return tf.image.resize_images(image_batch, new_image_size, new_image_size)

def main(argv=None):

    # Read inventory of training images and labels
    with tf.name_scope('batch_inputs'):
        train_file = "./train.txt"
        valid_file = "./valid.txt"

        image_size = IMAGE_SIZE / IMAGE_RESIZE_FACTOR

        train_image_batch, train_label_batch = inputs(train_file,
            batch_size=TRAIN_BATCH_SIZE, num_epochs=TRAIN_EPOCHS)
        valid_image_batch, valid_label_batch = inputs(valid_file,
            batch_size=VALID_BATCH_SIZE, num_epochs=VALID_EPOCHS)

    # These are image and label batch placeholders which we'll feed in during training
    x_ = tf.placeholder("float32", shape=[None, image_size, image_size,
        IMAGE_CHANNELS])

    y_ = tf.placeholder("float32", shape=[None, NUM_CLASSES])

    # Store weights for our convolution & fully-connected layers
    with tf.name_scope('weights'):
        weights = {
            # 5x5 conv, 1 * IMAGE_CHANNELS input, 32 * IMAGE_CHANNELS outputs
            'wc1': tf.Variable(tf.truncated_normal([5, 5, 1 * IMAGE_CHANNELS,
                32 * IMAGE_CHANNELS])),
            # 5x5 conv, 32 * IMAGE_CHANNELS inputs, 64 * IMAGE_CHANNELS outputs
            'wc2': tf.Variable(tf.truncated_normal([5, 5, 32 * IMAGE_CHANNELS,
                64 * IMAGE_CHANNELS])),
            # 5x5 conv, 64 * IMAGE_CHANNELS inputs, 128 * IMAGE_CHANNELS outputs
            'wc3': tf.Variable(tf.truncated_normal([5, 5, 64 * IMAGE_CHANNELS,
                128 * IMAGE_CHANNELS])),
            # 5x5 conv, 128 * IMAGE_CHANNELS inputs, 256 * IMAGE_CHANNELS outputs
            'wc4': tf.Variable(tf.truncated_normal([5, 5, 128 * IMAGE_CHANNELS,
                256 * IMAGE_CHANNELS])),
            # fully connected, 19 * 19 * 256 * IMAGE_CHANNELS inputs, 1024 outputs
            'wd1': tf.Variable(tf.truncated_normal([10 * 10 * 256 * IMAGE_CHANNELS, 1024])),
            # 1024 inputs, 2 class labels (prediction)
            'out': tf.Variable(tf.truncated_normal([1024, NUM_CLASSES]))
        }

    # Store biases for our convolution and fully-connected layers
    with tf.name_scope('biases'):
        biases = {
            'bc1': tf.Variable(tf.random_normal([IMAGE_CHANNELS * 32])),
            'bc2': tf.Variable(tf.random_normal([IMAGE_CHANNELS * 64])),
            'bc3': tf.Variable(tf.random_normal([IMAGE_CHANNELS * 128])),
            'bc4': tf.Variable(tf.random_normal([IMAGE_CHANNELS * 256])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([NUM_CLASSES]))
        }

    # Define dropout rate to prevent overfitting
    keep_prob = tf.placeholder(tf.float32)

    # Build our graph
    pred = conv_net(x_, weights, biases, image_size, keep_prob)

    # Calculate loss
    with tf.name_scope('cross_entropy'):
        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y_))
        cost_summary = tf.scalar_summary("cost_summary", cost)

    # Run optimizer step
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

    # Evaluate model accuracy
    with tf.name_scope('predict'):
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        accuracy_summary = tf.scalar_summary("accuracy_summary", accuracy)
        w_summary = tf.histogram_summary("weights", weights['wc1'])
        b_summary = tf.histogram_summary("biases", biases['bc1'])

    sess = tf.Session()

    # conv_filter_sample = tf.image_summary("conv_filter_sample",
    #    tf.reshape(weights['wc1'], [32, 5, 5, 1]), max_images=3)

    writer = tf.train.SummaryWriter("./logs", sess.graph)
    init_op = tf.initialize_all_variables()

    # we need init_local_op step only on tensorflow 0.10rc due to a regression from 0.9
    # https://github.com/tensorflow/models/pull/297
    init_local_op = tf.initialize_local_variables()

    step = 0

    with sess.as_default():
        sess.run(init_op)
        sess.run(init_local_op) # we need this only with tensorflow 0.10rc
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                step += 1
                x, y = sess.run([train_image_batch, train_label_batch])
                train_step.run(feed_dict={keep_prob: 0.75,
                    x_: x, y_: y})

                if step % TRAIN_BATCH_SIZE == 0:
                    x, y = sess.run([valid_image_batch, valid_label_batch])
                    conv_summary, relu_summary = generate_image_summary(x_, weights, biases, step, image_size)
                    result = sess.run([cost_summary, accuracy_summary, accuracy, conv_summary, relu_summary, w_summary, b_summary],
                        feed_dict={keep_prob: 1.0, x_: x, y_: y})

                    cost_summary_str = result[0]
                    accuracy_summary_str = result[1]
                    acc = result[2]
                    conv_summary_str = result[3]
                    relu_summary_str = result[4]
                    w_summary_str = result[5]
                    b_summary_str = result[6]

                    # write summaries for viewing in Tensorboard
                    writer.add_summary(accuracy_summary_str, step)
                    writer.add_summary(cost_summary_str, step)
                    writer.add_summary(conv_summary_str, step)
                    writer.add_summary(relu_summary_str, step)
                    writer.add_summary(w_summary_str, step)
                    writer.add_summary(b_summary_str, step)

                    print("Accuracy at step %s: %s" % (step, acc))


        except tf.errors.OutOfRangeError:
            x, y = sess.run([valid_image_batch, valid_label_batch])
            result = sess.run([accuracy], feed_dict={keep_prob: 1.0,
                x_: x, y_: y})
            print("Validation accuracy: %s" % result[0])

        finally:
            coord.request_stop()
            coord.join(threads)
            sess.close()

    return 0

if __name__ == '__main__':
    tf.app.run()
