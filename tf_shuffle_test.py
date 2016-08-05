'''
An example to demonstrate TensorFlow input queue shuffling
Author: Win Woo

'''

import tensorflow as tf

x_ = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y_ = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def next(input_queue):
    return input_queue[0], input_queue[1]

input_queue = tf.train.slice_input_producer([x_, y_],
    num_epochs=None, shuffle=True)

x_, y_ = next(input_queue)

sess = tf.Session()
init_op = tf.initialize_all_variables()

with sess.as_default():
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        x, y = sess.run([x_, y_])
        print(x, y)


    except tf.errors.OutOfRangeError:
        print("done!")

    finally:
        coord.request_stop()
        coord.join(threads)
        sess.close()
