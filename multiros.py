"""
Detects Cars in an image using KittiSeg.

Input: Image
Output: Image (with Cars plotted in Green)

Utilizes: Trained KittiSeg weights. If no logdir is given,
pretrained weights will be downloaded and used.

Usage:
python demo.py --input_image data/demo.png [--output_image output_image]
                [--logdir /path/to/weights] [--gpus 0]

--------------------------------------------------------------------------------

The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann

Details: https://github.com/MarvinTeichmann/KittiSeg/blob/master/LICENSE
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import sys
import time
import multiprocessing as mp
import collections
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
# configure logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import numpy as np
import scipy as scp
import scipy.misc
import tensorflow as tf


flags = tf.app.flags
FLAGS = flags.FLAGS

sys.path.insert(1, 'incl')
queue = mp.Queue(maxsize=2)

from seg_utils import seg_utils as seg

try:
    # Check whether setup was done correctly

    import tensorvision.utils as tv_utils
    import tensorvision.core as core
except ImportError:
    # You forgot to initialize submodules
    logging.error("Could not import the submodules.")
    logging.error("Please execute:"
                  "'git submodule update --init --recursive'")
    exit(1)


flags.DEFINE_string('logdir', None,
                    'Path to logdir.')
flags.DEFINE_string('input_image', None,
                    'Image to apply KittiSeg.')
flags.DEFINE_string('output_image', None,
                    'Image to apply KittiSeg.')


default_run = 'KittiSeg_pretrained'
weights_url = ("ftp://mi.eng.cam.ac.uk/"
               "pub/mttt2/models/KittiSeg_pretrained.zip")


def maybe_download_and_extract(runs_dir):
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)
    logdir = os.path.join(runs_dir, default_run)

    if os.path.exists(logdir):
        # weights are downloaded. Nothing to do
        return

    import zipfile
    download_name = tv_utils.download(weights_url, runs_dir)

    logging.info("Extracting KittiSeg_pretrained.zip")

    zipfile.ZipFile(download_name, 'r').extractall(runs_dir)

    return


def resize_label_image(image, gt_image, image_height, image_width):
    image = scp.misc.imresize(image, size=(image_height, image_width),
                              interp='cubic')
    shape = gt_image.shape
    gt_image = scp.misc.imresize(gt_image, size=(image_height, image_width),
                                 interp='nearest')

    return image, gt_image

class Imageinput:
    def __init__(self):
        self.image_pub = rospy.Publisher("visual_image", Image)
        self.bridge = CvBridge()
	rospy.loginfo('deeplab ros init successfully!')
        self.image_sub=rospy.Subscriber("/camera_array/cam0/image_raw",Image,self.callback)
    def callback(self,data):
        try:
            img0=self.bridge.imgmsg_to_cv2(data)
            img = cv2.resize(img0,(1242,375)) 

            queue.put(img)
            queue.get() if queue.qsize() > 1 else time.sleep(0.01)


        except CvBridgeError as e:
            print (e)




def thread_ros():
    #rospy.init_node("deeplab")
    #global image_read
    #image_read = Imageinput()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("shutting down")
    cv2.destroyAllWindows()


def model_path():
    tv_utils.set_gpus_to_use()

    # if FLAGS.input_image is None:
    #     logging.error("No input_image was given.")
    #     logging.info(
    #         "Usage: python demo.py --input_image data/test.png "
    #         "[--output_image output_image] [--logdir /path/to/weights] "
    #         "[--gpus GPUs_to_use] ")
    #     exit(1)

    if FLAGS.logdir is None:
        # Download and use weights from the MultiNet Paper
        if 'TV_DIR_RUNS' in os.environ:
            runs_dir = os.path.join(os.environ['TV_DIR_RUNS'],
                                    'KittiSeg')
        else:
            runs_dir = 'RUNS'
        maybe_download_and_extract(runs_dir)
        logdir = os.path.join(runs_dir, default_run)
    else:
        logging.info("Using weights found in {}".format(FLAGS.logdir))
        logdir = FLAGS.logdir

    # Loading hyperparameters from logdir
    hypes = tv_utils.load_hypes_from_logdir(logdir, base_path='hypes')

    logging.info("Hypes loaded successfully.")

    # Loading tv modules (encoder.py, decoder.py, eval.py) from logdir
    modules = tv_utils.load_modules_from_logdir(logdir)
    logging.info("Modules loaded successfully. Starting to build tf graph.")

    # Create tf graph and build module.
    with tf.Graph().as_default():
        # Create placeholder for input
        image_pl = tf.placeholder(tf.float32)
        image = tf.expand_dims(image_pl, 0)

        # build Tensorflow graph using the model from logdir
        prediction = core.build_inference_graph(hypes, modules,
                                                image=image)

        logging.info("Graph build successfully.")

        # Create a session for running Ops on the Graph.
        sess = tf.Session()
        saver = tf.train.Saver()

        # Load weights from logdir
        core.load_weights(logdir, sess, saver)

        logging.info("Weights loaded successfully.")
    cv2.namedWindow("visual_image", flags=cv2.WINDOW_FREERATIO)
    while True:
        frame=queue.get()
        cv2.imshow("image",frame)
        #cv2.waitKey(1)
        image = frame
    # logging.info("Starting inference using {} as input".format(input_image))

    # Load and resize input image
    #image = scp.misc.imread(input_image)
        if hypes['jitter']['reseize_image']:
        # Resize input only, if specified in hypes
            image_height = hypes['jitter']['image_height']
            image_width = hypes['jitter']['image_width']
            image = scp.misc.imresize(image, size=(image_height, image_width),
                                  interp='cubic')

    # Run KittiSeg model on image
        feed = {image_pl: image}
        softmax = prediction['softmax']
        output = sess.run([softmax], feed_dict=feed)

    # Reshape output from flat vector to 2D Image
        shape = image.shape
        output_image = output[0][:, 1].reshape(shape[0], shape[1])

    # Plot confidences as red-blue overlay
        rb_image = seg.make_overlay(image, output_image)

    # Accept all pixel with conf >= 0.5 as positive prediction
    # This creates a `hard` prediction result for class street
        threshold = 0.5
        street_prediction = output_image > threshold

    # Plot the hard prediction as green overlay
        green_image = tv_utils.fast_overlay(image, street_prediction)
        cv2.imshow("visual_image",green_image)
        cv2.waitKey(1)

def run_two_task():
    #mp.set_start_method(method='spawn')
    rospy.init_node("deeplab")
    #global image_read
    image_read = Imageinput()

    processes= [mp.Process(target=thread_ros,args=()),
                mp.Process(target=model_path,args=())]

    [process.start() for process in processes]
    [process.join() for process in processes]


run_two_task()
if __name__ == '__main__':
    run_two_task()
