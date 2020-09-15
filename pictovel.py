from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import sys
import numpy as np
import cv2
from PIL import Image

import json
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
import kitti_util as utils
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)
try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3
import scipy as scp
import scipy.misc
import tensorflow as tf
import mayavi.mlab as mlab
from viz_util import draw_lidar, draw_lidar_simple, draw_gt_boxes3d

flags = tf.app.flags
FLAGS = flags.FLAGS

sys.path.insert(1, 'incl')

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



class kitti_object(object):
    '''Load and parse object data into a usable format.'''

    def __init__(self, root_dir, split='training'):
        '''root_dir contains training and testing folders'''
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)

        if split == 'training':
            self.num_samples = 7481
        elif split == 'testing':
            self.num_samples = 7518
        else:
            print('Unknown split: %s' % (split))
            exit(-1)

        self.image_dir = os.path.join(self.split_dir, 'image_2')
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.lidar_dir = os.path.join(self.split_dir, 'velodyne')
        self.label_dir = os.path.join(self.split_dir, 'label_2')

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert (idx < self.num_samples)
        img_filename = os.path.join(self.image_dir, '%06d.png' % (idx))
        return utils.load_image(img_filename)

    def get_lidar(self, idx): 
        assert (idx < self.num_samples)
        lidar_filename = os.path.join(self.lidar_dir, '%06d.bin' % (idx))
        return utils.load_velo_scan(lidar_filename)


    def get_calibration(self, idx):
        assert (idx < self.num_samples)
        calib_filename = os.path.join(self.calib_dir, '%06d.txt' % (idx))
        return utils.Calibration(calib_filename)


    def get_label_objects(self, idx):
        assert (idx < self.num_samples and self.split == 'training')
        label_filename = os.path.join(self.label_dir, '%06d.txt' % (idx))
        return utils.read_label(label_filename)


    def get_depth_map(self, idx):
        pass


    def get_top_down(self, idx):
        pass


def maybe_download_and_extract(runs_dir):
    logdir = os.path.join(runs_dir, default_run)

    if os.path.exists(logdir):
        # weights are downloaded. Nothing to do
        return

    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)
    download_name = tv_utils.download(weights_url, runs_dir)
    logging.info("Extracting KittiSeg_pretrained.zip")

    import zipfile
    zipfile.ZipFile(download_name, 'r').extractall(runs_dir)

    return


def resize_label_image(image, gt_image, image_height, image_width):
    image = scp.misc.imresize(image, size=(image_height, image_width),
                              interp='cubic')
    shape = gt_image.shape
    gt_image = scp.misc.imresize(gt_image, size=(image_height, image_width),
                                 interp='nearest')

    return image, gt_image


def main(_):
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

    dataset=kitti_object(os.path.join(ROOT_DIR,'free-space/dataset/KITTI/object'))

    for data_idx in range(len(dataset)):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
        objects = dataset.get_label_objects(data_idx)


        # Load and resize input image
        image = dataset.get_image(data_idx)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        scp.misc.imsave('new.png', image)
        if hypes['jitter']['reseize_image']:
            # Resize input only, if specified in hypes
            image_height = hypes['jitter']['image_height']
            image_width = hypes['jitter']['image_width']
            image = scp.misc.imresize(image, size=(image_height, image_width),
                                      interp='cubic')
        img_height, img_width, img_channel = image.shape
        pc_velo = dataset.get_lidar(data_idx)[:,0:3]
        print(len(pc_velo))
        velo_len=len(pc_velo)
        calib = dataset.get_calibration(data_idx)

        # Run KittiSeg model on image
        feed = {image_pl: image}
        softmax = prediction['softmax']                                                                                                                                    
        output = sess.run([softmax], feed_dict=feed)

        # Reshape output from flat vector to 2D Image
        shape = image.shape
        output_image = output[0][:, 1].reshape(shape[0], shape[1])

        # Plot confidences as red-blue overlay
        rb_image = seg.make_overlay(image, output_image)
        scp.misc.imsave('new0.png', rb_image)

        # Accept all pixel with conf >= 0.5 as positive prediction
        # This creates a `hard` prediction result for class street
        threshold = 0.5
        street_prediction = output_image > threshold

        index=np.where(street_prediction==True)
        chang = len(index[0])
        print(chang)
        test = np.zeros((velo_len,2),dtype=np.int)
        for tmp0 in range(chang):
            test[tmp0][0]=index[0][tmp0]
            test[tmp0][1]=index[1][tmp0]
        print("suoyindayin")
        # if (chang>0):
        #     print(test[0][0])
        #     print(test[0][1])

        pts_2d=calib.project_velo_to_image(pc_velo)
        print(pts_2d.shape)
        # print(pts_2d[1][0])
        # print(pts_2d[1][1])


        fig = mlab.figure(figure=None, bgcolor=(0, 0, 0),
                          fgcolor=None, engine=None, size=(1000, 500))
        fov_inds = (pts_2d[:,0]<1242) & (pts_2d[:,0]>=0) & \
            (pts_2d[:,1]<370) & (pts_2d[:,1]>=0)
        print(fov_inds.shape)
        # print(fov_inds[1000])
        # print(pc_velo.shape)
        print("okok")
        fov_inds = fov_inds & (pc_velo[:,0]>2.0)
        print(fov_inds.shape)
        imgfov_pts_2d=pts_2d[fov_inds,:]
        imgfov_pc_velo = pc_velo[fov_inds, :]
        pts_2d0=calib.project_velo_to_image(imgfov_pc_velo)
        fov_inds0 = (pts_2d0[:,0]<1242) & (pts_2d0[:,0]>=0) & \
            (pts_2d0[:,1]<370) & (pts_2d0[:,1]>=0)
        fov_inds0 = fov_inds0 & (imgfov_pc_velo[:,0]>2.0)
        print(fov_inds0.shape)
        print(fov_inds0[2])
        print(imgfov_pts_2d.shape)
        # if(chang>0):
        #     print(int(imgfov_pts_2d[5,0]))
        #     print(int(imgfov_pts_2d[5,1]))
        #     print(street_prediction[int(imgfov_pts_2d[5,1]),int(imgfov_pts_2d[5,0])])
        
        if(chang>0):
            for i in range(len(fov_inds0)):
                fov_inds0[i]=fov_inds0[i] & (street_prediction[int(pts_2d0[i,1]),int(pts_2d0[i,0])]==True)
        imgfov_pc_velo0 = imgfov_pc_velo[fov_inds0, :]
        draw_lidar(imgfov_pc_velo0, fig=fig)

        # for obj in objects:
        #     if obj.type == 'DontCare': continue
        #     # Draw 3d bounding box
        #     box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        #     box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        #     # Draw heading arrow
        #     ori3d_pts_2d, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
        #     ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
        #     x1, y1, z1 = ori3d_pts_3d_velo[0, :]
        #     x2, y2, z2 = ori3d_pts_3d_velo[1, :]
        #     draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig)
        #     mlab.plot3d([x1, x2], [y1, y2], [z1, z2], color=(0.5, 0.5, 0.5),
        #                 tube_radius=None, line_width=1, figure=fig)
        mlab.show(1)

        # Plot the hard prediction as green overlay
        green_image = tv_utils.fast_overlay(image, street_prediction)

        # Save output images to disk.
        if FLAGS.output_image is None:
            output_base_name = image
        else:
            output_base_name = FLAGS.output_image

        # raw_image_name = output_base_name.split('.')[0] + '_raw.png'
        # rb_image_name = output_base_name.split('.')[0] + '_rb.png'
        # green_image_name = output_base_name.split('.')[0] + '_green.png'

        scp.misc.imsave('1.png', output_image)
        scp.misc.imsave('2.png', rb_image)
        scp.misc.imsave('3.png', green_image)
        raw_input()

if __name__ == '__main__':
    tf.app.run()
