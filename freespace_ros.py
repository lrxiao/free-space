from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import sys
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import PointCloud2,PointField
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image as newImage
from cv_bridge import CvBridge,CvBridgeError
#from geometry_msgs import Point
import std_msgs.msg
import sensor_msgs.point_cloud2 as pc2
import time
#import pcl
import pcl_msgs
import pcl_ros
from PIL import Image

import json
import logging
import message_filters

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


default_run = 'KittiSeg_2020_10_18_01.26'
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
    brideg = CvBridge()

    def expend3(pts_3d):
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d,np.ones((n,1))))
        return pts_3d_hom

    P_0 = [[665.373765,0.00,308.477032],[0.00,660.876949,341.540759],[0.00,0.00,1.00]]
    P_1 = [[-0.0121,-0.9999,0.0010,-0.0504],[-0.0109,-0.0009,-0.9999,0.0030],[0.9999,-0.0122,-0.0109,-0.0963]]
    distort = [0.163407,-0.182330,0.005264,0.011895,0.00]
    def callback(image,Pointcloud):
        print("ros topic input")
        gen = point_cloud2.read_points(Pointcloud,field_names=("x","y","z"),skip_nans=True)
        n=30000
        # for q in gen:
        #     n=n+1
        # print(n)
        pc_velo = np.zeros([n,3])
        i=0
        for p in gen:
            pc_velo[i,0]=p[0]
            pc_velo[i,1]=p[1]
            pc_velo[i,2]=p[2]
            i=i+1
        print(i)

        image = brideg.imgmsg_to_cv2(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # undistort
        h,w = image.shape[:2]
        k = np.array(P_0)
        d = np.array(distort)
        mapx,mapy = cv2.initUndistortRectifyMap(k,d,None,k,(w,h),5)
        image = cv2.remap(image,mapx,mapy,cv2.INTER_LINEAR)

        if hypes['jitter']['reseize_image']:
            # Resize input only, if specified in hypes
            image_height = hypes['jitter']['image_height']
            image_width = hypes['jitter']['image_width']
            image = scp.misc.imresize(image, size=(image_height, image_width),
                                      interp='cubic')
        img_height, img_width, img_channel = image.shape

        # fig = mlab.figure(figure=None, bgcolor=(0, 0, 0),
        #                   fgcolor=None, engine=None, size=(1000, 500))

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

        index=np.where(street_prediction==True)
        chang = len(index[0])
        print(chang)
        #pts_2d=calib.project_velo_to_image(pc_velo)
        pc_4=expend3(pc_velo)
        cam_3d = np.dot(pc_4,np.transpose(P_1))

        pts_3d=np.dot(cam_3d,np.transpose(P_0))
        pts_3d[:,0] /= cam_3d[:,2]
        pts_3d[:,1] /= cam_3d[:,2]
        pts_2d=pts_3d[:,0:2]
        print(pts_2d.shape)
        print("image coordinate")
        print(pts_2d[1,0])
        print(pts_2d[1,1])

        # fov_inds = (pts_2d[:,0]<640) & (pts_2d[:,0]>=0) & \
        #     (pts_2d[:,1]<480) & (pts_2d[:,1]>=0)
        # fov_inds = (pts_2d[:,0]<640) & (pts_2d[:,0]>0) & \
        #     (pts_2d[:,1]<480) & (pts_2d[:,1]>0)
        # fov_inds = fov_inds & (pc_velo[:,0]<0)
        fov_inds = pc_velo[:,0]>0
        print(fov_inds.shape)
        print(pts_2d.shape)

        imgfov_pc_velo = pc_velo[fov_inds, :]
        print(imgfov_pc_velo.shape)
        #pts_2d0=calib.project_velo_to_image(imgfov_pc_velo)
        pc_4_0=expend3(imgfov_pc_velo)
        cam_3d_0 = np.dot(pc_4_0,np.transpose(P_1))
        pts_3d_0=np.dot(cam_3d_0,np.transpose(P_0))
        pts_3d_0[:,0] /= cam_3d_0[:,2]
        pts_3d_0[:,1] /= cam_3d_0[:,2]
        pts_2d0=pts_3d_0[:,0:2]
        print("camera")
        print(pts_2d0.shape)
        print("image size")
        print(len(image[0]))
        print(len(image))
        fov_inds0 = (pts_2d0[:,0]<len(image[0])) & (pts_2d0[:,0]>=0) & \
            (pts_2d0[:,1]<len(image)) & (pts_2d0[:,1]>=0)
        imgfov_pc_velo = pc_velo[fov_inds, :]
        #fov_inds0 = fov_inds0 & (imgfov_pc_velo[:,0]>2.0)
        print(fov_inds0.shape)
        #imgfov_pc_velo0 = imgfov_pc_velo[fov_inds0, :]
        #print(imgfov_pc_velo0.shape)
        if(chang>0):
            for i in range(len(fov_inds0)):
                if((pts_2d0[i,1]<len(street_prediction))&(pts_2d0[i,0]<len(street_prediction[0]))&(pts_2d0[i,0]>=0)&(pts_2d0[i,1]>=0)):
                    fov_inds0[i]=fov_inds0[i] & (street_prediction[int(pts_2d0[i,1]),int(pts_2d0[i,0])]==True)

        # if(chang>0):
        #     for i in range(len(fov_inds)):
        #         if((pts_2d0[i,1]<len(street_prediction))&(pts_2d0[i,0]<len(street_prediction[0]))):
        #             fov_inds[i]=fov_inds[i] & (street_prediction[int(pts_2d0[i,1]),int(pts_2d0[i,0])]==True)
        #imgfov_pc_velo0 = imgfov_pc_velo[fov_inds0, :]
        print("number")
        green_image = tv_utils.fast_overlay(image, street_prediction)
        imgfov_pc_velo0 = imgfov_pc_velo[fov_inds0, :]
        # pub point-cloud topic
        print(imgfov_pc_velo0.shape)
        videoWriter.write(green_image)
        number=len(imgfov_pc_velo0)

        # draw_lidar(pc_velo, fig=fig)

        header=std_msgs.msg.Header()
        header.stamp=rospy.Time.now()
        header.frame_id="velodyne"
        points=pc2.create_cloud_xyz32(header,imgfov_pc_velo0)
        point_pub.publish(points)
        
        # raw_input()

    # make a video 
    video_dir='/home/user/Data/lrx_work/free-space/hitsz.avi'
    fps=10
    num=4541
    img_size=(640,480)
    fourcc='mp4v'
    videoWriter=cv2.VideoWriter(video_dir,cv2.VideoWriter_fourcc(*fourcc),fps,img_size)

    # get transform martix 
    calib = dataset.get_calibration(0)

    point_pub = rospy.Publisher('new_cloud',PointCloud2,queue_size=50)
    rospy.init_node('point-cloud',anonymous=True)
    image_sub = message_filters.Subscriber("/usb_cam/image_raw",newImage)
    point_sub = message_filters.Subscriber("/velodyne_points",PointCloud2)
    # ts = message_filters.TimeSynchronizer([image_sub, point_sub], 10)
    ts = message_filters.ApproximateTimeSynchronizer([image_sub, point_sub], 10, 0.1, allow_headerless=True) 
    print("here")
    ts.registerCallback(callback)


    rospy.spin()

    # h = std_msgs.msg.Header()
    # h.frame_id="base_link"
    # h.stamp=rospy.Time.now()
    #rate = rospy.Rate(10)
    #point_msg=PointCloud2()

if __name__ == '__main__':
    tf.app.run()
