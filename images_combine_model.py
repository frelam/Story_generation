import tensorflow as tf
import model
import numpy as np
from utils.nn import NN
from model import CaptionGenerator
class StoryGenerator():
    def __init__(self):
        self.conv_feats = []
        self.reshaped_conv5_3_feats = []
        self.nn = NN

    def build_vgg16(self):
        """ Build the VGG16 net. """
        config = self.config

        images = tf.placeholder(
            dtype = tf.float32,
            shape = [config.batch_size] + self.image_shape)

        conv1_1_feats = self.nn.conv2d(images, 64, name = 'conv1_1')
        conv1_2_feats = self.nn.conv2d(conv1_1_feats, 64, name = 'conv1_2')
        pool1_feats = self.nn.max_pool2d(conv1_2_feats, name = 'pool1')

        conv2_1_feats = self.nn.conv2d(pool1_feats, 128, name = 'conv2_1')
        conv2_2_feats = self.nn.conv2d(conv2_1_feats, 128, name = 'conv2_2')
        pool2_feats = self.nn.max_pool2d(conv2_2_feats, name = 'pool2')

        conv3_1_feats = self.nn.conv2d(pool2_feats, 256, name = 'conv3_1')
        conv3_2_feats = self.nn.conv2d(conv3_1_feats, 256, name = 'conv3_2')
        conv3_3_feats = self.nn.conv2d(conv3_2_feats, 256, name = 'conv3_3')
        pool3_feats = self.nn.max_pool2d(conv3_3_feats, name = 'pool3')

        conv4_1_feats = self.nn.conv2d(pool3_feats, 512, name = 'conv4_1')
        conv4_2_feats = self.nn.conv2d(conv4_1_feats, 512, name = 'conv4_2')
        conv4_3_feats = self.nn.conv2d(conv4_2_feats, 512, name = 'conv4_3')
        pool4_feats = self.nn.max_pool2d(conv4_3_feats, name = 'pool4')

        conv5_1_feats = self.nn.conv2d(pool4_feats, 512, name = 'conv5_1')
        conv5_2_feats = self.nn.conv2d(conv5_1_feats, 512, name = 'conv5_2')
        conv5_3_feats = self.nn.conv2d(conv5_2_feats, 512, name = 'conv5_3')

        reshaped_conv5_3_feats = tf.reshape(conv5_3_feats,
                                            [config.batch_size, 196, 512])

        self.conv_feats.append(reshaped_conv5_3_feats)
        self.reshaped_conv5_3_feats.append(conv5_3_feats)

    def images_combine(self):

        for conv_feats in self.conv_feats:

            # shape = [batch_size, height ,weight ,channel * 5]

            self.images_feat = tf.concat(conv_feats,axis=0)

        '''
        image_feat = tf.placeholder(
            dtype = tf.float32,
            shape = [config.batch_size] + 16 + 16 + config.image_nums * config.images_num)
        '''
        conv1_1_feats = self.nn.conv2d(self.images_feat, 2048, name='conv1_1')
        conv1_2_feats = self.nn.conv2d(conv1_1_feats, 2048, name='conv1_2')
        pool1_feats = self.nn.max_pool2d(conv1_2_feats, name='pool1')

        conv2_1_feats = self.nn.conv2d(pool1_feats, 1024, name='conv2_1')
        conv2_2_feats = self.nn.conv2d(conv2_1_feats, 1024, name='conv2_2')
        pool2_feats = self.nn.max_pool2d(conv2_2_feats, name='pool2')

        fc1 = self.nn.dense(pool2_feats, units=config.combine_fc1, activation=None, name='fc_combine_image')
        fc2 = self.nn.dense(fc1, units=config.combine_fc2, activation=None, name='fc_combine_image')

        return fc2
