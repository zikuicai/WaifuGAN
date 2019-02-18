from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np

from ops import *
from utils import *


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class DCGAN(object):
    def __init__(self, sess, input_height=108, input_width=108, crop=True,
                 batch_size=64, sample_num=64, output_height=64, output_width=64,
                 z_dim=100, gf_dim=64, df_dim=64, gfc_dim=1024, dfc_dim=1024, c_dim=3,
                 data_dir='default', input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None):
        """Initialize some default parameters.

        Args:
          sess: TensorFlow session
          batch_size: The size of batch. Should be specified before training.
          y_dim: (optional) Dimension of dim for conditional variable y (see paper page 10). [None]
          z_dim: (optional) Dimension of dim for random noise Z. [100]
          gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
          df_dim: (optional) Dimension of discriminator filters in the first conv layer. [64]
          gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
          dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
          c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.crop = crop

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        # batch normalization : deals with poor initialization helps gradient flow
        # it's a class, used as a function
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        self.data_dir = data_dir
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.data = glob(os.path.join(self.data_dir, self.input_fname_pattern))

        imreadImg = imread(self.data[0])
        if len(imreadImg.shape) >= 3:  # check color channel number
            self.c_dim = imread(self.data[0]).shape[-1]
        else:
            self.c_dim = 1

        self.grayscale = (self.c_dim == 1)

        self.build_model()

    def build_model(self):

        if self.crop:  # crop the input image into output size
            image_dims = [self.output_height, self.output_width, self.c_dim]
        else:
            image_dims = [self.input_height, self.input_width, self.c_dim]

        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='real_images')
        # inputs are real images
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        # z is noise
        self.g = self.generator(self.z)
        # G is the generated image
        self.sampler = self.sampler(self.z)

        self.d_real, self.d_real_logits = self.discriminator(self.inputs)
        # Logit is a function that maps probabilities ( [0, 1] ) to R ( (-inf, inf) )
        # D is the probability that real images being recognized as real
        self.d_fake, self.d_fake_logits = self.discriminator(self.g, reuse=True)
        # D_ is the probability that fake images being recognized as real

        self.z_smry = histogram_summary("z", self.z)
        self.d_real_smry = histogram_summary("d_real", self.d_real)
        self.d_fake_smry = histogram_summary("d_fake", self.d_fake)
        self.g_smry = image_summary("g", self.g)

        self.d_loss_real = tf.reduce_mean(
            loss(self.d_real_logits, tf.ones_like(self.d_real)))
        # the loss of recognizing real images
        # for the real images, we want them to be classified as positives,  
        # so we want their labels to be all ones.
        
        self.d_loss_fake = tf.reduce_mean(
            loss(self.d_fake_logits, tf.zeros_like(self.d_fake)))
        # the loss of recognizing fake images
        # for the fake images produced by the generator, 
        # we want the discriminator to clissify them as false images,
        # so we set their labels to be all zeros.

        self.d_loss = self.d_loss_real + self.d_loss_fake
        # the loss for discriminator to recognize all images
        self.g_loss = tf.reduce_mean(
            loss(self.d_fake_logits, tf.ones_like(self.d_fake)))
        # the loss for generator to make real enough iamges

        self.d_loss_real_smry = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_smry = scalar_summary("d_loss_fake", self.d_loss_fake)
        self.d_loss_smry = scalar_summary("d_loss", self.d_loss)
        self.g_loss_smry = scalar_summary("g_loss", self.g_loss)

        t_vars = tf.trainable_variables()  # define all variables

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()  # save trained variables

    def train(self, config):  # define the optimizer for discriminator and generator
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        tf.initializers.global_variables().run()

        self.g_smry = merge_summary([self.z_smry, self.d_fake_smry,
                                    self.g_smry, self.d_loss_fake_smry, self.g_loss_smry])
        self.d_smry = merge_summary(
            [self.z_smry, self.d_real_smry, self.d_loss_real_smry, self.d_loss_smry])
        self.writer = SummaryWriter("./logs", self.sess.graph)
        # merge the variables that are related to generator and discriminator

        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))
        # initialize the noise

        sample_files = self.data[0:self.sample_num]
        sample = [
            get_image(sample_file,
                      input_height=self.input_height,
                      input_width=self.input_width,
                      resize_height=self.output_height,
                      resize_width=self.output_width,
                      crop=self.crop,
                      grayscale=self.grayscale) for sample_file in sample_files]
        if (self.grayscale):
            sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_inputs = np.array(sample).astype(np.float32)

        counter = 1
        # number of batches has trained on
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        # load checkpoint and decide whether to train from scratch
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in range(config.epoch):

            self.data = glob(os.path.join(config.data_dir, self.input_fname_pattern))
            # get all images in arbitrary order
            batch_idxs = min(len(self.data), config.train_size) // config.batch_size
            # how many batches the whole data set can be divided into

            for idx in range(0, batch_idxs):
                batch_files = self.data[idx *config.batch_size:(idx+1)*config.batch_size]
                batch = [
                    get_image(batch_file,
                              input_height=self.input_height,
                              input_width=self.input_width,
                              resize_height=self.output_height,
                              resize_width=self.output_width,
                              crop=self.crop,
                              grayscale=self.grayscale) for batch_file in batch_files]
                if self.grayscale:
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)

                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                    .astype(np.float32)

                # Update D network
                _, summary_str = self.sess.run(
                    [d_optim, self.d_smry],
                    feed_dict={self.inputs: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_smry],
                                               feed_dict={self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_smry],
                                               feed_dict={self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                errD_real = self.d_loss_real.eval({self.inputs: batch_images})
                errG = self.g_loss.eval({self.z: batch_z})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"
                      % (epoch + 1, idx, batch_idxs,
                         time.time() - start_time, errD_fake+errD_real, errG))

                if np.mod(counter, 100) == 1:
                    # save generated sample images every 100 batches
                    try:
                        samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={
                                self.z: sample_z,
                                self.inputs: sample_inputs,
                            },
                        )
                        save_images(samples, image_manifold_size(samples.shape[0]),
                                    './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" %
                              (d_loss, g_loss))
                    except:
                        print("one pic error!...")

                if np.mod(counter, 500) == 2:
                    # save checkpoint every 500 batches
                    self.save(config.checkpoint_dir, counter)

    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            # last dimension of output: h0 is 64, h3 is 512
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
            h4 = dense(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_dense')
            # return the probability and logits
            return tf.nn.sigmoid(h4), h4

    def generator(self, z):
        with tf.variable_scope("generator") as scope:
            # generator is the reverse of discriminator
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            # project `z` and reshape
            z_ = dense(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_dense')

            h0 = tf.reshape(z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(h0))

            h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, 
                               self.gf_dim*4], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1))

            h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, 
                               self.gf_dim*2], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, 
                               self.gf_dim*1], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4 = deconv2d(h3, [self.batch_size, s_h, s_w, 
                               self.c_dim], name='g_h4')

            return tf.nn.tanh(h4)

    def sampler(self, z):
        # sampler is just a copy of generator without updating variables
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            # project `z` and reshape
            h0 = tf.reshape(
                dense(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_dense'),
                [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(h0, train=False))

            h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8,
                               self.gf_dim*4], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1, train=False))

            h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4,
                               self.gf_dim*2], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2, train=False))

            h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2,
                               self.gf_dim*1], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3, train=False))

            h4 = deconv2d(h3, [self.batch_size, s_h, s_w, 
                               self.c_dim], name='g_h4')

            return tf.nn.tanh(h4)

    # property decorator
    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.data_dir, self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(
                checkpoint_dir, ckpt_name))
            counter = int(
                next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
