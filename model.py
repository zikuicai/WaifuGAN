import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np

from ops import *
from utils import *


class DCGAN(object):
    def __init__(self, sess, flags, z_dim=100, c_dim=3,
                 gf_dim=64, df_dim=64, gfc_dim=1024, dfc_dim=1024):
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
          c_dim: (optional) Dimension of image color. [3]
        """
        self.sess = sess
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        # Parameters from main
        self.crop = flags.crop
        self.epoch = flags.epoch
        self.batch_size = flags.batch_size
        self.input_height = flags.input_height
        self.input_width = flags.input_width
        self.output_height = flags.output_height
        self.output_width = flags.output_width

        self.data_dir = flags.data_dir
        self.input_fname_pattern = flags.input_fname_pattern
        self.checkpoint_dir = flags.checkpoint_dir
        self.logs_dir = flags.logs_dir
        self.sample_dir = flags.sample_dir
        self.data = glob(os.path.join(self.data_dir, self.input_fname_pattern))

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

        self.z_summary = tf.summary.histogram("z", self.z)
        self.d_real_summary = tf.summary.histogram("d_real", self.d_real)
        self.d_fake_summary = tf.summary.histogram("d_fake", self.d_fake)
        self.g_summary = tf.summary.image("g", self.g)

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

        self.d_loss_real_summary = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_summary = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
        self.d_loss_summary = tf.summary.scalar("d_loss", self.d_loss)
        self.g_loss_summary = tf.summary.scalar("g_loss", self.g_loss)

        t_vars = tf.trainable_variables()  # define all variables

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()  # save trained variables

    def train(self, config):  # define the optimizer for discriminator and generator
        
        # setup optimizer
        d_optimizer = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1)
        g_optimizer = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1)

        d_train_op = d_optimizer.minimize(self.d_loss, var_list=self.d_vars)
        g_train_op = g_optimizer.minimize(self.g_loss, var_list=self.g_vars)
        
        tf.initializers.global_variables().run()

        # merge the variables related to generator and discriminator
        self.g_summary = tf.summary.merge([self.z_summary, self.d_fake_summary,
                                    self.g_summary, self.d_loss_fake_summary, self.g_loss_summary])
        self.d_summary = tf.summary.merge(
            [self.z_summary, self.d_real_summary, self.d_loss_real_summary, self.d_loss_summary])
        self.writer = tf.summary.FileWriter(self.logs_dir, self.sess.graph)

        # initialize noise that uniformly distributes between -1 and 1
        sample_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

        sample_files = self.data[0:self.batch_size]
        sample = [get_image(sample_file, self.input_width, self.input_height) 
                  for sample_file in sample_files]
        sample_inputs = np.array(sample).astype(np.float32)

        # number of total batches
        number_images = len(os.listdir(self.data_dir))
        number_batches = number_images // self.batch_size

        batch_counter = 1
        # number of batches has trained on
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        # load checkpoint and decide whether to train from scratch
        if could_load:
            batch_counter = number_batches * checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in range(checkpoint_counter+1, self.epoch+1):
            # shuffle images
            self.data = glob(os.path.join(self.data_dir, self.input_fname_pattern))

            for idx in range(0, number_batches):
                batch_files = self.data[idx *self.batch_size:(idx+1)*self.batch_size]
                batch = [get_image(batch_file, self.input_width, self.input_height) 
                         for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]) \
                    .astype(np.float32)

                # Update D network
                _, summary_str = self.sess.run(
                    [d_train_op, self.d_summary],
                    feed_dict={self.inputs: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, batch_counter)

                # Update G network
                _, summary_str = self.sess.run([g_train_op, self.g_summary],
                                               feed_dict={self.z: batch_z})
                self.writer.add_summary(summary_str, batch_counter)

                # Run g_optimizer twice to make sure that d_loss does not go to zero (different from paper)
                # _, summary_str = self.sess.run([g_train_op, self.g_summary],
                #                                feed_dict={self.z: batch_z})
                # self.writer.add_summary(summary_str, batch_counter)

                errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                errD_real = self.d_loss_real.eval({self.inputs: batch_images})
                errD = errD_fake + errD_real
                errG = self.g_loss.eval({self.z: batch_z})

                batch_counter += 1
                print("Epoch: [%4d] batch: [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"
                      % (epoch, idx + 1, number_batches, time.time() - start_time, errD, errG))

            # save generated sample images every epoch
            try:
                samples, d_loss, g_loss = self.sess.run(
                    [self.sampler, self.d_loss, self.g_loss],
                    feed_dict={
                        self.z: sample_z,
                        self.inputs: sample_inputs,
                    },
                )
                save_images(samples, image_manifold_size(samples.shape[0]),
                            './{}/train_{:04d}.png'.format(self.sample_dir, epoch))
                print("[Sample] d_loss: %.8f, g_loss: %.8f" %(d_loss, g_loss))
            except:
                print("save pic error!...")
            
            # save checkpoint every epoch
            self.save(self.checkpoint_dir, epoch)


    def discriminator(self, x, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            
            # conv2d, batch_norm, leaky_relu

            h0 = conv2d(x, self.df_dim, name='d_h0_conv')
            h0 = lrelu(h0)
            
            h1 = conv2d(h0, self.df_dim*2, name='d_h1_conv')
            h1 = batch_norm(name='d_bn1')(h1)
            h1 = lrelu(h1)

            h2 = conv2d(h1, self.df_dim*4, name='d_h2_conv')
            h2 = batch_norm(name='d_bn2')(h2)
            h2 = lrelu(h2)

            h3 = conv2d(h2, self.df_dim*8, name='d_h3_conv')
            h3 = batch_norm(name='d_bn3')(h3)
            h3 = lrelu(h3)

            h4 = tf.reshape(h3, [self.batch_size, -1])
            h4 = dense(h4, 1)
            
            # return the probability and logits
            return tf.nn.sigmoid(h4), h4


    def generator(self, z):
        """4 layers of convolution
        """
        # generator is the reverse of discriminator
        with tf.variable_scope("generator") as scope:
            # size of the output image, no matter how big the output image is
            s_h0, s_w0 = self.output_height, self.output_width
            # size of the filter in each one of the 4 layers
            s_h1, s_w1 = conv_out_size(s_h0, s_w0, 2, 2)
            s_h2, s_w2 = conv_out_size(s_h1, s_w1, 2, 2)
            s_h3, s_w3 = conv_out_size(s_h2, s_w2, 2, 2)
            s_h4, s_w4 = conv_out_size(s_h3, s_w3, 2, 2)

            # project z to z1 and reshape
            z1 = dense(z, self.gf_dim * 8 * s_h4 * s_w4, 'g_h0_dense')
            h0 = tf.reshape(z1, [-1, s_h4, s_w4, self.gf_dim * 8])
            h0 = batch_norm(name='g_bn0')(h0)
            h0 = tf.nn.relu(h0)

            h1 = deconv2d(h0, [self.batch_size, s_h3, s_w3, self.gf_dim * 4], name='g_h1')
            h1 = batch_norm(name='g_bn1')(h1)
            h1 = tf.nn.relu(h1)

            h2 = deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')
            h2 = batch_norm(name='g_bn2')(h2)
            h2 = tf.nn.relu(h2)

            h3 = deconv2d(h2, [self.batch_size, s_h1, s_w1, self.gf_dim * 1], name='g_h3')
            h3 = batch_norm(name='g_bn3')(h3)
            h3 = tf.nn.relu(h3)

            h4 = deconv2d(h3, [self.batch_size, s_h0, s_w0, self.c_dim], name='g_h4')

            return tf.nn.tanh(h4)


    def sampler(self, z):
        # sampler is just a copy of generator without training
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            s_h0, s_w0 = self.output_height, self.output_width
            s_h1, s_w1 = conv_out_size(s_h0, s_w0, 2, 2)
            s_h2, s_w2 = conv_out_size(s_h1, s_w1, 2, 2)
            s_h3, s_w3 = conv_out_size(s_h2, s_w2, 2, 2)
            s_h4, s_w4 = conv_out_size(s_h3, s_w3, 2, 2)

            z1 = dense(z, self.gf_dim * 8 * s_h4 * s_w4, 'g_h0_dense')
            h0 = tf.reshape(z1, [-1, s_h4, s_w4, self.gf_dim * 8])
            h0 = tf.nn.relu(batch_norm(name='g_bn0')(h0, train=False))

            h1 = deconv2d(h0, [self.batch_size, s_h3, s_w3, 
                               self.gf_dim * 4], name='g_h1')
            h1 = batch_norm(name='g_bn1')(h1, train=False)
            h1 = tf.nn.relu(h1)

            h2 = deconv2d(h1, [self.batch_size, s_h2, s_w2, 
                               self.gf_dim * 2], name='g_h2')
            h2 = batch_norm(name='g_bn2')(h2, train=False)
            h2 = tf.nn.relu(h2)

            h3 = deconv2d(h2, [self.batch_size, s_h1, s_w1, 
                               self.gf_dim * 1], name='g_h3')
            h3 = batch_norm(name='g_bn3')(h3, train=False)
            h3 = tf.nn.relu(h3)

            h4 = deconv2d(h3, [self.batch_size, s_h0, s_w0, 
                               self.c_dim], name='g_h4')

            return tf.nn.tanh(h4)

    # property decorator
    @property
    def model_dir(self):
        dataset_name = self.data_dir.split('/')[-1]
        return "{}_{}_{}_{}".format(
            dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        # how to save only 5 or even less
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
