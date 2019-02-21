import os
import numpy as np
import tensorflow as tf

from model import DCGAN
from utils import visualize, show_all_variables

flags = tf.app.flags
flags.DEFINE_integer("epoch", 100, "Epoch to train [25]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")

flags.DEFINE_integer("input_height", 96, "The height of image to use. [96]")
flags.DEFINE_integer("input_width", None, "The width of image to use. [None]"
                                          "If None, same value as input_height.")
flags.DEFINE_integer("output_height", 64, "The height of the output images to produce. [64]")
flags.DEFINE_integer("output_width", None, "The width of the output images to produce. [None]"
                                           "If None, same value as output_height.")

flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of adam optimizer. [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam. [0.5]")

flags.DEFINE_string("data_dir", "./data/anime-faces", "The path of images.")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images. [*.jpg]")

flags.DEFINE_string("checkpoint_dir", "./checkpoint", "Directory name to save the checkpoints.")
flags.DEFINE_string("logs_dir", "./logs", "Directory name to save the summary logs.")
flags.DEFINE_string("sample_dir", "./samples", "Directory name to save the image samples.")

flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")

flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")



def main(_):
    print(FLAGS.train)
    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    # Prevent tensorflow from allocating the totality of a GPU memory
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=True

    with tf.Session(config=run_config) as sess:

        dcgan = DCGAN(
            sess,
            input_width=FLAGS.input_width,
            input_height=FLAGS.input_height,
            output_width=FLAGS.output_width,
            output_height=FLAGS.output_height,
            batch_size=FLAGS.batch_size,
            data_dir=FLAGS.data_dir,
            input_fname_pattern=FLAGS.input_fname_pattern,
            crop=FLAGS.crop,
            checkpoint_dir=FLAGS.checkpoint_dir,
            logs_dir=FLAGS.logs_dir,
            sample_dir=FLAGS.sample_dir)

 


        show_all_variables()

        if FLAGS.train:
            dcgan.train(FLAGS)
        else:
            if not dcgan.load(FLAGS.checkpoint_dir)[0]:
                raise Exception("[!] Train a model first, then run test mode")

        visualize(sess, dcgan, FLAGS)


if __name__ == '__main__':
    tf.app.run()
