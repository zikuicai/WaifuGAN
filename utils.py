import math
import random
import scipy.misc
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


def show_all_variables():
    # print all trainable variables
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def image_manifold_size(num_images):
    # see if the height and width are the same
    manifold_h = int(np.floor(np.sqrt(num_images)))
    manifold_w = int(np.ceil(np.sqrt(num_images)))
    assert manifold_h * manifold_w == num_images
    return manifold_h, manifold_w

def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64):
    image = scipy.misc.imread(image_path).astype(np.float)
    return transform(image, input_height, input_width,
                     resize_height, resize_width)

def save_images(images, size, image_path):
    images = inverse_transform(images)
    image = np.squeeze(merge(images, size))
    scipy.misc.imsave(image_path, image)

def merge(images, size):  
    """Merge size[0]*size[1] small pictures into a big one
    :param size: an 1x2 array, like [8,8]
    """
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3, 4)):  # RGB
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')

def transform(image, resize_height=64, resize_width=64):
    resized_image = scipy.misc.imresize(image, [resize_height, resize_width])
    image = np.array(resized_image)/127.5 - 1. 
    return image

def inverse_transform(images):
    return (images+1.)/2.

def visualize(sess, dcgan, config):
    image_frame_dim = int(math.ceil(config.batch_size**.5))
    # set the checkerboard dimension to be the square root of batch size
    # generate n x batch-size images and save them in the samples folder
    n = 10
    for idx in range(n):
        print(" [*] generating pic %d" % idx)
        z_sample = np.random.uniform(-1, 1, size=(config.batch_size, dcgan.z_dim))
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
        save_images(samples, [image_frame_dim, image_frame_dim],
                    config.sample_dir+'/test_%04d.png' % (idx))

def make_gif(images, fname, duration=2, true_image=False):
    """make a gif of certain duartion from a bunch of images
    """
    import moviepy.editor as mpy

    def make_frame(t):
        try:
            x = images[int(len(images)/duration*t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x+1)/2*255).astype(np.uint8)

    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.write_gif(fname, fps=len(images) / duration)
