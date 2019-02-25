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


def imread(path):
        return scipy.misc.imread(path).astype(np.float)


def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True):
    image = imread(image_path)
    return transform(image, input_height, input_width,
                     resize_height, resize_width, crop)



def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


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


def random_crop(img, width, height):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    return img


def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(
        x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])


def transform(image, input_height, input_width,
              resize_height=64, resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(
            image, input_height, input_width,
            resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(
            image, [resize_height, resize_width])
    return np.array(cropped_image)/127.5 - 1.


def inverse_transform(images):
    return (images+1.)/2.


def visualize(sess, dcgan, config):
    image_frame_dim = int(math.ceil(config.batch_size**.5))
    # set the checkerboard dimension to be the square root of batch size
    
    # generate n x batch-size images and save them in the samples folder
    n = 100
    for idx in range(n):
        print(" [*] generating pic %d" % idx)
        z_sample = np.random.uniform(-1, 1, size=(config.batch_size, dcgan.z_dim))
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
        save_images(samples, [image_frame_dim, image_frame_dim],
                    './samples/test_%04d.png' % (idx))


def image_manifold_size(num_images):
    # see if the height and width are the same
    manifold_h = int(np.floor(np.sqrt(num_images)))
    manifold_w = int(np.ceil(np.sqrt(num_images)))
    assert manifold_h * manifold_w == num_images
    return manifold_h, manifold_w
