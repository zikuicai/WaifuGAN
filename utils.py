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


def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True):
    image = scipy.misc.imread(image_path).astype(np.float)
    return transform(image, input_height, input_width,
                     resize_height, resize_width, crop)


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


def random_crop(img, width, height):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    return img


def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    """Crop an image around the center
    :param x: input image
    :param crop_h: the height of the crop
    :return: an image of size 64x64
    """
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


def image_manifold_size(num_images):
    # see if the height and width are the same
    manifold_h = int(np.floor(np.sqrt(num_images)))
    manifold_w = int(np.ceil(np.sqrt(num_images)))
    assert manifold_h * manifold_w == num_images
    return manifold_h, manifold_w
