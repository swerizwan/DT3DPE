# Import necessary libraries
import os
import numpy as np
from PIL import Image
from utils import paramUtil
import math
import time
import matplotlib.pyplot as plt


# Function to create a directory if it does not exist
def mkdir(path):
    """
    Creates a directory at the specified path if it does not already exist.
    :param path: The path where the directory should be created.
    """
    if not os.path.exists(path):
        os.makedirs(path)

# List of colors used for visualization
COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

# Placeholder value for missing data
MISSING_VALUE = -1

# Function to save an image from a numpy array
def save_image(image_numpy, image_path):
    """
    Saves an image from a numpy array to the specified path.
    :param image_numpy: The numpy array representing the image.
    :param image_path: The path where the image should be saved.
    """
    img_pil = Image.fromarray(image_numpy)
    img_pil.save(image_path)


# Function to save a log file with loss values
def save_logfile(log_loss, save_path):
    """
    Saves a log file containing loss values.
    :param log_loss: A dictionary with loss values.
    :param save_path: The path where the log file should be saved.
    """
    with open(save_path, 'wt') as f:
        for k, v in log_loss.items():
            w_line = k
            for digit in v:
                w_line += ' %.3f' % digit
            f.write(w_line + '\n')


# Function to print current loss values during training
def print_current_loss(start_time, niter_state, total_niters, losses, epoch=None, sub_epoch=None,
                       inner_iter=None, tf_ratio=None, sl_steps=None):
    """
    Prints the current loss values during training.
    :param start_time: The start time of the training process.
    :param niter_state: The current iteration number.
    :param total_niters: The total number of iterations.
    :param losses: A dictionary with loss values.
    :param epoch: The current epoch number (optional).
    :param sub_epoch: The current sub-epoch number (optional).
    :param inner_iter: The current inner iteration number (optional).
    :param tf_ratio: The teacher forcing ratio (optional).
    :param sl_steps: The number of steps for scheduled sampling (optional).
    """
    def as_minutes(s):
        """
        Converts seconds to a string in the format 'Xm Ys'.
        :param s: The number of seconds.
        :return: A string representing the time in minutes and seconds.
        """
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def time_since(since, percent):
        """
        Calculates the elapsed time and the estimated remaining time.
        :param since: The start time.
        :param percent: The percentage of completion.
        :return: A string with the elapsed time and the estimated remaining time.
        """
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

    if epoch is not None:
        print('ep/it:%2d-%4d niter:%6d' % (epoch, inner_iter, niter_state), end=" ")

    message = ' %s completed:%3d%%)' % (time_since(start_time, niter_state / total_niters), niter_state / total_niters * 100)

    for k, v in losses.items():
        message += ' %s: %.4f ' % (k, v)
    print(message)


# Function to print current loss values during decomposition training
def print_current_loss_decomp(start_time, niter_state, total_niters, losses, epoch=None, inner_iter=None):
    """
    Prints the current loss values during decomposition training.
    :param start_time: The start time of the training process.
    :param niter_state: The current iteration number.
    :param total_niters: The total number of iterations.
    :param losses: A dictionary with loss values.
    :param epoch: The current epoch number (optional).
    :param inner_iter: The current inner iteration number (optional).
    """
    def as_minutes(s):
        """
        Converts seconds to a string in the format 'Xm Ys'.
        :param s: The number of seconds.
        :return: A string representing the time in minutes and seconds.
        """
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def time_since(since, percent):
        """
        Calculates the elapsed time and the estimated remaining time.
        :param since: The start time.
        :param percent: The percentage of completion.
        :return: A string with the elapsed time and the estimated remaining time.
        """
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

    print('epoch: %03d inner_iter: %5d' % (epoch, inner_iter), end=" ")
    message = '%s niter: %07d completed: %3d%%)' % (time_since(start_time, niter_state / total_niters), niter_state, niter_state / total_niters * 100)
    for k, v in losses.items():
        message += ' %s: %.4f ' % (k, v)
    print(message)


# Function to compose and save a GIF from a list of images
def compose_gif_img_list(img_list, fp_out, duration):
    """
    Composes and saves a GIF from a list of images.
    :param img_list: A list of images.
    :param fp_out: The output file path for the GIF.
    :param duration: The duration of each frame in the GIF.
    """
    img, *imgs = [Image.fromarray(np.array(image)) for image in img_list]
    img.save(fp=fp_out, format='GIF', append_images=imgs, optimize=False,
             save_all=True, loop=0, duration=duration)


# Function to save multiple images to a directory
def save_images(visuals, image_path):
    """
    Saves multiple images to a directory.
    :param visuals: A dictionary with image labels and numpy arrays.
    :param image_path: The directory path where the images should be saved.
    """
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    for i, (label, img_numpy) in enumerate(visuals.items()):
        img_name = '%d_%s.jpg' % (i, label)
        save_path = os.path.join(image_path, img_name)
        save_image(img_numpy, save_path)


# Function to save multiple images to a directory with specific naming
def save_images_test(visuals, image_path, from_name, to_name):
    """
    Saves multiple images to a directory with specific naming.
    :param visuals: A dictionary with image labels and numpy arrays.
    :param image_path: The directory path where the images should be saved.
    :param from_name: The source name for the images.
    :param to_name: The target name for the images.
    """
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    for i, (label, img_numpy) in enumerate(visuals.items()):
        img_name = "%s_%s_%s" % (from_name, to_name, label)
        save_path = os.path.join(image_path, img_name)
        save_image(img_numpy, save_path)


# Function to compose and save an image from a list of images
def compose_and_save_img(img_list, save_dir, img_name, col=4, row=1, img_size=(256, 200)):
    """
    Composes and saves an image from a list of images.
    :param img_list: A list of images.
    :param save_dir: The directory path where the composed image should be saved.
    :param img_name: The file name for the composed image.
    :param col: The number of columns in the composed image.
    :param row: The number of rows in the composed image.
    :param img_size: The size of each image in the composed image.
    """
    compose_img = compose_image(img_list, col, row, img_size)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img_path = os.path.join(save_dir, img_name)
    compose_img.save(img_path)


# Function to compose an image from a list of images
def compose_image(img_list, col, row, img_size):
    """
    Composes an image from a list of images.
    :param img_list: A list of images.
    :param col: The number of columns in the composed image.
    :param row: The number of rows in the composed image.
    :param img_size: The size of each image in the composed image.
    :return: The composed image.
    """
    to_image = Image.new('RGB', (col * img_size[0], row * img_size[1]))
    for y in range(0, row):
        for x in range(0, col):
            from_img = Image.fromarray(img_list[y * col + x])
            paste_area = (x * img_size[0], y * img_size[1],
                          (x + 1) * img_size[0], (y + 1) * img_size[1])
            to_image.paste(from_img, paste_area)
    return to_image


# Function to plot loss curves
def plot_loss_curve(losses, save_path, intervals=500):
    """
    Plots loss curves and saves the plot to a file.
    :param losses: A dictionary with loss values.
    :param save_path: The file path where the plot should be saved.
    :param intervals: The interval for averaging the loss values.
    """
    plt.figure(figsize=(10, 5))
    plt.title("Loss During Training")
    for key in losses.keys():
        plt.plot(list_cut_average(losses[key], intervals), label=key)
    plt.xlabel("Iterations/" + str(intervals))
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path)
    plt.show()


# Function to average a list of values over specified intervals
def list_cut_average(ll, intervals):
    """
    Averages a list of values over specified intervals.
    :param ll: The list of values.
    :param intervals: The interval for averaging.
    :return: A list of averaged values.
    """
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new