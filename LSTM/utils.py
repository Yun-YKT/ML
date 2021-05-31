import os
import time
import math
import torch
import random
import logging
import librosa
import argparse
import pescador
import numpy as np
from config import *
from torch import autograd
from torch.autograd import Variable
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


LOGGER = logging.getLogger('lstm')
LOGGER.setLevel(logging.DEBUG)


def make_path(output_path):
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    return output_path


traindata = DATASET_NAME
output = make_path(OUTPUT_PATH)


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def get_all_audio_filepaths(audio_dir):
    return [os.path.join(root, fname)
            for (root, dir_names, file_names) in os.walk(audio_dir, followlinks=True)
            for fname in file_names
            if (fname.lower().endswith('.wav') or fname.lower().endswith('.mp3'))]


def batch_generator(audio_path_list, batch_size):
    streamers = []
    for audio_path in audio_path_list:
        s = pescador.Streamer(sample_generator, audio_path)
        streamers.append(s)

    mux = pescador.ShuffledMux(streamers)
    batch_gen = pescador.buffer_stream(mux, batch_size)

    return batch_gen


def split_data(audio_path_list, valid_ratio, test_ratio, batch_size):
    num_files = len(audio_path_list)
    num_valid = int(np.ceil(num_files * valid_ratio))
    num_test = int(np.ceil(num_files * test_ratio))
    num_train = num_files - num_valid - num_test

    if not (num_valid > 0 and num_test > 0 and num_train > 0):
        LOGGER.error("Please download DATASET '{}' and put it under current path !".format(DATASET_NAME))

    # Random shuffle the audio_path_list for splitting.
    random.shuffle(audio_path_list)

    valid_files = audio_path_list[:num_valid]
    test_files = audio_path_list[num_valid:num_valid + num_test]
    train_files = audio_path_list[num_valid + num_test:]
    train_size = len(train_files)

    train_data = batch_generator(train_files, batch_size)
    valid_data = batch_generator(valid_files, batch_size)
    test_data = batch_generator(test_files, batch_size)

    return train_data, valid_data, test_data, train_size



def numpy_to_var(numpy_data, cuda):
    """
    Convert numpy array to Variable.
    """
    data = numpy_data[:, np.newaxis, :]
    data = torch.Tensor(data)
    if cuda:
        data = data.cuda()
    return Variable(data, requires_grad=False)


def plot_loss(D_cost_train, D_wass_train, D_cost_valid, D_wass_valid,
              G_cost, save_path):
    assert len(D_cost_train) == len(D_wass_train) == len(D_cost_valid) == len(D_wass_valid) == len(G_cost)

    save_path = os.path.join(save_path, "loss_curve.png")

    x = range(len(D_cost_train))

    y1 = D_cost_train
    y2 = D_wass_train
    y3 = D_cost_valid
    y4 = D_wass_valid
    y5 = G_cost

    plt.plot(x, y1, label='D_loss_train')
    plt.plot(x, y2, label='D_wass_train')
    plt.plot(x, y3, label='D_loss_valid')
    plt.plot(x, y4, label='D_wass_valid')
    plt.plot(x, y5, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path)


def parse_arguments():
    """
    Get command line arguments
    """
    parser = argparse.ArgumentParser(description='Train a LSTM on a given set of sequence data')

    parser.add_argument('-ms', '--model-size', dest='model_size', type=int, default=64,
                        help='Model size parameter used in LSTM')
    parser.add_argument('-lra', '--lrelu-alpha', dest='alpha', type=float, default=0.2,
                        help='Slope of negative part of LReLU used by discriminator')
    parser.add_argument('-bs', '--batch-size', dest='batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size used for training')
    parser.add_argument('-ne', '--num-epochs', dest='num_epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('-ng', '--ngpus', dest='ngpus', type=int, default=4,
                        help='Number of GPUs to use for training')
    parser.add_argument('-lr', '--learning-rate', dest='learning_rate', type=float, default=1e-4,
                        help='Initial ADAM learning rate')
    parser.add_argument('-bo', '--beta-one', dest='beta1', type=float, default=0.5, help='beta_1 ADAM parameter')
    parser.add_argument('-bt', '--beta-two', dest='beta2', type=float, default=0.9, help='beta_2 ADAM parameter')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true')
    parser.add_argument('-audio_dir', '--audio_dir', dest='audio_dir', type=str, default=traindata, help='Path to directory containing audio files')
    parser.add_argument('-output_dir', '--output_dir', dest='output_dir', type=str, default=output, help='Path to directory where model files will be output')
    args = parser.parse_args()
    return vars(args)
