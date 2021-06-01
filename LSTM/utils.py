import os
import time
import math
import torch
import logging
import argparse
from config import *
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


LOGGER = logging.getLogger('lstm')
LOGGER.setLevel(logging.DEBUG)


def make_path(output_path):
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    return output_path


# traindata = DATASET_NAME
output = make_path(OUTPUT_PATH)


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def mkDataSet(data_size, data_length=50, freq=60., noise=0.00):
    """
    params\n
    data_size : データセットサイズ\n
    data_length : 各データの時系列長\n
    freq : 周波数\n
    noise : ノイズの振幅\n
    returns\n
    train_x : トレーニングデータ（t=1,2,...,size-1の値)\n
    train_t : トレーニングデータのラベル（t=sizeの値）\n
    """
    train_x = []
    train_t = []

    for offset in range(data_size):
        train_x.append([[math.sin(2 * math.pi * (offset + i) / freq) + np.random.normal(loc=0.0, scale=noise)] for i in range(data_length)])
        train_t.append([math.sin(2 * math.pi * (offset + data_length) / freq)])

    return train_x, train_t

def mkRandomBatch(train_x, train_t, batch_size=10):
    """
    train_x, train_tを受け取ってbatch_x, batch_tを返す。
    """
    batch_x = []
    batch_t = []

    for _ in range(batch_size):
        idx = np.random.randint(0, len(train_x) - 1)
        batch_x.append(train_x[idx])
        batch_t.append(train_t[idx])
    
    return torch.tensor(batch_x), torch.tensor(batch_t)

def plot_loss(loss_train, loss_test, save_path):
    save_path = os.path.join(save_path, "loss_curve.png")
    assert len(loss_train) == len(loss_test)

    x = range(len(loss_train))

    y1 = loss_train
    y2 = loss_test

    plt.plot(x, y1, label='loss_train')
    plt.plot(x, y2, label='loss_test')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path)


def parse_arguments():
    """
    Get command line arguments
    """
    parser = argparse.ArgumentParser(description='Train a LSTM on a given set of sequence data')

    parser.add_argument('-ms', '--model-size', dest='model_size', type=int, default=1,
                        help='Model size parameter used in LSTM')
    parser.add_argument('-lra', '--lrelu-alpha', dest='alpha', type=float, default=0.2,
                        help='Slope of negative part of LReLU used by discriminator')
    parser.add_argument('-bs', '--batch-size', dest='batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size used for training')
    parser.add_argument('-hs', '--hiddenlayer_size', dest='hiddenlayer_size', type=int, default=100, help='hiddenlayer size used for training')
    parser.add_argument('-ne', '--num-epochs', dest='num_epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('-ng', '--ngpus', dest='ngpus', type=int, default=4,
                        help='Number of GPUs to use for training')
    parser.add_argument('-lr', '--learning-rate', dest='learning_rate', type=float, default=1e-4,
                        help='Initial ADAM learning rate')
    parser.add_argument('-bo', '--beta-one', dest='beta1', type=float, default=0.5, help='beta_1 ADAM parameter')
    parser.add_argument('-bt', '--beta-two', dest='beta2', type=float, default=0.9, help='beta_2 ADAM parameter')
    #parser.add_argument('-data_dir', '--data_dir', dest='data_dir', type=str, default=traindata, help='Path to directory containing sequence data files')
    parser.add_argument('-output_dir', '--output_dir', dest='output_dir', type=str, default=output, help='Path to directory where model files will be output')
    args = parser.parse_args()
    return vars(args)
