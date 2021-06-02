import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from  torch.utils.data import DataLoader
from torch import autograd
from torch import optim
from torch.optim import SGD

import time
import pprint
import datetime
from lstm import *
from utils import *
from logger import *

cuda = True if torch.cuda.is_available() else False

# =============ログ===============
LOGGER = logging.getLogger('lstm')
LOGGER.setLevel(logging.DEBUG)

LOGGER.info('Initialized logger.')
init_console_logger(LOGGER)

# =============パラメータ===============
args = parse_arguments()
epochs = args['num_epochs']
batch_size = args['batch_size']
ngpus = args['ngpus']
model_size = args['model_size']
hiddenlayer_size = args['hiddenlayer_size']
model_dir = make_path(os.path.join(args['output_dir'],
                                   datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
args['model_dir'] = model_dir


# ディレクトリまでのパス
#data_dir = args['data_dir']
output_dir = args['output_dir']

# =============ネットワーク===============
training_size = 10000
test_size = 1000
hiddenlayer_size = 5
#model = LSTM(model_size = model_size, batch_size = batch_size, hidden_layer_size = hiddenlayer_size)
model = LSTM(1, hiddenlayer_size, 1)
#if cuda:
    #model = torch.nn.DataParallel(model).cuda()
loss_function = nn.MSELoss()
#optimizer_lstm = optim.Adam(model.parameters(), lr=args['learning_rate'], betas=(args['beta1'], args['beta2']))
optimizer_lstm = SGD(model.parameters(), lr=0.01)

train_x, train_t = mkDataSet(training_size)
test_x, test_t = mkDataSet(test_size)

# =============学習===============
start = time.time()

loss_train_history = []
loss_test_history = []

LOGGER.info('Starting training...EPOCHS={}, BATCH_SIZE={}'.format(epochs, batch_size))
for epoch in range(1, epochs+1):
    LOGGER.info("{} Epoch: {}/{}".format(time_since(start), epoch, epochs))
    loss_train_epoch = []
    for i in range(int(training_size / batch_size)):
        optimizer_lstm.zero_grad()

        data, label = mkRandomBatch(train_x, train_t, batch_size)
        if cuda:
            data = data.cuda()
        output = model(data)

        loss = loss_function(output, label)
        loss.backward(retain_graph=True)
        optimizer_lstm.step()

        if cuda:
            loss = loss.cpu()

        loss_train_epoch.append(loss.data.numpy())

    loss_train_epoch_avg = sum(loss_train_epoch) / float(len(loss_train_epoch))
    loss_train_history.append(loss_train_epoch_avg)

    #test
    loss_test_epoch = []
    for i in range(int(test_size / batch_size)):
        offset = i * batch_size
        data, label = torch.tensor(test_x[offset:offset+batch_size]), torch.tensor(test_t[offset:offset+batch_size])

        if cuda:
            data = data.cuda()
        output = model(data, None)
        loss = loss_function(output, label)
        if cuda:
            loss = loss.cpu()
        loss_test_epoch.append(loss.data.numpy())

    loss_test_epoch_avg = sum(loss_test_epoch) / float(len(loss_test_epoch))
    loss_test_history.append(loss_test_epoch_avg)

    LOGGER.info("{} loss_train:{:.4f}| loss_test:{:.4f}".format(time_since(start), loss_train_epoch_avg, loss_test_epoch_avg))

elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

LOGGER.info('>>>>>>>Training finished !<<<<<<<')

# Save model
LOGGER.info("Saving models...")
lstm_path = os.path.join(output_dir, "lstm.pkl")
torch.save(model.state_dict(), lstm_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)


# Plot loss curve.
LOGGER.info("Saving loss curve...")
plot_loss(loss_train_history, loss_test_history, output_dir)

LOGGER.info("All finished!")
