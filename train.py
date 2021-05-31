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
latent_dim = args['latent_dim']
ngpus = args['ngpus']
model_size = args['model_size']
model_dir = make_path(os.path.join(args['output_dir'],
                                   datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
args['model_dir'] = model_dir
# Nエポック毎のサンプルを保存
epochs_per_sample = args['epochs_per_sample']
# gradient penalty regularization factor.
lmbda = args['lmbda']

# ディレクトリまでのパス
audio_dir = args['audio_dir']
output_dir = args['output_dir']

# =============ネットワーク===============
lstm = LSTM()

if cuda:
    lstm = torch.nn.DataParallel(lstm).cuda()

optimizer_lstm = optim.Adam(lstm.parameters(), lr=args['learning_rate'], betas=(args['beta1'], args['beta2']))

# データのロード.
LOGGER.info('Loading sequence data...')
df_dic = {}
inp_dim = 0
for i in os.listdir(audio_dir):
    with open(audio_dir + i, 'rb') as f:
        df = pickle.load(f)
        #入力ベクトルのサイズを取得
        if len(df) > inp_dim:
            inp_dim = len(df)
    df_dic[i] = df

#入力ベクトルの次元を持つデータの抽出
key = []
for k, v in df_dic.itmes():
    if len(v) == inp_dim:
        key.append(k)

#訓練データとテストデータに分割
train_k, test_k = train_test_split(key, test_size = 0.3, shuffle = True)
train = torch.zeros((len(train_k), inp_dim))
test = torch.zeros((len(test_k), inp_dim))
for n, k in enumerate(train_k):
    nd_cast = df_dic[k].調整後終値.values.astype(np.float32)
    train[n] =  torch.from_numpy(nd_cast).clone()
for n, k in enumerate(test_k):
    nd_cast = df_dic[k].調整後終値.values.astype(np.float32)
    test[n] = torch.from_numpy(nd_cast).clone()

#minmax標準化(正規分布で標準化すると外れ値に引きずられるため)
X_max, X_min = max(train.flatten()), min(train.flatten())
train_norm = (train - X_min) / (X_max - X_min)
test_norm = (test - X_min) / (X_max - X_min)
TOTAL_TRAIN_SAMPLES = train_size
BATCH_NUM = TOTAL_TRAIN_SAMPLES // batch_size

#時系列データに分割する。
window = 6
X_train, y_train, X_test, y_test= [], [], [], []
for n in range(len(train)):
    for i in range(inp_dim - window):
        X_train.append(train_norm[n][i:i+window])
        y_train.append(train_norm[n][i+window])
        
for n in range(len(test)):
    for i in range(inp_dim - window):
        X_test.append(test_norm[n][i:i+window])
        y_test.append(test_norm[n][i+window])
train_iter = iter(train_data)
valid_iter = iter(valid_data)
test_iter = iter(test_data)

#Dataloaderに引数として渡すためのデータセット作成
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, X, y):
        self.data = X
        self.teacher = y

    def __len__(self):
        return len(self.teacher)

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label =  self.teacher[idx]

        return out_data, out_label

# =============学習===============
epochs = 50
batch_size = 32

dataset = MyDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=batch_size)

model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

sample_len = len(X_train)
for i in range(epochs):
    for b, tup in enumerate(dataloader):
        X, y = tup
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, len(X), model.hidden_layer_size),
                             torch.zeros(1, len(X), model.hidden_layer_size))

        y_pred = model(X)

        single_loss = loss_function(y_pred, y)
        single_loss.backward()
        optimizer.step()

    print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
history = []
D_costs_train = []
D_wasses_train = []
D_costs_valid = []
D_wasses_valid = []
G_costs = []

start = time.time()
LOGGER.info('Starting training...EPOCHS={}, BATCH_SIZE={}, BATCH_NUM={}'.format(epochs, batch_size, BATCH_NUM))
for epoch in range(1, epochs+1):
    LOGGER.info("{} Epoch: {}/{}".format(time_since(start), epoch, epochs))

    D_cost_train_epoch = []
    D_wass_train_epoch = []
    D_cost_valid_epoch = []
    D_wass_valid_epoch = []
    G_cost_epoch = []
    for i in range(1, BATCH_NUM+1):
        # Set Discriminator parameters to require gradients.
        for p in netD.parameters():
            p.requires_grad = True

        one = torch.tensor(1, dtype=torch.float)
        neg_one = one * -1
        if cuda:
            one = one.cuda()
            neg_one = neg_one.cuda()
        #############################
        # (1) Train Discriminator
        #############################
        for iter_dis in range(5):
            netD.zero_grad()

            # Noise
            noise = torch.Tensor(batch_size, latent_dim).uniform_(-1, 1)
            if cuda:
                noise = noise.cuda()
            noise_Var = Variable(noise, requires_grad=False)

            real_data_Var = numpy_to_var(next(train_iter)['X'], cuda)


            # a) compute loss contribution from real training data
            D_real = netD(real_data_Var)
            D_real = D_real.mean()  # avg loss
            D_real.backward(neg_one)  # loss * -1

            # b) compute loss contribution from generated data, then backprop.
            fake = autograd.Variable(netG(noise_Var).data)
            D_fake = netD(fake)
            D_fake = D_fake.mean()
            D_fake.backward(one)

            # c) compute gradient penalty and backprop
            gradient_penalty = calc_gradient_penalty(netD, real_data_Var.data,
                                                     fake.data, batch_size, lmbda,
                                                     use_cuda=cuda)
            gradient_penalty.backward(one)

            # Compute cost * Wassertein loss..
            D_cost_train = D_fake - D_real + gradient_penalty
            D_wass_train = D_real - D_fake

            # Update gradient of discriminator.
            optimizerD.step()

            #############################
            # (2) Compute Valid data
            #############################
            netD.zero_grad()

            valid_data_Var = numpy_to_var(next(valid_iter)['X'], cuda)
            D_real_valid = netD(valid_data_Var)
            D_real_valid = D_real_valid.mean()  # avg loss

            # b) compute loss contribution from generated data, then backprop.
            fake_valid = netG(noise_Var)
            D_fake_valid = netD(fake_valid)
            D_fake_valid = D_fake_valid.mean()

            # c) compute gradient penalty and backprop
            gradient_penalty_valid = calc_gradient_penalty(netD, valid_data_Var.data,
                                                           fake_valid.data, batch_size, lmbda,
                                                           use_cuda=cuda)
            # Compute metrics and record in batch history.
            D_cost_valid = D_fake_valid - D_real_valid + gradient_penalty_valid
            D_wass_valid = D_real_valid - D_fake_valid

            if cuda:
                D_cost_train = D_cost_train.cpu()
                D_wass_train = D_wass_train.cpu()
                D_cost_valid = D_cost_valid.cpu()
                D_wass_valid = D_wass_valid.cpu()

            # Record costs
            D_cost_train_epoch.append(D_cost_train.data.numpy())
            D_wass_train_epoch.append(D_wass_train.data.numpy())
            D_cost_valid_epoch.append(D_cost_valid.data.numpy())
            D_wass_valid_epoch.append(D_wass_valid.data.numpy())

        #############################
        # (3) Train Generator
        #############################
        # Prevent discriminator update.
        for p in netD.parameters():
            p.requires_grad = False

        # Reset generator gradients
        netG.zero_grad()

        # Noise
        noise = torch.Tensor(batch_size, latent_dim).uniform_(-1, 1)
        if cuda:
            noise = noise.cuda()
        noise_Var = Variable(noise, requires_grad=False)

        fake = netG(noise_Var)
        G = netD(fake)
        G = G.mean()

        # Update gradients.
        G.backward(neg_one)
        G_cost = -G

        optimizerG.step()

        # Record costs
        if cuda:
            G_cost = G_cost.cpu()
        G_cost_epoch.append(G_cost.data.numpy())

        if i % (BATCH_NUM // 5) == 0:
            LOGGER.info("{} Epoch={} Batch: {}/{} D_c:{:.4f} | D_w:{:.4f} | G:{:.4f}".format(time_since(start), epoch,
                                                                                             i, BATCH_NUM,
                                                                                             D_cost_train.data.numpy(),
                                                                                             D_wass_train.data.numpy(),
                                                                                             G_cost.data.numpy()))

    # Save the average cost of batches in every epoch.
    D_cost_train_epoch_avg = sum(D_cost_train_epoch) / float(len(D_cost_train_epoch))
    D_wass_train_epoch_avg = sum(D_wass_train_epoch) / float(len(D_wass_train_epoch))
    D_cost_valid_epoch_avg = sum(D_cost_valid_epoch) / float(len(D_cost_valid_epoch))
    D_wass_valid_epoch_avg = sum(D_wass_valid_epoch) / float(len(D_wass_valid_epoch))
    G_cost_epoch_avg = sum(G_cost_epoch) / float(len(G_cost_epoch))

    D_costs_train.append(D_cost_train_epoch_avg)
    D_wasses_train.append(D_wass_train_epoch_avg)
    D_costs_valid.append(D_cost_valid_epoch_avg)
    D_wasses_valid.append(D_wass_valid_epoch_avg)
    G_costs.append(G_cost_epoch_avg)

    LOGGER.info("{} D_cost_train:{:.4f} | D_wass_train:{:.4f} | D_cost_valid:{:.4f} | D_wass_valid:{:.4f} | "
                "G_cost:{:.4f}".format(time_since(start),
                                       D_cost_train_epoch_avg,
                                       D_wass_train_epoch_avg,
                                       D_cost_valid_epoch_avg,
                                       D_wass_valid_epoch_avg,
                                       G_cost_epoch_avg))

elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    # TODO
    # Early stopping by Inception Score(IS)

LOGGER.info('>>>>>>>Training finished !<<<<<<<')

# Save model
LOGGER.info("Saving models...")
lstm_path = os.path.join(output_dir, "lstm.pkl")
torch.save(model.state_dict(), lstm_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)


# Plot loss curve.
LOGGER.info("Saving loss curve...")
plot_loss(D_costs_train, D_wasses_train,
          D_costs_valid, D_wasses_valid, G_costs, output_dir)

LOGGER.info("All finished!")
