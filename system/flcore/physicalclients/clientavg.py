from utils.privacy import *
import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
import socket
import time
import pickle as pkl
import sys
sys.path.append('../../')

socket.setdefaulttimeout(180)


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, ip, port):
        # socket based
        self.id = id
        self.address = (ip, port+id)
        self.client = socket.socket()
        self.client.connect(self.address)
        print('successfully connect')
        self.args = args
        self.main_loop()

    def main_loop(self):
        while True:
            # try:
            received_data = pkl.loads(self.receive_long_data())
            order, data = received_data
            if order == 'stay':
                continue
            print(f'Agent {self.id} {order}')
            if order == 'init':  # initialize work
                id, train_samples, test_samples, param = data
                self.model = copy.deepcopy(self.args.model)
                self.algorithm = self.args.algorithm
                self.dataset = self.args.dataset
                self.device = self.args.device
                self.id = id  # integer
                self.save_folder_name = self.args.save_folder_name

                self.num_classes = self.args.num_classes
                self.train_samples = train_samples
                self.test_samples = test_samples
                self.batch_size = self.args.batch_size
                self.learning_rate = self.args.local_learning_rate
                self.local_epochs = self.args.local_epochs

                # check BatchNorm
                self.has_BatchNorm = False
                for layer in self.model.children():
                    if isinstance(layer, nn.BatchNorm2d):
                        self.has_BatchNorm = True
                        break

                self.train_slow = param['train_slow']
                self.send_slow = param['send_slow']
                self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
                self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

                self.privacy = self.args.privacy
                self.dp_sigma = self.args.dp_sigma

                self.loss = nn.CrossEntropyLoss()
                self.optimizer = torch.optim.SGD(
                    self.model.parameters(), lr=self.learning_rate)
                self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer=self.optimizer,
                    gamma=self.args.learning_rate_decay_gamma
                )
                self.learning_rate_decay = self.args.learning_rate_decay
                # send feedback message
                self.client.sendall(pkl.dumps(('finish init', 'placeholder')))
                print(f'Agent {self.id} initialize done')

            elif order == 'train':
                self.train()

                # send feedback message
                self.client.sendall(pkl.dumps(('finish train', 'placeholder')))
                print(f'Agent {self.id} train done')
            elif order == 'set_parameters':
                self.model.load_state_dict(data)

                # send feedback message
                self.client.sendall(
                    pkl.dumps(('finish set_parameters', 'placeholder')))
                print(f'Agent {self.id} set parameters done')

            elif order == 'test_metrics':
                test_acc, test_num, auc = self.test_metrics()

                # send feedback message
                self.client.sendall(
                    pkl.dumps(('finish test_metrics', [test_acc, test_num, auc])))
                print(f'Agent {self.id} test metrics done')
            elif order == 'train_metrics':
                losses, train_num = self.train_metrics()

                # send feedback message
                self.client.sendall(
                    pkl.dumps(('finish train_metrics', [losses, train_num])))
                print(f'Agent {self.id} train metrics done')
            elif order == 'synchronize':
                time.sleep(2)  # 艹，sb，这个地方必须放一个空语句，吗的智障！！
                # send feedback message
                self.client.sendall(
                    pkl.dumps(('finish synchronize', self.model.state_dict())))
                print(f'Agent {self.id} synchronize done')
            elif order == 'synchronize1':
                time.sleep(2)  # 艹，sb，这个地方必须放一个空语句，吗的智障！！
                # send feedback message
                self.client.sendall(
                    pkl.dumps(('finish synchronize1', self.train_time_cost)))
                print(f'Agent {self.id} synchronize1 done')
            elif order == 'synchronize2':
                time.sleep(2)  # 艹，sb，这个地方必须放一个空语句，吗的智障！！
                # send feedback message
                self.client.sendall(
                    pkl.dumps(('finish synchronize2', self.send_time_cost)))
                print(f'Agent {self.id} synchronize2 done')

            # except:
            #     print('exception')
            #     break

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=False)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(),
                                    classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y

    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" +
                   str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))

    def receive_long_data(self):
        '''
        处理过长的tcp内容
        :return:
        '''
        total_data = bytes()
        while True:
            data = self.client.recv(2048)
            total_data += data
            if len(data) < 2048:
                break
        return total_data

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        # differential privacy
        if self.privacy:
            self.model, self.optimizer, trainloader, privacy_engine = \
                initialize_dp(self.model, self.optimizer,
                              trainloader, self.dp_sigma)

        start_time = time.time()

        max_local_steps = self.local_epochs
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")
