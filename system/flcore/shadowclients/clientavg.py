import copy
import sys
import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
import socket
import pickle as pkl
from threading import Thread
import time
import queue

socket.setdefaulttimeout(180)

class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        self.params = copy.deepcopy(kwargs)
        # sockets
        self.id = id  # integer
        self.address = ('127.0.0.1', 8085+id)
        self.socket_server = socket.socket()
        self.socket_server.bind(self.address)
        self.socket_server.listen(5)
        self.events = queue.Queue(1)
        self.end = False
        self.can_add = True

        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device

        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay
        self.returned_data = None

        self.create_client()

    def create_client(self):
        while True:
            # 等待客户端连接
            print(f'waiting for connection from {self.address}')
            client, info = self.socket_server.accept()
            print(f'socket established {info}')
            time.sleep(0.5)
            # 给每个客户端创建一个独立的线程进行管理
            thread = Thread(target=self.main_loop, args=(client, info))
            # 设置成守护线程，在主进程退出时可以直接退出
            thread.setDaemon(True)
            thread.start()
            if client:
                break
        self.init()

    def init(self):
        while(self.can_add == False):
            time.sleep(1)
        self.events.put('init')
        self.can_add = False
    
    def synchronize(self):
        while(self.can_add == False):
            time.sleep(1)
        self.events.put('synchronize')
        self.can_add = False
        while(self.can_add == False):
            time.sleep(1)
        param, train_time_cost,send_time_cost = self.returned_data
        self.model.load_state_dict(param)
        self.train_time_cost = train_time_cost
        self.send_time_cost = send_time_cost

        
    def main_loop(self, client, info):
        while True:
            try:
                if self.events.empty():
                    continue
                status = self.events.get()
                print(f'Agent {self.id} {status}')
                self.can_add = False
                if status == 'init':
                    client.sendall(pkl.dumps(('init', [
                                self.id, self.train_samples, self.test_samples]+[self.params])))
                elif status == 'train':
                    client.sendall(pkl.dumps(('train','placeholder')))
                elif status == 'set_parameters':
                    client.sendall(
                        pkl.dumps(('set_parameters', self.model.state_dict())))
                elif status == 'test_metrics':
                    client.sendall(pkl.dumps(('test_metrics','placeholder')))
                elif status == 'train_metrics':
                    client.sendall(pkl.dumps(('train_metrics','placeholder')))
                elif status == 'static':
                    client.sendall(pkl.dumps(('static','placeholder')))
                elif status == 'synchronize':
                    client.sendall(pkl.dumps(('synchronize','placeholder')))
                
                
                received_message,data = pkl.loads(self.receive_long_data(client))
                self.returned_data = data
                print(f'Agent {self.id} {received_message}')
                self.can_add = True
            except:
                sys.exit()

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
        while(self.can_add == False):
            pass

        self.events.put('set_parameters')
        self.can_add = False
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    # def clone_model(self, model, target):
    #     for param, target_param in zip(model.parameters(), target.parameters()):
    #         target_param.data = param.data.clone()
    #         # target_param.grad = param.grad.clone()

    # def update_parameters(self, model, new_params):
    #     self.status = 'update_parameters'
    #     for param, new_param in zip(model.parameters(), new_params):
    #         param.data = new_param.data.clone()

    def test_metrics(self):
        while(self.can_add == False):
            time.sleep(1)

        self.events.put('test_metrics')
        self.can_add = False
        while(self.can_add == False):
            time.sleep(1)
        test_acc, test_num, auc = self.returned_data
        return test_acc,test_num,auc
        

    def train_metrics(self):
        while(self.can_add == False):
            time.sleep(1)

        self.events.put('train_metrics')
        self.can_add = False
        while(self.can_add == False):
            time.sleep(1)
        losses, train_num = self.returned_data
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

    # def save_item(self, item, item_name, item_path=None):
    #     if item_path == None:
    #         item_path = self.save_folder_name
    #     if not os.path.exists(item_path):
    #         os.makedirs(item_path)
    #     torch.save(item, os.path.join(item_path, "client_" +
    #                str(self.id) + "_" + item_name + ".pt"))

    # def load_item(self, item_name, item_path=None):
    #     if item_path == None:
    #         item_path = self.save_folder_name
    #     return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))
    
    def train(self):
        while(self.can_add == False):
            time.sleep(1)
        self.events.put('train')
        self.can_add = False
        self.synchronize()
        

    def receive_long_data(self, client):
        '''
        TCP一个数据报只能接收1024字节，所以需要把多次接受的拼起来，返回整个数据
        '''
        total_data = bytes()
        while True:
            data = client.recv(1024)
            total_data += data
            if len(data) < 1024:
                break
        return total_data
