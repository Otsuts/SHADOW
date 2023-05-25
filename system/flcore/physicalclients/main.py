#!/usr/bin/env python
import sys
import copy
sys.path.append('../../')
import torch
import torch.nn as nn
import argparse
import os
import warnings
import torchvision
from flcore.trainmodel.models import *
from flcore.trainmodel.bilstm import *
from flcore.trainmodel.resnet import *
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.mobilenet_v2 import *
from clientavg import Client

warnings.simplefilter("ignore")
torch.manual_seed(0)

# hyper-params for Text tasks
vocab_size = 98635
max_len = 200
emb_dim = 32
class BaseHeadSplit(nn.Module):
    def __init__(self, base, head):
        super(BaseHeadSplit, self).__init__()

        self.base = base
        self.head = head
        
    def forward(self, x):
        out = self.base(x)
        out = self.head(out)

        return out

def get_args():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test",
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="mnist")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay",
                        type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma",
                        type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=2000)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1,
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=2,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-dp', "--privacy", type=bool, default=False,
                        help="differential privacy")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)
    parser.add_argument('-sfn', "--save_folder_name",
                        type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-fte', "--fine_tuning_epoch", type=int, default=0)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP
    parser.add_argument('-bt', "--beta", type=float, default=0.0,
                        help="Average moving parameter for pFedMe, Second learning rate of Per-FedAvg, \
                        or L1 regularization weight of FedTransfer")
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight")
    parser.add_argument('-mu', "--mu", type=float, default=0,
                        help="Proximal rate for FedProx")
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")
    # FedFomo
    parser.add_argument('-M', "--M", type=int, default=5,
                        help="Server only sends M client models to one client at each round")
    # FedMTL
    parser.add_argument('-itk', "--itk", type=int, default=4000,
                        help="The iterations for solving quadratic subproblems")
    # FedAMP
    parser.add_argument('-alk', "--alphaK", type=float, default=1.0,
                        help="lambda/sqrt(GLOABL-ITRATION) according to the paper")
    parser.add_argument('-sg', "--sigma", type=float, default=1.0)
    # APFL
    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    # Ditto / FedRep
    parser.add_argument('-pls', "--plocal_steps", type=int, default=1)
    # MOON
    parser.add_argument('-ta', "--tau", type=float, default=1.0)
    # FedBABU
    parser.add_argument('-fts', "--fine_tuning_steps", type=int, default=10)
    # APPLE
    parser.add_argument('-dlr', "--dr_learning_rate", type=float, default=0.0)
    parser.add_argument('-L', "--L", type=float, default=1.0)
    # FedGen
    parser.add_argument('-nd', "--noise_dim", type=int, default=512)
    parser.add_argument('-glr', "--generator_learning_rate",
                        type=float, default=0.005)
    parser.add_argument('-hd', "--hidden_dim", type=int, default=512)
    parser.add_argument('-se', "--server_epochs", type=int, default=1000)
    parser.add_argument('-lf', "--localize_feature_extractor",
                        type=bool, default=False)
    # SCAFFOLD
    parser.add_argument('-slr', "--server_learning_rate",
                        type=float, default=1.0)
    # FedALA
    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-s', "--rand_percent", type=int, default=80)
    parser.add_argument('-p', "--layer_idx", type=int, default=2,
                        help="More fine-graind than its original paper.")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    model_str = args.model

    # Generate args.model
    if model_str == "mlr":  # convex
        if "mnist" in args.dataset:
            args.model = Mclr_Logistic(
                1*28*28, num_classes=args.num_classes).to(args.device)
        elif "Cifar10" in args.dataset:
            args.model = Mclr_Logistic(
                3*32*32, num_classes=args.num_classes).to(args.device)
        else:
            args.model = Mclr_Logistic(
                60, num_classes=args.num_classes).to(args.device)

    elif model_str == "cnn":  # non-convex
        if "mnist" in args.dataset:
            args.model = FedAvgCNN(
                in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
        elif "Cifar10" in args.dataset:
            args.model = FedAvgCNN(
                in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
        elif "omniglot" in args.dataset:
            args.model = FedAvgCNN(
                in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
            # args.model = CifarNet(num_classes=args.num_classes).to(args.device)
        elif "Digit5" in args.dataset:
            args.model = Digit5CNN().to(args.device)
        else:
            args.model = FedAvgCNN(
                in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

    elif model_str == "dnn":  # non-convex
        if "mnist" in args.dataset:
            args.model = DNN(
                1*28*28, 100, num_classes=args.num_classes).to(args.device)
        elif "Cifar10" in args.dataset:
            args.model = DNN(
                3*32*32, 100, num_classes=args.num_classes).to(args.device)
        else:
            args.model = DNN(
                60, 20, num_classes=args.num_classes).to(args.device)

    elif model_str == "resnet":
        args.model = torchvision.models.resnet18(
            pretrained=False, num_classes=args.num_classes).to(args.device)

        # args.model = torchvision.models.resnet18(pretrained=True).to(args.device)
        # feature_dim = list(args.model.fc.parameters())[0].shape[1]
        # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        # args.model = resnet18(num_classes=args.num_classes, has_bn=True, bn_block_num=4).to(args.device)

    elif model_str == "alexnet":
        args.model = alexnet(
            pretrained=False, num_classes=args.num_classes).to(args.device)

        # args.model = alexnet(pretrained=True).to(args.device)
        # feature_dim = list(args.model.fc.parameters())[0].shape[1]
        # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

    elif model_str == "googlenet":
        args.model = torchvision.models.googlenet(
            pretrained=False, aux_logits=False, num_classes=args.num_classes).to(args.device)

        # args.model = torchvision.models.googlenet(pretrained=True, aux_logits=False).to(args.device)
        # feature_dim = list(args.model.fc.parameters())[0].shape[1]
        # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

    elif model_str == "mobilenet_v2":
        args.model = mobilenet_v2(
            pretrained=False, num_classes=args.num_classes).to(args.device)

        # args.model = mobilenet_v2(pretrained=True).to(args.device)
        # feature_dim = list(args.model.fc.parameters())[0].shape[1]
        # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

    elif model_str == "lstm":
        args.model = LSTMNet(hidden_dim=emb_dim, vocab_size=vocab_size,
                             num_classes=args.num_classes).to(args.device)

    elif model_str == "bilstm":
        args.model = BiLSTM_TextClassification(input_size=vocab_size, hidden_size=emb_dim, output_size=args.num_classes,
                                               num_layers=1, embedding_dropout=0, lstm_dropout=0, attention_dropout=0,
                                               embedding_length=emb_dim).to(args.device)

    elif model_str == "fastText":
        args.model = fastText(hidden_dim=emb_dim, vocab_size=vocab_size,
                              num_classes=args.num_classes).to(args.device)

    elif model_str == "TextCNN":
        args.model = TextCNN(hidden_dim=emb_dim, max_len=max_len, vocab_size=vocab_size,
                             num_classes=args.num_classes).to(args.device)

    elif model_str == "Transformer":
        args.model = TransformerModel(ntoken=vocab_size, d_model=emb_dim, nhead=2, d_hid=emb_dim, nlayers=2,
                                      num_classes=args.num_classes).to(args.device)

    elif model_str == "AmazonMLP":
        args.model = AmazonMLP().to(args.device)

    elif model_str == "harcnn":
        if args.dataset == 'har':
            args.model = HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, conv_kernel_size=(
                1, 9), pool_kernel_size=(1, 2)).to(args.device)
        elif args.dataset == 'pamap':
            args.model = HARCNN(9, dim_hidden=3712, num_classes=args.num_classes, conv_kernel_size=(
                1, 9), pool_kernel_size=(1, 2)).to(args.device)

    else:
        raise NotImplementedError
    if args.algorithm == "FedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)

    return args


def main(args,id=0):
    client = Client(args,id,'127.0.0.1',8085)

if __name__ == "__main__":
    args = get_args()
    main(args)
