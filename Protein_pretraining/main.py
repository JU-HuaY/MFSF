import os
import time
import torch
import torch.nn as nn
from torchvision.datasets.folder import default_loader  # 或者自己实现一个图像加载函数
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from data_loader import Protein_pkl_Dataset
from data_loader import my_collate
from network.model import Representation_model
import torch.optim as optim
import timeit
import warnings
warnings.filterwarnings("ignore")
from evaluate import *
from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve, auc

class Trainer(object):
    def __init__(self, model, batch_size, lr, weight_decay):
        self.model = model
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        # self.schedule = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=150, eta_min=0)
        self.batch_size = batch_size
        # self.optimizer = Ranger(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train(self, dataset):
        loss_total, loss1_, loss2_, loss3_ = 0, 0, 0, 0
        self.optimizer.zero_grad()
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                                collate_fn=my_collate)
        for data in dataloader:
            loss_all, loss1, loss2, loss3 = model(data, device, task="pretrain")
            loss_all.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss_total += loss_all.mean().item()
            loss1_ += loss1.mean().item()
            loss2_ += loss2.mean().item()
            loss3_ += loss3.mean().item()
            # loss_total2 += loss3.item()
        return loss_total, loss1_, loss2_, loss3_


class Tester(object):
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size

    def test(self, dataset):
        BS_t, BS_p_s, BS_p_l = [], [], []
        pred_bs_st_collect, pred_bs_lm_collect, label_collect = [], [], []

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                                collate_fn=my_collate)
        for data in dataloader:
            # print(model(data, device, train=False))
            BS, pre_st, pre_lm = model(data, device, task="pretrain", train=False)
            for i in range(len(BS)):
              BS_t.append(BS[i])
              BS_p_s.append(np.argmax(pre_st[i], axis=-1))
              BS_p_l.append(np.argmax(pre_lm[i], axis=-1))
              pred_bs_st_collect.extend(np.argmax(pre_st[i], axis=-1))
              pred_bs_lm_collect.extend(np.argmax(pre_lm[i], axis=-1))
              label_collect.extend(BS[i])

        spec_s, ACC_s, MCC_s, F1_score_s = get_results(BS_t, BS_p_s)
        spec_l, ACC_l, MCC_l, F1_score_l = get_results(BS_t, BS_p_l)
        AUC_s = roc_auc_score(label_collect, pred_bs_st_collect)
        AUC_l = roc_auc_score(label_collect, pred_bs_lm_collect)
        return MCC_s, AUC_s, MCC_l, AUC_l


    def save_Losses(self, Losses, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, Losses)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)


if __name__ == "__main__":
    batchs = 1
    lr = 5e-4
    weight_decay = 0.1
    iteration = 80
    decay_interval = 8
    lr_decay = 0.5
    setting = "pretrain_test"
    device_count = torch.cuda.device_count()
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    train_dataset = Protein_pkl_Dataset(root_dir='BS_data/train')
    validation_dataset = Protein_pkl_Dataset(root_dir='BS_data/validation')
    model = Representation_model(3, 512, 256, 64, 128, batchs, device)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")
    # model = nn.DataParallel(model, device_ids=[0, 1]).to(device)
    # trainer = Trainer(model, batchs, lr, weight_decay)
    # tester = Tester(model, batchs)
    # """Output files."""
    # file_Losses = 'output/result/loss--' + setting + '.txt'
    # file_model = 'output/model/' + setting
    # Losses = ('Epoch\tTime(sec)\tLoss_all\tLoss_1\tLoss_2\tLoss_3\tMCC_s\t'
    #           'F1_score_s\tMCC_l\tF1_score_l')
    # with open(file_Losses, 'w') as f:
    #     f.write(Losses + '\n')
    #
    # """Start training."""
    # print('Training...')
    # print(Losses)
    # start = timeit.default_timer()
    # MCC_ls = 0
    # for epoch in range(1, iteration):
    #
    #     if epoch % decay_interval == 0:
    #         trainer.optimizer.param_groups[0]['lr'] *= lr_decay
    #
    #     # loss_total, loss1_, loss2_, loss3_ = trainer.train(train_dataset)
    #     MCC_s, AUC_s, MCC_l, AUC_l = tester.test(validation_dataset)
    #     end = timeit.default_timer()
    #     time = end - start
    #     print(MCC_s)
    #     print(AUC_s)
    #     print(MCC_l)
    #     print(AUC_l)
        # Losses = [epoch, time, loss_total, loss1_, loss2_, loss3_, MCC_s, F1_score_s, MCC_l, F1_score_l]
        # print('\t'.join(map(str, Losses)))
        # tester.save_Losses(Losses, file_Losses)
        # if MCC_ls < MCC_l:
        #     MCC_ls = MCC_l
        #     tester.save_model(model, file_model)


