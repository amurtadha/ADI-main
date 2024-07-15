
import logging
import argparse
import math
import os
from datetime import datetime

import sys
import random
import numpy
from sklearn import metrics
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset,random_split, Subset
from data_utils import   Process_Corpus,Process_Corpus_ads

from tqdm import tqdm
from transformers import  AdamW

import json
from transformers import  AutoTokenizer
from MyModel import Vanilla
import pickle as pk
from torch.utils.tensorboard import SummaryWriter
import copy
from collections import defaultdict

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))




class Instructor:
    def __init__(self, opt):
        self.opt = opt
        print('here')
        opt.plm = opt.pretrained_bert_name.split('/')[-1]
        self.labels  = json.load(open('{}/datasets/{}/labels.json'.format(opt.workspace,opt.dataset)))

        tokenizer = AutoTokenizer.from_pretrained(opt.pretrained_bert_name)

        self.opt.lebel_dim = len(self.labels)
        self.trainset = Process_Corpus(opt.dataset_file['train'], tokenizer, opt.max_seq_len, self.labels)
        self.valset = Process_Corpus(opt.dataset_file['dev'], tokenizer, opt.max_seq_len, self.labels)
        self.testset = Process_Corpus(opt.dataset_file['test'], tokenizer, opt.max_seq_len, self.labels)

        self.labels= list(self.labels.keys())
        logger.info('labeled train: {}, test: {}, dev: {}'.format(len( self.trainset),len( self.testset), len( self.valset)))

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))






    def warmup_linear(self, x, warmup=0.002):
        if x < warmup:
            return x / warmup
        return 1.0 - x



    def _evaluate(self, model, criterion, val_data_loader, getreps=False):
        with torch.no_grad():
            pred_list, true_all = [], []
            test_loss = test_acc = 0.0
            # logger.info('testing')
            for i, v_sample_batched in enumerate(tqdm(val_data_loader)):
                labels = v_sample_batched['label']

                labels = labels.to(self.opt.device)

                inputs = [v_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                logits= model(inputs)

                loss = criterion(logits, labels)
                test_loss += inputs[0].size(0) * loss.data

                _, pred = torch.max(logits.data, -1)
                acc = float((pred == labels.data).sum())
                test_acc += acc
                pred_list.extend(pred.detach().cpu().tolist())
                true_all.extend(labels.data.detach().cpu().tolist())


            test_loss /= len(val_data_loader.dataset)
            test_acc /= len(val_data_loader.dataset)
            f1_sc = metrics.f1_score(true_all, pred_list, average='macro')
            f1_micro = metrics.f1_score(true_all, pred_list, average='micro')
            precisions = metrics.precision_score(true_all, pred_list, average=None)
            recalls = metrics.recall_score(true_all, pred_list, average=None)
            f1s = metrics.f1_score(true_all, pred_list, average=None)

            return test_loss, f1_sc, f1_micro, test_acc, precisions, recalls, f1s

    def _full(self, model, criterion, val_data_loader, getreps=False):
        with torch.no_grad():
            pred_list, true_all = [], []
            all_reps = []
            test_loss = test_acc = 0.0
            # logger.info('testing')
            for i, v_sample_batched in enumerate(tqdm(val_data_loader)):
                labels = v_sample_batched['label']

                labels = labels.to(self.opt.device)

                inputs = [v_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                logits = model(inputs)

                loss = criterion(logits, labels)
                test_loss += inputs[0].size(0) * loss.data

                _, pred = torch.max(logits.data, -1)
                acc = float((pred == labels.data).sum())
                test_acc += acc
                pred_list.extend(pred.detach().cpu().tolist())
                true_all.extend(labels.data.detach().cpu().tolist())

                if getreps:
                    all_reps.append(reps)

            test_loss /= len(val_data_loader.dataset)
            test_acc /= len(val_data_loader.dataset)
            f1_sc = metrics.f1_score(true_all, pred_list, average='macro')
            f1_micro = metrics.f1_score(true_all, pred_list, average='micro')
           # confusion = metrics.confusion_matrix(true_all, pred_list)
            precisions = metrics.precision_score(true_all, pred_list, average=None)
            recalls = metrics.recall_score(true_all, pred_list, average=None)
            f1s = metrics.f1_score(true_all, pred_list, average=None)
            misclassifications = np.where(np.array(true_all) != np.array(pred_list))[0]
            conf_matrix = metrics.confusion_matrix(true_all, pred_list)

            if getreps:
                all_reps = torch.cat(all_reps).detach().cpu().numpy()

            return test_loss, f1_sc, f1_micro, test_acc, precisions, recalls, f1s, np.array(pred_list), \
                misclassifications, conf_matrix, all_reps


    def _train(self,model,optimizer,criterion,train_data_loader, val_data_loader, test_data_loader, t_total, lamd=0.8):

        best_acc_test=0
        global_step = 0
        best_f1_micro_test = 0.0
        best_valid_acc = 0.0
        best_f1_test = 0.0
        len_dataloader= len(train_data_loader.dataset)

        for epoch in range(self.opt.num_epoch):
            train_loss = train_acc = 0.0
            model.train()
            if epoch != 0:
                lr_this_step = self.opt.learning_rate * self.warmup_linear(global_step / t_total,self.opt.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                    self.opt.learning_rate = param_group['lr']

            for i_batch, sample_batched in enumerate(tqdm(train_data_loader)):

                model.zero_grad()

                global_step += 1
                optimizer.zero_grad()
                labels = sample_batched['label'].to(self.opt.device)


                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                logits = model(inputs)
                loss= criterion(logits, labels)
                train_loss += inputs[0].size(0) * loss.data


                loss.backward()
                optimizer.step()



            # with torch.no_grad():
            train_loss /= global_step


            logger.info(
                '[%6d/%6d] loss: %5f, lr: %7f'
                % (epoch, self.opt.num_epoch, train_loss,  self.opt.learning_rate, ))


            model.eval()
            with torch.no_grad():
                logger.info('validating')
                val_loss, val_f1_sc,val_f1_micro, val_acc, val_precisions, val_recalls, val_f1s = self.(model, criterion, val_data_loader)

                best_valid_acc = max(val_acc, best_valid_acc)

                logger.info('\t valid ...loss: %5f, acc: %5f,f1: %5f,f1 micro: %5f, best_acc: %5f' % (
                    val_loss, val_acc, val_f1_sc,val_f1_micro,best_valid_acc))
                if val_acc > best_f1_test:
                    path = copy.deepcopy(model.state_dict())
                best_f1_test = max(best_f1_test, val_acc)
        return path



    def balanced_train_sample(sefl, trainset, train_sample_ratio, seed=None):
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        # Group samples by their labels
        label_to_indices = defaultdict(list)
        for idx, d in enumerate(trainset.data):

            label_to_indices[d['label']].append(idx)

        sampled_indices = []
        for label, indices in label_to_indices.items():
            sample_size = int(len(indices) * train_sample_ratio)
            sampled_indices.extend(random.sample(indices, sample_size))

        # Create a subset of the trainset with the sampled indices
        trainset = Subset(trainset, sampled_indices)

        return trainset



    def run(self):

        trainset = self.trainset
        testset = self.testset
        valset = self.valset


        if self.opt.train_sample >0:
            trainset=self.balanced_train_sample(trainset, self.opt.train_sample)

        logger.info('train sample ratio {}, training {}, test {}, dev {}'.format(self.opt.train_sample, len(trainset), len(testset), len(valset)))
        train_data_loader = DataLoader(dataset=trainset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=testset, batch_size=self.opt.batch_size_val, shuffle=False)
        val_data_loader = DataLoader(dataset=valset, batch_size=self.opt.batch_size_val, shuffle=False)
        t_total = int(len(train_data_loader) * self.opt.num_epoch)


        model = Vanilla(self.opt)
        #model = nn.DataParallel(model)
        model.to(self.opt.device)

        optimizer =AdamW(model.parameters(), lr=self.opt.learning_rate, weight_decay=0.01)


        criterion = nn.CrossEntropyLoss()

        best_model_path = self._train(model,optimizer,criterion,  train_data_loader, val_data_loader, test_data_loader, t_total)
        model.load_state_dict(best_model_path)

        model.to(self.opt.device)

        test_loss, test_f1_sc, test_f1_micro, test_acc, test_precisions, test_recalls, test_f1s, test_preds, \
            misclass, conf_matrix, reps = self._evaluate_full(model, criterion_y, test_data_loader, getreps=True)

        logger.info(
            '\t test ...loss: %5f, acc: %5f,f1 macro: %5f , f1 micro: %5f' % (
                test_loss, test_acc, test_f1_sc, test_f1_micro))

        with open('results_vanila.txt', 'a+') as f:
            f.write(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M')} model {self.opt.pretrained_bert_name.split('/')[-1]} dataset {self.opt.dataset} train_sample {self.opt.train_sample} f1  {test_f1_sc:.4} acc {test_acc:.4} \n")
        f.close()



def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Corpus-26', type=str, help=' Corpus-26, Corpus-6')
    parser.add_argument('--workspace', default='/workspace/ArNLP/', type=str, help=' workspace')
    parser.add_argument('--learning_rate', default=3e-5, type=float,)
    parser.add_argument('--num_epoch', default=6, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--batch_size_val', default=64, type=int)
    parser.add_argument('--warmup_proportion', default=0.01, type=float)
    parser.add_argument('--pretrained_bert_name', default='rahbi/alclam-base-v1',type=str)
    parser.add_argument('--max_seq_len', default=128, type=int)
    parser.add_argument('--lebel_dim', default=26, type=int)
    parser.add_argument('--train_sample', default=0.1, type=float)
    parser.add_argument('--device', default='cuda' , type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=42, type=int, help='set seed for reproducibility')
    opt = parser.parse_args()

    opt.seed= random.randint(0,300)

    if opt.seed is not None:

        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    opt.dataset_file = {
        'train': '{}/datasets/{}/train.json'.format(opt.workspace, opt.dataset),
        'unlabel': '{}/datasets/large_corpus/unlabeled_corpus.json'.format(opt.workspace,opt.dataset),
        'test': '{}/datasets/{}/dev.json'.format(opt.workspace,opt.dataset),
        'dev': '{}/datasets/{}/dev.json'.format(opt.workspace,opt.dataset),
    }

    opt.inputs_cols = ['input_ids', 'segments_ids', 'input_mask', 'label']
    opt.device = torch.device(opt.device if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    if not os.path.exists('models'):os.makedirs('models')
    if not os.path.exists('outputs'):os.makedirs('outputs')
    log_file = 'AADI-{}.log'.format(opt.dataset)
    logger.addHandler(logging.FileHandler(log_file))
    logger.info('seed {}'.format(opt.seed))
    ins = Instructor(opt)
    ins.run()



if __name__ == '__main__':
    main()

