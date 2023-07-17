
import logging
import argparse
import math
import os
from datetime import datetime

import sys
#from sentence_transformers import SentenceTransformer
import random
import numpy
from sklearn import metrics
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset,random_split
from data_utils import   Process_Corpus,Process_Corpus_ads

from tqdm import tqdm

import json
from transformers import  AutoTokenizer
from MyModel import ADI_Classifier
import pickle as pk
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))



class Instructor:
    def __init__(self, opt):
        self.opt = opt
        opt.plm = opt.pretrained_bert_name.split('/')[-1]
        cache = 'cache/AADI_{}_{}.pk'.format(opt.dataset, opt.plm)
        self.labels  = json.load(open('{}/datasets/{}/labels.json'.format(opt.workspace,opt.dataset)))

        if os.path.exists(cache):
            d = pk.load(open(cache, 'rb'))
            self.trainset = d['train']
            self.trainset_unlabel = d['unlabel']
            self.testset = d['test']
            self.valset = d['dev']

        else:
            tokenizer = AutoTokenizer.from_pretrained(opt.pretrained_bert_name)

            self.opt.lebel_dim = len(self.labels)
            self.trainset = Process_Corpus(opt.dataset_file['train'], tokenizer, opt.max_seq_len, self.labels )
            self.trainset_unlabel = Process_Corpus_ads(opt.dataset_file['unlabel'], tokenizer, opt.max_seq_len,self.labels, train_len=len(self.trainset))
            self.valset = Process_Corpus(opt.dataset_file['dev'], tokenizer, opt.max_seq_len, self.labels )
            self.testset = Process_Corpus(opt.dataset_file['test'], tokenizer, opt.max_seq_len, self.labels )

            if not os.path.exists('cache'):
                os.mkdir('cache')
            d = {'train': self.trainset,'unlabel': self.trainset_unlabel, 'test': self.testset, 'dev': self.valset}
            pk.dump(d, open(cache, 'wb'))

        self.labels= list(self.labels.keys())
        logger.info('labeled train: {}, unlabeled train: {}, test: {}, dev: {}'.format(len( self.trainset), len( self.trainset_unlabel),len( self.testset), len( self.valset)))

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
                reps, logits, _ = model(inputs, alpha=0)

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

    def _evaluate_full(self, model, criterion, val_data_loader, getreps=False):
        with torch.no_grad():
            pred_list, true_all = [], []
            all_reps = []
            test_loss = test_acc = 0.0
            # logger.info('testing')
            for i, v_sample_batched in enumerate(tqdm(val_data_loader)):
                labels = v_sample_batched['label']

                labels = labels.to(self.opt.device)

                inputs = [v_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                reps, logits, _ = model(inputs, alpha=0)

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


    def _train(self,model,optimizer,criterion_y,criterion_d,train_data_loader, val_data_loader, test_data_loader, t_total):

        best_acc_test=0
        global_step = 0
        best_f1_micro_test = 0.0
        best_valid_acc = 0.0
        best_f1_test = 0.0
        len_dataloader= len(train_data_loader.dataset)

        for epoch in range(self.opt.num_epoch):
            train_loss =train_rev= t_total_c = train_acc = 0.0

            model.train()
            if epoch != 0:
                lr_this_step = self.opt.learning_rate * self.warmup_linear(global_step / t_total,
                                                                           self.opt.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                    self.opt.learning_rate = param_group['lr']

            for i_batch, sample_batched in enumerate(tqdm(train_data_loader)):

                model.zero_grad()

                global_step += 1
                optimizer.zero_grad()
                labels = sample_batched['label']
                evid = sample_batched['is_evidence']
                evid=evid.to(self.opt.device)


                p = float(global_step + epoch * len_dataloader) /self.opt.num_epoch  / len_dataloader
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

                #
                label_clean = labels[evid == 1]
                label_clean = label_clean.to(self.opt.device)


                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                _,logits,logits_rev = model(inputs, alpha)

                loss_d = criterion_d(logits_rev, evid)
                loss_y = criterion_d(logits[evid==1], label_clean)

                loss = loss_d+loss_y

                with torch.no_grad():

                    train_rev += evid.size(0) * loss_d.data
                    if logits[evid==1].size(0):
                        train_loss += logits[evid == 1].size(0) * loss_y.data
                        _, pred = torch.max(logits[evid==1].data, -1)
                        acc = float((pred == label_clean.data).sum())
                        train_acc += acc
                        t_total_c+=label_clean.size(0)


                loss.backward()
                optimizer.step()



            # with torch.no_grad():
            train_loss /= t_total_c
            train_rev /= len(train_data_loader.dataset)
            train_acc/=t_total_c

            logger.info(
                '[%6d/%6d] loss: %5f,train_rev: %5f, acc: %5f, lr: %7f'
                % (epoch, self.opt.num_epoch, train_loss, train_rev, train_acc, self.opt.learning_rate, ))


            model.eval()
            with torch.no_grad():
                logger.info('validating')
                val_loss, val_f1_sc,val_f1_micro, val_acc, val_precisions, val_recalls, val_f1s = self._evaluate(model, criterion_y, val_data_loader)
                # best_test_acc = max(val_f1_sc, best_test_acc)
                # best_test_acc = max(val_acc, best_test_acc)
                best_valid_acc = max(val_acc, best_valid_acc)

                logger.info('\t valid ...loss: %5f, acc: %5f,f1: %5f,f1 micro: %5f, best_acc: %5f' % (
                    val_loss, val_acc, val_f1_sc,val_f1_micro,best_valid_acc))

            is_best = val_acc >= best_valid_acc
            if is_best:
                model.eval()

                with torch.no_grad():
                    logger.info('testing')
                    # test_loss, f1_sc, f1_micro, test_acc

                    test_loss, test_f1_sc, test_f1_micro, test_acc, test_precisions, test_recalls, \
                        test_f1s = self._evaluate(model, criterion_y,  test_data_loader, getreps=False)

                    if test_f1_sc > best_f1_test:
                        path = f"models/aadi_{self.opt.pretrained_bert_name.split('/')[-1]}_{self.opt.dataset}_{datetime.now().strftime('%Y-%m-%d')}"
                        torch.save(model.state_dict(), path)

                    best_acc_test = max(best_acc_test, test_acc)
                    best_f1_test = max(best_f1_test, test_f1_sc)
                    best_f1_micro_test = max(best_f1_micro_test, test_f1_micro)


                    logger.info(
                        '\t test ...loss: %5f, acc: %5f,f1 macro: %5f , f1 micro: %5f best_acc: %5f best_f1: %5f best_f1 micro: %5f ' % (
                            test_loss, test_acc, test_f1_sc, test_f1_micro, best_acc_test, best_f1_test,
                            best_f1_micro_test))

                    writer = SummaryWriter('runs/AADI/Corpus_6_camelbert-mix_5')
                    writer.add_scalar('Testing Accuracy', best_f1_micro_test, global_step)
                    writer.add_scalar('Testing loss', test_loss, global_step)
                    writer.add_scalar('Validation Accuracy', best_valid_acc, global_step)
                    writer.add_scalar('Validation', val_loss, global_step)

        with open('results_ads.txt', 'a+') as f :
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M')} model {self.opt.pretrained_bert_name.split('/')[-1]} dataset {self.opt.dataset} train_sample {self.opt.train_sample} f1  {best_f1_test:.4} acc {best_acc_test:.4} \n")
        f.close()

    def run(self):

        trainset = self.trainset
        trainset_unlabel = self.trainset_unlabel
        testset = self.testset
        valset = self.valset


        if self.opt.train_sample >0:
            ratio = int(len(trainset) * self.opt.train_sample)
            _, trainset = random_split(trainset, (len(trainset) - ratio, ratio))

        for i in range(len(trainset)):
            trainset[i]['is_evidence'] =1
            # trainset[i]['new_index'] =len(trainset)



        for i in range(len(trainset_unlabel)):
            trainset_unlabel[i]['is_evidence'] = 0
            trainset_unlabel[i]['new_index'] = i
        for i in range(len(trainset)):
            trainset[i]['new_index'] = len(trainset_unlabel)+i
        trainset= ConcatDataset([trainset, trainset_unlabel])
        logger.info('train sample ratio {}, label {}, unlabel {}, test {}, dev {}'.format(self.opt.train_sample, len(trainset), len(trainset_unlabel), len(testset), len(valset)))


        train_data_loader = DataLoader(dataset=trainset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=testset, batch_size=self.opt.batch_size_val, shuffle=False)
        val_data_loader = DataLoader(dataset=valset, batch_size=self.opt.batch_size_val, shuffle=False)
        t_total = int(len(train_data_loader) * self.opt.num_epoch)


        model = ADI_Classifier(self.opt)
        #model = nn.DataParallel(model)
        model.to(self.opt.device)
        _params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = self.opt.optimizer(model.parameters(), lr=self.opt.learning_rate,
                                            weight_decay=self.opt.l2reg)


        criterion_y = nn.CrossEntropyLoss()
        criterion_d = nn.CrossEntropyLoss()

        self._train(model,optimizer,criterion_y, criterion_d,  train_data_loader, val_data_loader, test_data_loader, t_total)

        path = f"models/aadi_{self.opt.pretrained_bert_name.split('/')[-1]}_{self.opt.dataset}_{datetime.now().strftime('%Y-%m-%d')}"
        model.load_state_dict(torch.load(path))
        model.to(self.opt.device)

        test_loss, test_f1_sc, test_f1_micro, test_acc, test_precisions, test_recalls, test_f1s, test_preds, \
            misclass, conf_matrix, reps = self._evaluate_full(model, criterion_y, test_data_loader, getreps=True)

        logger.info(
            '\t test ...loss: %5f, acc: %5f,f1 macro: %5f , f1 micro: %5f' % (
                test_loss, test_acc, test_f1_sc, test_f1_micro))
        test_raw = open(self.opt.dataset_file['test']).read().splitlines()
        test_raw = np.asarray([e.split('\t') for e in test_raw])
        stats = list(zip(test_precisions, test_recalls, test_f1s))


        with open(f'outputs/aadi_{self.opt.dataset}_stats.csv', 'w') as f:
            f.write('Class,Precision,Recall,F1\n')
            for j, (prec, rec, f1) in enumerate(stats):
                f.write(f'{self.labels[j - 1]},{prec},{rec},{f1}\n')

        with open(f'outputs/aadi_{self.opt.dataset}_misclassified.csv', 'w') as f:
            f.write('Text, Label, Predicition\n')
            for idx in misclass:
                line = test_raw[idx]
                # f.write(f"{line['text']}, {line['label']}, {self.labels[test_preds[idx]]}\n")
                # f.write(f"['text'], ['label'], {self.labels[test_preds[idx]]}\n")
                f.write(f"{line[0]}, {line[1]}, {self.labels[test_preds[idx]]}\n")
        np.save(f"outputs/aadi_{self.opt.dataset}_conf_matrix.npy", conf_matrix)

        np.save(f"outputs/aadi_{self.opt.dataset}_reps.npy", reps)


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Corpus-26', type=str, help=' Corpus-26, Corpus-6')
    parser.add_argument('--workspace', default='/workspace/June/NLP_ADI', type=str, help=' workspace')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=3e-5, type=float, help='try 5e-5, 3e-5 for BERT, 1e-3 for others')
    parser.add_argument('--adam_epsilon', default=2e-8, type=float, help='')
    parser.add_argument('--weight_decay', default=0, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--reg', type=float, default=0.00005, help='regularization constant for weight penalty')
    parser.add_argument('--num_epoch', default=10, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=32, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--batch_size_val', default=32, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=35500, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--max_grad_norm', default=10, type=int)
    parser.add_argument('--warmup_proportion', default=0.01, type=float)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='/workspace/plm/arbert',type=str)
    parser.add_argument('--max_seq_len', default=128, type=int)
    parser.add_argument('--lebel_dim', default=26, type=int)
    parser.add_argument('--train_sample', default=0.1, type=float)
    parser.add_argument('--device', default='cuda' , type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=85, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0.1, type=float, help='set ratio between 0 and 1 for validation support')
    opt = parser.parse_args()


    if opt.seed is not None:

        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



    dataset_files = {
        'train': '{}/datasets/{}/train.json'.format(opt.workspace, opt.dataset),
        'unlabel': '{}/datasets/large_corpus/unlabeled_corpus.txt'.format(opt.workspace,opt.dataset),
        'test': '{}/datasets/{}/dev.json'.format(opt.workspace,opt.dataset),
        'dev': '{}/datasets/{}/dev.json'.format(opt.workspace,opt.dataset),
    }


    input_colses =  ['input_ids', 'segments_ids', 'input_mask', 'label']
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.AdamW,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }

    opt.dataset_file = dataset_files
    opt.inputs_cols = input_colses
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
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

