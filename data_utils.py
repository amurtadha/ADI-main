import logging

from torch.utils.data import Dataset
import  json
from tqdm import tqdm
import numpy as np
class Process_Corpus(Dataset):
    def __init__(self, fname, tokenizer, max_seq_len, labels):
        self.tokenizer=tokenizer
        self.max_seq_len=max_seq_len
        data_file = json.load(open(fname))

        all_data=[]
        for d in tqdm(data_file):
            text, label = d['text'], d['label']
            if label not in labels:continue

            example = tokenizer.encode_plus(text, None, add_special_tokens=True, truncation=True,
                                            padding='max_length', max_length=self.max_seq_len,
                                            return_token_type_ids=True)
            data = {
                'input_ids': np.asarray(example['input_ids'], dtype='int64'),
                'segments_ids': np.asarray(example['token_type_ids'], dtype='int64'),
                'input_mask': np.asarray(example['attention_mask'], dtype='int64'),
                'label':  labels[label],
            }

            all_data.append(data)
        self.data = all_data


    def __getitem__(self, index):

        return self.data[index]

    def __len__(self):
        return len(self.data)
class Process_Corpus_ads(Dataset):
    def __init__(self, fname, tokenizer, max_seq_len, labels, train_len):
        # MyModel_ads.resize_token_embeddings(len(tokenizer))
        self.tokenizer=tokenizer
        self.max_seq_len=max_seq_len
        data_file = json.load(open(fname))
        indexes = np.random.choice(np.arange(1000000), train_len)
        all_data=[]
        logging.info('the number of unlabeled data : {}'.format(len(indexes)))
        label = labels[ next(iter(labels))]

        for i in tqdm(indexes):
            d= data_file[i]
            text = d['text']

            example = tokenizer.encode_plus(text, None, add_special_tokens=True, truncation=True,
                                            padding='max_length', max_length=self.max_seq_len,
                                            return_token_type_ids=True)
            data = {
                'input_ids': np.asarray(example['input_ids'], dtype='int64'),
                'segments_ids': np.asarray(example['token_type_ids'], dtype='int64'),
                'input_mask': np.asarray(example['attention_mask'], dtype='int64'),
                'label': label,
            }



            all_data.append(data)
        self.data = all_data


    def __getitem__(self, index):

        return self.data[index]

    def __len__(self):
        return len(self.data)
