import logging

from torch.utils.data import Dataset
import  json
from tqdm import tqdm
import numpy as np
class Process_Corpus(Dataset):
    def __init__(self, fname, tokenizer, max_seq_len, labels):
        self.tokenizer=tokenizer
        self.max_seq_len=max_seq_len
        data = json.load(open(fname))

        all_data=[]
        for d in tqdm(data):
            text, label = d['text'], d['label']
            if label not in labels:continue
            inputs = tokenizer.encode_plus(text.strip().lower(), None, add_special_tokens=True,
                                           max_length=max_seq_len, truncation='only_first', padding='max_length',
                                           return_token_type_ids=True)

            input_ids = inputs['input_ids']
            input_mask = inputs['attention_mask']
            segment_ids = inputs["token_type_ids"]

            assert len(input_ids) <= max_seq_len
            input_ids = np.asarray(input_ids, dtype='int64')
            input_mask = np.asarray(input_mask, dtype='int64')
            segment_ids = np.asarray(segment_ids, dtype='int64')

            data = {
                'input_ids': input_ids,
                'segments_ids': segment_ids,
                'input_mask': input_mask,
                'label': labels[label]
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
        data = open(fname)
        indexes = np.random.choice(np.arange(1000000), train_len)
        all_data=[]
        logging.info('the number of unlabeled data : {}'.format(len(indexes)))
        label = labels[ next(iter(labels))]
        for i, text in enumerate(data):
            if i not in indexes:continue
            inputs = tokenizer.encode_plus(text.strip().lower(), None, add_special_tokens=True,
                                           max_length=max_seq_len, truncation='only_first', padding='max_length',
                                           return_token_type_ids=True)

            input_ids = inputs['input_ids']
            input_mask = inputs['attention_mask']
            segment_ids = inputs["token_type_ids"]

            assert len(input_ids) <= max_seq_len
            input_ids = np.asarray(input_ids, dtype='int64')
            input_mask = np.asarray(input_mask, dtype='int64')
            segment_ids = np.asarray(segment_ids, dtype='int64')

            data = {
                'input_ids': input_ids,
                'segments_ids': segment_ids,
                'input_mask': input_mask,
                'label': label
            }
            all_data.append(data)
        self.data = all_data


    def __getitem__(self, index):

        return self.data[index]

    def __len__(self):
        return len(self.data)
