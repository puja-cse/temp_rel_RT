from torch.utils.data import Dataset, DataLoader
import re
import torch
import torch.nn as nn
from transformers import AutoTokenizer
import itertools


tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", return_tensors = 'pt')



class customDataset(Dataset):
    def __init__(self, path, language='it'):
        f= open(path, "r")
        pattern = 's1:|s2:|\t|\n'
        sen1=[]; sen2=[]; location1=[]; location2=[]; event1=[]; event2=[]; labels=[]
        label2idx = {'BEFORE': 0, 'AFTER': 1, 'EQUAL':2, 'VAGUE': 3}
        idx2label = {0: 'BEFORE', 1: 'AFTER', 2: 'EQUAL', 3: 'VAGUE'}
        for line in f:
            s = re.split(pattern, line)
            s1 = s[1]
            s2 = s[2]
            loc1 = int (s[4])
            e1 = s[5]
            loc2 =int  (s[7])
            e2 = s[8]
            if s[9] in label2idx:
                label = label2idx[s[9]]
            else:
                label=4
            loc2 += ( len(s1.strip().split(' ')) + 2 ) # since we will have <s> and </s> token before sentence 2
            s1 = s1 +' </s> '
            sen1.append(s1.strip()), sen2.append(s2.strip())
            location1.append(loc1+1), location2.append(loc2)
            event1.append(e1), event2.append(e2)
            labels.append(label)
        self.path = path
        self.sentence1 = sen1
        self.sentence2 = sen2
        self.location1 = location1
        self.location2 = location2
        self.event1 = event1
        self.event2 = event2
        self.label = labels
        X = list(map(''.join, itertools.zip_longest(sen1, sen2)))
        self.encodings = tokenizer(X, truncation=True, padding=True)


    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        sample = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        sample['sentence1'] =  self.sentence1[idx]
        sample['sentence2'] = self.sentence2[idx]
        sample['location1'] = self.location1[idx]
        sample['location2'] = self.location2[idx]
        sample['label'] = torch.tensor(self.label[idx])
        return sample