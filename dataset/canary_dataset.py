import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, dataloader
from dataset.data_process import get_sent_list
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import random
from random import sample
import pickle

with open(os.path.join(BASE_DIR, "default_canary.json"), "rb") as f:
    default_canary = pickle.load(f)

def random_insert_seq(lst, seq):
    insert_locations = sample(range(len(lst) + len(seq)), len(seq))
    inserts = dict(zip(insert_locations, seq))
    input = iter(lst)
    lst[:] = [inserts[pos] if pos in inserts else next(input)
              for pos in range(len(lst) + len(seq))]
    return lst, insert_locations


def generate_evaluation_sequence_for_gpt(canary_type="email", use_full_text=False, insert_proportion=0.4):
    canary_format = default_canary[canary_type]["canary_format"]
    fill_list = default_canary[canary_type]["fill_list"]
    length = round(len(fill_list) * insert_proportion)
    inserted = fill_list[:length]
    not_inserted = fill_list[length:]
    if use_full_text:
        inserted = [canary_format.replace('*', fill) for fill in inserted]
        not_inserted = [canary_format.replace('*', fill) for fill in not_inserted]
    return inserted, not_inserted


class CanaryDataset(Dataset):
    def __init__(self, canary_type_list, insert_proportion_list, insert_time_base_list, dataset_name,
                 data_type="train"):
        self.canary_type_list = canary_type_list
        self.insert_proportion_list = insert_proportion_list
        self.insert_time_base_list = insert_time_base_list
        self.dataset_name = dataset_name
        if self.dataset_name in ['personachat', 'qnli', "mnli", 'sst2']:
            self.data_list = get_sent_list(dataset_name=dataset_name, data_type=data_type)
        else:
            raise NotImplementedError("Given dataset is not supported now")
        self.canary_format_list = []
        self.insert_time_list = []
        self.canary_infill_list = []
        self.canary_text_list = []
        for canary_type, insert_proportion, insert_time_base in zip(canary_type_list,
                                                                    insert_proportion_list, insert_time_base_list):
            canary_format = default_canary[canary_type]["canary_format"]
            fill_list = default_canary[canary_type]["fill_list"]
            length = round(len(fill_list) * insert_proportion)
            fill_list = fill_list[:length]
            insert_time_list = [i * insert_time_base for i in range(1, length + 1)]
            canary_text_list = [canary_format.replace('*', fill) for fill in fill_list]
            self.insert_time_list.extend(insert_time_list)
            self.canary_format_list.extend([canary_format] * length)
            self.canary_infill_list.extend(fill_list)
            self.canary_text_list.extend(canary_text_list)

        self.canary_loc_list = []
        self.insert_canary()

    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)

    def insert_canary(self):
        insert_canary_text_list = []
        if self.dataset_name == "personachat":
            for canary_text, insert_time in zip(self.canary_text_list, self.insert_time_list):
                insert_canary_text_list.extend([canary_text] * insert_time)
            idx = random.sample(range(len(self.data_list)), len(insert_canary_text_list))
            for i_0, canary_text in zip(idx, insert_canary_text_list):
                sentence = self.data_list[i_0]
                dialog = sentence.split(" <SEP> ")
                i_1 = random.randint(0, len(dialog))
                self.canary_loc_list.append((i_0, i_1))
                dialog.insert(i_1, canary_text)
                self.data_list[i_0] = " <SEP> ".join(dialog)
        else:
            for canary_text, insert_time in zip(self.canary_text_list, self.insert_time_list):
                insert_canary_text_list.extend([canary_text] * insert_time)
            self.data_list, self.canary_loc_list = random_insert_seq(self.data_list, insert_canary_text_list)


class EvaluationDataset(Dataset):
    def __init__(self, dataset_name, data_type):
        assert data_type != 'train'
        self.dataset_name = dataset_name
        if self.dataset_name in ['qnli', "mnli", 'sst2']:
            self.data_list = get_sent_list(dataset_name=dataset_name, data_type=data_type, return_all=True)
        else:
            raise NotImplementedError("Given dataset is not supported now")
        self.label_list = []
        self.sent_list = []
        for sent in self.data_list:
            self.sent_list.append(sent[:-2])
            if dataset_name == 'qnli':
                self.label_list.append(' ' + sent[-1])
            elif dataset_name == 'sst2':
                self.label_list.append(' ' + sent[-1])
            elif dataset_name == 'mnli':
                self.label_list.append(' ' + sent[-1])

    def __getitem__(self, index):
        return self.sent_list[index], self.label_list[index]

    def __len__(self):
        return len(self.sent_list)

