from datasets import load_dataset
from pprint import pprint
import json
import sys
import os
from config import config
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(DIR_PATH, "..","data","cache")
### 60% training data for LMs, 40% training data for attacker
#SPLIT_THRESHOLD = 0.6
SPLIT_THRESHOLD = config["insert_proportion"]
#SEED_NUMBER = 42
import numpy as np

'''
Datasets from GLUE:
Generative LMs (GPT-2/BART ...)
This py file is used to get the list of sentences from the dataset.
For dialog datasets like PC, each index sample refers to a dialog.
    - We evaluate on dialogs to make longer data samples for evaluation.
    - ''hi how are you? <SEP> i am fine <eos> canary <SEP> .... <eos>''
    - Canary insetion: canary randomly inserted in the dialog (between utterances).
For NLI datasets like QNLI, each index sample refers to a pair of sentences (premise and hypothesis).
    - '' [premise] <SEP> [hypothesis] <SEP> The relation is [neutral/contradiction/entailment] <eos>''
    - Canary insetion: canary can be used as samples
For sentiment datasets like SST, each index sample refers to a sentence. Then we append label for prediction task.
    - ''saw how bad this movie was. <SEP> The sentiment is [negative/positive/netural] <eos>''
    - Canary insetion: canary can be used as samples

Canary:
1- Keep track of categories
2- Keep track of canary text (position)
3- Keep track of times of occurance in the dataset

    - Numbers: my phone number is xxxx
    - Words/attributes: my name is xxxx / I live in xxxx
    - Emails: my email is xxxx@xx.xxx (Enron dataset)
    
Masked LMs (BERT/RoBERTa ...)
Tips: The original BERT was pretrained on Wikipedia and BookCorpus datasets. 

For NLI datasets like QNLI, each index sample refers to a pair of sentences (premise and hypothesis). 
    - Classification pipeline to determine neutral/contradiction/entailment
For sentiment datasets like SST, each index sample refers to a sentence.
    - Classification pipeline to determine positive/negative


list of supported datasets:
['personachat'.'qnli']
'''
#### this function is used to get the list of sentences from the dataset for CLS/MLM tasks
#### If return sent_a (list), sent_b (list), label (list), label_num if single sentence, sent_b is None
def get_sent_list_cls(dataset_name, data_type,is_aux=False,return_all=False, SEED_NUMBER = 42):
    if dataset_name == 'qnli':
        fn = get_qnli_data_cls

    elif dataset_name == "sst2":
        fn = get_sst2_data_cls

    elif dataset_name == "mnli":
        fn = get_mnli_data_cls

    else:
        print('Name of dataset only supports: personachat, qnli, sst2, mnli')
        sys.exit(-1)
    sent_dict = fn(data_type)
    split_idx = int(SPLIT_THRESHOLD * len(sent_dict['sent_a']))
    np.random.seed(SEED_NUMBER)
    shuffle_idx = np.random.permutation(len(sent_dict['sent_a']))
    sent_a = np.array(sent_dict['sent_a'])[shuffle_idx].tolist()
    sent_b = sent_dict['sent_b']
    if sent_b is not None:
        sent_b = np.array(sent_dict['sent_b'])[shuffle_idx].tolist()
    label = np.array(sent_dict['label'])[shuffle_idx].tolist()
    if return_all:
        return {'sent_a': sent_a,
                'sent_b': None if sent_b is None else sent_b, 
                'label': label,
                'label_num': sent_dict['label_num']
            }
    if is_aux:
        return {'sent_a': sent_a[split_idx:],
                'sent_b': None if sent_b is None else sent_b[split_idx:], 
                'label': label[split_idx:],
                'label_num': sent_dict['label_num']
            }
    else:
        return {'sent_a': sent_a[:split_idx],
                'sent_b': None if sent_b is None else sent_b[:split_idx], 
                'label': label[:split_idx],
                'label_num': sent_dict['label_num']
            }

def get_sent_list_cls_ori_order(dataset_name, data_type,is_aux=False,SEED_NUMBER=42):
    if dataset_name == 'qnli':
        fn = get_qnli_data_cls

    elif dataset_name == "sst2":
        fn = get_sst2_data_cls

    elif dataset_name == "mnli":
        fn = get_mnli_data_cls

    else:
        print('Name of dataset only supports: personachat, qnli, sst2, mnli')
        sys.exit(-1)
    sent_dict = fn(data_type)
    split_idx = int(SPLIT_THRESHOLD * len(sent_dict['sent_a']))
    np.random.seed(SEED_NUMBER)
    shuffle_idx = np.random.permutation(len(sent_dict['sent_a']))

    sent_a = np.array(sent_dict['sent_a']).tolist()
    sent_b = sent_dict['sent_b']
    if sent_b is not None:
        sent_b = np.array(sent_dict['sent_b']).tolist()
    label = np.array(sent_dict['label']).tolist()
    return {'sent_a': sent_a,
            'sent_b': None if sent_b is None else sent_b,
            'label': label,
            'label_num': sent_dict['label_num']
        }   ,shuffle_idx[:split_idx],shuffle_idx[split_idx:] # return the original order of the dataset. train index and test index



def get_qnli_data_cls(data_type):
    if (data_type == 'dev'):
        data_type = 'validation'
    dataset = load_dataset('glue', 'qnli', cache_dir=CACHE_PATH, split=data_type)
    senta_list = []
    sentb_list = []
    label_list = []
    label_num = 2
    for data in dataset:
        question = data["question"]
        ans = data["sentence"]
        label = data["label"]

        senta_list.append(question)
        sentb_list.append(ans)
        label_list.append(label)
    return {'sent_a': senta_list,
            'sent_b': sentb_list, 
            'label': label_list,
            'label_num': label_num
        }


def get_sst2_data_cls(data_type):
    if(data_type == 'dev'):
        data_type = 'validation'
    dataset = load_dataset('glue', 'sst2', cache_dir=CACHE_PATH, split=data_type)
    senta_list = []
    sentb_list = []
    label_list = []
    label_num = 2
    for data in dataset:
        sentence = data["sentence"]
        label = data["label"]
        senta_list.append(sentence)
        label_list.append(label)
    return {'sent_a': senta_list,
            'sent_b': None, 
            'label': label_list,
            'label_num': label_num
        }

def get_mnli_data_cls(data_type):
    if(data_type == 'test'):
        data_type = 'test_matched'
    if(data_type == 'dev'):
        data_type = 'validation_matched'
    dataset = load_dataset('glue', 'mnli', cache_dir=CACHE_PATH, split=data_type)
    senta_list = []
    sentb_list = []
    label_list = []
    label_num = 3
    for data in dataset:
        premise = data['premise']
        hypothesis = data['hypothesis']
        label = data["label"]
        senta_list.append(premise)
        sentb_list.append(hypothesis)
        label_list.append(label)
    return {'sent_a': senta_list,
            'sent_b': sentb_list, 
            'label': label_list,
            'label_num': label_num
        }
'''
This is for generative LMs
'''
def get_sent_list(dataset_name, data_type,is_aux=False,return_all=False):
    if dataset_name == 'personachat':
        processed_persona_path = "data/personachat/processed_persona"
        processed_persona_path = os.path.join(DIR_PATH, "..","data","personachat","processed_persona")
        #processed_persona_path = DIR_PATH + "/../data/personachat/processed_persona"
        sent_list = get_personachat_data(data_type, processed_persona_path)
        #return sent_list
    elif dataset_name == 'qnli':
        sent_list = get_qnli_data(data_type)
        #return sent_list
    elif dataset_name == "sst2":
        sent_list = get_sst2_data(data_type)
        #return sent_list
    elif dataset_name == "mnli":
        sent_list = get_mnli_data(data_type)
        #return sent_list
    else:
        print('Name of dataset only supports: personachat, qnli, sst2, mnli')
        sys.exit(-1)
    split_idx = int(SPLIT_THRESHOLD * len(sent_list))
    np.random.seed(SEED_NUMBER)
    shuffle_idx = np.random.permutation(len(sent_list))
    sent_list = np.array(sent_list)[shuffle_idx].tolist()
    if return_all:
        return sent_list
    if is_aux:
        return sent_list[split_idx:]
    else:
        return sent_list[:split_idx]

def get_sent_list_ori_order(dataset_name, data_type,is_aux=False,SEED_NUMBER=42):
    if dataset_name == 'personachat':
        processed_persona_path = "data/personachat/processed_persona"
        processed_persona_path = os.path.join(DIR_PATH, "..","data","personachat","processed_persona")
        #processed_persona_path = DIR_PATH + "/../data/personachat/processed_persona"
        sent_list = get_personachat_data(data_type, processed_persona_path)
        #return sent_list
    elif dataset_name == 'qnli':
        sent_list = get_qnli_data(data_type)
        #return sent_list
    elif dataset_name == "sst2":
        sent_list = get_sst2_data(data_type)
        #return sent_list
    elif dataset_name == "mnli":
        sent_list = get_mnli_data(data_type)
        #return sent_list
    else:
        print('Name of dataset only supports: personachat, qnli, sst2, mnli')
        sys.exit(-1)
    split_idx = int(SPLIT_THRESHOLD * len(sent_list))
    np.random.seed(SEED_NUMBER)
    shuffle_idx = np.random.permutation(len(sent_list))
    # sent_list = np.array(sent_list)[shuffle_idx].tolist()
    return sent_list,shuffle_idx[:split_idx],shuffle_idx[split_idx:] # return the original order of the dataset. train index and test index


def get_personachat_data(data_type, processed_persona_path):
    dialog_list = get_persona_dict(data_type=data_type, processed_persona_path=processed_persona_path)
    sent_list = []
    for dialog in dialog_list:
        sent = " <SEP> ".join(dialog)
        sent_list.append(sent)
    return sent_list


def get_qnli_data(data_type):
    if (data_type == 'dev'):
        data_type = 'validation'
    dataset = load_dataset('glue', 'qnli', cache_dir=CACHE_PATH, split=data_type)
    sentence_list = []
    for data in dataset:
        question = data["question"]
        ans = data["sentence"]
        label = data["label"]
        if label == 0:
            #suffix = "The relation is not entailment"
            suffix = "The relation is 0"
        elif label == 1:
            #suffix = "The relation is entailment"
            suffix = "The relation is 1"
        else:
            sys.exit(-1)
        sentence = question + " <SEP> " + ans + " <SEP> " + suffix
        sentence_list.append(sentence)
    return sentence_list


def get_sst2_data(data_type):
    if(data_type == 'dev'):
        data_type = 'validation'
    dataset = load_dataset('glue', 'sst2', cache_dir=CACHE_PATH, split=data_type)
    sentence_list = []
    for data in dataset:
        sentence = data["sentence"]
        label = data["label"]
        if label == 0:
            #suffix = "The sentiment is negative"
            suffix = "The sentiment is 0"
        elif label == 1:
            #suffix = "The sentiment is positive"
            suffix = "The sentiment is 1"
        else:
            sys.exit(-1)
        sentence = sentence + " <SEP> " + suffix
        sentence_list.append(sentence)
    return sentence_list


def get_mnli_data(data_type):
    if(data_type == 'test'):
        data_type = 'test_matched'
    if(data_type == 'dev'):
        data_type = 'validation_matched'
    dataset = load_dataset('glue', 'mnli', cache_dir=CACHE_PATH, split=data_type)
    sentence_list = []
    for data in dataset:
        premise = data['premise']
        hypothesis = data['hypothesis']
        if data['label'] == 0:
            #suffix = "The relation is entailment"
            suffix = "The relation is 0"
        elif data['label'] == 1:
            #suffix = "The relation is neutral"
            suffix = "The relation is 1"
        elif data['label'] == 2:
            #suffix = "The relation is contradiction"
            suffix = "The relation is 2"
        else:
            sys.exit(-1)
        sent = premise + " <SEP> " + hypothesis + " <SEP> " + suffix
        sentence_list.append(sent)
    return sentence_list


def get_processed_persona(kind, processed_persona_path, require_label=False):
    # processed_persona_path = config.processed_persona
    if (require_label):
        path = processed_persona_path + '/%s_merged_shuffle.txt' % kind
    else:
        path = processed_persona_path + '/%s.txt' % kind
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def process_persona(data):
    '''
    get only list of texts for batch training for
    '''
    sentence_list = []
    for i, dict_i in enumerate(data):
        conv = dict_i['conv']
        sentence_list.append(conv)

    return sentence_list


def get_persona_dict(data_type, processed_persona_path):
    data = get_processed_persona(data_type, processed_persona_path)
    processed_data = process_persona(data)

    return processed_data


if __name__ == '__main__':
    dataset_name = 'mnli'
    data_type = 'dev'
    sent_list_aux = get_sent_list(dataset_name, data_type,is_aux=True)
    sent_list = get_sent_list(dataset_name, data_type)
    sent_list_all = get_sent_list(dataset_name, data_type,return_all=True)
    print(sent_list)