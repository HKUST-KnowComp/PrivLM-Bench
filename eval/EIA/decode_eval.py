'''
This file is used to evaluate the generation performance from attackers.
The metrics include BLEU, ROUGE, accuracy.

'''
import os
from sklearn import metrics
import numpy as np
import argparse
import torch
import json
from scipy.spatial.distance import cosine
from simcse import SimCSE
from nltk.tokenize import word_tokenize
import string
import re
from transformers import AutoTokenizer


import nltk
import evaluate
import editdistance


rouge = evaluate.load('rouge')

# remove punctuation from list of sentences 
def punctuation_remove(sent_list):
    removed_list = []
    for sent in sent_list:
        word_list = []
        for word in sent.split():
            word_strip = word.strip(string.punctuation)
            if word_strip:  # cases for not empty string
                word_list.append(word_strip)
        removed_sent = ' '.join(word_list)
        removed_list.append(removed_sent)
    return removed_list

def get_rouge(data):
    gt = data["gt"]
    pred = data["pred"]
    results = rouge.compute(predictions=pred,references=gt)
    print(results)
    return results

def get_bleu(data):
    gt = data['gt']
    pred = data["pred"]
    cands_list_bleu = [sentence.split() for sentence in pred] 
    refs_list_bleu = [[sentence.split()] for sentence in gt]
    bleu_score = nltk.translate.bleu_score.corpus_bleu(refs_list_bleu, cands_list_bleu) 
    bleu_score_1 = nltk.translate.bleu_score.corpus_bleu(refs_list_bleu, cands_list_bleu,weights=(1, 0, 0, 0)) 
    bleu_score_2 = nltk.translate.bleu_score.corpus_bleu(refs_list_bleu, cands_list_bleu,weights=(0.5, 0.5, 0, 0)) 
    #print(f'bleu1 : {bleu_score_1}')
    #print(f'bleu2 : {bleu_score_2}')
    #print(f'bleu : {bleu_score}')
    return bleu_score_1, bleu_score_2, bleu_score




def get_edit_dist(data):
    gt = data['gt']
    pred = data["pred"]
    assert len(gt) == len(pred)
    edit_dist_list = []
    for i,d in enumerate(pred):
        gt_str = gt[i]
        pred_str = pred[i]
        dist = editdistance.distance(gt_str, pred_str)
        edit_dist_list.append(dist)
    ### now we return mean and median
    edit_dist_list = np.array(edit_dist_list)
    edit_median  = np.median(edit_dist_list)
    edit_mean = np.mean(edit_dist_list)
    #print(f'edit_mean: {edit_mean}')
    #print(f'edit_median: {edit_median}')
    return edit_mean,edit_median

def exact_match(data):
    gt = data['gt']
    pred = data["pred"]

    gt_remove = punctuation_remove(gt)       
    pred_remove = punctuation_remove(pred) 

    assert len(gt) == len(pred)
    count = 0 
    for i,d in enumerate(pred):
        gt_str = gt[i]
        pred_str = pred[i]
        if(gt_str == pred_str):
            count += 1
    ratio = count/len(gt)
    count = 0 
    for i,d in enumerate(pred):
        gt_str = gt_remove[i]
        pred_str = pred_remove[i]
        if(gt_str == pred_str):
            count += 1
    ratio_remove = count/len(gt_remove)
    #print(f'exact_match ratio: {ratio}')

    #print(f'exact_match ratio after removing punctuation: {ratio_remove}')

    return ratio, ratio_remove

def remove_eos_gen(data):
    gt = data['gt']
    pred = data["pred"]
    for i,s in enumerate(pred):
        pred[i] = s.replace('<|endoftext|>','')

def report_metrics(sent_dict):
    data = sent_dict

    result_dict = {}
    remove_eos_gen(data)
    rouge_results = get_rouge(data)
    result_dict['rouge_results'] = rouge_results
    bleu_score_1, bleu_score_2, bleu_score = get_bleu(data)
    result_dict['bleu_score_1'] = bleu_score_1
    result_dict['bleu_score_2'] = bleu_score_2
    result_dict['bleu_score_4'] = bleu_score
    #print(f'exact_match ratio: {ratio}')

    #print(f'exact_match ratio after removing punctuation: {ratio_remove}')

    ratio, ratio_remove = exact_match(data)
    result_dict['exact_match_ratio'] = ratio
    result_dict['exact_match_ratio_remove'] = ratio_remove
    edit_mean,edit_median = get_edit_dist(data)
    result_dict['edit_mean'] = edit_mean
    result_dict['edit_median'] = edit_median

    return result_dict




###### CLS METRICS ######
def vectorize(sent_list,tokenizer):
    turn_ending = tokenizer.encode(tokenizer.eos_token)
    token_num = len(tokenizer)
    dial_tokens = [tokenizer.encode(item) + turn_ending for item in sent_list]
    dial_tokens_np = np.array(dial_tokens)
    input_labels = []
    for i in dial_tokens_np:
        temp_i = np.zeros(token_num)
        temp_i[i] = 1
        input_labels.append(temp_i)
    input_labels = np.array(input_labels)


    return input_labels





def report_score(y_true,y_pred):
    # micro result should be reported
    precision = metrics.precision_score(y_true, y_pred, average='micro')
    recall = metrics.recall_score(y_true, y_pred, average='micro')
    f1 = metrics.f1_score(y_true, y_pred, average='micro')
    #logger.info(f"micro precision_score on token level: {str(precision)}")
    #logger.info(f"micro recall_score on token level: {str(recall)}")
    #logger.info(f"micro f1_score on token level: {str(f1)}")
    return precision,recall,f1

# remove punctuation from list of sentences 
def punctuation_remove(sent_list):
    removed_list = []
    for sent in sent_list:
        word_list = []
        for word in sent.split():
            word_strip = word.strip(string.punctuation)
            if word_strip:  # cases for not empty string
                word_list.append(word_strip)
        removed_sent = ' '.join(word_list)
        removed_list.append(removed_sent)
    return removed_list

# remove space before punctuation from list of sentences 
def space_remove(sent_list):
    removed_list = []
    for sent in sent_list:
        sent_remove = re.sub(r'\s([?.!"](?:\s|$))', r'\1', sent)
        removed_list.append(sent_remove)
    return removed_list

def metrics_word_level(token_true,token_pred):
    len_pred = len(token_pred)
    len_ture = len(token_true)
    recover_pred = 0
    recover_true = 0
    for p in token_pred:
        if p in token_true:
            recover_pred += 1
    for t in token_true:
        if t in token_pred:
            recover_true += 1
    ### return for precision recall calculation        
    return len_pred,recover_pred,len_ture,recover_true
            
    
def word_level_metrics(y_true,y_pred):
    assert len(y_true) == len(y_pred)
    recover_pred_all = 0
    recover_true_all = 0
    len_pred_all = 0
    len_ture_all = 0
    for i in range(len(y_true)):
        sent_true = y_true[i]
        sent_pred = y_pred[i]
        token_true = word_tokenize(sent_true)
        token_pred = word_tokenize(sent_pred)
        len_pred,recover_pred,len_ture,recover_true = metrics_word_level(token_true,token_pred)
        len_pred_all += len_pred
        recover_pred_all += recover_pred
        len_ture_all += len_ture
        recover_true_all += recover_true
        
        
    ### precision and recall are based on micro (but not exactly)
    precision = recover_pred_all/len_pred_all
    recall = recover_true_all/len_ture_all
    f1 = 2*precision*recall/(precision+recall)
    return precision,recall,f1

def remove_eos(sent_list):
    for i,s in enumerate(sent_list):
        sent_list[i] = s.replace('<|endoftext|>','')

def classification_metrics(sent_dict):
    y_true = sent_dict['gt']     # list of sentences
    y_pred = sent_dict['pred']   # list of sentences   
    result_config = {}
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    y_true_token = vectorize(y_true,tokenizer)
    y_pred_token = vectorize(y_pred,tokenizer)

    ### token-level metrics are reported
    precision,recall,f1 = report_score(y_true_token,y_pred_token)
    # logger.info(f"micro precision_score on token level: {str(precision)}")
    # logger.info(f"micro recall_score on token level: {str(recall)}")
    # logger.info(f"micro f1_score on token level: {str(f1)}")
    result_config['micro_token_level_precision'] = precision
    result_config['micro_token_level_recall'] = recall
    result_config['micro_token_level_f1'] = f1



    remove_eos(y_pred)           # make sure to remove <eos>
    ### scores for word level
    y_true_removed_p = punctuation_remove(y_true)       
    y_pred_removed_p = punctuation_remove(y_pred)  
    y_true_removed_s = space_remove(y_true)       
    y_pred_removed_s = space_remove(y_pred)  
    precision,recall,f1 = word_level_metrics(y_true_removed_s,y_pred_removed_s)
    #logger.info(f'word level precision: {str(precision)}')
    #logger.info(f'word level recall: {str(recall)}')
    #logger.info(f'word level f1: {str(f1)}')
    result_config['word_level_precision'] = precision
    result_config['word_level_recall'] = recall
    result_config['word_level_f1'] = f1

    precision,recall,f1 = word_level_metrics(y_true_removed_p,y_pred_removed_p)
    #logger.info(f'word level precision without punctuation: {str(precision)}')
    #logger.info(f'word level recall without punctuation: {str(recall)}')
    #logger.info(f'word level f1 without punctuation: {str(f1)}')
    result_config['word_level_precision_without_punctuation'] = precision
    result_config['word_level_recall_without_punctuation'] = recall
    result_config['word_level_f1_without_punctuation'] = f1

    return result_config


def eval_eia(sent_dict,log_path):
    result_dict = {}
    result_dict['cls'] = classification_metrics(sent_dict)
    result_dict['gen'] = report_metrics(sent_dict)
    save_path = os.path.join(log_path,'eia_result.json')
    with open(save_path,'w') as f:
        json.dump(result_dict,f,indent=4)
    return result_dict