'''
This file is used for GLUE dataset testing stage performance evaluation.

'''
import torch
import torch.nn as nn
import os
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForCausalLM
import torch.nn.functional as F


def utility_evaluation(model, tokenizer, dataloader, dataset_name, eva_method, device='cuda',
                       num_virtual_token=0,
                       num_beams=5,
                       max_length=20,
                       num_return_sequences=10):
    length = len(dataloader.dataset)
    tokenizer.pad_token = tokenizer.eos_token
    right_count = 0
    for batch_sentence, batch_label in dataloader:
        batch_right_count = clm_utility_prediction_for_batch(model, tokenizer, batch_sentence,
                                                        batch_label, dataset_name, eva_method, device,
                                                        num_virtual_token,
                                                        num_beams,
                                                        max_length,
                                                        num_return_sequences
                                                        )
        right_count += batch_right_count
    return right_count/length


def clm_utility_prediction_for_batch(model, tokenizer, batch_sentence, batch_label,
                           dataset_name, eva_method="counting", device='cuda', num_virtual_token=0,
                                     num_beams=5, max_length=20, num_return_sequences=10):
    # return 0 for missing and 1 for hitting
    batch_sentence = list(batch_sentence)
    batch_label = list(batch_label)
    if dataset_name in ['qnli', 'sst2']:
        prediction_range = ['0', '1']
    elif dataset_name == 'mnli':
        prediction_range = ['0', '1', '2']
    else:
        raise NotImplementedError
    prediction_idx_range = []
    for i in prediction_range:
        prediction_idx_range.append(tokenizer.encode(i)[0])
    batch_label = [label[0] for label in tokenizer(batch_label)['input_ids']]
    if eva_method == 'counting':
        right_count = calculate_by_counting(model, tokenizer, batch_sentence, batch_label, prediction_idx_range, device,
                                            num_beams, max_length, num_return_sequences)
    elif eva_method == 'logits':
        right_count = calculate_by_logits(model, tokenizer, batch_sentence, batch_label, prediction_idx_range, device, num_virtual_token)
    else:
        raise NotImplementedError
    return right_count


def calculate_by_counting(model, tokenizer, batch_sentence, batch_label, prediction_idx_range, device,
                          num_beams=20, max_length=5, num_return_sequences=10):
    right_count = 0
    for sentence, label in zip(batch_sentence, batch_label):
        encoding = tokenizer(sentence, return_tensors='pt').to(device)
        sent_len = len(encoding['input_ids'][0])
        with torch.no_grad():
            generated_ids = model.generate(
                **encoding,
                max_new_tokens=1,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                eos_token_id=tokenizer.eos_token_id,
            )
        pred_ids = generated_ids[:, -1].tolist()
        counting = [0] * (len(prediction_idx_range)+1)
        for pred_id in pred_ids:
            if pred_id in prediction_idx_range:
                counting[prediction_idx_range.index(pred_id)] += 1
            else:
                counting[-1] += 1
        max_counting_idx = counting.index(max(counting))
        if max_counting_idx < len(prediction_idx_range) and prediction_idx_range[max_counting_idx] == label:
            right_count += 1
        return right_count


def calculate_by_logits(model, tokenizer, batch_sentence, batch_label, prediction_idx_range,
                        device, num_virtual_token=0):
    batch_right_count = 0
    with torch.no_grad():
        inputs = tokenizer(batch_sentence, return_tensors='pt', padding=True)
        input_ids = inputs['input_ids'].to(device)
        att_mask = inputs['attention_mask'].to(device)
        last_idx = torch.sum(att_mask, dim=1, keepdim=True) - 1 + num_virtual_token
        outputs = model(input_ids=input_ids, return_dict=True)
        full_logits = outputs.logits
        #past = outputs.past_key_values
        # [batch_size, 1, range_num]
        index_0 = torch.arange(0, full_logits.shape[0]).unsqueeze(dim=-1).unsqueeze(dim=-1).repeat(1, 1, len(prediction_idx_range)).to(device)
        index_1 = last_idx.unsqueeze(dim=-1).repeat(1, 1, len(prediction_idx_range))
        index_2 = torch.tensor(prediction_idx_range).unsqueeze(dim=0).unsqueeze(dim=0).repeat(index_1.shape[0], 1, 1).to(device)
        logits = full_logits[index_0, index_1, index_2].squeeze(dim=1)
        prob = F.softmax(logits)
    pred_idx_list = torch.argmax(prob, dim=-1).tolist()
    for pred_idx, label in zip(pred_idx_list, batch_label):
        if prediction_idx_range[pred_idx] == label:
            batch_right_count += 1
    return batch_right_count

