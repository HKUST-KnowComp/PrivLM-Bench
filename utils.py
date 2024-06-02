import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SequenceCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets, mask, label_smoothing=-1, reduce=None):
        """
        reduce: None, "batch", "sentence"
        """
        return sequence_cross_entropy_with_logits(logits, targets, mask, label_smoothing, reduce)


def sequence_cross_entropy_with_logits(logits, targets, mask, label_smoothing, reduce):
    # type: (Tensor, Tensor, Tensor, float, bool)-> Tensor
    """
    label_smoothing : ``float``, optional (default = 0.0)
        It should be smaller than 1.
    """
    # shape : (batch * sequence_length, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = F.log_softmax(logits_flat, dim=-1)
    # shape : (batch * max_len, 1)
    targets_flat = targets.reshape(-1, 1).long()

    if label_smoothing > 0.0:
        num_classes = logits.size(-1)
        smoothing_value = label_smoothing / float(num_classes)
        # Fill all the correct indices with 1 - smoothing value.
        one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(-1, targets_flat, 1.0 - label_smoothing)
        smoothed_targets = one_hot_targets + smoothing_value
        negative_log_likelihood_flat = -log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
    else:
        # shape : (batch * sequence_length, 1)
        negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)

    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(-1, logits.shape[1])

    # shape : (batch, sequence_length)
    loss = negative_log_likelihood * mask

    if reduce:
        # shape : (batch,)
        loss = loss.sum(1) / (mask.sum(1) + 1e-13)

        if reduce == "batch":
            # shape : scalar
            loss = loss.mean()

    return loss


def calculate_perplexity_for_gpt(batch_text, model, tokenizer, num_decode_virtual_tokens, tuning_method, device):
    model.eval()
    model.to(device)
    padding_token_id = tokenizer.encode(tokenizer.eos_token)[0]
    log_ppl_list = []
    with torch.no_grad():
        tokenizer.pad_token = tokenizer.eos_token
        for text in batch_text:
            inputs = tokenizer(text, return_tensors='pt')
            input_ids = inputs['input_ids'].to(device)
            labels = input_ids.clone()
            logits, past = model(input_ids=input_ids, return_dict=False)

            if tuning_method == 'prompt-tuning':
                soft_logits = F.softmax(logits, dim=-1)[0, num_decode_virtual_tokens-1:-1]
                target = labels
            else:
                soft_logits = F.softmax(logits, dim=-1)[0, :-1]
                target = labels[0, 1:]

            index_0 = torch.arange(0, soft_logits.shape[0])
            index_1 = target
            probs = soft_logits[index_0, index_1].cpu().numpy()

            log_ppl = np.sum(-1.0 * np.log2(probs))
            log_ppl_list.append(log_ppl)

    log_ppl_list = np.stack(log_ppl_list)
    return log_ppl_list


def calculate_perplexity_for_t5(batch_text_pair, model, tokenizer, tuning_method, device):
    model.eval()
    model.to(device)
    log_ppl_list = []
    with torch.no_grad():
        tokenizer.pad_token = tokenizer.eos_token
        for text_pair in batch_text_pair:
            tokenized_source = tokenizer(text_pair[0], return_tensors='pt')
            tokenized_target = tokenizer(text_pair[1], return_tensors='pt')

            input_ids = tokenized_source['input_ids'].to(device)
            src_attention_mask = tokenized_source['attention_mask'].to(device)

            labels = tokenized_target['input_ids'].to(device)

            tgt_attention_mask = tokenized_target['attention_mask'].to(device)

            if tuning_method == 'prompt-tuning':
                logits = model(
                    input_ids=input_ids,
                    attention_mask=src_attention_mask,
                    labels=labels
                ).logits
            else:
                logits = model(
                    input_ids=input_ids,
                    attention_mask=src_attention_mask,
                    labels=labels,
                    decoder_attention_mask=tgt_attention_mask
                ).logits

            soft_logits = F.softmax(logits, dim=-1)[0]
            target = labels.clone()

            index_0 = torch.arange(0, soft_logits.shape[0])
            index_1 = target
            probs = soft_logits[index_0, index_1].cpu().numpy()

            log_ppl = np.sum(-1.0 * np.log2(probs))
            log_ppl_list.append(log_ppl)

        log_ppl_list = np.stack(log_ppl_list)
        return log_ppl_list



def calculate_exposures(canary_losses, reference_losses):
    combination = np.append(canary_losses, reference_losses)
    n = combination.shape[0]
    exposures = []
    for idx, canary_loss in enumerate(canary_losses):
        exposure = np.log2(n) - np.log2(np.argsort(combination).tolist().index(idx) + 1)
        exposures.append(exposure)
    return np.stack(exposures)
