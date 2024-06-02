import numpy as np
import pandas as pd
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import operator
from queue import PriorityQueue



def eval_on_batch(batch_X,batch_D,model,tokenizer,device,config):
    decode_method = config['decode']
    padding_token_id = tokenizer.encode(tokenizer.eos_token)[0]
    if(not config['use_opt']):
        tokenizer.pad_token = tokenizer.eos_token
    batch_X = batch_X.to(device)
    print(f'batch_X:{batch_X.size()}')
    sent_list = []
    gt_list = batch_D
    for i,hidden in enumerate(batch_X):
        inputs_embeds = hidden
        if(decode_method == 'beam'):
            #print('Using beam search decoding')

            sentence = beam_decode_sentence(hidden_X=inputs_embeds, config = config,num_generate=1, beam_size = 5)

            #print(sentence)
            sentence = sentence[0]
        else:
            ## greedy decoding
            sentence = generate_sentence(config,hidden_X=inputs_embeds)
        sent_list.append(sentence)



    return sent_list, gt_list


def generate_sentence(config,hidden_X):
    temperature = 0.9
    top_k = -1
    top_p = 0.9
    sent = []
    prev_input = None
    past = None
    model = config['model']
    tokenizer =config['tokenizer']
    #eos = [tokenizer.encoder["<|endoftext|>"]]
    eos = tokenizer.encode("<|endoftext|>")
    hidden_X_unsqueeze = torch.unsqueeze(hidden_X, 0)
    hidden_X_unsqueeze = torch.unsqueeze(hidden_X_unsqueeze, 0)  #[1,1,embed_dim]
    logits, past = model(inputs_embeds=hidden_X_unsqueeze,past_key_values  = past,return_dict=False)
    logits = logits[:, -1, :] / temperature
    logits = top_filtering(logits, top_k=top_k, top_p=top_p)

    probs = torch.softmax(logits, dim=-1)

    prev_input = torch.multinomial(probs, num_samples=1)
    prev_word = prev_input.item()
    sent.append(prev_word)

    for i in range(50):
        #logits, past = model(prev_input, past=past)
        logits, past = model(prev_input,past_key_values  = past,return_dict=False)
        logits = logits[:, -1, :] / temperature
        logits = top_filtering(logits, top_k=top_k, top_p=top_p)

        probs = torch.softmax(logits, dim=-1)

        prev_input = torch.multinomial(probs, num_samples=1)
        prev_word = prev_input.item()

        if prev_word == eos[0]:
            break
        sent.append(prev_word)
    
    output = tokenizer.decode(sent)

    return output


def top_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
    """
    # batch support!
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[-1])
        logits = torch.where(logits < min_values, 
                             torch.ones_like(logits, dtype=logits.dtype) * -float('Inf'), 
                             logits)
    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        sorted_logits = sorted_logits.masked_fill_(sorted_indices_to_remove, filter_value)
        logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)
    
    return logits




class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

    def __lt__(self, x):
        if(self.eval() < x.eval()):
            return True

        else:
            return False



def beam_decode_sentence(hidden_X, config,num_generate=1, beam_size = 5, batch_size = 1):
    '''
    generate a sentence based on beam search
    :param hidden_X: hidden_X of sentence embedding  (1024) with/without projection
    :param model: GPT-2 model
    :param tokenizer: GPT-2 tokenizer
    :return: decoded_batch
    '''
    #SOS_token = tokenizer.encode("<|endoftext|>")
    beam_width = beam_size
    topk = num_generate  # how many sentence do you want to generate

    past = None
    model = config['model']
    tokenizer =config['tokenizer']
    eos = [tokenizer.encode(tokenizer.eos_token)]
    EOS_token = eos
    hidden_X_unsqueeze = torch.unsqueeze(hidden_X, 0)
    hidden_X_unsqueeze = torch.unsqueeze(hidden_X_unsqueeze, 0)  #[1,1,embed_dim]

    decoded_batch = []

    # decoding goes sentence by sentence
    for idx in range(batch_size):

        # Number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        # starting node -  hidden vector, previous node, word id, logp, length              hiddenstate, previousNode, wordId, logProb, length
        node = BeamSearchNode(past, None, torch.tensor([[220]]).cuda(), 0, 1)                    # 220 refers to single space ' ' on GPT
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        # start beam search
        for text_len in range(50):
            # give up when decoding takes too long
            if qsize > 2000: break

            # fetch the best node
            try:
                score, n = nodes.get()
            except:
                print('Cannot get nodes')
                while not nodes.empty():
                    next_item = nodes.get()
                    print(next_item)
            prev_input = n.wordid
            past = n.h

            if n.wordid.item() == EOS_token[0] and n.prevNode != None:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break                    
                else:
                    print('continue')
                    continue
            # decode for one step using decoder
            if(text_len == 0):
                logits, past = model(inputs_embeds=hidden_X_unsqueeze,past_key_values  = past,return_dict=False)
                
            else:
                logits, past = model(prev_input,past_key_values = past, attention_mask=None, return_dict=False)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1) 
            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(probs, beam_width)
            nextnodes = []
            
            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(1, -1)
                log_p = log_prob[0][new_k].item()
                #### hiddenstate, previousNode, wordId, logProb, length
                node = BeamSearchNode(past, n, decoded_t, n.logp + log_p, n.leng + 1)
                score = -node.eval()
                nextnodes.append((score, node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                try:
                    nodes.put((score, nn))
                except:
                    print('Cannot put nodes')
                    print(score)
                    print(nn)
                # increase qsize
            qsize += len(nextnodes) - 1
        # for loop ends here
        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        text = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.wordid.item())
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordid.item())

            utterance = utterance[::-1]
            utterances.append(utterance)
            decode_process = tokenizer.decode(utterance[1:-1])
            text.append(decode_process)
        decoded_batch.append(utterances)

    return text