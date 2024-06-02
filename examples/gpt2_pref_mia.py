'''
This file is for GPT-2 casual LM prefix tuning
'''
import os
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.append(BASE_DIR)
import config
from training_interface import DP_trainer
from peft import PrefixTuningConfig, TaskType, get_peft_model, PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import wandb
from private_transformers import PrivacyEngine

from torch.utils.data import DataLoader, Dataset

from dataset.canary_dataset import CanaryDataset, EvaluationDataset, generate_evaluation_sequence_for_gpt
from utils import SequenceCrossEntropyLoss, calculate_perplexity_for_gpt, calculate_exposures

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rcParams
from sklearn import metrics
import numpy as np
from gpt2_casual_lm_pref import GPT2_Casual_LM_Pref_Trainer
MAX_SEQ_LEN = 1024


class GPT2_Casual_LM_Pref_Trainer_for_MIA(GPT2_Casual_LM_Pref_Trainer):
    def __init__(self, config, tuning_config):
        super().__init__(config, **tuning_config)

    def get_probs_mia(self,data_type="dev"):
        self.data_type = data_type
        self.eval_dataloader = self.get_dataloader_for_mia(self.dataset_name)
        self.model.eval()
        self.model.to(self.device)
        sentences=[]
        for s in self.eval_dataloader:
            # print(s)
            sentences.extend(s)
        sentences_count=min(3276,len(sentences))
        print("sentences_count",sentences_count)
        ppls=calculate_perplexity_for_gpt(sentences[0:sentences_count],model=self.model, tokenizer=self.tokenizer,num_decode_virtual_tokens=self.num_decode_virtual_tokens,device=self.device,tuning_method=self.tuning_method,)
        return torch.tensor(ppls)
        
    def mia_evaluate(self,train_lira,test_prob_lira):
        fpr, tpr, thresholds = metrics.roc_curve(torch.cat( [torch.zeros_like(test_prob_lira),torch.ones_like(train_lira)] ).cpu().numpy(), torch.cat([test_prob_lira,train_lira]).cpu().numpy())
        auc=metrics.auc(fpr, tpr)
        log_tpr,log_fpr=np.log10(tpr),np.log10(fpr)
        log_tpr[log_tpr<-5]=-5
        log_fpr[log_fpr<-5]=-5
        log_fpr=(log_fpr+5)/5.0
        log_tpr=(log_tpr+5)/5.0
        log_auc=metrics.auc( log_fpr,log_tpr )
        tprs={}
        for fpr_thres in [0.1,0.02,0.01,0.001,0.0001]:
            tpr_index = np.sum(fpr<fpr_thres)
            tprs[str(fpr_thres)]=tpr[tpr_index-1]
        return auc,log_auc,tprs

    def get_dataloader_for_mia(self, dataset_name):
        if dataset_name in ['personachat', 'qnli', "mnli", 'sst2']:
            # data = EvaluationDataset(dataset_name=dataset_name, data_type=self.data_type)
            data = CanaryDataset(self.canary_type_list, self.insert_proportion_list, self.insert_time_base_list,
                                 dataset_name=dataset_name, data_type=self.data_type)
        else:
            raise NotImplementedError("Given dataset is not supported")

        return DataLoader(
            dataset=data,
            shuffle=False,
            batch_size=self.batch_size
        )



'''
CUDA_VISIBLE_DEVICES=0 python gpt2_pref_mia.py &
CUDA_VISIBLE_DEVICES=1 python gpt2_pref_mia.py &
CUDA_VISIBLE_DEVICES=2 python gpt2_pref_mia.py &
CUDA_VISIBLE_DEVICES=3 python gpt2_pref_mia.py &

for model in  gpt2-xl
do 
CUDA_VISIBLE_DEVICES=3 python gpt2_pref_mia.py --model $model --dataset mnli --dp 1  
CUDA_VISIBLE_DEVICES=3 python gpt2_pref_mia.py --model $model --dataset mnli --dp 0
done

for model in  gpt2-xl
do 
CUDA_VISIBLE_DEVICES=3 python gpt2_prom_mia.py --model $model --dataset mnli --dp 1  
CUDA_VISIBLE_DEVICES=3 python gpt2_prom_mia.py --model $model --dataset mnli --dp 0
done

for model in  gpt2 gpt2-medium gpt2-large
do 
CUDA_VISIBLE_DEVICES=2 python gpt2_prom_mia.py --model $model --dataset sst2 --dp 1  
CUDA_VISIBLE_DEVICES=2 python gpt2_prom_mia.py --model $model --dataset sst2 --dp 0
done

for model in  gpt2 gpt2-medium gpt2-large
do 
CUDA_VISIBLE_DEVICES=2 python gpt2_pref_mia.py --model $model --dataset sst2 --dp 1  
CUDA_VISIBLE_DEVICES=2 python gpt2_pref_mia.py --model $model --dataset sst2 --dp 0
done

'''

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='openai-community/gpt2')
    parser.add_argument("--dataset", type=str, default='sst2')
    parser.add_argument("--dp", type=int, default=0)
    args = parser.parse_args()

    config = config.config
    config['model'] = args.model
    config['dataset_name'] = args.dataset
    config['dp'] = args.dp == 1
    config['lr'] = 1e-2
    config['canary_type_list'] = []
    config["insert_proportion_list"] = []
    config["insert_time_base_list"] = []
    # assert 0

    tuning_config = {}
    tuning_config["num_virtual_token"] = 15
    tuning_config["adapter_name"] = "gpt2_casual_lm_pref"
    tuning_config["training_mode"] = "prefix-tuning"
    # tuning_config["training_mode"] = "prefix-tuning"
    tuning_config["task_type"] = "casual_lm"
    trainer = GPT2_Casual_LM_Pref_Trainer_for_MIA(config, tuning_config)
    
    with torch.no_grad():
        dev_prob_before=trainer.get_probs_mia(data_type="dev")
        train_prob_before=trainer.get_probs_mia(data_type="train")
        dp_str = "dp" if trainer.dp else "non_dp"
        trainer.save_path = os.path.join(BASE_DIR, 'checkpoints_gpt_all', trainer.config_str,
                                     trainer.dataset_name + '_' + trainer.model_name + '_' + trainer.tuning_method + '_' + dp_str)
        print("loading saved model")
        torch.cuda.empty_cache()
    # trainer.train_our_model()
    trainer.utility_evaluate()
    with torch.no_grad():
        train_prob_after=trainer.get_probs_mia(data_type="train")
        dev_prob_after=trainer.get_probs_mia(data_type="dev")
        res=trainer.mia_evaluate(train_prob_before-train_prob_after,dev_prob_before-dev_prob_after)
        with open("MIA_RES_pref.txt","a") as f:
            f.write( f"{ config['model'] }{config['dataset_name']}{config['dp']}"  +"\t"+ str(res)+"\n" )
        print(res)