import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import wandb
import torch, transformers
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from private_transformers import PrivacyEngine
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from dataset.canary_dataset import CanaryDataset, EvaluationDataset, generate_evaluation_sequence_for_gpt
from utils import SequenceCrossEntropyLoss, calculate_perplexity_for_gpt, calculate_exposures
import numpy as np
from transformers import AutoModel, AutoTokenizer
from eval.DEA.exposure_metric import calculate_single_exposure
from transformers import AutoModelForCausalLM
import argparse
import config
from tqdm import tqdm
from eval.utility.utility_evaluation import utility_evaluation
from training_interface import DP_trainer

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rcParams
from sklearn import metrics
rcParams['font.size'] = 30   




MAX_SEQ_LEN = 1024

class GPT_Trainer_For_MIA(DP_trainer):
    def __init__(self, config):
        super().__init__(**config)
    def get_scheduler(self):
        #self.gradient_accumulation_steps
        t_total = int(len(self.dataloader) // self.n_accumulation_steps * self.epochs)
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=t_total
        )

    def freeze_model_embedding(self):
        self.model.get_input_embeddings().requires_grad_(False)
        print("embedding layer is frozen")

    def get_tokenizer(self, model):
        # return AutoTokenizer.from_pretrained(model)
        tokenizer = AutoTokenizer.from_pretrained(model)
        return tokenizer

    def get_model(self, model):
        # hf_model = AutoModelForCausalLM.from_pretrained(model)
        # hf_model.save_pretrained(os.path.join('hf_models', model))
        model = AutoModelForCausalLM.from_pretrained(model)
        return model

    def load_model(self, model_path):
        del self.model
        self.model = AutoModelForCausalLM.from_pretrained(model_path)

    def get_optimizer(self, optimizer, lr):
        if optimizer == "adam":
            optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        elif optimizer == "adamw":
            optimizer = AdamW(params=self.model.parameters(), lr=lr)
        else:
            raise NotImplementedError("Given optimizer type is not supported")
        return optimizer

    def get_dataloader(self, dataset_name):
        if dataset_name in ['personachat', 'qnli', "mnli", 'sst2']:
            if self.data_type == 'train':
                data = CanaryDataset(self.canary_type_list, self.insert_proportion_list, self.insert_time_base_list,
                                     dataset_name=dataset_name, data_type=self.data_type)
            else:
                data = EvaluationDataset(dataset_name=dataset_name, data_type=self.data_type)
        else:
            raise NotImplementedError("Given dataset is not supported")

        return DataLoader(
            dataset=data,
            shuffle=False,
            batch_size=self.batch_size
        )


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

    def utility_evaluate(self):
        self.data_type = 'dev'
        self.eval_dataloader = self.get_dataloader(self.dataset_name)
        self.model.eval()
        self.model.to(self.device)
        acc = utility_evaluation(model=self.model, tokenizer=self.tokenizer,dataloader=self.eval_dataloader,
                                 dataset_name=self.dataset_name,
                                 ### eva_method = 'counting' or 'logits'
                                 eva_method=self.eva_method,
                                 device=self.device,
                                 num_virtual_token=self.num_decode_virtual_tokens,
                                 num_beams=self.num_beams,
                                 num_return_sequences=self.num_return_sequence,
                                 max_length=self.generation_max_length
                                 )
        print(f"acc = {acc}")
        # wandb.log({"Validation Accuracy": acc})
    
    def get_probs_mia(self,data_type="dev"):
        self.data_type = data_type
        self.eval_dataloader = self.get_dataloader_for_mia(self.dataset_name)
        self.model.eval()
        self.model.to(self.device)
        sentences=[]
        for s in self.eval_dataloader:
            # print(s)
            sentences.extend(s)
        ppls=calculate_perplexity_for_gpt(sentences[0:3276],model=self.model, tokenizer=self.tokenizer,num_decode_virtual_tokens=self.num_decode_virtual_tokens,device=self.device,tuning_method=self.tuning_method,)
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
        
"""
CUDA_VISIBLE_DEVICES=0 python mia_attack.py &
CUDA_VISIBLE_DEVICES=1 python mia_attack.py &
CUDA_VISIBLE_DEVICES=2 python mia_attack.py &
CUDA_VISIBLE_DEVICES=3 python mia_attack.py &

for model in gpt2 gpt2-medium gpt2-large 
do
CUDA_VISIBLE_DEVICES=1 python mia_attack.py --model $model --dataset sst2 --dp 1
CUDA_VISIBLE_DEVICES=1 python mia_attack.py --model $model --dataset sst2 --dp 0
done

"""



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
    config['lr'] = 1e-4
    config['canary_type_list'] = []
    config["insert_proportion_list"] = []
    config["insert_time_base_list"] = []
    trainer = GPT_Trainer_For_MIA(config)
    with torch.no_grad():

        dev_prob_before=trainer.get_probs_mia(data_type="dev")
        train_prob_before=trainer.get_probs_mia(data_type="train")
        dp_str = "dp" if trainer.dp else "non_dp"
        trainer.save_path = os.path.join(BASE_DIR, 'checkpoints_gpt_all',trainer.config_str, trainer.dataset_name + '_' + trainer.model_name + '_' + dp_str)
        print("loading saved model")
        torch.cuda.empty_cache()

    trainer.train_our_model()
    trainer.utility_evaluate()

    with torch.no_grad():
        trainer.load_model(trainer.save_path)
        train_prob_after=trainer.get_probs_mia(data_type="train")
        dev_prob_after=trainer.get_probs_mia(data_type="dev")
        res=trainer.mia_evaluate(train_prob_before-train_prob_after,dev_prob_before-dev_prob_after)
        with open("MIA_RES.txt","a") as f:
            f.write( f"{ config['model'] }{config['dataset_name']}{config['dp']}"  +"\t"+ str(res)+"\n" )
        print(res)
    
