import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import config
from training_interface import DP_trainer
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

MAX_SEQ_LEN = 1024


class Llama_Finetune(DP_trainer):
    def __init__(self, config):
        super().__init__(**config)

    def get_tokenizer(self, model):
        # return AutoTokenizer.from_pretrained(model)
        tokenizer = AutoTokenizer.from_pretrained(model)
        return tokenizer

    def get_model(self, model):
        # hf_model = AutoModelForCausalLM.from_pretrained(model)
        # hf_model.save_pretrained(os.path.join('hf_models', model))
        model = AutoModelForCausalLM.from_pretrained(model)
        return model

if __name__ == "__main__":
    config = config.config
    ### reload config

    config['model'] = "llama-2-7b"
    config['lr'] = 1e-4
    config['epochs'] = 5
    #config['dataset_name'] = 'qnli'

    for dataset_name in ['mnli']:
        for dp in [False,True]:
            for training_mode in ['finetune']:
                config['dp'] = dp
                config['dataset_name'] = dataset_name

                print("Config is loaded")

                trainer = Llama_Finetune(config)
                config_str = trainer.config_str
                ### add wanbd
                dp_str = "dp" if config["dp"]  else "non_dp"
                name_=f"{config['dataset_name']}-{config['model']}-{dp_str}"
                name_ = name_ + '-' + config_str
                run = wandb.init(
                # set the wandb project where this run will be logged
                project=f"llama-{config['dataset_name']}",
                name=name_,
                # track hyperparameters and run metadata
                config=config)
                # trainer.utility_evaluate()
                trainer.train_our_model()
                run.finish()