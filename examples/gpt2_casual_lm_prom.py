'''
This file is for GPT-2 casual LM prompt tuning
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['HF_HOME'] = './hf_models/cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
access_token = "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"
os.environ['HF_TOKEN'] =access_token
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import config
import os
from training_interface import DP_trainer
from peft import PromptTuningInit, PromptTuningConfig, TaskType, get_peft_model, PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from private_transformers import PrivacyEngine
from tqdm import tqdm
import wandb
MAX_SEQ_LEN = 1024


class GPT2_Casual_LM_Prom_Trainer(DP_trainer):
    def __init__(self, config, **kwargs):
        ### tuning method for save functions
        self.num_virtual_token = kwargs.get("num_virtual_token", 15)
        self.adapter_name = kwargs.get("adapter_name", "gpt2_casual_lm_prom")
        self.training_mode = kwargs.get("training_mode", "prompt_tuning")
        self.task_type = kwargs.get("task_type", "casual_lm")
        
        self.dataset_name = config["dataset_name"]
        if self.dataset_name == "qnli":
            self.prompt_tuning_init_text = "Classify if the relation of these two sentence is entailment or not"
        elif self.dataset_name == "sst2":
            self.prompt_tuning_init_text = "Classify if the sentiment is negative or positive"
        elif self.dataset_name == "mnli":
            self.prompt_tuning_init_text = "Classify if the relation of these two sentence is entailment, neutral or contradiction"
        else:
            self.prompt_tuning_init_text = "The text is a dialog between peoples"

        super().__init__(**config)

        dp_str = "dp" if self.dp else "non_dp"
        self.tuning_method = "prompt-tuning"
        self.num_decode_virtual_tokens = self.num_virtual_token
        self.config_str = f"tuning_method-{self.tuning_method}-lr-{self.lr}-batch_size-{self.batch_size}-n_accumulation_steps-{self.n_accumulation_steps}-freeze_embedding-{self.freeze_embedding}-epochs-{self.epochs}-target_epsilon-{self.target_epsilon}"
        if len(self.canary_type_list) > 0:
            self.config_str = "canary_inserted-" + self.config_str
        self.save_path = os.path.join(BASE_DIR, 'checkpoints', self.config_str, self.dataset_name + '_' + self.model_name + '_' + dp_str)

    def get_tokenizer(self, model):
        tokenizer = AutoTokenizer.from_pretrained(model)
        return tokenizer


    def get_model(self, model):
        model = AutoModelForCausalLM.from_pretrained(model)
        assert self.training_mode == "prompt-tuning"
        assert self.num_virtual_token is not None
        assert self.task_type == "casual_lm"
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=self.num_virtual_token,
            prompt_tuning_init_text=self.prompt_tuning_init_text,
            # tokenizer_name_or_path=self.model_name
            tokenizer_name_or_path=self.model_name
        )
        model = get_peft_model(model, peft_config, adapter_name=self.adapter_name)
        model.print_trainable_parameters()
        return model

    def train_on_batch(self, batch_text, model, tokenizer, criterion, device):
        tokenizer.pad_token = tokenizer.eos_token
        if (self.tokenizer_len):
            inputs = tokenizer(batch_text, return_tensors='pt', padding='max_length', truncation=True,
                               max_length=self.tokenizer_len)
        else:
            inputs = tokenizer(batch_text, return_tensors='pt', padding=True, truncation=True)

        input_ids = inputs['input_ids']  # tensors of input ids

        ### GPT-2 tokenizer requires extra <eos> padding for batch training (no eos in the end)
        if ('gpt' in self.model_name and not self.tokenizer_len and input_ids.shape[1] < MAX_SEQ_LEN):
            pad_id = tokenizer.encode(tokenizer.pad_token)[0]
            pad_pt = torch.ones((input_ids.shape[0], 1), dtype=input_ids.dtype) * pad_id
            input_ids = torch.cat((input_ids, pad_pt), dim=1)

        input_ids = input_ids.to(device)
        labels = input_ids.clone()
        logits, past = model(input_ids=input_ids, return_dict=False)
        logits = logits[:, self.num_virtual_token-1:-1].contiguous()
        target = labels[:, :]

        target_mask = torch.ones_like(target).float()
        if self.dp:
            loss = criterion(logits, target, target_mask, label_smoothing=0.02, reduce=True)
        else:
            loss = criterion(logits, target, target_mask, label_smoothing=0.02, reduce="batch")
        return loss

    def save_checkpoints(self):
        dp_str = "dp" if self.dp else "non_dp"
        save_path = os.path.join(BASE_DIR,  'checkpoints', self.config_str, self.dataset_name + '_' + self.model_name + '_' + self.tuning_method + '_' + dp_str)
        self.model.save_pretrained(save_directory=save_path, selected_adapters=[self.adapter_name])

    def load_model(self, model_path):
        del self.model
        base_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model = PeftModel.from_pretrained(model=base_model, model_id=os.path.join(model_path, self.adapter_name), adapter_name=self.adapter_name)



if __name__ == "__main__":
    config = config.config
    config['model'] = 'openai-community/gpt2'
    config['dataset_name'] = 'sst2'
    config['dp'] = True
    config['lr'] = 1e-2

    tuning_config = {}
    tuning_config["num_virtual_token"] = 15
    tuning_config["adapter_name"] = "gpt2_casual_lm_prom"
    tuning_config["training_mode"] = "prompt-tuning"
    # tuning_config["training_mode"] = "prefix-tuning"
    tuning_config["task_type"] = "casual_lm"
    trainer = GPT2_Casual_LM_Prom_Trainer(config, **tuning_config)
    trainer.train_our_model()
    trainer.utility_evaluate()
