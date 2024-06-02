import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"


### Global vars
BASE_DIR = ''
### cnonfig args
config = {}
#gpt2 (124M), gpt2-medium (355M), gpt2-large (774M), gpt2-xl (1558M~1.5B)
#bert-base-uncased,roberta-large
config['model'] = "openai-community/gpt2"
config['optimizer'] = "adamw"
config['lr'] = 1e-4
config['batch_size'] = 4
config['virtual_batch_size'] = 1024
config['n_accumulation_steps'] = int(config['virtual_batch_size']/config['batch_size'])
config['epochs'] = 5
#config['dp'] = True
config['dp'] = False
config['seed'] = 42
### freeze embedding layers
config['freeze_embedding'] = True
#### qnli,sst2,mnli
config['dataset_name'] = "mnli"
config['insert_proportion'] = 0.6
#### canary config
config['canary_type_list'] = ['name', 'city', 'email', 'phone_number', 'letters', 'setting_1', 'setting_2']
config["insert_proportion_list"] = [0.4] * 7
config["insert_time_base_list"] = [10] * 7
# config['canary_type_list'] = []
# config["insert_proportion_list"] = []
# config["insert_time_base_list"] = []
### hr appended config args
config['tokenizer_len'] = 256           ### tokenizer's max lenth with truncation, if None, perform 'longest' padding
config['data_type'] = "train"
### privacy engine args
config['target_delta'] = 1e-5
config['target_epsilon'] = 8
config['max_grad_norm'] = 0.1


#utility evaluation config
config['eva_method'] = 'logits' # logits or counting
config['num_decode_virtual_tokens'] = 0
config['num_beams'] = 30
config['generation_max_length'] = 20
config['num_return_sequence'] = 25
