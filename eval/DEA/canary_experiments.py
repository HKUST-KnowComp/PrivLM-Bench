"""
Experiments for different models / training method / dp or non_dp
"""
import os
import pickle
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import sys
current_dir = os.path.dirname(__file__)
parent_of_B_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(parent_of_B_dir)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(CURRENT_DIR)

## debug
with open(os.path.join("/home/data/hlibt/tosave/P-bench-0529", "dataset", "default_canary.json"), "rb") as f:
    default_canary = pickle.load(f)
def generate_evaluation_sequence_for_t5(canary_type="email", use_full_text=False, insert_proportion=0.4):
    canary_format = default_canary[canary_type]["canary_format"]
    fill_list = default_canary[canary_type]["fill_list"]
    length = round(len(fill_list) * insert_proportion)
    inserted = fill_list[:length]
    not_inserted = fill_list[length:]
    if use_full_text:
        inserted = [canary_format.replace('*', fill).split(" <SEP> ") for fill in inserted]
        not_inserted = [canary_format.replace('*', fill).split(" <SEP> ") for fill in not_inserted]
    return inserted, not_inserted


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration
from peft import PeftModel
from utils import calculate_perplexity_for_gpt, calculate_exposures, calculate_perplexity_for_t5
from dataset.canary_dataset import generate_evaluation_sequence_for_gpt
#from examples.t5_s2s_finetune import generate_evaluation_sequence_for_t5

from matplotlib import pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from training_interface import DP_trainer
import config

rcParams['font.size'] = 60

BASE_DIR = "privacy_benchmark-main"

checkpoints_path = os.path.join(BASE_DIR, "checkpoints_dp")

canary_type_list = ['name', 'city', 'email', 'phone_number', 'letters', 'setting_1', 'setting_2']
insert_proportion_list = [0.4] * 7
insert_time_base_list = [10] * 7


def mean(numbers):
    return sum(numbers) / len(numbers)


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def get_tokenizer(model):
    # return AutoTokenizer.from_pretrained(model)
    tokenizer = AutoTokenizer.from_pretrained(model)
    return tokenizer


def load_model_finetune(model_path, model_type='gpt'):
    if model_type == 'gpt':
        model = AutoModelForCausalLM.from_pretrained(model_path)
    else:
        model = T5ForConditionalGeneration.from_pretrained(model_path)
    return model


def load_model_peft(base_model_name, model_path, tuning_method, model_type='gpt'):
    if model_type == 'gpt':
        base_model = AutoModelForCausalLM.from_pretrained(os.path.join(BASE_DIR, 'hf_models', base_model_name))
        adapter_name = "gpt2_casual_lm_pref" if tuning_method == "prefix-tuning" else "gpt2_casual_lm_prom"
        model = PeftModel.from_pretrained(model=base_model, model_id=os.path.join(model_path, adapter_name),
                                          adapter_name=adapter_name)
    else:
        base_model = T5ForConditionalGeneration.from_pretrained(os.path.join(BASE_DIR, 'hf_models', base_model_name))
        model = PeftModel.from_pretrained(model=base_model, model_id=model_path)
    return model


def load_models(tuning_method, dataset_name, model_name, dp, model_type='gpt', epsilon=8):
    lr = 1e-4 if tuning_method == "finetune" else 1e-2
    dp_str = 'dp' if dp else 'non_dp'
    config_str = f"canary_inserted-tuning_method-{tuning_method}-lr-{lr}-batch_size-4-n_accumulation_steps-256-freeze_embedding-True-epochs-5-target_epsilon-{epsilon}"
    if tuning_method == "finetune":
        save_path = os.path.join(checkpoints_path, config_str, dataset_name + '_' + model_name + '_' + dp_str)
    else:
        save_path = os.path.join(checkpoints_path, config_str,
                                 dataset_name + '_' + model_name + '_' + tuning_method + '_' + dp_str)

    if tuning_method == "finetune":
        model = load_model_finetune(save_path, model_type=model_type)
    else:
        model = load_model_peft(base_model_name=model_name, model_path=save_path, tuning_method=tuning_method,
                                model_type=model_type)
    return model


def calculate_expose_rate(model_name, tuning_method, dataset_name, dp,
                    use_full_text=True, model_type='gpt', device=torch.device('cuda')):
    model = load_models(tuning_method=tuning_method, dataset_name=dataset_name, model_name=model_name, dp=dp, model_type=model_type)
    tokenizer = get_tokenizer(model_name)
    num_decode_virtual_tokens = 0
    if model_type == 'gpt' and tuning_method == 'prompt-tuning':
        num_decode_virtual_tokens = 15
    total_insert_num = 0
    expose_num = 0
    for canary_type, insert_proportion, insert_time_base in zip(canary_type_list,
                                                                insert_proportion_list,
                                                                insert_time_base_list
                                                                ):
        if model_type == 'gpt':
            insert, not_insert = generate_evaluation_sequence_for_gpt(canary_type=canary_type,
                                                                      use_full_text=use_full_text,
                                                                      insert_proportion=insert_proportion)
            total_insert_num = total_insert_num + len(insert)
            insert_seqs_perplexity = calculate_perplexity_for_gpt(insert, model, tokenizer,
                                                                  num_decode_virtual_tokens=num_decode_virtual_tokens,
                                                                  tuning_method=tuning_method,
                                                                  device=device)
            not_insert_seqs_perplexity = calculate_perplexity_for_gpt(not_insert, model, tokenizer,
                                                                      num_decode_virtual_tokens=num_decode_virtual_tokens,
                                                                      tuning_method=tuning_method,
                                                                      device=device)
        else:
            insert, not_insert = generate_evaluation_sequence_for_t5(canary_type=canary_type,
                                                                     use_full_text=use_full_text,
                                                                     insert_proportion=insert_proportion)
            total_insert_num = total_insert_num + len(insert)
            insert_seqs_perplexity = calculate_perplexity_for_t5(insert, model, tokenizer,
                                                                 tuning_method=tuning_method,
                                                                 device=device)
            not_insert_seqs_perplexity = calculate_perplexity_for_t5(not_insert, model, tokenizer,
                                                                     tuning_method=tuning_method,
                                                                     device=device)
        sorted_not_insert_seqs_perplexity = sorted(not_insert_seqs_perplexity)

        for insert_ppl in insert_seqs_perplexity:
            if insert_ppl < sorted_not_insert_seqs_perplexity[0]:
                expose_num = expose_num + 1
    expose_rate = expose_num/total_insert_num
    return expose_rate
    # dp_str = 'dp' if dp else 'non_dp'
    # print(f"{model_name} {tuning_method} {dataset_name} {dp_str} expose_rate: {expose_rate}")


def draw_figs_for_expose_rate():
    models = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
    models_ = ['small', 'medium', 'large', 'xl']
    model_type = 'gpt'

    tuning_methods = ['finetune', 'prefix-tuning', 'prompt-tuning']
    plt.figure(figsize=(35, 30), dpi=80)
    plt.xlabel('GPT-2 Models')
    plt.ylabel('Exposure Rate')
    for tuning_method in tuning_methods:
        for dp in [False, True]:
            expose_rate_list = []

            for model in models:
                expose_rate = calculate_expose_rate(model_name=model, tuning_method=tuning_method, dataset_name='mnli',
                                      dp=dp, model_type=model_type
                                      )
                expose_rate_list.append(expose_rate)
            # 线形
            if dp:
                line_style = '--'
            else:
                line_style = '-'
            # 颜色
            if tuning_method == 'finetune':
                color = 'red'
            elif tuning_method == 'prefix-tuning':
                color = 'green'
            else:
                color = 'blue'
            # 点型
            if tuning_method == 'finetune':
                marker = 'p'
            elif tuning_method == 'prefix-tuning':
                marker = 'x'
            else:
                marker = '*'
            dp_str = "dp" if dp else "non_dp"
            plt.plot(models_, expose_rate_list, color=color,
                     linestyle=line_style, linewidth=5, marker=marker, markersize=20,
                     label=dp_str + ' ' + tuning_method)
            plt.legend(loc=2)
    plt.savefig(f"exposure-rate-{model_type}-mnli.pdf", bbox_inches='tight')


def draw_figs_for_different_dp():
    epsilons = [4, 8, 20, 100]
    for model_name in ['gpt2-medium', 't5-base']:
        plt.figure(figsize=(35, 30), dpi=80)
        plt.xlabel('Number of insertions')
        plt.ylabel('Mean Exposure')
        for epsilon, color in zip(epsilons, ['r', 'g', 'b', 'y']):
            if model_name == 'gpt2-medium':
                model_type = 'gpt'
            else:
                model_type = 't5'
            model = load_models(tuning_method='finetune', dataset_name='mnli', model_name=model_name, dp=True,
                                model_type=model_type, epsilon=epsilon)
            tokenizer = get_tokenizer(model_name)
            return_dict = canary_evaluate(model=model, tokenizer=tokenizer,
                                          num_decode_virtual_tokens=0,
                                          tuning_method='finetune',
                                          device=torch.device('cuda'),
                                          model_type=model_type
                                          )
            insert_time_list = []
            exposures_lists = []
            for canary_type in canary_type_list:
                if len(return_dict[canary_type]['insert_time_list']) > len(insert_time_list):
                    insert_time_list = return_dict[canary_type]['insert_time_list']
                exposures_lists.append(return_dict[canary_type]['exposures'])
            mean_exposure_list = []
            for i in range(len(insert_time_list)):
                temp = []
                for exposures_list in exposures_lists:
                    if i < len(exposures_list):
                        temp.append(exposures_list[i])
                mean_exposure_list.append(mean(temp))
            plt.scatter(insert_time_list, mean_exposure_list, s=300, c=color,
                        label=r'$\epsilon$' + ' = ' + str(epsilon))
            sns.regplot(x=insert_time_list, y=mean_exposure_list, ci=95, color=color, line_kws={'alpha': 0.1}, order=2)
            # plt.fill_between(insert_time_list, [i-0.5 for i in mean_exposure_list], [i-0.5 for i in mean_exposure_list], facecolor=color, alpha=0.3)
            plt.legend(loc=2)
        plt.savefig(f"different_epsilon_{model_name}.pdf")


def canary_evaluate(model, tokenizer, num_decode_virtual_tokens, tuning_method, device,
                    use_full_text=True, model_type='gpt'):
    epochs = 5
    return_dict = {}
    for canary_type, insert_proportion, insert_time_base in zip(canary_type_list,
                                                                insert_proportion_list,
                                                                insert_time_base_list
                                                                ):
        if model_type == 'gpt':
            insert, not_insert = generate_evaluation_sequence_for_gpt(canary_type=canary_type,
                                                                      use_full_text=use_full_text,
                                                                      insert_proportion=insert_proportion)
            insert_seqs_perplexity = calculate_perplexity_for_gpt(insert, model, tokenizer,
                                                                  num_decode_virtual_tokens=num_decode_virtual_tokens,
                                                                  tuning_method=tuning_method,
                                                                  device=device)
            not_insert_seqs_perplexity = calculate_perplexity_for_gpt(not_insert, model, tokenizer,
                                                                      num_decode_virtual_tokens=num_decode_virtual_tokens,
                                                                      tuning_method=tuning_method,
                                                                      device=device)
        else:
            insert, not_insert = generate_evaluation_sequence_for_t5(canary_type=canary_type,
                                                                     use_full_text=use_full_text,
                                                                     insert_proportion=insert_proportion)
            insert_seqs_perplexity = calculate_perplexity_for_t5(insert, model, tokenizer,
                                                                 tuning_method=tuning_method,
                                                                 device=device)
            not_insert_seqs_perplexity = calculate_perplexity_for_t5(not_insert, model, tokenizer,
                                                                     tuning_method=tuning_method,
                                                                     device=device)
        exposures = calculate_exposures(insert_seqs_perplexity, not_insert_seqs_perplexity)
        insert_time_list = [i * insert_time_base * epochs for i in range(1, len(insert) + 1)]
        exposures = exposures.tolist()
        return_dict[canary_type] = {}
        return_dict[canary_type]["insert_time_list"] = insert_time_list
        return_dict[canary_type]["exposures"] = exposures
    return return_dict


def compare_of_different_dp(dataset_name='qnli', model_type='gpt'):
    """
    compare the canary exposure of dp and nondp
    """
    model_name_list = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
    # model_name_list = ['t5-small']
    for model_name in model_name_list:
        plt.figure(figsize=(25, 20), dpi=80)
        plt.xlabel('Number of insertions')
        plt.ylabel('Mean Exposure')
        # plt.title(f"Exposures-{model_name}")

        for tuning_method in ['finetune', 'prefix-tuning', 'prompt-tuning']:
            tokenizer = get_tokenizer(model_name)
            for dp in [False, True]:
                model = load_models(tuning_method=tuning_method, dataset_name=dataset_name, model_name=model_name, dp=dp,
                                    model_type=model_type)
                if model_type == 'gpt':
                    num_virtual_tokens = 15 if tuning_method == "prompt-tuning" else 0
                else:
                    num_virtual_tokens = 0
                return_dict = canary_evaluate(model=model, tokenizer=tokenizer,
                                              num_decode_virtual_tokens=num_virtual_tokens,
                                              tuning_method=tuning_method,
                                              device=torch.device('cuda'),
                                              model_type=model_type
                                              )
                insert_time_list = []
                exposures_lists = []
                for canary_type in canary_type_list:
                    if len(return_dict[canary_type]['insert_time_list']) > len(insert_time_list):
                        insert_time_list = return_dict[canary_type]['insert_time_list']
                    exposures_lists.append(return_dict[canary_type]['exposures'])
                mean_exposure_list = []
                for i in range(len(insert_time_list)):
                    temp = []
                    for exposures_list in exposures_lists:
                        if i < len(exposures_list):
                            temp.append(exposures_list[i])
                    mean_exposure_list.append(mean(temp))
                # 线形
                if dp:
                    line_style = '--'
                else:
                    line_style = '-'
                # 颜色
                if tuning_method == 'finetune':
                    if dp:
                        color = 'darkred'
                    else:
                        color = 'lightcoral'

                elif tuning_method == 'prefix-tuning':
                    if dp:
                        color = 'darkgreen'
                    else:
                        color = 'lightgreen'
                else:
                    if dp:
                        color = "midnightblue"
                    else:
                        color = 'cyan'
                # 点型
                if tuning_method == 'finetune':
                    marker = 'p'
                elif tuning_method == 'prefix-tuning':
                    marker = 'x'
                else:
                    marker = '*'
                dp_str = "dp" if dp else "non_dp"
                # plt.plot(insert_time_list, mean_exposure_list, color=color,
                #          linestyle=line_style, linewidth=5, marker=marker, markersize=20,
                #          label=dp_str + ' ' + tuning_method)
                if dp:
                    linewidths=5
                else:
                    linewidths=1
                plt.scatter(insert_time_list, mean_exposure_list, s=300, c=color,
                            marker=marker, linewidths=linewidths,
                            label=dp_str + ' ' + tuning_method)
                sns.regplot(x=insert_time_list, y=mean_exposure_list, ci=95, color=color, line_kws={'alpha':0.1}, order=2)
                # plt.fill_between(insert_time_list, [i-0.5 for i in mean_exposure_list], [i-0.5 for i in mean_exposure_list], facecolor=color, alpha=0.3)
                plt.legend()
        # plt.show()
        plt.savefig(f"exposure-{model_name}-{dataset_name}.pdf", bbox_inches='tight')



def compare_of_different_tuning_method():
    """
    compare the canary exposure of gpt2, gpt2-medium, gpt2-large
    """
    for dataset_name in ['mnli', 'qnli', 'sst2']:
        for dp in [False, True]:
            for model_name in ['gpt2', 'gpt2-medium', 'gpt2-large']:
                tokenizer = get_tokenizer(model=model_name)
                tuning_method_dict = {}
                for tuning_method in ['finetune', 'prefix-tuning', 'prompt-tuning']:
                    model = load_models(tuning_method=tuning_method, dataset_name=dataset_name, model_name=model_name,
                                        dp=dp)
                    num_virtual_tokens = 15 if tuning_method == "prompt-tuning" else 0
                    return_dict = canary_evaluate(model=model, tokenizer=tokenizer,
                                                  num_decode_virtual_tokens=num_virtual_tokens,
                                                  tuning_method=tuning_method, device='cuda')
                    tuning_method_dict[tuning_method] = return_dict

                for canary_type in canary_type_list:
                    insert_time_list = tuning_method_dict['finetune'][canary_type]['insert_time_list']
                    plt.figure(figsize=(15, 10), dpi=80)
                    plt.xlabel('Number of insertions')
                    plt.ylabel('Exposure')
                    dp_str = "dp" if dp else "non_dp"
                    plt.title(
                        f"Exposures of canary type {canary_type} by different tuning method trained on {dataset_name} of {model_name}, {dp_str}")
                    for tuning_method in ['finetune', 'prefix-tuning', 'prompt-tuning']:
                        exposures = tuning_method_dict[tuning_method][canary_type]['exposures']
                        plt.plot(insert_time_list, exposures, label=tuning_method)
                    save_path = os.path.join(BASE_DIR, "canary_exp", "different_tuning",
                                             f"{canary_type}_{model_name}_{dataset_name}_{dp_str}")
                    mkdir(save_path)
                    plt.savefig(save_path)


def compare_of_different_models():
    """
    compare the canary exposure of gpt2, gpt2-medium, gpt2-large
    """
    for tuning_method in ['prompt-tuning']:
        for dataset_name in ['mnli', 'qnli', 'sst2']:
            for dp in [False, True]:
                return_dict_list = []
                model_name_list = ['gpt2', 'gpt2-medium', 'gpt2-large']
                for model_name in model_name_list:
                    tokenizer = get_tokenizer(model=model_name)
                    model = load_models(tuning_method=tuning_method, dataset_name=dataset_name, model_name=model_name,
                                        dp=dp)
                    num_virtual_tokens = 15 if tuning_method == "prompt-tuning" else 0
                    return_dict = canary_evaluate(model=model, tokenizer=tokenizer,
                                                  num_decode_virtual_tokens=num_virtual_tokens,
                                                  tuning_method=tuning_method, device='cuda')
                    return_dict_list.append(return_dict)

                for canary_type in canary_type_list:
                    insert_time_list = return_dict_list[0][canary_type]['insert_time_list']
                    exposures_list = [return_dict[canary_type]['exposures'] for return_dict in return_dict_list]

                    plt.figure(figsize=(15, 10), dpi=80)
                    plt.xlabel('Number of insertions')
                    plt.ylabel('Exposure')
                    dp_str = "dp" if dp else "non_dp"
                    plt.title(
                        f"Exposures of canary type {canary_type} on different models trained on {dataset_name} by {tuning_method}, {dp_str}")

                    for exposures, model_name in zip(exposures_list, model_name_list):
                        plt.plot(insert_time_list, exposures, label=model_name)
                    save_path = os.path.join(BASE_DIR, "canary_exp", "different_model",
                                             f"{canary_type}_{dataset_name}_{tuning_method}_{dp_str}")
                    # mkdir(save_path)
                    plt.savefig(save_path)


if __name__ == "__main__":
    config = config.config
    config['canary_type_list'] = ['name', 'city', 'email', 'phone_number', 'letters', 'setting_1', 'setting_2']
    config["insert_proportion_list"] = [0.4] * 7
    config["insert_time_base_list"] = [10] * 7

    trainer = DP_trainer(**config)
    trainer.train_our_model()
    trainer.canary_evaluate()
