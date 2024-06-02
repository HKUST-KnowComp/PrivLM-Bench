"""
This code is written to prove the zero shot capability of llama-2-7b and llama-2-13b
"""
import os
import sys

import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import random
from dataset.data_process import get_sent_list
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import random

random.seed(42)
MAX_SEQ_LEN = 1024 * 2


def get_tokenizer(model_name_or_path):
    # tokenizer = AutoTokenizer.from_pretrained(os.path.join('..', 'tokenizers', model_name_or_path))
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return tokenizer


def get_model(model_name_or_path):
    # hf_model = AutoModelForCausalLM.from_pretrained(model)
    # hf_model.save_pretrained(os.path.join('hf_models', model))
    # model = AutoModelForCausalLM.from_pretrained(os.path.join('..', 'hf_models', model_name_or_path))
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    return model


def prepare_training_dataset(dataset_name):
    data_list = get_sent_list(dataset_name=dataset_name, data_type="train", is_aux=False, return_all=True)
    if dataset_name == 'mnli':
        entailment_data_list = []
        neutral_data_list = []
        contra_data_list = []
        for data in data_list:
            if data[-1] == '0':
                entailment_data_list.append(data)
            elif data[-1] == '1':
                neutral_data_list.append(data)
            else:
                contra_data_list.append(data)
        del data_list
        label_split_training_data = [
            entailment_data_list,
            neutral_data_list,
            contra_data_list
        ]
        return label_split_training_data
    else:
        raise NotImplementedError


def sample_from_training_set(num_shots, dataset_name, split_dataset):
    if num_shots == 0:
        return []
    # samples = random.sample(data_list, num_shots)
    if dataset_name == "mnli":
        samples = []
        samples.extend(random.sample(split_dataset[0], int(num_shots/3)))
        samples.extend(random.sample(split_dataset[1], int(num_shots/3)))
        samples.extend(random.sample(split_dataset[2], int(num_shots/3)))
        if num_shots - len(samples) > 0:
            for _ in range(num_shots - len(samples)):
                random_index = random.randint(0, 2)
                chosen_list = split_dataset[random_index]
                samples.append(random.sample(chosen_list, 1)[0])
        # begin_sentence = f"Here are {num_shots} pairs of sentence and their relationship.\n"
        return_samples = []
        for sentence in samples:
            if sentence[-1] == '0':
                return_samples.append(sentence[:-1] + "entailment.")
            elif sentence[-1] == '1':
                return_samples.append(sentence[:-1] + "neutral.")
            else:
                return_samples.append(sentence[:-1] + "contradiction.")

    # elif dataset_name == "qnli":
    #     samples = random.sample(data_list, num_shots)
    #     begin_sentence = f"Here are {num_shots} pairs of sentences and their relationships.\n"
    #     for sentence in samples:
    #         if sentence[-1] == '0':
    #             begin_sentence = begin_sentence + sentence[:-1] + "not entailment." + '\n'
    #         else:
    #             begin_sentence = begin_sentence + sentence[:-1] + "entailment." + '\n'
    # elif dataset_name == "sst2":
    #     samples = random.sample(data_list, num_shots)
    #     begin_sentence = f"Here are {num_shots} sentences and their sentiments.\n"
    #     for sentence in samples:
    #         if sentence[-1] == '0':
    #             begin_sentence = begin_sentence + sentence[:-1] + "negative" + '\n'
    #         else:
    #             begin_sentence = begin_sentence + sentence[:-1] + "positive" + '\n'
    else:
        raise NotImplementedError
    return return_samples


def get_test_data(dataset_name):
    test_data_list = get_sent_list(dataset_name=dataset_name, data_type='dev', return_all=True)
    sentence_list = []
    label_list = []
    for test_data in test_data_list:
        label_list.append(test_data[-1])
        sentences = test_data.split(" <SEP> ")[:-1]
        sentence_list.append(" <SEP> ".join(sentences))
    return sentence_list, label_list


def evaluate_single_mnli_data(generated_ids, label):
    generated_ids = generated_ids[-10:]
    if label == '0':  # entailment
        if 875 in generated_ids and 737 in generated_ids:
            return 1
        else:
            return 0
    if label == '1': # neutral
        if 21104 in generated_ids:
            return 1
        else:
            return 0
    if label == '2': # contradiction
        if 23949 in generated_ids:
            return 1
        else:
            return 0



def tokenizer_test(tokenizer):
    input_ids = tokenizer(" The relation is contradiction.")['input_ids']
    print(input_ids)
    print(tokenizer.convert_ids_to_tokens(input_ids))


def few_shot_evaluation(model, tokenizer, num_shots, dataset_name, device):
    # if dataset_name == "mnli":
    #     prompt_sentence = prompt_sentence + "Question: Classify whether the relationship of two context sentences \n"
    #     options = "Options: (A) entailment (B) neutral (C) contradiction \n"
    # elif dataset_name == "qnli":
    #     prompt_sentence = prompt_sentence + "Question: Classify whether the relationship of following two sentences is " \
    #                                         "entailment or not entailment:\n "
    #     options = ""
    # else:
    #     prompt_sentence = prompt_sentence + "Question: Classify whether the sentiment of the following sentence is positive or " \
    #                                         "negative:\n "
    #     options = ""
    model.to(device)
    model.eval()
    label_split_training_data = prepare_training_dataset(dataset_name=dataset_name)
    test_sentence_list, label_list = get_test_data(dataset_name)
    right_count = 0
    for test_sentence, label in tqdm(zip(test_sentence_list, label_list)):
        if dataset_name == 'mnli':
            if num_shots == 0:
                options = "Options: (A) entailment (B) neutral (C) contradiction \n"
                full_prompt = "Context: " + test_sentence + '\n' + options + "\n" + "The answer is"
            else:
                training_examples = sample_from_training_set(num_shots=num_shots,
                                                             dataset_name=dataset_name,
                                                             split_dataset=label_split_training_data)
                options = "Please find the relation from the Options: \n (A) entailment (B) neutral (C) contradiction \n"
                examples_text = options
                for training_example in training_examples:
                    examples_text = examples_text + '<example>' + training_example + '\n'
                full_prompt = examples_text + '<example>' + test_sentence + "The relation is"
                # full_prompt = examples_text + test_sentence + "The relation is"
        else:
            raise NotImplementedError
        encoding = tokenizer(full_prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            generated_ids = model.generate(
                **encoding,
                max_new_tokens=10,
                # do_sample=False,
                num_beams=3,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_ids = generated_ids[0].tolist()
        right_count = right_count + evaluate_single_mnli_data(generated_ids, label)
    acc = right_count / len(test_sentence_list)
    print(acc)


if __name__ == "__main__":
    # model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
    # model_name_or_path = "meta-llama/Llama-2-7b-hf"
    for num_shots in [0, 5, 10]:
        # for model_name_or_path in ["meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-7b-chat-hf"]:
        for model_name_or_path in ["meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-13b-chat-hf"]:
            print(f"num_shots: {num_shots}")
            print(model_name_or_path)
            model = get_model(model_name_or_path)
            tokenizer = get_tokenizer(model_name_or_path)
            # tokenizer_test(tokenizer)
            label_split_dataset = prepare_training_dataset(dataset_name='mnli')
            few_shot_evaluation(model=model, tokenizer=tokenizer, num_shots=num_shots, dataset_name="mnli", device='cuda')



