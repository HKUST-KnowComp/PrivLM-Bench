from transformers import AutoTokenizer
from config import config
import random
import json
import pickle
from default_canary_config import default_canary

num_list = [' ' + str(i) for i in range(10)]
alphabet_list = [' ' + chr(i) for i in range(ord('a'), ord('a') + 26)]


def create_canary(canary_num_list, random_position_num_list, model_name_or_path='gpt2-large'):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # add phone number canary
    canary_format = "My phone number is*. You can call me by this number."
    fill_list = []
    canary_num = canary_num_list[0]
    random_position_num = random_position_num_list[0]
    for i in range(canary_num):
        fill_text = ''.join(random.sample(num_list, random_position_num))
        fill_list.append(fill_text)

    default_canary["phone_number"] = {}
    default_canary["phone_number"]["canary_format"] = canary_format
    default_canary["phone_number"]["fill_list"] = fill_list

    print(fill_list)

    # add alphabet canary
    canary_format = "The letters are*. It is my password."
    fill_list = []
    canary_num = canary_num_list[1]
    random_position_num = random_position_num_list[1]
    for i in range(canary_num):
        fill_text = ''.join(random.sample(alphabet_list, random_position_num))
        fill_list.append(fill_text)
    default_canary["letters"] = {}
    default_canary["letters"]["canary_format"] = canary_format
    default_canary["letters"]["fill_list"] = fill_list

    print(canary_format)
    print(fill_list)
    # print(fill_list[0])
    # example = tokenizer(fill_list)["input_ids"][0]
    # print(example)
    # print(tokenizer.convert_ids_to_tokens(example))


    # original paper setting 1

    known_word_ids = random.sample(range(len(tokenizer)), 4)
    fill_list = []
    for i in range(canary_num):
        fill_id = random.sample(range(len(tokenizer)), 1)
        if tokenizer.convert_tokens_to_ids(tokenizer.eos_token) not in fill_id:
            fill_list.append(' '.join(tokenizer.convert_ids_to_tokens(fill_id)))
    words = tokenizer.convert_ids_to_tokens(known_word_ids[:2]) + \
                    tokenizer.convert_ids_to_tokens(known_word_ids[2:])
    canary_format = ' * '.join([' '.join(words[:2]), ' '.join(words[2:])])

    default_canary["setting_1"] = {}
    default_canary["setting_1"]["canary_format"] = canary_format
    default_canary["setting_1"]["fill_list"] = fill_list

    print(canary_format)
    print(fill_list)

    # original paper setting 2
    known_word_ids = random.sample(range(len(tokenizer)), 4)
    fill_list = []
    for i in range(canary_num):
        fill_id = random.sample(range(len(tokenizer)), 3)
        if tokenizer.convert_tokens_to_ids(tokenizer.eos_token) not in fill_id:
            fill_list.append(' '.join(tokenizer.convert_ids_to_tokens(fill_id)))
    words = tokenizer.convert_ids_to_tokens(known_word_ids[:2]) + \
            tokenizer.convert_ids_to_tokens(known_word_ids[2:])
    canary_format = ' * '.join([' '.join(words[:2]), ' '.join(words[2:])])
    #
    default_canary["setting_2"] = {}
    default_canary["setting_2"]["canary_format"] = canary_format
    default_canary["setting_2"]["fill_list"] = fill_list

    print(canary_format)
    print(fill_list)

    with open("default_canary.json", "wb") as f:
        pickle.dump(default_canary, f)

    print("Default canary ready")


if __name__ == "__main__":
    canary_num_list = [100, 100, 100, 100]
    random_num_list = [5, 6]
    create_canary(canary_num_list=canary_num_list, random_position_num_list=random_num_list)
