## P-Bench

Code for ACL 2024 paper: P-Bench: A Multi-level Privacy Evaluation Benchmark for Language Models

**TODO**

We are working on code cleaning and our code for membership inference attacks, data extraction attacks and embedding inversion attacks has been uploaded.

1. Improve code readability (In Progress)
2. Add backdoor attacks on generative LLMs ❌




**Support Features** 

1. Parameter-Efficient Fine-Tuning (🤗PEFT) + PrivacyEngine (Opacus) ✅
2. Membership Inference Attacks ✅
3. Training Data Extraction Attacks ✅
4. Embedding Inversion Attacks ✅



### Preparation

1. numpy
2. torch
3. transformers
4. wandb
5. tqdm
6. typing
7. ml-swissknife
8. datasets


## Fine-tune LMs with/without Differential Privacy
Our base trainer is put in ```training_interface.py```, you may find all fine-tuning examples for different LMs and tuning methods under ```examples/```.

For instance, ```bert_cls_p-tuning.py``` under ```examples/``` includes our implementations for BertForSequenceClassification with PEFT methods (you may also try LoRA) with privacy engine.


## Run Attacks


### Membership Inference Attacks (MIAs)
For MIAs, we implement the Likelihood Ratio Attack (LiRA) by training multiple (50) shallow LMs. You may refer to ```run_mia.sh``` for more details. 

For evaluations on MIAs, you may use scripts under ```eval/MIA/``` to calculate AUC scores as well as likelihood.

### Training Data Extraction Attacks (DEAs)
To run DEAs， the first thing to do is to modify the canary_type_list, insert_proportion_list, and insert_time_base_list in the ```config.py``` file to control the type, proportion, and frequency of canary insertions. Then you can run ```eval/DEA/canary_experiments.py``` or run the examples under ```examples/``` for GPT or T5 training. Then run the canary_evaluate function provided by the trainer class to calculate exposures. We also provide some plotting functions for comparing different models in ```eval/DEA/canary_experiments.py```.

### Embedding Inversion Attacks (EIAs)
To run EIAs, you may first fine-tune LMs with/without DP under ```examples/```. Then you can use attacker code inside ```eval/EIA/``` to train the attacker and perform evaluation.



### Citation

Please kindly cite the following paper if you found our method and resources helpful!

```
@misc{li2024privlmbench,
      title={PrivLM-Bench: A Multi-level Privacy Evaluation Benchmark for Language Models}, 
      author={Haoran Li and Dadi Guo and Donghao Li and Wei Fan and Qi Hu and Xin Liu and Chunkit Chan and Duanyi Yao and Yuan Yao and Yangqiu Song},
      year={2024},
      eprint={2311.04044},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```



### Miscellaneous

Please send any questions about the code and/or the algorithm to hlibt@connect.ust.hk
