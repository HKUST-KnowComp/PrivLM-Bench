attacker_config = {
    'warmup_steps': 100,
    'epochs': 5,
    'batch_size': 8,
    'model_dir': "microsoft/DialoGPT-medium",
    'device': 'cuda:1',
    'tokenizer_len': 128 ,           ### tokenizer's max lenth with truncation, if None, perform 'longest' padding

    
}