def get_config():
    return {
        "experiment_name": "model-M68",
        "batch_size": 2,
        "num_epochs": 50,
        "lr": 3e-5,
        "seq_len": 512,
        "d_model": 768,
        "n_layers": 12,
        "head": 12,
        "d_ff": 3072,
        "dropout": 0.1,
        "masking_prob": 0.15,
        "model_file_path": "M68.pt",
        "tokenizer_file": "tokenizer/tokenizer.json",
    }