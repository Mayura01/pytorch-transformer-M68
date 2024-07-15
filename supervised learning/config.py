def get_config():
    return {
        "experiment_name": "Project_M68",
        "batch_size": 4,
        "num_epochs": 100,
        "lr": 10**-4,
        "seq_len": 400,
        "d_model": 512,
        "n_layers": 6,
        "head": 8,
        "d_ff": 2048,
        "dropout": 0.1,
        "model_file_path": "model_M68/M68.pt",
        "tokenizer_file": "dataset_and_tokenizer/tokenizer_{0}.json",
    }