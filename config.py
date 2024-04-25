from pathlib import Path

def get_config():
    return {
        "batch_size": 4,
        "num_epochs": 100,
        "lr": 10**-4,
        "seq_len": 400,
        "d_model": 512,
        "model_file_path": "model_M68/M68.pt",
        "tokenizer_file": "dataset_and_tokenizer/tokenizer_{0}.json",
    }

def get_weights_file_path(config):
    model_file_path = config.get('model_file_path', '')
    if Path(model_file_path).exists():
        return str(model_file_path)
    else:
        return None
