from pathlib import Path

def get_config():
    return {
        "batch_size": 4,
        "num_epochs": 100,
        "lr": 10**-4,
        "seq_len": 1024,
        "d_model": 512,
        "model_folder": "model_M68",
        "model_basename": "M68",
        "tokenizer_file": "tokenizer_{0}.json",
    }

def get_weights_file_path(config):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}.pt"
    return str(Path('.') / model_folder / model_filename)
