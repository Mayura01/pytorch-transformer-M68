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
        "tokenizer_file": "dataset_and_tokenizers/tokenizer_{0}.json",
    }

def get_weights_file_path(config):
    model_folder = config.get('model_folder', '')
    model_filename = f"{config.get('model_basename', '')}.pt"
    
    model_path = Path(model_folder) / model_filename
    if model_path.exists():
        return str(model_path)
    else:
        return None

