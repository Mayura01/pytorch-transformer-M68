from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

def get_all_sentences(ds, field):
    for item in ds:
        yield item[field]

def build_or_get_tokenizer(config, ds):
    tokenizer_path = Path(config['tokenizer_file'])
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]", "[MASK]"], min_frequency=1)
        tokenizer.train_from_iterator(get_all_sentences(ds, "text"), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer
