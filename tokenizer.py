from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

def get_all_sentences(ds, field):
    for item in ds:
        yield item[field]

def get_or_build_tokenizer(config, ds, sequence):
    tokenizer_path = Path(config['tokenizer_file'].format(sequence))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=1)
        tokenizer.train_from_iterator(get_all_sentences(ds, sequence), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer
