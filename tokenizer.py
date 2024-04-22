from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

def get_all_sentences(ds, contexts):
    for context in contexts:
        for item in ds:
            yield item[context]

def get_or_build_tokenizer(config, ds, contexts):
    tokenizer_path = Path(config['tokenizer'])
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=1)
        tokenizer.train_from_iterator(get_all_sentences(ds, contexts), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer
