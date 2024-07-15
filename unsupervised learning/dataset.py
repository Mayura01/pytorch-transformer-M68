import torch
from torch.utils.data import Dataset

class tokenize_raw_dataset(Dataset):
    def __init__(self, config, ds, tokenizer):
        super().__init__()
        self.seq_len = config['seq_len']
        self.ds = ds
        self.tokenizer = tokenizer
        self.sos_token = torch.tensor([tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = tokenizer.token_to_id("[PAD]")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_text = self.ds[idx]['text']
        input_tokens = self.tokenizer.encode(src_text).ids
        num_padding_tokens = self.seq_len - len(input_tokens) - 2

        if num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        tokenized_text = torch.cat([
            self.sos_token,
            torch.tensor(input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * num_padding_tokens, dtype=torch.int64),
        ], dim=0)

        assert tokenized_text.size(0) == self.seq_len

        # Create mask
        tokenized_text_mask = (tokenized_text != self.pad_token).unsqueeze(0).unsqueeze(0).int()

        return {
            "tokenized_text": tokenized_text,  # (seq_len)
            "tokenized_text_mask": tokenized_text_mask,  # (1, 1, seq_len)
            "src_text": src_text
        }

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
