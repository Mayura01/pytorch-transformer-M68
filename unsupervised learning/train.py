from config import get_config
from tokenizer import build_or_get_tokenizer
from mlm_dataset import MLMDataset, causal_mask
from model import build_transformer
import json
import sys
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

def get_weights_file_path(config):
    model_file_path = config.get('model_file_path', '')
    return model_file_path if Path(model_file_path).exists() else None

def get_model(config, vocab_size):
    model = build_transformer(
        vocab_size, vocab_size, config["seq_len"], config["seq_len"], 
        d_model=config['d_model'], N=config['n_layers'], h=config['head'], 
        dropout=config['dropout'], d_ff=config['d_ff']
    )
    return model

def create_masks(tokenized_text, pad_token_id):
    # Create masks
    encoder_mask = (tokenized_text != pad_token_id).unsqueeze(1).unsqueeze(2)
    decoder_input = tokenized_text[:, :]
    decoder_mask = causal_mask(decoder_input.size(1)).to(tokenized_text.device)
    
    return encoder_mask, decoder_input, decoder_mask

def greedy_decode(model, source, source_mask, tokenizer, max_len, device):
    sos_idx = tokenizer.token_to_id('[SOS]')
    eos_idx = tokenizer.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def validation(model, tokenizer, max_len, device):
    model.eval()

    predicted = []
    input_text = "Astronomy is a natural science that studies"

    console_width = 80

    with torch.no_grad():
        source = tokenizer.encode(input_text).ids
        encoder_input = torch.cat([
            torch.tensor([tokenizer.token_to_id('[SOS]')], dtype=torch.int64),
            torch.tensor(source, dtype=torch.int64),
            torch.tensor([tokenizer.token_to_id('[EOS]')], dtype=torch.int64),
            torch.tensor([tokenizer.token_to_id('[PAD]')] * (max_len - len(source) - 2), dtype=torch.int64)
        ], dim=0).unsqueeze(0).to(device)
        encoder_mask = (encoder_input != tokenizer.token_to_id('[PAD]')).unsqueeze(1).unsqueeze(2).to(device)

        model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer, max_len, device)
        model_out_text = tokenizer.decode(model_out.detach().cpu().numpy())

        predicted.append(model_out_text)
            
        print('-'*console_width)
        print(f"{f'SOURCE: ':>12}{input_text}")
        print(f"{f'PREDICTED: ':>12}{model_out_text}")
        print('-'*console_width)
        

def train_model(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    if device == 'cuda':
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3} GB")
    device = torch.device(device)

    # Load raw dataset
    with open('dataset.json', 'r', encoding='utf-8') as f:
        raw_ds = json.load(f)
    
    # Build or get tokenizer (BPE)
    tokenizer = build_or_get_tokenizer(config, raw_ds)

    # Masked Language Model (MLM) Dataset
    train_ds = MLMDataset(config, raw_ds, tokenizer)

    # Create data loader
    data_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)

    # Get model
    model = get_model(config, tokenizer.get_vocab_size()).to(device)

    # Define loss function and optimizer
    pad_token_id = tokenizer.token_to_id("[PAD]")
    criterion = CrossEntropyLoss(ignore_index=pad_token_id)
    optimizer = Adam(model.parameters(), lr=config['lr'])
    
    # TensorBoard
    writer = SummaryWriter(log_dir=config['experiment_name'])

    initial_epoch = 0
    global_step = 0
    model_filename = get_weights_file_path(config)
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    # Training loop
    model.train()
    for epoch in range(initial_epoch, config['num_epochs']):
        total_loss = 0
        num_batches = len(data_loader)
        for batch in data_loader:
            tokenized_text = batch['tokenized_text'].to(device)
            original_tokens = batch['original_tokens'].to(device)

            # Create masks
            encoder_mask, decoder_input, decoder_mask = create_masks(tokenized_text, pad_token_id)

            # Forward pass
            optimizer.zero_grad()
            encoder_output = model.encode(tokenized_text, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)
            
            # sys.exit(0)
            # Compute loss
            loss = criterion(proj_output.view(-1, proj_output.size(-1)), original_tokens.view(-1))
            total_loss += loss.item()

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            global_step += 1

        # Log the loss
        avg_loss = total_loss / num_batches
        writer.add_scalar('train_loss', avg_loss, epoch)
        writer.flush()

        print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {avg_loss}")

        #validation
        validation(model, tokenizer, config['seq_len'], device)

        # Save the model state
        model_save_path = config['model_file_path']
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_save_path)

    writer.close()

if __name__ == '__main__':
    config = get_config()
    train_model(config)