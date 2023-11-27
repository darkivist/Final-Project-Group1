from pathlib import Path
from config import get_config, latest_weights
from Transformer_from_scratch import build_transformer
from tokenizers import Tokenizer
from datasets import load_dataset
from dataset import MathWordQuestion, causal_mask
import torch
import sys


def translate(sentence: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    config = get_config()
    tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
    tokenizer_tgt = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_tgt']))))
    model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), config["seq_len"],
                              config['seq_len'], d_model=config['d_model']).to(device)

    # Load the pretrained weights
    model_filename = latest_weights(config)
    print(model_filename)
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])


    seq_len = config['seq_len']

    # translate the sentence
    model.eval()
    with torch.no_grad():
        # Precompute the encoder output and reuse it for every generation step
        source = tokenizer_src.encode(sentence)
        source = torch.cat([
            torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64),
            torch.tensor(source.ids, dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[PAD]')] * (seq_len - len(source.ids) - 2), dtype=torch.int64)
        ], dim=0).to(device)
        source_mask = (source != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)
        encoder_output = model.encode(source, source_mask)

        # Initialize the decoder input with the sos token
        sos_token = tokenizer_tgt.encode('[SOS]').ids[0]
        decoder_input = torch.tensor([[sos_token]]).to(device)

        predicted_text = ""
        while decoder_input.size(1) < seq_len:
            # build mask for target and calculate output
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

            # project next token
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat(
                [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

            # concatenate the translated word to the predicted_text
            predicted_text += tokenizer_tgt.decode([next_word.item()])

            # break if we predict the end of sentence token
            if next_word.item() == tokenizer_tgt.token_to_id('[EOS]'):
                break

        # Print the source sentence and predicted output
        print(f"{f'SOURCE: ':>12}{sentence}")


        # convert ids to tokens
        return tokenizer_tgt.decode(decoder_input[0].tolist())