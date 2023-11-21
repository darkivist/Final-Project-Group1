import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import MathWordQuestion, causal_mask
from Transformer_from_scratch import build_transformer

def get_all_sentences(ds, lang):
    '''
    :param ds: this is just the data set
    :param lang: this is to choose to tokenize the question or the answer.
    :return:
    '''
    for item in ds:
        yield item[lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        '''
        Change tokenizer for your data, special characters are important.
        '''
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency = 2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    '''
    The lang_src, and lang_tgt might give some trouble later. will have to change it.
    '''

    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    train_ds_size = int(0.9* len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = MathWordQuestion(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = MathWordQuestion(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item[config['lang_src']]).ids
        tgt_ids = tokenizer_src.encode(item[config['lang_tgt']]).ids
        max_len_src = max(max_len_src, src_ids)
        max_len_tgt = max(max_len_tgt,tgt_ids)

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle = True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle = True)

    return  train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    



