import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer, trainers, pre_tokenizers
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import MathWordQuestion, causal_mask
from Transformer_from_scratch import build_transformer

from torch.utils.tensorboard import SummaryWriter
from config import get_config, get_weights_file_path
from tqdm import tqdm


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        prob = model.project(out[:,-1])

        _, next_word = torch.max(prob, dim =1)

        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_example=2):
    model.eval()
    count = 0



    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1

            model_output = greedy_decode(model, encoder_input,encoder_mask, tokenizer_src,tokenizer_tgt,max_len,device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]

            model_out_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())




            print_msg('-'*console_width)
            print_msg(f'Source:{source_text}')
            print_msg(f'Target:{target_text}')
            print_msg(f'Predicted:{model_out_text}')

            if count == num_example:
                break




def get_all_sentences(ds, lang):
    '''
    :param ds: this is just the data set
    :param lang: this is to choose to tokenize the question or the answer.
    :return:
    '''
    for item in ds:
        yield item[lang]


def get_or_build_tokenizer(config, ds, text_column):
    assert isinstance(text_column, str), f"Text column should be a string, but received {type(text_column)}"

    tokenizer_path = Path(config['tokenizer_file'].format(text_column))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)

        #  'question' or 'answer' column contains the text data
        sentences = ds[text_column]

        tokenizer.train_from_iterator(sentences, trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset(f"{config['datasource']}", "main", split='train')
    '''
    The lang_src, and lang_tgt might give some trouble later. will have to change it.
    '''
    print(ds_raw)

    tokenizer_src = get_or_build_tokenizer(config, ds_raw, 'question')

    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, 'answer')

    train_ds_size = int(0.9* len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = MathWordQuestion(train_ds_raw, tokenizer_src, tokenizer_tgt,  config['seq_len'])
    val_ds = MathWordQuestion(val_ds_raw, tokenizer_src, tokenizer_tgt, config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['question']).ids
        tgt_ids = tokenizer_tgt.encode(item['answer']).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt,len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle = True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle = True)

    return  train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], d_model=config['d_model'])
    return model




def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using device {device}')

    print(f"Device name: {torch.cuda.get_device_name(device.index)}")
    print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")



    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config,tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    writer = SummaryWriter(config['experiment_name'])

    optimizer  = torch.optim.Adam(model.parameters(), lr = config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing= 0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc = f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:
                encoder_input = batch['encoder_input'].to(device)
                decoder_input = batch['decoder_input'].to(device)
                encoder_mask = batch['encoder_mask'].to(device)
                decoder_mask = batch['decoder_mask'].to(device)

                encoder_output = model.encode(encoder_input, encoder_mask)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)

                proj_output = model.project(decoder_output)

                label = batch['label'].to(device)

                loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))

                batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

                # log the loss
                writer.add_scalar('train loss', loss.item(), global_step)
                writer.flush()

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                global_step += 1

        #run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device,
                      # lambda msg: batch_iterator.write(msg), global_step, writer)

        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global steps': global_step
        }, model_filename)

if __name__ == "__main__":
    config = get_config()
    train_model(config)







