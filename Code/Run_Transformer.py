from pathlib import Path
from config import get_config, latest_weights
from Transformer_from_scratch import build_transformer
from tokenizers import Tokenizer
from datasets import load_dataset
from dataset import MathWordQuestion
import torch
import sys

