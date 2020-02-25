import os

file_path = os.path.dirname(os.path.abspath(__file__))

HOME = os.path.join("..")
CHECKPOINTS_LM = os.path.join(HOME, "checkpoints_LM")
CHECKPOINTS_TOKENIZER = os.path.join(HOME, "checkpoints_tokenizer")