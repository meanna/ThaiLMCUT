import os

file_path = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.abspath(os.path.join(file_path, os.pardir))
CHECKPOINTS_LM = os.path.join(file_path,"checkpoints_LM")
CHECKPOINTS_TOKENIZER = os.path.join(file_path, "checkpoints_tokenizer")