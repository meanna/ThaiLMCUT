import os

file_path = os.path.dirname(os.path.abspath(__file__))
print(file_path)
HOME = os.path.abspath(os.path.join(file_path, os.pardir))
print(HOME)
CHECKPOINTS_LM = os.path.join(file_path,"checkpoints_LM")
CHECKPOINTS_TOKENIZER = os.path.join(HOME, "checkpoints_tokenizer")