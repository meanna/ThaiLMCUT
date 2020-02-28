ThaiLMCut - Deep Learning Tokenizer for Thai language based on Transfer Learning and bidirectional-LSTM
=====================================================

## About
- the tokenizer utilizes transfer learning from a character language model which is trained on a large Thai hotel review corpus and InterBEST2009.
- at the moment, the tokenizer supports only Thai text only. Texts that includes English characters or special symbols will not be tokenized correctly, since the model was trained only on Thai characters.
- we will soon release the model that supports English characters as well.


## Requirements

- Python 3.5+
- PyTorch 1.0+
- numpy 

# How to setup

### Download the weight file from:
```
https://drive.google.com/file/d/1e39tNMfUFzYQ4MDHTMyNWfNUxu9RoaTA/view?usp=sharing
```

### Move the weight file to this path:
```
lmcut/weight/.
```

### Create a package wheel using:
```python setup.py bdist_wheel```

### Install the package using:
```
pip install dist/lmcut*
```

# How to use

### Tokenize a given Thai text

After importing the package, you can tokenize any Thai text by using:
```
from lmcut import tokenize
text = "โรงแรมดี สวยงามน่าอยู่มากๆ"
result = tokenize(text)
print(result)
```

Result will be a list of tokens:
```
['โรง', 'แรม', 'ดี', 'สวยงาม', 'น่า', 'อยู่', 'มาก', 'ๆ']
```


### Train a language model

Define the training and development dataset in `train/get_corpus_lm.py`
See expected input in `data/news_00001.txt`

To train a new language model, you could run:
```
python train/LanguageModel.py --dataset [dataset name] --batchSize 60  --char_dropout_prob 0.01  --char_embedding_size 200   --hidden_dim 500  --layer_num 3  --learning_rate 0.0001 --sequence_length 100  --epoch 20 --len_lines_per_chunk 1000 --optim [adam or sgd] --lstm_num_direction [2 for bidirectional LSTM]  --add_note "..add some note.."
```

To resume the training of a language model, you could run
```
python train/LanguageModel.py   `--load_from [model name]`  --dataset [dataset name]  --learning_rate 0.0001 --epoch 20  --optim [adam or sgd] --add_note "..add some note.."
```

### Train a new tokenizer

( this section is not completed yet)

* news_00001.txt in the `data` folder is from InterBEST2009 corpus which is publicly available at [NECTEC](https://www.nectec.or.th/corpus/index.php?league=pm)


# License

All original code in this project is licensed under the MIT License. See the included LICENSE file.
