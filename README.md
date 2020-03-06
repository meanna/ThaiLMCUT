ThaiLMCut - Word Tokenizer for Thai Language based on Transfer Learning and bidirectional-LSTM
=====================================================

## About
- the tokenizer utilizes transfer learning from a character language model which is trained on a large Thai hotel review corpus and InterBEST2009.
- at the moment, the tokenizer supports only Thai texts. Texts that includes English characters or special symbols will not be tokenized correctly, since the model was trained exclusively using Thai texts (also with out any spaces, special symbols, and digits).
- we will soon release the model that supports those characters as well.

<img src="https://github.com/meanna/upload/blob/master/pic_lm.png?raw=true" width="200">
<img src="https://github.com/meanna/upload/blob/master/pic_ws.png?raw=true" width="200">

<img src="https://github.com/meanna/upload/blob/master/example_final.jpg?raw=true" width="200">

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

* [Try ThaiLMCut in Colab](https://colab.research.google.com/drive/1LpMsxP1xddodZTUPzOXdiiBR6IjMnk0E)

### Train a language model

* Define the training and development dataset in `train/get_corpus_lm.py`
* Input data can be any text. Example of an input text can be found in `data/TEST_100K.txt`
* If you use InterBEST2009, the boundary markers must be removed (see `train/get_corpus.py`)
To train a new language model, you could run:
```
python train/LanguageModel.py --dataset [dataset name] --batchSize 60  --char_dropout_prob 0.01  --char_embedding_size 200   --hidden_dim 500  --layer_num 3  --learning_rate 0.0001 --sequence_length 100  --epoch 20 --len_lines_per_chunk 1000 --optim [adam or sgd] --lstm_num_direction [2 for bidirectional LSTM]  --add_note "..add some note.."
```

To resume the training of a language model, you could run
```
python train/LanguageModel.py   --load_from [model name]  --dataset [dataset name]  --learning_rate 0.0001 --epoch 20  --optim [adam or sgd] "
```

### Train a new tokenizer
* Expected input is InterBEST2009 or any corpus with boundary marker `|`
* Define the train, dev, test dataset in `train/get_corpus.py`
* Example of an input text can be found in `data/news_00001.txt`

* To train a new tokenizer, you could run:

```
python Tokenizer.py --epoch 5 --lstm_num_direction 2 --batchSize 30 --sequence_length 80 --char_embedding_size 100 --hidden_dim 60 --layer_num 2 [adam or sgd] --learning_rate 0.0001
```

* to transfer the embedding layer and recurrent layer of a pre-trained language model, you could run

```
python Tokenizer.py --load_from [language model name] --epoch 5  --learning_rate 0.0001
```
* to resume the training of a tokenizer, you could run
```
python Tokenizer.py --load_from [tokenizer name] --epoch 5  --learning_rate 0.0001 
```

* use `--over_write 1` if you want to over write the weights to the resumed model
* with `--over_write 0` it will save the trained model as a new model

* More detail about other arguments, see `train/Tokenizer.py` and `train/LanguageModel.py`

* `data/news_00001.txt` and `data/TEST_100K.txt` is from InterBEST2009 corpus which is publicly available at [NECTEC](https://www.nectec.or.th/corpus/index.php?league=pm)

# Credits
* Most of the code are borrowed from [Tabula nearly rasa: Probing the Linguistic Knowledge of Character-Level Neural Language Models Trained on Unsegmented Text](https://github.com/m-hahn/tabula-rasa-rnns)
* Some codes are borrowed from [DeepCut](https://github.com/rkcosmos/deepcut) and [Attacut](https://github.com/PyThaiNLP/attacut/) 
* We would like to thank all the contributors

# License

All original code in this project is licensed under the MIT License. See the included LICENSE file.
