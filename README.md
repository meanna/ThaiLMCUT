ThaiLMCut - Word Tokenizer for Thai Language based on Transfer Learning and bidirectional-LSTM
=====================================================

## About
- the tokenizer utilizes transfer learning from a character language model which is trained on a large Thai hotel review corpus and [InterBEST2009](https://www.nectec.or.th/corpus/index.php?league=pm).
- at the moment, the tokenizer supports only Thai texts.
Texts that includes English characters or special symbols will not be tokenized correctly, since the model was trained exclusively using Thai texts (also without any spaces, special symbols, and digits).
- we will soon release the model that supports those characters as well.
- [Try ThaiLMCut in Colab](https://colab.research.google.com/drive/1LpMsxP1xddodZTUPzOXdiiBR6IjMnk0E)
- [Paper: ThaiLMCut: Unsupervised Pretraining for Thai Word Segmentation](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.858.pdf)
- [bibtex](http://www.lrec-conf.org/proceedings/lrec2020/bib/2020.lrec-1.858.bib)

<p align="center"><img src="https://github.com/meanna/ThaiLMCUT/blob/master/graphic_lmcut/pic_lm.png?raw=true" width="368"><img src="https://github.com/meanna/ThaiLMCUT/blob/master/graphic_lmcut/pic_ws.png?raw=true" width="368"></p>


* an example input from a hotel review
<img src="https://github.com/meanna/ThaiLMCUT/blob/master/graphic_lmcut/example_final.jpg?raw=true">

## Requirements

- Python 3.5+
- PyTorch 1.0+
- numpy 

# Install LMCut as package

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

# How to use LMCut

### Tokenize a given Thai text

```
from lmcut import tokenize
text = "โรงแรมดี สวยงามน่าอยู่มากๆ"
result = tokenize(text)
print(result)
```

Result is a list of tokens:
```
['โรง', 'แรม', 'ดี', 'สวยงาม', 'น่า', 'อยู่', 'มาก', 'ๆ']
```


### Train a language model

#### Prepare dataset for training
- To train a language model, you should provide two .txt files of raw text.
One for training and one for development.
The model supports only Thai and English texts. Characters from other languages will be treated as unknown characters.
- See an example in `data/lm_data/`

* You can define your custom datasets in the function `get_path_data_LM` in `train/get_corpus.py`.
* The dataset name defined there will be use in the command line flag `--dataset`.
* note: if you use InterBEST2009, the boundary markers must be removed first.
To train a new language model, run
```
python train/LanguageModel.py \
--dataset [dataset name] \
--batchSize 60 \
--char_dropout_prob 0.01 \
--char_embedding_size 200 \
--hidden_dim 500 \
--layer_num 3 \
--learning_rate 0.0001 \
--sequence_length 100 \
--epoch 20 \
--len_lines_per_chunk 1000 \
--optim [adam or sgd] \
--lstm_num_direction [2 means bidirectional and 1 means uni directional] \
--add_note "..add some note.."
```
Example command.
```
python train/LanguageModel.py \
--dataset default \
--batchSize 32 \
--char_dropout_prob 0.01 \
--char_embedding_size 100 \
--hidden_dim 100 \
--layer_num 2 \
--learning_rate 0.0001 \
--sequence_length 100 \
--epoch 3 \
--len_lines_per_chunk 100 \
--optim adam \
--lstm_num_direction 2 \
--add_note "test the command"
```


To resume the training of a language model, run
```
python train/LanguageModel.py \
--load_from [model name to resume] \
--dataset [dataset name] \
--learning_rate 0.0001 \
--epoch 20 \
--optim [adam or sgd] "
```

- `[language model name]` must starts with `LM`, for example, `LM_2022-05-03_18.59.59`.


- Model artifacts are in `train/checkpoints_LM`
- After the training, 4 files will be generated.
```
LM_2022-05-03_18.28.05          (log file)
LM_2022-05-03_18.28.05.json     (data about model structure used when reloading the model)
LM_2022-05-03_18.28.05.pth.tar  (model weights)
LM_log.csv    (a log file in csv, for collecting experiment results)
```

### Train a new tokenizer
* The expected input is the InterBEST2009 dataset or any corpus with boundary marker `|`.
* You can split InterBEST2009 using `create_dataset.py`.
* See `data/toy` as an example.
* Define the train, dev, and test dataset paths in the function `get_path_data_tokenizer` in `train/get_corpus.py`

To train a new tokenizer, you could run:

```
python Tokenizer.py \
--dataset default \
--epoch 3 \
--lstm_num_direction 2 \
--batchSize 30 \
--sequence_length 80 \
--char_embedding_size 100 \
--hidden_dim 60 \
--layer_num 2 \
--optim adam \
--learning_rate 0.0001
```

To load a pre-trained language model(the embedding layer and recurrent layer) to the tokenizer and train, you could run

```
python Tokenizer.py \
--load_from [language model name] \
--epoch 5 \
--learning_rate 0.0001 \
```
- `[language model name]` must start with `LM`, for example, `LM_2022-05-03_18.59.59`.

To resume the training of a tokenizer, you could run
```
python Tokenizer.py \
--load_from [tokenizer name] \
--epoch 5 \
--learning_rate 0.0001 
```

- `[tokenizer name]` should begin with `Tokenizer`, for example, `Tokenizer_2022-05-03_20.46.35`.

- Use `--over_write 1` if you want to replace the loaded model with the trained model.
- With `--over_write 0` it will save the trained model as a separate model.

- For more detail about other arguments, see `train/Tokenizer.py` and `train/LanguageModel.py`

- `data/news_00001.txt` and `data/TEST_100K.txt` are from InterBEST2009 corpus which is publicly available at [NECTEC](https://www.nectec.or.th/corpus/index.php?league=pm)

# Credits
* Most of the code are from [Tabula nearly rasa: Probing the Linguistic Knowledge of Character-Level Neural Language Models Trained on Unsegmented Text](https://github.com/m-hahn/tabula-rasa-rnns)
* Some codes are from [DeepCut](https://github.com/rkcosmos/deepcut) and [Attacut](https://github.com/PyThaiNLP/attacut/)

# Acknowledgements
The project is funded by [TrustYou](https://www.trustyou.com/). The author would like to sincerely thank TrustYou and other contributors.

# Contributors
- Suteera Seeha
- Ivan Bilan
- Liliana Mamani Sanchez
- Johannes Huber
- Michael Matuschek

# License

All original code in this project is licensed under the MIT License. See the included LICENSE file.
