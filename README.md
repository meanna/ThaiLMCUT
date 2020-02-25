ThaiLMCut - Deep Learning Tokenizer for the Thai language
=====================================================

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

```

```



### Train a new tokenizer


* news_00001.txt in the `data` folder is from InterBEST2009 corpus which is publicly available at [NECTEC](https://www.nectec.or.th/corpus/index.php?league=pm)


# License

All original code in this project is licensed under the MIT License. See the included LICENSE file.
