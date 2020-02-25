ThaiLMCut - Deep Learning Tokenizer for the Thai language
=====================================================

## Requirements

- Python 3.5+
- PyTorch 1.0+
- numpy 

# How to setup

## Create a package wheel using:
```python setup.py bdist_wheel```

## Install it using:
```
pip install dist/lmcut*
```

# How to use

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

# License

All original code in this project is licensed under the MIT License. See the included LICENSE file.
