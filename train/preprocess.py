import re
import string
from typing import Dict, List



# Code below is taken from PyThaiNLP

_thai_arabic = {
    "๐": "0",
    "๑": "1",
    "๒": "2",
    "๓": "3",
    "๔": "4",
    "๕": "5",
    "๖": "6",
    "๗": "7",
    "๘": "8",
    "๙": "9",
}


def thai_digit_to_arabic_digit(text: str) -> str:
    """
    This function convert Thai digits (i.e. ๑, ๓, ๑๐) to Arabic digits
    (i.e. 1, 3, 10).
    :param str text: Text with Thai digits such as '๑', '๒', '๓'
    :return: Text with Thai digits being converted to Arabic digits
             such as '1', '2', '3'
    :rtype: str
    :Example:
    >>> from pythainlp.util import thai_digit_to_arabic_digit
    >>>
    >>> text = 'เป็นจำนวน ๑๒๓,๔๐๐.๒๕ บาท'
    >>> thai_digit_to_arabic_digit(text)
    เป็นจำนวน 123,400.25 บาท
    """
    if not text or not isinstance(text, str):
        return ""

    newtext = []
    for ch in text:
        if ch in _thai_arabic:
            newtext.append(_thai_arabic[ch])
        else:
            newtext.append(ch)

    return "".join(newtext)

DEFAULT_PREPROCESSING_STEPS = [
    "remove_tags",
    "thai_digit_to_arabic_digit",
    "new_line_as_space",
    "remove_first_pipe",
    "remove_last_pipe"
]

ARABIC_RX = re.compile(r"[A-Za-z]+")
CAMEL_CASE_RX = re.compile(r"([a-z])([A-Z])([a-z])")
EMAIL_RX = re.compile(r"^\w+\@\w+\.\w+$")
NUMBER_RX = re.compile(r"[0-9,]+")
TRAILING_SPACE_RX = re.compile(r"\n$")
URL_RX = re.compile(r"(https?:\/\/)?(\w+\.)?\w+\.\w+")

def syllable2token(syllable: str) -> str:
    if ARABIC_RX.match(syllable):
        return "<ENGLISH>"
    elif NUMBER_RX.match(syllable):
        return "<NUMBER>"
    else:
        return syllable


def syllable2ix(sy2ix: Dict[str, int], syllable: str) -> int:
    token = syllable2token(syllable)

    return sy2ix.get(token, sy2ix.get("<UNK>"))


def character2ix(ch2ix: Dict[str, int], character: str) -> int:
    if character == "":
        return ch2ix.get("<PAD>")
    elif character in string.punctuation:
        return ch2ix.get("<PUNC>")

    return ch2ix.get(character, ch2ix.get("<UNK>"))

def step_remove_tags(txt: str) -> str:
    return re.sub(r"<\/?[A-Z]+>", "", txt)


def step_thai_digit_to_arabic_digit(txt: str) -> str:
    return thai_digit_to_arabic_digit(txt)


def step_number_tag(txt: str, tag: str = "ttNumber") -> str:
    return re.sub(r"[0-9,]+", tag, txt)


def step_english_tag(txt: str, tag: str = "ttEnglish") -> str:
    return re.sub(r"[A-Za-z]+", tag, txt)


def step_new_line_as_space(txt: str) -> str:
    return re.sub(r"\n", " ", txt)


def step_remove_first_pipe(txt: str) -> str:
    return re.sub(r"^\|", "", txt)


def step_remove_last_pipe(txt: str) -> str:
    return re.sub(r"\|$", "", txt)


def preprocess(txt: str, steps=DEFAULT_PREPROCESSING_STEPS) -> str:
    for s in steps:
        if isinstance(s, str):
            txt = globals()["step_%s" % s](txt)
        elif callable(s):
            txt = s(txt)
    return txt

""" 
path = "../data/news_00001.txt"
with open(path, "r") as inFile:
    for _ in range(10):
        for line in inFile:
            line = line.strip()
            line =preprocess(line)
            print(line)
            line = line.split("|")
            print(line)
"""

