import random
from itertools import chain
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import preprocess

def load_data_classifier(path, num_lines_per_chunk, doShuffling=False):
    chunks = []
    with open(path, "r") as inFile:
        for line in inFile:
            line = line.strip()
            line = preprocess.preprocess(line)
            words = line.split("|")
            #logging.info(words)
            chunks.append(words)
            if len(chunks) >= num_lines_per_chunk:
                if doShuffling:
                    random.shuffle(chunks)
                chunks = list(chain(*chunks))
                yield chunks
                chunks = []
    chunks = list(chain(*chunks))
    yield chunks