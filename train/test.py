import random
from itertools import chain
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

path =  "../data/news_00001.txt"
chunks = load_data_classifier(path,1)
for i in range(10):
    print(next(chunks))