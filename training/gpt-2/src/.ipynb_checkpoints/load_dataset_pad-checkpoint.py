import glob
import numpy as np
import os
import tensorflow as tf
import tqdm
import random

def load_dataset(enc, path, combine):
    paths = []
    if os.path.isfile(path):
        # Simple file
        paths.append(path)
    elif os.path.isdir(path):
        # Directory
        for (dirpath, _, fnames) in os.walk(path):
            for fname in fnames:
                paths.append(os.path.join(dirpath, fname))
    else:
        # Assume glob
        paths = glob.glob(path)

    token_chunks = []
    raw_text = ''
    for path in tqdm.tqdm(paths):
        if path.endswith('.npz'):
            # Pre-encoded
            with np.load(path) as npz:
                for item in npz.files:
                    token_chunks.append(npz[item])
        else:
            # Plain text
            with open(path, 'r') as fp:
                raw_text = fp.read()
                tokens = enc.encode(raw_text) #+ '<|endoftext|>') 
                token_chunks.append(tokens)
                
    # list of list
    return token_chunks 


class Sampler(object):
    """Fairly samples a slice from a set of variable sized chunks.

    'Fairly' means that the distribution is the same as sampling from one concatenated chunk,
    but without crossing chunk boundaries."""

    def __init__(self, chunks, seed=None):
        self.chunks = chunks
        self.n_documents = len(chunks)
        self.rs = np.random.RandomState(seed=seed)
        self.seed = seed
        
        # shuffle-related
        self.targets = {' <start-directions>': [1279, 9688, 12, 12942, 507, 29],
                        ' <start-ingredients>': [1279, 9688, 12, 278, 23320, 29],
                        ' <start-title>': [1279, 9688, 12, 7839, 29]}
        self.end_tag = [1279, 437, 12, 278, 23320, 29]
        
    def sample(self, length, shuffle_ingredients = True, shuffle = True):
        while True:
            index = self.rs.randint(0, self.n_documents)
            tokens = self.chunks[index]
            if shuffle:
                tokens = self.shuffle(tokens, shuffle_ingredients)
            # BPE encoding for '<|endoftext|>'
            tokens += [27, 91, 437, 1659, 5239, 91, 29]
            # import pdb; pdb.set_trace()
            # if not no constraints
            if not length == 0:
                diff = length - len(tokens)
                if diff > 0:
                    tokens += [16791] * diff # 16791 corresponds to <<
                elif diff < 0:
                    start = self.rs.randint(0, abs(diff))
                    tokens = tokens[start:start+length]
            
            return np.array(tokens)

    def shuffle_ingredients(self, encoded_file):
        random.seed(self.seed)
        start, end, output = len(self.targets[' <start-ingredients>'])+1, 0, []
        for idx, token in enumerate(encoded_file):
            if idx >= start and encoded_file[idx] in [3]:
                end = idx+1
                output.append(encoded_file[start:end])
                start = idx+1
        random.shuffle(output)
        return  self.targets[' <start-ingredients>'] +sum(output, []) + self.end_tag

    def shuffle(self, encoded_file, shuffle_ingredients = True):
        ''' main version
        Args: encoded_file: a list encodes e.g. ' <start-title>easy, crunchy hot dogs <end-title> <start-ingr...'
        '''
        random.seed(self.seed)
        idx_targets = {}
        # read list
        start, end, output = 0, 0, []
        for idx, token in enumerate(encoded_file):
            if encoded_file[idx: idx+2] ==[1279, 9688]:
                end = idx
                field = encoded_file[start:end]
                if start != 0 and shuffle_ingredients:
                    field = self.shuffle_ingredients(field)
                output.append(field)
                start = idx
        output.append(encoded_file[start:])
        random.shuffle(output)
        return sum(output, [])