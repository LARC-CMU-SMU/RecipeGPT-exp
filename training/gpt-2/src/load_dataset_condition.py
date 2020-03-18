import glob
import numpy as np
import os
import tensorflow as tf
import tqdm
import random
from .save import load_pickle

def load_dataset(enc, path, combine):
    paths = []
    if os.path.isfile(path):
        # Simple file
        paths.append(path)
        if 'chunk' in path:
            return load_pickle(path)
        
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
    
    '''
    this version only shuffle the order of ingredients; the rest remains the same
    '''

    def __init__(self, 
                 chunks, 
                 mode = 'condition',
                 seed=None,
                 max_ingred = None, 
                 max_token = 512):
        assert mode in ['fkg',' condition']
        self.mode = mode
        self.chunks = chunks #[recipe for recipe in chunks if len(recipe)<= max_token]
        self.n_documents = len(self.chunks)
        self.rs = np.random.RandomState(seed=seed)
        self.seed = seed
        self.delimiter = {'ingredients':3, # the # token
                          'tag':2, # the $ token
                          }
        # shuffle-related
        self.targets = {' <start-ingredients>': [1279, 9688, 12, 278, 23320, 29],
                        ' <end-ingredients>': [1279,437,12,278,23320,29],
                        ' <start-tag>':[1279, 9688, 12, 12985, 29],
                        ' <end-tag>': [1279, 437, 12, 12985, 29]
                       }
        
    def sample(self, length):
        while True:
            index = self.rs.randint(0, self.n_documents)
            tokens = self.chunks[index]
            tokens = self.shuffle(tokens)
            
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

    def within(self, encoded_file, field):
        ''' shuffle withing field '''
        random.seed(self.seed)
        output, prev, idx, delimiter = [], 0, 0, self.delimiter[field]
        while idx < len(encoded_file)-1:
            idx +=1
            if encoded_file[idx] == delimiter:
                output.append(encoded_file[prev:idx]+[delimiter])
                prev = idx+1
        random.shuffle(output)
        return  self.targets[' <start-%s>'%(field)] +sum(output, []) + self.targets[' <end-%s>'%(field)]

    def shuffle_field(self, encoded_file, field):
        ''' main version
        Args: encoded_file: a list encodes e.g. ' <start-title>easy, crunchy hot dogs <end-title> <start-ingr...'
        field = 'ingredients'
        '''
        random.seed(self.seed)
        taglen = len(self.targets[' <start-%s>'%(field)])
        idx, start, end, output = 0, 0, 0, []
        while idx < len(encoded_file):
            idx +=1
            if encoded_file[idx: idx+taglen] == self.targets[' <start-%s>'%(field)]:
                start = idx
            if encoded_file[idx: idx+taglen] == self.targets[' <end-%s>'%(field)]: 
                end = idx
        if not start or not end:
            return encoded_file
        encoded_file = encoded_file[:start]+  self.within(encoded_file[start+taglen:end], field) + encoded_file[end+taglen:]
        return encoded_file
    
    def shuffle(self, encoded_file):
        if self.mode == 'condition':
            encoded_file = self.shuffle_field(encoded_file, 'ingredients')
            encoded_file = self.shuffle_field(encoded_file, 'tag')
        elif self.mode == 'fkg':
            encoded_file = self.shuffle_field(encoded_file, 'ingredients')
        return encoded_file