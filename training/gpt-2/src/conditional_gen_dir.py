# new version on interact
#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf
import tqdm
import re

from .import model, sample, encoder
from .path import path_to_model
from .load_dataset import load_dataset
from .save import save, make_dir, datetime

def interact_model(
    model_name='117M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    top_p=0.0,
    filename='',
    overwrite=False,
    tag = '',
    max_document = None
):
    """
    Interactively run the model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    :top_p=0.0 : Float value controlling diversity. Implements nucleus sampling,
     overriding top_k if set to a value > 0. A good setting is 0.9.
    :overwrite=False : whether to overwrite the y_pred
    :filename: A directory that contains many .txt files, each of them is contains a input we want to use
    :tag: The name of the new directory that saves the generated text in corresponding .txt files
    
    """
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join(path_to_model, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)
    
    start_encode = datetime.now()
    # load every files
    documents = []

    if os.path.isdir(filename):
        # Directory
        for (dirpath, _, fnames) in os.walk(filename):
            fnames.sort()
            for fname in tqdm.tqdm(fnames):
                path = os.path.join(dirpath, fname)
                with open(path, 'r') as fp:
                    raw_text = fp.read()
                    # Reminder: the last token in raw_text should not be either \n or space
                    context_tokens = enc.encode(raw_text.replace('\n',''))
                    documents.append((path, context_tokens))
    print('time spent in encoding', datetime.now() - start_encode)
    if max_document:
        documents = documents[:max_document]
        print('conserve %d test cases' % max_document)
        
    
    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(path_to_model, model_name))
        saver.restore(sess, ckpt)
        
        start_decode = datetime.now()
        print('start decoding', start_decode)
        to_write = ''
        for (path, context_tokens) in tqdm.tqdm(documents):
            for _ in range(nsamples // batch_size):
                out = sess.run(output, 
                               feed_dict={context: [context_tokens for _ in range(batch_size)]
                                         })[:, len(context_tokens):]
                for i in range(batch_size):
                    text = enc.decode(out[i])
                    #to_write = text.replace('\n','').split('<')[0]
                    to_write = text.replace('\n','')
                    dir_path, fname = os.path.split(path)
                    save(os.path.join(dir_path[:-1], 'generation%s'%(tag), fname), to_write, overwrite, print_ = False)
                    
        print('time spent in decoding', datetime.now() - start_decode)
        
        
            
if __name__ == '__main__':
    fire.Fire(interact_model)

    
