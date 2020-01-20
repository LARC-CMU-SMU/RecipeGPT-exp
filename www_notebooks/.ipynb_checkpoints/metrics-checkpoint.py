# if cannot import the modules, add the parent directory to system path might help
import os, tqdm, sys
parent_dir = os.path.abspath(os.getcwd()+'/../')
sys.path.append(parent_dir)

from utils.path import dir_HugeFiles
from utils.save import make_dir, save_pickle, load_pickle, save
from utils.tree import instr2tree, tree_distance, build_tree, stem
from utils.evaluation import metrics, spacy_extension

import pandas as pd
import numpy as np
import re

treemaker = instr2tree()
sp = spacy_extension()

class evaluation:
    def __init__(self, filename, tag):
        self.dic = self.load_dic({}, filename, tag)
        self.ori = tag
        self.gens = []
        #from utils.evaluation import metrics
    '''
    loading data
    '''
    def append_dic(self, filename, tag):
        if tag in self.gens:
            print('already exist, will not load again')
            self.gen = tag
        else:
            self.dic = self.load_dic(self.dic, filename, tag)
            self.gen = tag
            self.gens += [tag]
        
    def load_dic(self, dic, filename, tag):
        if os.path.isdir(filename):
            print('load', filename)
            for (dirpath, _, fnames) in os.walk(filename):
                for fname in fnames:
                    path = os.path.join(dirpath, fname)
                    with open(path, 'r') as fp:
                        raw_text = fp.read()
                        raw_text = self.remove_end(raw_text)

                    name, field = int(fname[:-5]), fname[-5]

                    if name not in dic.keys() and field in ['d','i']:
                        dic.update({name: {}})

                    if field == 'd':
                        dic[name].update({'%s_instr'%(tag): raw_text})

                    if field == 'i':
                        raw_text = self.reverse_list(raw_text.split('$'))
                        dic[name].update({'%s_ingr'%(tag): raw_text})
        return dic
    
    '''
    instruction evaluation
    '''
    def instr_tree(self, stem_only = False):
        value = []
        for i, v in tqdm.tqdm(self.dic.items()):
            ori_instr, gen_instr = v['%s_instr'%(self.ori)], v['%s_instr'%(self.gen)]
            score = self.norm_dist(ori_instr, gen_instr, stem_only = stem_only)
            value.append(score)
        avg = sum(value)/len(value)
        print(avg)
        return avg

    '''
    cleaning data
    '''
    def remove_end(self, text):
        return text.replace('\n','').split('<')[0]
    
    def reverse(self, text):
        '''
        Important data cleaning before NY times parser
        '''
        # replace things in brace
        text = re.sub(r'\([^)]*\)', '', text)

        # remove space before punct
        text = re.sub(r'\s([?.!,"](?:\s|$))', r'\1', text)

        # remove consecutive spaces
        text = re.sub(' +',' ',text).strip()
        return text
    
    def reverse_list(self, listoftext):
        output = []
        for text in listoftext:
            rev = self.reverse(text)
            if rev:
                output.append(rev)
        return output
    
    def add_space(self, line):
        # add space before punct
        line = re.sub('([.,!?()])', r' \1 ', line)
        line = re.sub('\s{2,}', ' ', line)
        return line
    
    '''
    tree edit distance
    '''

    def str2tree(self, instr, stem_only):
        instr = [x for x in instr.split('. ') if x]
        instr = treemaker.sents2tree(instr)
        if stem_only:
            instr = stem(instr)
        n_nodes = sum([len(line['ingredient']) +1 for line in instr])
        return build_tree(instr), n_nodes

    def norm_dist(self, ori_instr, gen_instr, stem_only):
        '''
        Args: ori_instr: str
        Args: gen_instr: str
        '''
        ori_tree, ori_nodes = self.str2tree(ori_instr, stem_only = stem_only)
        gen_tree, gen_nodes = self.str2tree(gen_instr, stem_only = stem_only)
        tree_dist = tree_distance(ori_tree, gen_tree)
        normed = tree_dist/(ori_nodes+gen_nodes)
        return normed
    
    '''can only be call in notebooks
    def to_bleu(self):
        to_write = {'%s_i'%(self.ori):'',
                    '%s_i'%(self.gen):'',
                    '%s_d'%(self.ori):'',
                    '%s_d'%(self.gen):''}
        
        for i, v in self.dic.items():
            to_write['%s_i'%(self.ori)] += self.add_space(' $ '.join(v['%s_ingr'%(self.ori)]))+ ' $ \n'
            to_write['%s_i'%(self.gen)] += self.add_space(' $ '.join(v['%s_ingr'%(self.gen)])) + ' $ \n'
            
            to_write['%s_d'%(self.ori)] += self.add_space(v['%s_instr'%(self.ori)])+ '\n'
            to_write['%s_d'%(self.gen)] += self.add_space(v['%s_instr'%(self.gen)])+ '\n'
        
        for k, v in to_write.items():
            save('../../to_gpt2/generation_%s.txt'%(k), v ,overwrite = True)
        !eval {"perl multi-bleu.perl ../../to_gpt2/generation_%s_i.txt < ../../to_gpt2/generation_%s_i.txt" %(self.ori, self.gen)}
        !eval {"perl multi-bleu.perl ../../to_gpt2/generation_%s_d.txt < ../../to_gpt2/generation_%s_d.txt" %(self.ori, self.gen)}
    
        !eval {"rouge -f ../../to_gpt2/generation_%s_i.txt ../../to_gpt2/generation_%s_i.txt --avg"%(self.ori, self.gen)}
        !eval {"rouge -f ../../to_gpt2/generation_%s_d.txt ../../to_gpt2/generation_%s_d.txt --avg"%(self.ori, self.gen)}
        
        print()
        
    '''