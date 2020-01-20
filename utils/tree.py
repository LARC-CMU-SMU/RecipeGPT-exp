import pandas as pd
import numpy as np
import spacy
import copy
import re
import tqdm
import zss
from ete3 import Tree
from gensim.models import KeyedVectors
gensim_model = KeyedVectors.load_word2vec_format('../data/vocab.bin', binary = True)
vocabulary = gensim_model.vocab.keys() # usuage:gensim_model.get_vector('cook')


class Node(object):
    def __init__(self, label, nodetype):
        # assert type(label) == str
        self.label = label
        self.children = list()
        self.nodetype = nodetype

    @staticmethod
    def get_children(node):
        return node.children

    @staticmethod
    def get_label(node):
        return node.label
    
    @staticmethod
    def get_nodetype(node):
        return node.nodetype
    
    def addkid(self, node, before=False):
        if before:  self.children.insert(0, node)
        else:   self.children.append(node)
        return self
    
def make_nodes(sentence):
    action = Node(label=sentence['word'], nodetype='action')
    for ing in sentence['ingredient']:
        action.addkid(Node(label=ing, nodetype='ingredient'))
    return action

def build_tree(recipe_inst):
    '''
    recipe_inst = [{'word': 'verb1', 'ingredient':['A1','B1','C1','D1']},
                   {'word': 'verb2', 'ingredient':['A2','B2','C2','D2']},
                   {'word': 'verb3', 'ingredient':['A3','B3','C3','D3']}
                  ]
    '''
    tree1 = Node(label=recipe_inst[0]['word'], nodetype='action')
    for ing in recipe_inst[0]['ingredient']:
        tree1.addkid(Node(label=ing, nodetype='ingredient')
                    )
    myroot = tree1
    recipe_inst = recipe_inst[1:]
    for sentence in recipe_inst:
        myroot.addkid(make_nodes(sentence), before=True)
        myroot = Node.get_children(myroot)[0]
    return tree1

'''compare the number of edit'''
def strdist(a, b): return 0 if a == b else 1

def cosine_distance(vector_a, vector_b):
        cosine_similarity =  np.dot(vector_a, vector_b)/(np.linalg.norm(vector_a)* np.linalg.norm(vector_b))
        return 1 - cosine_similarity
    
'''compare the cosine distance of node'''
def wordvec_dist(a, b):
    assert a in vocabulary
    assert b in vocabulary
    if a == b: 
        return 0
    else:
        vector_a, vector_b = gensim_model.get_vector(a), gensim_model.get_vector(b)
        return cosine_distance(vector_a, vector_b)
    
def tree_distance(tree1, tree2):
    return zss.distance(tree1, tree2, 
             Node.get_children,
             insert_cost=lambda node: strdist('', Node.get_label(node)),
             remove_cost=lambda node: strdist(Node.get_label(node), ''),
             update_cost=lambda a, b: wordvec_dist(Node.get_label(a), Node.get_label(b))
            )

def draw_tree(recipe_inst):
    '''
    from ete3 import Tree
    recipe_inst = [{'word': 'heated', 'ingredient':['rice','banana','cookie','dishes']},
                   {'word': 'boil', 'ingredient':['apple','banana','cookie','dish']},
                   {'word': 'rince', 'ingredient':['apple','banana','cookie','dish']}
                  ]
    '''
    # sorting will not improve the tree edit distance
    # if sort:
    #    recipe_inst = [{'word':line['word'], 'ingredient': sorted(line['ingredient'])} for line in recipe_inst]
        
    output = Tree()
    temp = output
    for i in recipe_inst:
        t = Tree(name=i['word'])
        t.add_feature('type', 'action')
        if not i['ingredient']:
            pass
        else:
            for j in i['ingredient']:
                a = t.get_tree_root().add_child(name=j)
                a.add_feature('type', 'ingredient')
            temp = temp.add_child(t)
    print(output.get_ascii(show_internal=True))
    return output

class instr2tree:
    def __init__(self):
        self.spacy = spacy.load('en_core_web_lg')
        self.vocabulary= list(vocabulary)
        
    def sents2tree(self, sents):
        '''
        contatenate the leaves to a big tree
        '''
        tree = []
        for sent in sents:
            temp = self.leaf(sent)
            if temp:
                for t in temp:
                    if t['word']:
                        tree.append(t)
                    else:
                        ''' while creating the leaf, I force every leaf to have a content in "word"
                            so that I can avoid the IndexError in tree[-1] '''
                        tree[-1]['ingredient'] += t['ingredient']
        return tree
    
    def leaf(self, sent):
        '''
        transform a sentence to a leaf
        '''
        doc = self.spacy(sent)
        verbs = [(token.i, token.lemma_) for token in doc \
                 if token.pos_ == 'VERB' and token.lemma_ in self.vocabulary]
        nouns = [(chunk.root.i, chunk.root.lemma_) for chunk in doc.noun_chunks \
                 if chunk.root.lemma_ not in ['-PRON-'] and chunk.root.lemma_ in self.vocabulary]
        
        if not verbs and not nouns:
            return 
        
        # if do not have a noun, just add the verb
        elif not nouns: 
            return [{'word': v, 'ingredient': []} for vidx, v in verbs]
        
        # if do not have a verb, automatically set the first word in noun to verb
        elif not verbs: 
            return [{'word': nouns[0][1], 'ingredient': [n for nidx, n in nouns[1:]]}]
        
        '''
        verbs=[(0, 'v1'),(1,'v2'),(6, 'v3'),(8,'v4'),(10,'v5')]
        nouns=[(2, 'n1'),(3,'n2'),(4, 'n3'),(7,'n4'), (9,'n5'),(12,'n6')]
        1) loop through verb
        2) check verb whether > noun
        3) if > then next noun
        4) if not then next verb
        '''
        output, sent, vidx, nidx = [], [], 1, 0
        while vidx < len(verbs):
            if nidx < len(nouns) and nouns[nidx] < verbs[vidx]:
                sent.append(nouns[nidx][1])
                nidx += 1
            else:
                output.append({'word':verbs[vidx-1][1], 'ingredient': sent})
                vidx +=1
                sent = []
        while nidx < len(nouns):
            sent.append(nouns[nidx][1])
            nidx += 1
        output.append({'word':verbs[vidx-1][1], 'ingredient': sent})
        return output
    
def avg_embedding(tree):
    '''
    [{'word': 'stir', 'ingredient': ['ketchup', 'mustard', 'plate']},
    {'word': 'stir', 'ingredient': ['']}]
    '''
    words = [ing for line in tree for ing in line['ingredient'] if ing]
    words += [line['word'] for line in tree]
    X = np.array([gensim_model.get_vector(word) for word in words if word in vocabulary])
    return X.mean(axis=0)

def stem(x):
    return [{'word':d['word'], 'ingredient':[]} for d in x]

'''Example of usuage
from utils.tree import instr2tree, tree_distance, build_tree
from utils.evaluation import spacy_extension
treemaker = instr2tree()
sp = spacy_extension()
def stem(x):
    return [{'word':d['word'], 'ingredient':[]} for d in x]

def allinone(v_directions, v_generated_instr):
    true = treemaker.sents2tree(v_directions)
    pred = treemaker.sents2tree(v_generated_instr)
    tree_dist = tree_distance(build_tree(true), build_tree(pred))
    true, pred = stem(true), stem(pred)
    stem_dist = tree_distance(build_tree(true), build_tree(pred))
    return tree_dist, stem_dist

'''