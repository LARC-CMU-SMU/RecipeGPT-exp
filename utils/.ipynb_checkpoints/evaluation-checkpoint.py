import numpy as np
import copy
import spacy
from collections import Counter
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from .save import load_pickle
state_verbs = load_pickle('../big_data/state_verbs.pickle')


class metrics:
    def __init__(self, list_true, list_pred):
        self.y_true = [word for word in list_true if word != 'nan']
        self.y_pred = [word for word in list_pred if word != 'nan']
        self.y_true_string = ' %s '%(' '.join(self.y_true))
        self.y_pred_string = ' %s '%(' '.join(self.y_pred))
        # from collections import Counter
        
    # frequency weighted
    def f1_freq(self):
        precision = self.precision_freq()
        recall = self.recall_freq()
        try:
            f1 = 2*precision*recall/(precision + recall)
        except ZeroDivisionError:
            f1 = 0
        return f1
    
    def precision_freq(self):
        return self.scoring(self.y_pred, self.y_true)
    
    def recall_freq(self):
        return self.scoring(self.y_true, self.y_pred)
    
    def scoring(self, n1, n2):
        if not n1 or not n2:
            self.warn()
            return 0
        n1c, n2c = copy.deepcopy(n1), copy.deepcopy(n2)
        score = 0
        for word in n1c:
            if word in n2c:
                score +=1
                n2c.remove(word)
        if len(n1c): 
            return score/len(n1c)
        
    # without frequency weighted
    def precision(self):
        if len(self.y_pred): 
            return len(set(self.y_true) & set(self.y_pred))/len(set(self.y_pred))
        else:
            return 0
    def recall(self):
        if len(self.y_true):
            return len(set(self.y_true) & set(self.y_pred))/len(set(self.y_true))
        else:
            return 1
    def f1(self):
        precision = self.precision()
        recall = self.recall()
        try:
            f1 = 2*precision*recall/(precision + recall)
        except ZeroDivisionError:
            f1 = 0
        return f1
    
    # return scores in a dict    
    def all_recall(self, name):
        output = {}
        output['recall_%s'%(name)] = self.recall()
        output['recall_freq_%s'%(name)] = self.recall_freq()
        return output
    
    def all_precision(self, name):
        output = {}
        output['precision_%s'%(name)] = self.precision()
        output['precision_freq_%s'%(name)] = self.precision_freq()
        return output
    
    # frequency weighted
    def ngram_scoring(self, n1, n2):
        '''
        n1 = [' ddd d ',' der ',' w ',' w ']
        n2 = ' ddd d mnnm,n,m der ddd d w ow '
        '''
        n1c, n2c = copy.deepcopy([' %s ' % word for word in n1]), copy.deepcopy(n2)
        true_counter = Counter(n1)
        denomintater = sum(true_counter.values())
        score = 0
        for word in true_counter:
            occurrence = n2.count(word)
            occurrence = occurrence if occurrence < true_counter[word] else true_counter[word]
            score += occurrence
        try:
            score = score/sum(true_counter.values())
            return score
        except ZeroDivisionError:
            return 0
    
    def ngram_recall_freq(self):
        return self.ngram_scoring(self.y_true, self.y_pred_string)
    
    def ngram_precision_freq(self):
        return self.ngram_scoring(self.y_pred, self.y_true_string)
    
    #  unweighted
    def ngram_recall(self):
        return np.mean([True if word in self.y_pred_string else False for word in set(self.y_true)])

    def ngram_precision(self):
        return np.mean([True if word in self.y_true_string else False for word in set(self.y_pred)])
    
    def all_ngram_recall(self, name):
        output = {}
        output['recall_ngram_%s'%(name)] = self.ngram_recall()
        output['recall_ngram_freq_%s'%(name)] = self.ngram_recall_freq()
        return output
    
    def all_ngram_precision(self, name):
        output = {}
        output['precision_ngram_%s'%(name)] = self.ngram_precision()
        output['precision_ngram_freq_%s'%(name)] = self.ngram_precision_freq()
        return output
    
    def warn(self):
        message = 'input/inputs may be empty'
        warnings.warn(message, UndefinedMetricWarning, stacklevel=2)
    
class spacy_extension:
    def __init__(self):
        '''
        Args: sent: string
        '''
        self.spacy = spacy.load('en_core_web_lg')
        
    def instructions(self, sent): # for cooking instructions
        if type(sent) == list:
            sent = ''.join(sent)
            
        assert type(sent) == str
        
        doc = self.spacy(sent)
        root_noun = [chunk.root.lemma_ for chunk in doc.noun_chunks \
                if chunk.root.lemma_ not in ['-PRON-']]
        verb = [token.lemma_ for token in doc if token.pos_ == 'VERB']
        
        return root_noun, verb
    
    def ny_ingred(self, ny_ingred):  # for ny_ingred
        '''
        Args: ny_ingred: a list of ingredient names
        '''
        phrases_to_sentences = ' '.join(['Mix the %s and water.'%ingr for ingr in ny_ingred])
        doc = self.spacy(phrases_to_sentences)
        exact_match, root_match = [],[]
        for chunk in doc.noun_chunks:
            if chunk.text != 'water':
                root_lemma = [token.lemma_ for token in doc if token.text == chunk.root.text][0]
                exact_match.append(chunk.lemma_.replace('the ',''))
                root_match.append(root_lemma)
        return exact_match, root_match
    
    def root(self, lst):
        '''
        Args: lst: a list of ingredient names
        '''
        phrases_to_sentences = ' '.join(['Mix the %s and water.'%ingr for ingr in lst])
        doc = self.spacy(phrases_to_sentences)
        root_match = []
        for chunk in doc.noun_chunks:
            if chunk.text != 'water':
                root_lemma = [token.lemma_ for token in doc if token.text == chunk.root.text][0]
                root_match.append(root_lemma)
        return root_match
    
    def root_visual(self, lst):
        '''
        Args: lst: a list of ingredient names
        used when len(lst) must equal to root_match
        '''
        root_match = {}
        for ingr in lst:
            phrase = 'Mix the %s and water.'%ingr
            doc = self.spacy(phrase)
            for chunk in doc.noun_chunks:
                if chunk.text != 'water':
                    root_match.update({ingr: doc[chunk.end - 1]})
        return root_match
    
    def match_state(self, sent):
        doc = self.spacy(sent)
        state = []
        for word in doc.doc:
            if word.lemma_ in state_verbs.keys():
                for i, v in state_verbs[word.lemma_].items():
                    state.append('%s_%s'%(i, v))
        return state