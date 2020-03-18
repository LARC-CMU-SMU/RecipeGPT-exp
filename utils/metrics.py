import numpy as np
import copy
from collections import Counter
from common.save import load_pickle

#import warnings
#from sklearn.exceptions import UndefinedMetricWarning
#state_verbs = load_pickle('../big_data/state_verbs.pickle')


class metrics:
    def __init__(self, list_true, list_pred):
        self.y_true = [word for word in list_true if word != 'nan']
        self.y_pred = [word for word in list_pred if word != 'nan']
        
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
        
    #frequency weighted metrics
    ''' 
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
    '''
        
   