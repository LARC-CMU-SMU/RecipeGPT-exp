from common.basics import *
from common.save import load_pickle
from utils.tree import instr2tree, tree_distance, build_tree, stem
from utils.metrics import metrics

import subprocess
from rouge import Rouge 
from torchnlp._third_party.lazy_loader import LazyLoader
from torchnlp.metrics import get_moses_multi_bleu

treemaker = instr2tree()
database = load_pickle('../big_data/database3.pickle')

def full_moses_multi_bleu(original, generation):
    '''
    One pair at a time
    Will not match moses_multi_bleu.perl because that one is the average the sentence level scores
    
    Note: need to add space before punctuations
    :original: string
    :generation: string
    '''
    filereference = '../../to_gpt2/generation_ori.txt'
    filehypothesis = '../../to_gpt2/generation_gen.txt'
    save(filereference, original ,overwrite = True, print_ = False)
    save(filehypothesis, generation ,overwrite = True, print_ = False)

    bleu_cmd = ['perl', 'multi-bleu.perl', filehypothesis]
    with open(filereference, "r") as read_pred:
        bleu_out = subprocess.check_output(bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
        bleu_out = bleu_out.decode("utf-8")
        
        
        BLEU = re.search(r"BLEU = (.+?),", bleu_out).group(1)
        bleu_out = bleu_out.split(', ')
        B1, B2, B3, B4, BP = bleu_out[-4].replace(' (BP=', '/').split('/')
        ratio = bleu_out[-3].replace('ratio=', '')
        hyp_len=bleu_out[-2].replace('hyp_len=', '')
        ref_len=bleu_out[-1].replace('ref_len=', '').replace(')\n', '')

        ans = {'BLEU': BLEU, 
               'B1': B1, 'B2':B2, 'B3':B3, 'B4':B4, 'BP': BP, 
               'ratio':ratio, 'hyp_len': hyp_len, 'ref_len':ref_len}
        ans = {i: float(v) for i, v in ans.items()}
    return ans


            
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
    
    def ingr_f1(self, root = True):
        value, number = [], []
        for i, v in tqdm.tqdm(self.dic.items()):
            true, pred = v['%s_ingr'%(self.ori)], v['%s_ingr'%(self.gen)]
            if root:
                true, pred = self.ingr(true), self.ingr(pred)
            scores = metrics(true, pred)
            value.append(scores.f1())
            number.append(len(set(pred)))
        avg = sum(value)/len(value)
        print(avg)
        print('average ingr number', sum(number)/len(number))
        return avg
    
    def ingr_precision_recall(self, generate = 'ingr'):
        assert generate in ['ingr','instr','human']
        precision, recall = [], []
        for i, v in tqdm.tqdm(self.dic.items()):
            if generate == 'ingr':
                true, pred = v['%s_ingr'%(self.ori)], v['%s_ingr'%(self.gen)]
                true, pred = self.ingr(true), self.ingr(pred)
            elif generate == 'instr':
                true, pred = v['%s_ingr'%(self.ori)], v['%s_instr'%(self.gen)]
                true, pred = self.ingr(true), self.instr(pred)
            elif generate == 'human':   
                true, pred = v['%s_ingr'%(self.ori)], v['%s_instr'%(self.ori)]
                true, pred = self.ingr(true), self.instr(pred)
            scores = metrics(true, pred)
            precision.append(scores.precision())
            recall.append(scores.recall())
    
        print('precision', sum(precision)/len(precision))
        print('recall', sum(recall)/len(recall))
        
        
    def jaccard(self, generate = 'ingr'):
        assert generate in ['ingr','instr','human']
        jaccard = []
        for i, v in tqdm.tqdm(self.dic.items()):
            if generate == 'ingr':
                true, pred = v['%s_ingr'%(self.ori)], v['%s_ingr'%(self.gen)]
                true, pred = self.ingr(true), self.ingr(pred)
            elif generate == 'instr':
                true, pred = v['%s_ingr'%(self.ori)], v['%s_instr'%(self.gen)]
                true, pred = self.ingr(true), self.instr(pred)
            elif generate == 'human':   
                true, pred = v['%s_ingr'%(self.ori)], v['%s_instr'%(self.ori)]
                true, pred = self.ingr(true), self.instr(pred)
            true, pred = set(true), set(pred)
            
            intersect = len(true & pred)
            similarity = intersect /(len(true)+len(pred) - intersect)
            jaccard.append(similarity)
    
        print('jaccard', sum(jaccard)/len(jaccard))
    
    def instr(self, directions):
        instr = sp.spacy(directions)
        root_instr = []
        for chunk in instr.noun_chunks:
            idx_rootnoun = chunk.end - 1
            str_rootnoun = instr[idx_rootnoun].lemma_
            if str_rootnoun in database:
                root_instr.append(str_rootnoun)
        return root_instr
    
    def ingr(self, lst):
        '''
        Args: lst: a list of ingredient names
        used when len(lst) must equal to root_match
        '''
        hl = [[{'text':x, 'highlight': None} for x in i.split(' ')] for i in lst]
        root_match = []
        for i, ingr in enumerate(lst):
            if ' ' not in ingr:
                hl[i][0]['highlight'] = 'wrong'
                doc = sp.spacy(ingr)
                root_match.append(doc[0].lemma_)
            else:
                phrase = 'Mix the %s and water.'%ingr
                doc = sp.spacy(phrase)
                
                last_chunk = None
                for chunk in doc.noun_chunks:
                    if chunk.text != 'water':
                        last_chunk = chunk
                if not last_chunk:
                    root_match.append('CANNOT_DETECT')
                else:
                    found = False
                    for j, word in enumerate(hl[i]):
                        if doc[last_chunk.end - 1].text in word['text']:
                            hl[i][j]['highlight'] = 'wrong' 
                            root_match.append(doc[last_chunk.end - 1].lemma_)
                            found = True
                            break
                    if not found:
                        root_match.append('CANNOT_DETECT')
                        
        assert len(root_match) == len(lst)
        return root_match
    
    def pair_f1(self, v):
        '''
        campare at the set root noun level
        '''
        true, pred = v['%s_ingr'%(self.ori)], v['%s_ingr'%(self.gen)]
        scores = metrics(self.ingr(true), self.ingr(pred))
        return {'f1': scores.f1()}
    
    def pair_jaccard(self, v, generate = 'ingr'):
        assert generate in ['ingr','instr','human']
        if generate == 'ingr':
            true, pred = v['%s_ingr'%(self.ori)], v['%s_ingr'%(self.gen)]
            true, pred = self.ingr(true), self.ingr(pred)
        elif generate == 'instr':
            true, pred = v['%s_ingr'%(self.ori)], v['%s_instr'%(self.gen)]
            true, pred = self.ingr(true), self.instr(pred)
        elif generate == 'human':   
            true, pred = v['%s_ingr'%(self.ori)], v['%s_instr'%(self.ori)]
            true, pred = self.ingr(true), self.instr(pred)
        true, pred = set(true), set(pred)
        intersect = len(true & pred)
        similarity = intersect /(len(true)+len(pred) - intersect)
        return {'jaccard_%s' %(generate): similarity }
    
    def pair_bleu(self, v):
        ori_instr, gen_instr = v['%s_instr'%(self.ori)], v['%s_instr'%(self.gen)]
        ori_instr, gen_instr = self.add_space(ori_instr), self.add_space(gen_instr)
        ans = full_moses_multi_bleu(ori_instr, gen_instr)
        ''' calculate rouge-L as well'''
        scores = self.rouge.get_scores(gen_instr, ori_instr)
        ans.update({'RL': scores[0]['rouge-l']['f']})
        return ans
    
    def pair_nted(self, v):
        ori_instr, gen_instr = v['%s_instr'%(self.ori)], v['%s_instr'%(self.gen)]
        score = self.norm_dist(ori_instr, gen_instr, stem_only = False)
        return {'NTED': score}
        
    def multibatch(self):
        value = []
        for i, v in tqdm.tqdm(self.dic.items()):
            self.function(i, v)
            
            
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