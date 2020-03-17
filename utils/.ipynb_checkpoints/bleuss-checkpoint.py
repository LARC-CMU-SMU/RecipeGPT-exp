#import subprocess
#from rouge import Rouge 
from torchnlp._third_party.lazy_loader import LazyLoader
from torchnlp.metrics import get_moses_multi_bleu

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