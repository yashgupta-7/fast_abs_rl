""" ROUGE utils"""
import os
import threading
import subprocess as sp
from collections import Counter, deque
# from bert_score import score
import bert_score
from cytoolz import concat, curry

# import tensorflow as tf
# tf.compat.v1.flags.DEFINE_string('f','','')
# tf.compat.v1.flags.DEFINE_string('path','','')
# tf.compat.v1.flags.DEFINE_string('abs_dir','','')
# tf.compat.v1.flags.DEFINE_string('ext_dir','','')
# tf.compat.v1.flags.DEFINE_string('reward','','')
# tf.compat.v1.flags.DEFINE_string('ckpt_freq','','')
# tf.compat.v1.flags.DEFINE_string('batch','','')
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#   tf.config.experimental.set_memory_growth(gpu, True)
# from bleurt import score
# checkpoint = "/exp/yashgupta/bleurt/bleurt-base-512"
# scorer = score.BleurtScorer(checkpoint)

@curry
def compute_bleurt_score(output, reference, n=1, mode='f'):
    # pass
    # print(len(output), output)
    scores = scorer.score(reference, output)
    # print(len(scores), scores)
    return scores

import nltk
@curry
def compute_bleu_score(output, reference, n=1, mode='f'):
    BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], output)
    return BLEUscore

import spacy
import wmd
nlp = spacy.load('en', create_pipeline=wmd.WMD.create_spacy_pipeline)
@curry
def compute_wms_score(output, reference, n=1, mode='f'):
    doc1 = nlp(" ".join(output))
    doc2 = nlp(" ".join(reference))
    return doc1.similarity(doc2)
    
@curry
def compute_bert_score(output, reference, n=1, mode='f'):
    # print(" ".join(output), " ".join(reference))
    # P,R,F1 = score([" ".join(output)], [" ".join(reference)], lang='en', verbose=True)
    P,R,F1 = bert_score.score(output, reference, lang='en', verbose=False)
    return [f.item() for f in F1]

def make_n_grams(seq, n):
    """ return iterator """
    ngrams = (tuple(seq[i:i+n]) for i in range(len(seq)-n+1))
    return ngrams

def _n_gram_match(summ, ref, n):
    summ_grams = Counter(make_n_grams(summ, n))
    ref_grams = Counter(make_n_grams(ref, n))
    grams = min(summ_grams, ref_grams, key=len)
    count = sum(min(summ_grams[g], ref_grams[g]) for g in grams)
    return count

@curry
def compute_rouge_n(output, reference, n=1, mode='f'):
    """ compute ROUGE-N for a single pair of summary and reference"""
    assert mode in list('fpr')  # F-1, precision, recall
    match = _n_gram_match(reference, output, n)
    if match == 0:
        score = 0.0
    else:
        precision = match / len(output)
        recall = match / len(reference)
        f_score = 2 * (precision * recall) / (precision + recall)
        if mode == 'p':
            score = precision
        elif mode == 'r':
            score = recall
        else:
            score = f_score
    return score


def _lcs_dp(a, b):
    """ compute the len dp of lcs"""
    dp = [[0 for _ in range(0, len(b)+1)]
          for _ in range(0, len(a)+1)]
    # dp[i][j]: lcs_len(a[:i], b[:j])
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp

def _lcs_len(a, b):
    """ compute the length of longest common subsequence between a and b"""
    dp = _lcs_dp(a, b)
    return dp[-1][-1]

@curry
def compute_rouge_l(output, reference, mode='f'):
    """ compute ROUGE-L for a single pair of summary and reference
    output, reference are list of words
    """
    assert mode in list('fpr')  # F-1, precision, recall
    lcs = _lcs_len(output, reference)
    if lcs == 0:
        score = 0.0
    else:
        precision = lcs / len(output)
        recall = lcs / len(reference)
        f_score = 2 * (precision * recall) / (precision + recall)
        if mode == 'p':
            score = precision
        if mode == 'r':
            score = recall
        else:
            score = f_score
    # print(score)
    return score


def _lcs(a, b):
    """ compute the longest common subsequence between a and b"""
    dp = _lcs_dp(a, b)
    i = len(a)
    j = len(b)
    lcs = deque()
    while (i > 0 and j > 0):
        if a[i-1] == b[j-1]:
            lcs.appendleft(a[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] >= dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    assert len(lcs) == dp[-1][-1]
    return lcs

def compute_rouge_l_summ(summs, refs, mode='f'):
    """ summary level ROUGE-L"""
    assert mode in list('fpr')  # F-1, precision, recall
    tot_hit = 0
    ref_cnt = Counter(concat(refs))
    summ_cnt = Counter(concat(summs))
    for ref in refs:
        for summ in summs:
            lcs = _lcs(summ, ref)
            for gram in lcs:
                if ref_cnt[gram] > 0 and summ_cnt[gram] > 0:
                    tot_hit += 1
                ref_cnt[gram] -= 1
                summ_cnt[gram] -= 1
    if tot_hit == 0:
        score = 0.0
    else:
        precision = tot_hit / sum((len(s) for s in summs))
        recall = tot_hit / sum((len(r) for r in refs))
        f_score = 2 * (precision * recall) / (precision + recall)
        if mode == 'p':
            score = precision
        if mode == 'r':
            score = recall
        else:
            score = f_score
    return score


try:
    _METEOR_PATH = os.environ['METEOR']
except KeyError:
    print('Warning: METEOR is not configured')
    _METEOR_PATH = None
class Meteor(object):
    def __init__(self):
        assert _METEOR_PATH is not None
        cmd = 'java -Xmx2G -jar {} - - -l en -norm -stdio'.format(_METEOR_PATH)
        self._meteor_proc = sp.Popen(
            cmd.split(),
            stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE,
            universal_newlines=True, encoding='utf-8', bufsize=1
        )
        self._lock = threading.Lock()

    def __call__(self, summ, ref):
        self._lock.acquire()
        score_line = 'SCORE ||| {} ||| {}\n'.format(
            ' '.join(ref), ' '.join(summ))
        self._meteor_proc.stdin.write(score_line)
        stats = self._meteor_proc.stdout.readline().strip()
        eval_line = 'EVAL ||| {}\n'.format(stats)
        self._meteor_proc.stdin.write(eval_line)
        score = float(self._meteor_proc.stdout.readline().strip())
        self._lock.release()
        return score

    def __del__(self):
        self._lock.acquire()
        self._meteor_proc.stdin.close()
        self._meteor_proc.kill()
        self._meteor_proc.wait()
        self._lock.release()
