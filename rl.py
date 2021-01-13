""" RL training utilities"""
import math
from time import time
from datetime import timedelta

from toolz.sandbox.core import unzip
from cytoolz import concat

import numpy as np
import torch
from torch.nn import functional as F
from torch import autograd
from torch.nn.utils import clip_grad_norm_

from metric import compute_rouge_l, compute_rouge_n, compute_bert_score, compute_bleurt_score, compute_bleu_score, compute_wms_score 
from training import BasicPipeline

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys, os
sys.path.append(os.path.abspath(os.path.join('/exp/yashgupta/PreSumm/src/')))
from prepro.data_builder import presumm_reward, presumm_reward2, presumm_reward3, presumm_reward4

tokenizer = None #AutoTokenizer.from_pretrained("facebook/bart-base")
bart_model = None #AutoModelForSeq2SeqLM.from_pretrained("/exp/yashgupta/transformers/examples/seq2seq/absm_cnn_bart_2/")
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
# bart_model.to(torch_device)
use_bart = 0
wt_rge = 0.0

def set_abstractor(args):
    global use_bart, wt_rge, tokenizer, bart_model
    use_bart = args.bart
    wt_rge = args.wt_rouge
    if use_bart == 1:
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
        bart_model = AutoModelForSeq2SeqLM.from_pretrained("/exp/yashgupta/transformers/examples/seq2seq/absm_cnn_bart_eval/")
        bart_model.to(torch_device) 
    elif use_bart == 2:
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
        bart_model = AutoModelForSeq2SeqLM.from_pretrained("/exp/yashgupta/transformers/examples/seq2seq/absm_cnn_bart_large/")
        bart_model.to(torch_device) 
    return

def get_bart_summaries(sents, tokenizer, model, beam_size=2):
    # print(bart_model)
    sents = [" ".join(sent) for sent in sents]
    batch = tokenizer(sents,truncation=True,padding='longest',max_length=130,return_tensors='pt').to(torch_device)
    translated = model.generate(batch['input_ids'], max_length=90, num_beams=beam_size) #, early_stopping=True)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return [text.split(" ") for text in tgt_text]

def a2c_validate(agent, abstractor, loader):
    agent.eval()
    start = time()
    print('start running validation...', end='')
    avg_reward = 0
    i = 0
    with torch.no_grad():
        for art_batch, abs_batch in loader:
            ext_sents = []
            ext_inds = []
            for raw_arts, raw_abs in zip(art_batch, abs_batch):
                indices, raw_arts = agent(raw_arts, raw_abs_sents=raw_abs)
                ext_inds += [(len(ext_sents), len(indices)-1)]
                ext_sents += [raw_arts[idx.item()]
                              for idx in indices if idx.item() < len(raw_arts)]
            if use_bart:
                all_summs=get_bart_summaries(ext_sents, tokenizer, bart_model)
            else:
                all_summs = abstractor(ext_sents)
            for (j, n), abs_sents in zip(ext_inds, abs_batch):
                summs = all_summs[j:j+n]
                # python ROUGE-1 (not official evaluation)
                avg_reward += compute_rouge_n(list(concat(summs[:3])),    ####CHANGEEEEEEEEEE THISSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
                                              list(concat(abs_sents)), n=1)
                i += 1
    avg_reward /= (i/100)
    print('finished in {}! avg reward: {:.2f}'.format(
        timedelta(seconds=int(time()-start)), avg_reward))
    return {'reward': avg_reward}


def a2c_train_step(agent, abstractor, loader, opt, grad_fn,
                   gamma=0.99, reward_fn=compute_rouge_l,
                   stop_reward_fn=compute_rouge_n(n=1), stop_coeff=1.0): #, wt_rge = wt_rge):
    opt.zero_grad()
    indices = []
    probs = []
    baselines = []
    ext_sents = []
    art_batch, abs_batch = next(loader)
    for raw_arts, raw_abs in zip(art_batch, abs_batch):
        # print(raw_abs)
        (inds, ms), bs, raw_arts = agent(raw_arts, raw_abs_sents=raw_abs, n_abs=None) #, n_abs=3
        # print(inds, len(raw_arts), len(inds))
        baselines.append(bs)
        indices.append(inds)
        probs.append(ms)
        ext_sents += [raw_arts[idx.item()]
                      for idx in inds if idx.item() < len(raw_arts)]
    with torch.no_grad():
        if not ext_sents:
            summaries = []
        elif use_bart:
            summaries = get_bart_summaries(ext_sents, tokenizer, bart_model)
        else:
            summaries = abstractor(ext_sents)
    i = 0
    rewards = []
    avg_reward = 0
    # print(indices)
    # print(abs_batch)
    cands = []
    refs = []
    F1s = None
    if reward_fn == compute_bert_score or reward_fn == compute_bleurt_score:
        # print("bertscore -reward")
        for inds, abss in zip(indices, abs_batch):
            for j in range(min(len(inds)-1, len(abss))):
                cands.append(" ".join(summaries[i+j]))
                refs.append(" ".join(abss[j]))
            i += len(inds)-1
        # print(len(cands), len(refs)) #around 120 each
        F1s = reward_fn(cands, refs)
        # print(F1s)

    i = 0
    t = 0
    avg_len = 0
    for inds, abss in zip(indices, abs_batch):
        # print(abss)
        # abss_tot = [x for y in abss for x in y]
        # print([j for j in range(min(len(inds)-1, len(abss)))]) compute_rouge_lcompute_rouge_n(n=2) #PLEASE NOTE HERE
        x = len(inds) - 1
        if (reward_fn == compute_bert_score or reward_fn == compute_bleurt_score) and wt_rge >= 0.0001:
            rwd_lst = [(1-wt_rge)*F1s[t + j] + wt_rge*compute_rouge_n(summaries[i+j], abss[j],  n=2) for j in range(min(len(inds)-1, len(abss)))]
            # print(rwd_lst)
            t += min(len(inds)-1, len(abss))
        elif (reward_fn == compute_bert_score or reward_fn == compute_bleurt_score):
            rwd_lst = [F1s[t + j] for j in range(min(len(inds)-1, len(abss)))]
            # print(rwd_lst)
            t += min(len(inds)-1, len(abss))
        elif wt_rge >= 0.0001:
            # print("hi", wt_rge) 
            rwd_lst = [(1-wt_rge)*reward_fn(summaries[i+j], abss[j]) + wt_rge*compute_rouge_l(summaries[i+j], abss[j]) for j in range(min(len(inds)-1, len(abss)))]
        elif reward_fn == presumm_reward or reward_fn == presumm_reward2 or reward_fn == presumm_reward3 or reward_fn == presumm_reward4:
            rwd_lst = reward_fn(summaries[i:i+len(inds)-1], abss)
            x = 3
            # print("hi", x)
        elif reward_fn == "summ-rouge-l":
            tmp = list(concat(abss))
            rwd_lst = [compute_rouge_l(summaries[i+j], tmp, mode='r') for j in range(min(len(inds)-1, 3))]
        else:
            rwd_lst = [reward_fn(summaries[i+j], abss[j]) for j in range(min(len(inds)-1, len(abss)))] #abss_tot, mode=r
        rs = (rwd_lst
              + [0 for _ in range(max(0, len(inds)-1-len(rwd_lst)))] #len(inds)-1-len(abss)
              + [stop_coeff*stop_reward_fn(
                  list(concat(summaries[i:i+x])),
                  list(concat(abss)))])
        assert len(rs) == len(inds)
        avg_reward += rs[-1]/stop_coeff
        if reward_fn == presumm_reward or reward_fn == presumm_reward3 or reward_fn == presumm_reward4:
            rs[-1] = 0
        # if reward_fn == presumm_reward3:
        #     rs.pop()
        avg_len += (len(inds)-1)/len(abs_batch)# print(avg_reward)
        i += len(inds)-1
        # compute discounted rewards
        # print(rs)
        R = 0
        disc_rs = []
        for r in rs[::-1]:
            R = r + gamma * R
            disc_rs.insert(0, R)
        rewards += disc_rs
    # print(avg_len)
    indices = list(concat(indices))
    probs = list(concat(probs))
    baselines = list(concat(baselines))
    # standardize rewards
    reward = torch.Tensor(rewards).to(baselines[0].device)
    # print(reward.mean())
    reward = (reward - reward.mean()) / (
        reward.std() + float(np.finfo(np.float32).eps))
    baseline = torch.cat(baselines).squeeze()
    avg_advantage = 0
    losses = []
    avg_list = [] ##ARJUN 
    for action, p, r, b in zip(indices, probs, reward, baseline):
        advantage = (r - b).item()     ##ARJUN # (r-b)
        avg_advantage += advantage
        avg_list.append(advantage)     ##ARJUN 
    
    avg_list = torch.Tensor(avg_list).to(baselines[0].device)   ##ARJUN #
    avg_list = (avg_list - avg_list.mean()) / ( avg_list.std() + float(np.finfo(np.float32).eps)) ##ARJUN #

    for action, p, advantage in zip(indices, probs, avg_list): ##ARJUN 
        losses.append(-p.log_prob(action)
                      * (advantage/len(indices))) # divide by T*B
    critic_loss = F.mse_loss(baseline, reward).unsqueeze(dim=0)
    # backprop and update
    autograd.backward(
        [critic_loss] + losses,
        [torch.ones(1).to(critic_loss.device)]*(1+len(losses))
    )
    grad_log = grad_fn()
    opt.step()
    log_dict = {}
    log_dict.update(grad_log)
    log_dict['reward'] = avg_reward/len(art_batch)
    log_dict['advantage'] = avg_advantage/len(indices) #removed .item()
    log_dict['mse'] = critic_loss.item()
    log_dict['avg_len'] = avg_len
    assert not math.isnan(log_dict['grad_norm'])
    return log_dict


def get_grad_fn(agent, clip_grad, max_grad=1e2):
    """ monitor gradient for each sub-component"""
    params = [p for p in agent.parameters()]
    def f():
        grad_log = {}
        for n, m in agent.named_children():
            tot_grad = 0
            for p in m.parameters():
                if p.grad is not None:
                    tot_grad += p.grad.norm(2) ** 2
            tot_grad = tot_grad ** (1/2)
            try:
                grad_log['grad_norm'+n] = tot_grad.item()
            except:
                grad_log['grad_norm'+n] = tot_grad
        grad_norm = clip_grad_norm_(
            [p for p in params if p.requires_grad], clip_grad)
        grad_norm = grad_norm.item()
        if max_grad is not None and grad_norm >= max_grad:
            print('WARNING: Exploding Gradients {:.2f}'.format(grad_norm))
            grad_norm = max_grad
        grad_log['grad_norm'] = grad_norm
        return grad_log
    return f


class A2CPipeline(BasicPipeline):
    def __init__(self, name,
                 net, abstractor,
                 train_batcher, val_batcher,
                 optim, grad_fn,
                 reward_fn, gamma,
                 stop_reward_fn, stop_coeff):
        self.name = name
        self._net = net
        self._train_batcher = train_batcher
        self._val_batcher = val_batcher
        self._opt = optim
        self._grad_fn = grad_fn

        self._abstractor = abstractor
        self._gamma = gamma
        self._reward_fn = reward_fn
        self._stop_reward_fn = stop_reward_fn
        self._stop_coeff = stop_coeff

        self._n_epoch = 0  # epoch not very useful?

    def batches(self):
        raise NotImplementedError('A2C does not use batcher')

    def train_step(self):
        # forward pass of model
        self._net.train()
        log_dict = a2c_train_step(
            self._net, self._abstractor,
            self._train_batcher,
            self._opt, self._grad_fn,
            self._gamma, self._reward_fn,
            self._stop_reward_fn, self._stop_coeff
        )
        return log_dict

    def validate(self):
        return a2c_validate(self._net, self._abstractor, self._val_batcher)

    def checkpoint(self, *args, **kwargs):
        # explicitly use inherited function in case I forgot :)
        return super().checkpoint(*args, **kwargs)

    def terminate(self):
        pass  # No extra processs so do nothing
