import stanza
from dl_utils.misc import check_dir, set_experiment_dir
from datasets import load_dataset
from tqdm import tqdm
import os
import pickle
from copy import deepcopy
import re
import torch
from torch.optim import AdamW
from models.syntactic_pooling_encoder import RootParseNode, FlatRootParseNode
from models.merge_bart import MergeBart
from utils import rouge_from_multiple_refs, display_rouges, TreeError
import argparse
import string
from os.path import join


parser = argparse.ArgumentParser()
parser.add_argument('--small', action='store_true')
parser.add_argument('--recompute-dset', action='store_true')
parser.add_argument('--disallow-drops', action='store_true')
parser.add_argument('--n-train', type=int, default=3)
parser.add_argument('--n-test', type=int, default=3)
parser.add_argument('--n-epochs', type=int, default=1)
parser.add_argument('--start-from', type=int, default=1)
parser.add_argument('--verbose-enc', action='store_true')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--expname', type=str, default='tmp')
parser.add_argument('--reload-from', type=str)
ARGS = parser.parse_args()

set_experiment_dir(expdir:=join('experiments',ARGS.expname), overwrite=ARGS.overwrite, name_of_trials='experiments/tmp')
torch.manual_seed(0)

chkpt = 'lucadiliello/bart-small' if ARGS.small else 'kabita-choudhary/finetuned-bart-for-conversation-summary'
mb = MergeBart('syn-pool', chkpt, disallow_drops=ARGS.disallow_drops, verbose=ARGS.verbose_enc)
if ARGS.reload_from is not None:
    mb.load_state_dict(torch.load(join('experiments', ARGS.reload_from, 'checkpoints','latest.pt')))
failed = 0

def preproc(input_text):
    global failed
    trees = []
    # remove '\r\n' when it occurs in the middle of lines, as then parser
    # ignores but tokenizer does not, so they get out of sync
    input_text = re.sub(r'\r\n(?!(\[|[A-Z][a-z]*: ))', '', input_text)
    input_text = input_text.replace('\r\n','\n').replace('\n\r','\n')
    input_text = ''.join(c for c in input_text if c in string.printable)
    token_ids = [mb.tokenizer.bos_token_id]
    assert 'Ġ' not in input_text
    assert 'Ċ' not in input_text
    doc = nlp(input_text)
    for i,sent in enumerate(doc.sentences):

        matching_toks = mb.tokenizer(sent.text,add_special_tokens=False).input_ids
        matching_word_toks = [mb.tokenizer.decode(t) for t in matching_toks]
        token_ids += matching_toks
        try:
            new_tree = RootParseNode(sent, matching_word_toks)
        except TreeError:
            failed += 1
            print(failed, 'failed tree so far')
            new_tree = FlatRootParseNode(sent, matching_word_toks)
        assert new_tree.span_size == len(matching_word_toks)
        trees.append(new_tree)

    token_ids = torch.tensor(token_ids + [mb.tokenizer.eos_token_id]).cuda()
    assert sum(t.span_size for t in trees) == len(token_ids) - 2
    return token_ids, trees

def get_ss_inputs(dpoint_dict):
    #gt_summs = [v for k,v in dpoint_dict.items() if k in ('soapcentral_condensed','tvdb','tvmega_recap')]
    gt_summs = dpoint_dict['Recap']
    token_ids, trees = preproc('\n'.join(dpoint_dict['Transcript']))
    epname = ''.join(w[0].lower() for w in dpoint_dict['Show Title'].split()) + '-' + dpoint_dict['Episode Number']
    return epname, token_ids, trees, gt_summs

dset = {}
raw_dset = load_dataset("YuanPJ/summ_screen", 'tms')
for split in ['train', 'test']:
    n = ARGS.n_train if split=='train' else ARGS.n_test
    if ARGS.recompute_dset or not os.path.exists(f'datasets/{split}set-{n}dpoints.pkl'):
        nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
        if n==-1:
            n = len(raw_dset[split])
        dset[split] = [get_ss_inputs(raw_dset[split][i]) for i in range(n)]
        print(f'Computing {split}set from scratch for {n} dpoints')
        with open(f'datasets/{split}set-{n}dpoints.pkl', 'wb') as f:
            pickle.dump(dset[split], f)
    else:
        print(f'Loading {split}set')
        with open(f'datasets/{split}set-{n}dpoints.pkl', 'rb') as f:
            dset[split] = pickle.load(f)
mb.cuda()
opt = AdamW(mb.parameters(), lr=1e-6)
epoch_loss = 0
check_dir(chkpt_dir:=join(expdir, 'checkpoints'))
for epoch in range(ARGS.n_epochs):
    print(f'Epoch: {epoch}')
    mb.train()
    for i, (epname, token_ids, trees, gt_summs) in enumerate(pbar:=tqdm(dset['train'])):
        if i<ARGS.start_from:
            continue
        labelss = [torch.tensor(mb.tokenizer(g).input_ids).cuda().unsqueeze(0) for g in gt_summs]
        copied_trees = [deepcopy(t) for t in trees]
        # train together on all labels for the same input to minimize encoder fwds
        loss = mb(token_ids, trees=copied_trees, labelss=labelss)
        loss.backward()
        #befores = [p.detach().cpu().clone() for p in mb.model.parameters()]
        opt.step()
        opt.zero_grad()
        #afters = [p.detach().cpu().clone() for p in mb.model.parameters()]
        normed_loss = loss.item()/len(gt_summs) # because add for different summs
        epoch_loss = (i*epoch_loss + normed_loss)/(i+1)
        pbar.set_description(f'loss: {normed_loss:.3f} epoch loss: {epoch_loss:.3f}')
        torch.save(mb.state_dict(), join(chkpt_dir, 'latest.pt'))
        #assert ( all([(b!=a).any() for b,a in zip(befores,afters)]))

torch.save(mb.state_dict(), join(chkpt_dir, 'best.pt'))
check_dir(join(expdir, 'test_gens'))
with torch.no_grad():
    #mb.eval()
    for epname, token_ids, trees, ref_summs in dset['test']:
        copied_trees = [deepcopy(t) for t in trees]
        genned = mb.generate(token_ids, trees=copied_trees, min_len=100)
        text_pred = mb.tokenizer.decode(genned)
        print(text_pred)
        with open(join(expdir, 'test_gens', f'{epname}.txt'), 'w') as f:
            f.write(text_pred)
        rs = rouge_from_multiple_refs(text_pred, ref_summs, False, False)
        for n,val in display_rouges(rs):
            print(f'{n}: {val}')
