import stanza
import numpy as np
from datasets import load_from_disk
from torch.utils.data import DataLoader
from dl_utils.misc import check_dir, set_experiment_dir
from datasets import load_dataset
from tqdm import tqdm
import os
from copy import deepcopy
import re
import torch
from torch.optim import AdamW
from parse import RootParseNode, FlatRootParseNode
from models.merge_bart import MergeBart
from utils import rouge_from_multiple_refs, TreeError
import argparse
import string
import nltk
nltk.download('punkt')
from os.path import join


parser = argparse.ArgumentParser()
parser.add_argument('--small', action='store_true')
parser.add_argument('--recompute-dset', action='store_true')
parser.add_argument('--buffer-layers', type=int, default=0)
parser.add_argument('--n-contract', type=int, default=1000)
parser.add_argument('--bs', type=int, default=2)
parser.add_argument('--n-epochs', type=int, default=1)
parser.add_argument('--start-from', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-6)
parser.add_argument('--verbose-enc', action='store_true')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--use-trees', action='store_true')
parser.add_argument('--is-test','-t', action='store_true')
parser.add_argument('--run-checks', action='store_true')
parser.add_argument('--truncate-inputs', type=int, default=-1)
parser.add_argument('--expname', type=str, default='tmp')
parser.add_argument('--reload-from', type=str)
parser.add_argument('--expdir-prefix', type=str, default='.')
ARGS = parser.parse_args()

set_experiment_dir(expdir:=join(ARGS.expdir_prefix, 'experiments', ARGS.expname), overwrite=ARGS.overwrite, name_of_trials=join(ARGS.expdir_prefix,'experiments/tmp'))
torch.manual_seed(0)

chkpt = 'lucadiliello/bart-small' if ARGS.small else 'kabita-choudhary/finetuned-bart-for-conversation-summary'
mb = MergeBart(chkpt, buffer_layers=ARGS.buffer_layers, verbose=ARGS.verbose_enc, n_contract=ARGS.n_contract, run_checks=ARGS.run_checks)
if ARGS.reload_from is not None:
    mb.load_state_dict(torch.load(join(ARGS.expdir_prefix, 'experiments', ARGS.reload_from, 'checkpoints','best.pt')))
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
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
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

def get_ss_inputs(dpoint_dict, use_trees):
    gt_summs = dpoint_dict['Recap']
    if use_trees:
        token_ids, trees = preproc('\n'.join(dpoint_dict['Transcript']))
    else:
        token_ids = torch.tensor(mb.tokenizer('\n'.join(dpoint_dict['Transcript']))['input_ids']).cuda()
        trees = None
    epname = ''.join(w[0].lower() for w in dpoint_dict['Show Title'].split()) + '-' + dpoint_dict['Episode Number']
    return epname, token_ids, trees, gt_summs

def hf_preproc(inputs):
    outputs = mb.tokenizer(['\n'.join(t) for t in inputs['Transcript']])
    del outputs['attention_mask'] # will be computed dynamically in loader
    # I'm taking the different list items to be parts of the same summ, not different summs?
    outputs['labels'] = mb.tokenizer(['\n'.join(t) for t in inputs['Recap']])['input_ids']
    return outputs

def padded_to_max(x):
    xlens = [len(item) for item in x]
    pad_len = max(xlens)
    padded = torch.tensor([item+[0]*(pad_len - len(item)) for item in x])
    pad_mask = torch.arange(pad_len).tile(len(x),1) < torch.tensor(xlens).unsqueeze(1)
    pad_mask = pad_mask.int()
    return padded.cuda(), pad_mask.cuda()

def collate_and_pad(batch):
    padded_input_ids, attn_mask = padded_to_max([item['input_ids'] for item in batch])
    padded_labels, loss_mask = padded_to_max([item['labels'] for item in batch])
    return padded_input_ids, attn_mask, padded_labels, loss_mask

def epname_from_dpoint(x):
    return ''.join(w[0].lower() for w in x['Show Title'].split())+'-'+x['Episode Number']

dsets = {}
check_dir('datasets')
for split in ('train','validation', 'test'):
    cachepath = 'datasets/trainset-tokenized' if split=='train' else f'datasets/{split}set'
    if ARGS.recompute_dset or not os.path.exists(cachepath):
        if 'raw_dset' not in locals().keys():
            raw_dset = load_dataset("YuanPJ/summ_screen", 'tms')
        dsets[split] = raw_dset[split]
        if split=='train':
            dsets[split] = raw_dset[split].map(hf_preproc, batched=True).remove_columns(raw_dset[split].features)
        dsets[split].save_to_disk(cachepath)
    else:
        print(f'Loading {split}set')
        dsets[split] = load_from_disk(cachepath)

trainloader = DataLoader(dsets['train'], batch_size=ARGS.bs, shuffle=True, collate_fn=collate_and_pad)
mb.cuda()
opt = AdamW(mb.parameters(), lr=ARGS.lr)
epoch_loss = 0
check_dir(chkpt_dir:=join(expdir, 'checkpoints'))

def run_inference(dset, dname, ndpoints=None):
    print(f'running inference on {dname} up to {ndpoints} dpoints')
    with torch.no_grad():
        #mb.eval()
        avg_rouges = np.zeros(4)
        avg_perp = 0
        check_dir(gendir:=join(expdir, f'{dname}_gens'))
        for i, dpoint in enumerate(pbar:=tqdm(dset)):
            nl_input = '\n'.join(dpoint['Transcript'])
            tokked = mb.tokenizer(nl_input)
            tokked_labels = torch.tensor([mb.tokenizer('\n'.join(dpoint['Recap'])).input_ids]).cuda()
            input_ids = torch.tensor([tokked.input_ids]).cuda()
            attn_mask = torch.tensor([tokked.attention_mask]).cuda()
            #genned, enc_out = mb.generate(input_ids, attn_mask, min_len=100, labels=tokked_labels)
            genned, enc_out = mb.generate(input_ids, attn_mask, min_len=100)
            perp = mb(input_ids=input_ids, attention_mask=attn_mask, labels=tokked_labels, encoder_outputs=enc_out)
            avg_perp = ((avg_perp*i)+perp) / (i+1)
            text_pred = mb.tokenizer.decode(genned)
            if i%10 == 0:
                print(text_pred)
            epname = epname_from_dpoint(dpoint)
            with open(join(gendir, f'{epname}.txt'), 'w') as f:
                f.write(text_pred)
            rs = rouge_from_multiple_refs(text_pred, dpoint['Recap'], False, False)
            avg_rouges = ((avg_rouges*i)+rs) / (i+1)
            pbar.set_description(f'r1: {avg_rouges[0]:.4f}  r2: {avg_rouges[1]:.4f}  rl: {avg_rouges[2]:.4f}  rlsum: {avg_rouges[3]:.4f}  perplexity: {avg_perp:.4f}')
            #for n,val in display_rouges(rs):
                #print(f'{n}: {val}')
            if ARGS.is_test and i==3:
                break
            if ndpoints is not None and (i == ndpoints):
                break
    return avg_rouges, avg_perp

bestr2 = 0
best_perp = np.inf
patience = 0
for epoch in range(ARGS.n_epochs):
    print(f'Epoch: {epoch}')
    mb.train()
    for i, (input_ids, attn_mask, labels, labels_mask) in enumerate(pbar:=tqdm(trainloader)):
        if i<ARGS.start_from:
            continue
        if ARGS.truncate_inputs != -1:
            input_ids = input_ids[:,:ARGS.truncate_inputs]
            attn_mask=attn_mask[:,:ARGS.truncate_inputs]
        loss = mb(input_ids=input_ids, attention_mask=attn_mask, labels=labels, decoder_attention_mask=labels_mask)
        loss.backward()
        opt.step()
        opt.zero_grad()
        epoch_loss = (i*epoch_loss + loss.item())/(i+1)
        pbar.set_description(f'loss: {loss.item():.4f} epoch loss: {epoch_loss:.4f}')
        if ARGS.is_test and i==1:
            break

    new_rouges, new_perp = run_inference(dsets['validation'], 'val', 300)
    rimproved, pimproved = False, False
    if new_rouges[1] > bestr2:
        print(f'rouge 2 improved from {bestr2:.5f} to {new_rouges[1]:.5f}, ', end='')
        bestr2 = new_rouges[1]
        rimproved = True
    if new_perp < best_perp:
        print(f'perplexity improved from {best_perp:.5f} to {new_perp:.5f}, ', end='')
        best_perp = new_perp
        pimproved = True
    if rimproved or pimproved:
        print('resetting patience to 0')
        patience = 0
        torch.save(mb.state_dict(), join(chkpt_dir, 'best.pt'))
    else:
        patience += 1
        print(f'r2 of {new_rouges[1]:.5f} unimproved from best of {bestr2:.5f}')
        print(f'perplexity of {new_perp:.5f} unimproved from {best_perp:.4f}')
        print(f'incrementing patience to {patience}')
    if patience == 2:
        print('patience has reached tolerance of 5, stopping training')
        break

if ARGS.n_epochs != 0:
    print('reloading from best checkpoint')
    mb.load_state_dict(torch.load(join(chkpt_dir,'best.pt')))
run_inference(dsets['test'], 'test')
