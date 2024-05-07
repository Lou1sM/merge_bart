import stanza
from copy import deepcopy
import json
import re
import torch
from torch.optim import AdamW
from models.syntactic_pooling_encoder import RootParseNode
from models.merge_bart import MergeBart
from utils import equals_mod_whitespace, rouge_from_multiple_refs, display_rouges
import argparse
import pandas as pd
import string


parser = argparse.ArgumentParser()
parser.add_argument('--small', action='store_true')
parser.add_argument('--disallow-drops', action='store_true')
parser.add_argument('--n-train', type=int, default=3)
parser.add_argument('--n-test', type=int, default=3)
parser.add_argument('--n-epochs', type=int, default=1)
parser.add_argument('--verbose-enc', type=int, default=1)
ARGS = parser.parse_args()

torch.manual_seed(0)

nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
chkpt = 'lucadiliello/bart-small' if ARGS.small else 'kabita-choudhary/finetuned-bart-for-conversation-summary'
mb = MergeBart('syn-pool', chkpt, disallow_drops=ARGS.disallow_drops, verbose=ARGS.verbose_enc)

def preproc(input_text):
    trees = []
    # remove '\r\n' when it occurs in the middle of lines, as then parser
    # ignores but tokenizer does not, so they get out of sync
    input_text = re.sub(r'\r\n(?!(\[|[A-Z][a-z]*: ))', '', input_text)
    input_text = ''.join(c for c in input_text if c in string.printable)
    token_ids = torch.tensor(mb.tokenizer(input_text)['input_ids']).cuda()
    remaining_word_toks = [mb.tokenizer.decode(t) for t in mb.tokenizer(input_text,add_special_tokens=False).input_ids]
    assert 'Ġ' not in input_text
    assert 'Ċ' not in input_text
    doc = nlp(input_text)
    for i,sent in enumerate(doc.sentences):
        word_toks_to_match = ''
        matching_word_toks = []
        while not equals_mod_whitespace(word_toks_to_match, sent.text):
            new = remaining_word_toks.pop(0)
            word_toks_to_match += new
            matching_word_toks.append(new)
            if len(word_toks_to_match.strip())>len(sent.text.strip()):
                print(word_toks_to_match.strip(),sent.text.strip())
                breakpoint()

        new_tree = RootParseNode(sent, matching_word_toks)
        trees.append(new_tree)
    return token_ids, trees

def get_ss_inputs(epname):
    with open(f'../amazon_video/SummScreen/transcripts/{epname}.json') as f:
        trans_text = '\n'.join(json.load(f)['Transcript'])

    with open(f'../amazon_video/SummScreen/summaries/{epname}.json') as f:
        gt_summ_dict = json.load(f)

    gt_summs = [v for k,v in gt_summ_dict.items() if k in ('soapcentral_condensed','tvdb','tvmega_recap')]
    token_ids, trees = preproc(trans_text)
    return token_ids, trees, gt_summs

df = pd.read_csv('../amazon_video/dset_info.csv', index_col=0)
trainepnames = df.loc[df['split']=='train'].index[:ARGS.n_train]
testepnames = df.loc[df['split']=='test'].index[:ARGS.n_test]
trainset = [get_ss_inputs(en) for en in trainepnames] # list of (text, tree, list-of-summs)
testset = [get_ss_inputs(en) for en in testepnames]
mb.cuda()
opt = AdamW(mb.parameters(), lr=1e-5)
for epoch in range(ARGS.n_epochs):
    mb.train()
    for token_ids, trees, gt_summs in trainset:
        labelss = [torch.tensor(mb.tokenizer(g).input_ids).cuda().unsqueeze(0) for g in gt_summs]
        copied_trees = [deepcopy(t) for t in trees]
        # train together on all labels for the same input to minimize encoder fwds
        loss = mb(token_ids, trees=copied_trees, labelss=labelss)
        loss.backward()
        print(loss.item())
        befores = [p.detach().cpu().clone() for p in mb.model.parameters()]
        opt.step()
        opt.zero_grad()
        afters = [p.detach().cpu().clone() for p in mb.model.parameters()]
        if not ( all([(b!=a).any() for b,a in zip(befores,afters)])):
            breakpoint()
with torch.no_grad():
    #mb.eval()
    for token_ids, trees, ref_summs in testset:
        copied_trees = [deepcopy(t) for t in trees]
        genned = mb.generate(token_ids, trees=copied_trees, min_len=10)
        text_pred = mb.tokenizer.decode(genned)
        print(text_pred)
        rs = rouge_from_multiple_refs(text_pred, ref_summs, False, False)
        for n,val in display_rouges(rs):
            print(f'{n}: {val}')
