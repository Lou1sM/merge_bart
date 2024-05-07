import stanza
from datasets import load_from_disk
from copy import deepcopy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
import re
import torch
from torch.optim import AdamW
from models.syntactic_pooling_encoder import SyntacticPoolingEncoder, RootParseNode
from models.merge_bart import MergeBart
from contractions import contractions
from utils import equals_mod_whitespace, rouge_from_multiple_refs, display_rouges
import argparse
import pandas as pd
import string


parser = argparse.ArgumentParser()
parser.add_argument('--small', action='store_true')
parser.add_argument('--n-train', type=int, default=3)
parser.add_argument('--n-test', type=int, default=3)
ARGS = parser.parse_args()

torch.manual_seed(0)

chkpt = 'lucadiliello/bart-small' if ARGS.small else 'kabita-choudhary/finetuned-bart-for-conversation-summary'
nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
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

def get_model_inputs(epname):
    with open(f'../amazon_video/SummScreen/transcripts/{epname}.json') as f:
        trans_text = '\n'.join(json.load(f)['Transcript'])

    with open(f'../amazon_video/SummScreen/summaries/{epname}.json') as f:
        gt_summs = json.load(f)

    token_ids, trees = preproc(trans_text)
    return token_ids, trees, gt_summs

#trainset = load_from_disk('../amazon_video/SummScreen/cached_tokenized/nocaptions_test_10dps_train_cache')
treestrainset = load_from_disk('../amazon_video/SummScreen/cached_tokenized/nocaptions_test_10dps_train_cache')
#testset = load_from_disk('../amazon_video/SummScreen/cached_tokenized/nocaptions_test_10dps_train_cache')
df = pd.read_csv('../amazon_video/dset_info.csv', index_col=0)
trainepnames = df.loc[df['split']=='train'].index
testepnames = df.loc[df['split']=='test'].index
mb = MergeBart('syn-pool', chkpt)
mb.cuda()
opt = AdamW(mb.parameters(), lr=1e-4)
for en in trainepnames[:ARGS.n_train]:
    mb.train()
    token_ids, trees, gt_summs = get_model_inputs(en)
    for k, gts in gt_summs.items():
        if k in ('soapcentral_condensed','tvdb','tvmega_recap'):
            print(f'training on summ from {k}')
            gt_toks = torch.tensor(mb.tokenizer(gts).input_ids).cuda()
            copied_trees = [deepcopy(t) for t in trees]
            outputs = mb(token_ids, trees=copied_trees, labels=gt_toks.unsqueeze(0))
            outputs.loss.backward()
            opt.zero_grad(); opt.step()
with torch.no_grad():
    mb.eval()
    for en in testepnames[:ARGS.n_test]:
        token_ids, trees, gt_summs = get_model_inputs(en)
        genned = mb.generate(token_ids, trees=trees, min_len=10)
        print(genned)
        text_pred = mb.tokenizer.decode(genned)
        refs = [v for k,v in gt_summs.items() if k in ('soapcentral_condensed','tvdb','tvmega_recap')]
        rs = rouge_from_multiple_refs(text_pred, refs, False, False)
        for n,val in display_rouges(rs):
            print(f'{n}: {val}')
    #for nc in [700,900,1100,1300]:
        #mb.model.encoder.n_contract = 1000
        #copied_trees = [deepcopy(t) for t in trees]
        #print(mb.tokenizer.decode(genned, cleanup_special_tokens=True))
