import stanza
import os
import pickle
from transformers import AutoTokenizer
import json
import re
import torch
from models.syntactic_pooling_encoder import SyntacticPoolingEncoder, RootParseNode
#import argparse
from contractions import contractions
from utils import equals_mod_whitespace


torch.manual_seed(0)

#parser = argparse.ArgumentParser()
#parser.add_argument('--recompute_trees', action='store_true')
#ARGS = parser.parse_args()

m = SyntacticPoolingEncoder()
epname_list = ['oltl-10-18-10']
tokenizer  = AutoTokenizer.from_pretrained('kabita-choudhary/finetuned-bart-for-conversation-summary')
for en in epname_list:
    with open(f'../amazon_video/SummScreen/transcripts/{en}.json') as f:
        trans = json.load(f)['Transcript']

    trans_segment = trans[5:200]
    full_text = '\n'.join(trans_segment)
    full_text = full_text.replace('é','e').replace('."', '. "').replace(',"', ', "')
    for k,v in contractions.items():
        full_text = full_text.replace(k,v)
        full_text = full_text.replace(k[0].upper()+k[1:],v[0].upper()+v[1:])
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
    doc = nlp(full_text)
    all_anchors = []
    if not os.path.isdir(epdir:=f'parse_trees/{en}'):
        os.makedirs(epdir)
    trees = []
    token_ids = torch.tensor(tokenizer(full_text)['input_ids'])
    remaining_word_toks = tokenizer._tokenizer.encode_batch([full_text], add_special_tokens=False)[0].tokens
    #remaining_word_toks = [w.replace('Ã©','é') for w in remaining_word_toks]
    remaining_token_ids = list(token_ids)
    if not ( ''.join(remaining_word_toks).replace('Ġ',' ').replace('Ċ','\n') == full_text):
        breakpoint()
    #remaining_word_ids = torch.tensor(tokenizer(full_text)).word_ids()
    for i,sent in enumerate(doc.sentences):
        word_toks_to_match = ''
        #if 'Vivian: Take a deep breath...' in sent.text:
            #breakpoint()
        matching_word_toks = []
        #print(sent.text, '\t&&&\t', ''.join(remaining_word_toks).replace('Ġ',' ').replace('Ċ','\n') )
        while not equals_mod_whitespace(word_toks_to_match, sent.text):
            new = remaining_word_toks.pop(0).replace('Ġ',' ').replace('Ċ','\n')
            word_toks_to_match += new
            matching_word_toks.append(new)
            #print(repr(word_toks_to_match.lstrip()), repr(sent.text.lstrip()), len(word_toks_to_match.lstrip()), len(sent.text.lstrip()))
        new_tree = RootParseNode(sent, matching_word_toks)
        trees.append(new_tree)
        with open(f'parse_trees/{en}/sent{i}.pkl','wb') as f:
            pickle.dump(new_tree, f)
        with open(f'parse_trees/{en}/sent{i}.pkl','rb') as f:
            treloaded = pickle.load(f)
        assert new_tree.parse==treloaded.parse

    m(input_ids=token_ids, trees=trees)
