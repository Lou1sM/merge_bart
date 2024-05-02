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

m = SyntacticPoolingEncoder()
epname_list = ['oltl-10-18-10']
tokenizer  = AutoTokenizer.from_pretrained('kabita-choudhary/finetuned-bart-for-conversation-summary')
for en in epname_list:
    with open(f'../amazon_video/SummScreen/transcripts/{en}.json') as f:
        trans = json.load(f)['Transcript']

    trans_segment = trans
    full_text = '\n'.join(trans_segment)
    full_text = full_text.replace('é','e').replace('."', '. "').replace(',"', ', "').replace('?"', '? "')#.replace('"?', '" ?')
    #for k,v in contractions.items():
        #full_text = full_text.replace(k,v)
        #full_text = full_text.replace(k[0].upper()+k[1:],v[0].upper()+v[1:])
    all_anchors = []
    trees = []
    token_ids = torch.tensor(tokenizer(full_text)['input_ids'])
    #remaining_word_toks = tokenizer._tokenizer.encode_batch([full_text], add_special_tokens=False)[0].tokens
    remaining_word_toks = [tokenizer.decode(t) for t in tokenizer(full_text,add_special_tokens=False).input_ids]
    #remaining_token_ids = [t.replace('Ġ',' ').replace('Ċ','\n').replace('ÃŃ','í') for t in remaining_word_toks]
    if not ( ''.join(remaining_word_toks).replace('Ġ',' ').replace('Ċ','\n') == full_text):
        breakpoint()
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
    doc = nlp(full_text)
    for i,sent in enumerate(doc.sentences):
        word_toks_to_match = ''
        matching_word_toks = []
        while not equals_mod_whitespace(word_toks_to_match, sent.text):
            new = remaining_word_toks.pop(0).replace('Ġ',' ').replace('Ċ','\n')
            word_toks_to_match += new
            matching_word_toks.append(new)
        new_tree = RootParseNode(sent, matching_word_toks)
        trees.append(new_tree)

    m(input_ids=token_ids, trees=trees)
