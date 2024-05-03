import stanza
from transformers import AutoTokenizer
import json
import re
import torch
from models.syntactic_pooling_encoder import SyntacticPoolingEncoder, RootParseNode
from models.merge_bart import MergeBart
from contractions import contractions
from utils import equals_mod_whitespace


torch.manual_seed(0)

m = SyntacticPoolingEncoder()
mb = MergeBart()
epname_list = ['oltl-10-18-10']
tokenizer  = AutoTokenizer.from_pretrained('kabita-choudhary/finetuned-bart-for-conversation-summary')
for en in epname_list:
    with open(f'../amazon_video/SummScreen/transcripts/{en}.json') as f:
        trans = json.load(f)['Transcript']

    trans_segment = trans
    full_text = '\n'.join(trans_segment)
    all_anchors = []
    trees = []
    token_ids = torch.tensor(tokenizer(full_text)['input_ids'])
    remaining_word_toks = [tokenizer.decode(t) for t in tokenizer(full_text,add_special_tokens=False).input_ids]
    assert 'Ġ' not in full_text
    assert 'Ċ' not in full_text
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
    #mb.generate(input_ids=token_ids, trees=trees)
