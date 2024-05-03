import stanza
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
import re
import torch
from models.syntactic_pooling_encoder import SyntacticPoolingEncoder, RootParseNode
from models.merge_bart import MergeBart
from contractions import contractions
from utils import equals_mod_whitespace


torch.manual_seed(0)

#m = SyntacticPoolingEncoder(load_from='kabita-choudhary/finetuned-bart-for-conversation-summary')
#tokenizer  = AutoTokenizer.from_pretrained('kabita-choudhary/finetuned-bart-for-conversation-summary')
#loaded_bartgen  = AutoModelForSeq2SeqLM.from_pretrained('kabita-choudhary/finetuned-bart-for-conversation-summary')
#toks = torch.tensor([tokenizer("stately plump buck mulligan came from").input_ids])
#loaded_bartgen.generate(toks)
mb = MergeBart('syn-pool', 'kabita-choudhary/finetuned-bart-for-conversation-summary')
epname_list = ['oltl-10-18-10']
for en in epname_list:
    with open(f'../amazon_video/SummScreen/transcripts/{en}.json') as f:
        trans = json.load(f)['Transcript']

    trans_segment = trans
    full_text = '\n'.join(trans_segment)
    all_anchors = []
    trees = []
    token_ids = torch.tensor(mb.tokenizer(full_text)['input_ids'])
    remaining_word_toks = [mb.tokenizer.decode(t) for t in mb.tokenizer(full_text,add_special_tokens=False).input_ids]
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
        #toks = torch.tensor([tokenizer(sent.text).input_ids])
        toks = torch.tensor([mb.tokenizer.bos_token_id]+[mb.tokenizer(wt,add_special_tokens=False).input_ids[0] for wt in matching_word_toks] + [mb.tokenizer.eos_token_id])
        #genned = mb.generate(toks, trees=[new_tree])
        trees.append(new_tree)

    #m(input_ids=token_ids, trees=trees)
    genned = mb.generate(token_ids, trees=trees, min_len=100)
    breakpoint()
    print(mb.tokenizer.decode(genned[0], cleanup_special_tokens=True))
