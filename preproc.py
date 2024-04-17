import stanza
import torch
import json
from transformers import AutoTokenizer
import nltk
import re
nltk.download('punkt')


class ParseTreeToTokens():
    def __init__(self, sent, tokenizer):
        word_tokens = [tokenizer.decode(t) for t in tokenizer(sent.text)['input_ids']]
        prev_anchor_idx = 0
        self.tok_split_idxs = []
        for i,w in enumerate(word_tokens):
            if w.startswith(' ') or w in ['<s>','\n','[',']',':',';',]: # at an anchor
                if prev_anchor_idx < i-1: # something was split
                    self.tok_split_idxs.append((prev_anchor_idx,i-1))
                prev_anchor_idx = i
        self.parse = sent.constituency
        assert self.parse.label=='ROOT'
        assert len(self.parse.children)==1
        self.split_idxs = []
        self.tokenizer = tokenizer
        self.root = ParseNode(self.parse, 'ROOT', 0, tokenizer)

    def list_mergables(self):
        return [n.children for n in self.descendents if not n.is_leaf and all(c.is_leaf for c in n.children)]

    def list_droppables(self):
        return [c for c in self.root.children if c.is_leaf]

    def determine_drops_and_merges(self, attention_weights):
        possible_drops_costs = [(d, attention_weights[d.offset]) for d in self.list_droppables()]
        possible_merges_costs = [(m, merge_cost(attention_weights[m[0].offset:m[0].offset+len(m)])) for m in self.list_mergables()]
        return possible_drops_costs, possible_merges_costs

    @property
    def descendents(self):
        return self.root.descendents

def merge_cost(merge_attns):
    return merge_attns @ (1-merge_attns/merge_attns.sum())

class ParseNode():
    def __init__(self, stanza_node_or_text, parent, child_idx, tokenizer):
        self.parent = parent
        self.tokenizer = tokenizer
        self.is_root = self.parent=='ROOT'
        self.child_idx = child_idx
        self.is_leaf = False
        self.is_orig_leaf = False
        if isinstance(stanza_node_or_text, str):
            self.text = stanza_node_or_text
            self.parse = None
            _children = []
        else:
            self.parse = stanza_node_or_text
            self.text = ' '.join([str(n)[1:-1].split()[1] for n in self.parse.yield_preterminals()])
            n = self.parse
            while True:
                if len(n.children) == 0:
                    toks = self.tokenizer(self.text, add_special_tokens=False)['input_ids']
                    _children = [] if len(toks) == 1 else toks
                    break
                elif len(n.children) == 1: # ignore internal nodes with just one child
                    n = n.children[0]
                else:
                    _children = n.children
                    break

        self.children = [ParseNode(c, self, i, tokenizer) for i,c in enumerate(_children)]
        if len(_children) == 0:
            self.children = []
            self.is_leaf = True
            self.is_orig_leaf = True
            self.span_size = 1
        else:
            self.span_size = sum(c.span_size for c in self.children)

    @property
    def offset(self):
        if self.is_root:
            return 0
        return self.parent.offset + sum(c.span_size for c in self.parent.children[:self.child_idx])

    @property
    def descendents(self):
        return [self] + sum([c.descendents for c in self.children],[])

    def merge(self):
        self.is_leaf = True
        self.span_size = 1 #sum(c.span_size for c in self.children)

    def __repr__(self):
        return f'Offset: {self.offset}\nSpan size: {self.span_size}\n Num children: {len(self.children)} Child idx: {self.child_idx}\n' + \
        f'Preterminals: {list(self.parse.yield_preterminals())}\n'

        #frontier = [self.parse.children[0]]
        #while True:
        #    offset = 0
        #    next_frontier = []
        #    merges_at_this_level = []
        #    all_leaves = True
        #    for new in frontier:
        #        print(new if new.is_leaf() else list(new.yield_preterminals()))
        #        maybe_append = new.children
        #        if len(maybe_append)==0:
        #            assert new.is_leaf()
        #            next_frontier.append(new)
        #            offset += 1
        #        else:
        #            next_frontier += maybe_append
        #            all_leaves = False
        #            if len(maybe_append) > 1:
        #                merges_at_this_level.append((offset,offset+len(maybe_append)-1))
        #        offset += len(maybe_append)
        #    print(merges_at_this_level)
        #    if all_leaves:
        #        break
        #    frontier = next_frontier
        #    if len(merges_at_this_level) > 0: # can be internal nodes with just one child, leading to no splits at that level
        #        self.split_idxs.append(merges_at_this_level)


if __name__ == '__main__':
    epname_list = ['oltl-10-18-10']
    tokenizer  = AutoTokenizer.from_pretrained('kabita-choudhary/finetuned-bart-for-conversation-summary')
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
    for en in epname_list:
        with open(f'../amazon_video/SummScreen/transcripts/{en}.json') as f:
            trans = json.load(f)['Transcript']

        #sents = nltk.sent_tokenize('\n'.join(trans))
        all_anchors = []
        for tline in trans[106:]:
            if bool(maybe_match:=re.match(r'\w+:', tline)):
                start_of_speech = maybe_match.span()[1] + 1
                inner = tline[start_of_speech:]
            else:
                assert tline.startswith('[') and inner.endswith(']')
                inner = tline[1:-1]
            doc = nlp(inner)
            trees = [ParseTreeToTokens(sent, tokenizer) for sent in doc.sentences]
            for sent in doc.sentences:
                t = ParseTreeToTokens(sent, tokenizer)
                print(t.root.children[1].offset)
                drop_options, merge_options = t.determine_drops_and_merges(torch.rand(len(sent.tokens)))
                costs = torch.tensor([x[1] for x in drop_options]+[x[1] for x in merge_options])
                to_reduce = costs.topk(2).indices

                breakpoint()
