from transformers.models.bart.modeling_bart import BartEncoder
from utils import strip_equals, equals_mod_whitespace
from stanza.models.constituency.parse_tree import Tree
import stanza
import json
import re
import math
import torch
import torch.nn as nn
#from transformers.modeling_outputs import BaseModelOutput
from transformers import BartConfig
#from ...modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa
from transformers import AutoTokenizer
import nltk
#import re
nltk.download('punkt')



class SyntacticPoolingEncoder(BartEncoder):
    def __init__(self):
        default_bart_config = BartConfig()
        super().__init__(default_bart_config)

    def forward(self, input_ids, trees):
        r"""attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. - 1 means **not masked**, - 0 means **masked**.
        """

        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        hidden_states_nop = inputs_embeds # don't add embed_pos now, will do so in the for loop, 'nop' means no positional embeds
        hidden_states_nop = self.layernorm_embedding(hidden_states_nop)
        hidden_states_nop = nn.functional.dropout(hidden_states_nop, p=self.dropout, training=self.training)

        n_to_drop_at_each_layer = 500

        for idx, encoder_layer in enumerate(self.layers):
            running_span_sizes = [sum(t.span_size for t in trees[:i]) for i in range(len(trees)+1)]
            if not ( running_span_sizes[-1] == len(hidden_states_nop) - 2): # -2 for bos and eos
                breakpoint()
            if (contracting:=len(hidden_states_nop) > 1024):
                pseudo_batch_size = math.ceil(hidden_states_nop.shape[0]/1024) # assuming hidden_states shape is (n_tokens, emb_size)
                padding_needed = pseudo_batch_size*1024 - len(hidden_states_nop)
                padded = torch.cat([hidden_states_nop] + [torch.zeros_like(hidden_states_nop[:1]) for _ in range(padding_needed)])
                hidden_states_nop = padded.reshape(-1,1024,hidden_states_nop.shape[1])
                attention_mask = torch.ones_like(hidden_states_nop[:,:,0])
                attention_mask[-1,-padding_needed:] = 0
                attention_mask = _prepare_4d_attention_mask_for_sdpa(attention_mask, inputs_embeds.dtype)
            else:
                attention_mask = None
                hidden_states_nop = hidden_states_nop.unsqueeze(0)
                padding_needed = 0
            embed_pos = self.embed_positions(hidden_states_nop[:,:,-1])
            hidden_states = hidden_states_nop + embed_pos
            layer_outputs = encoder_layer(hidden_states, attention_mask=attention_mask, layer_head_mask=None, output_attentions=True)
            hidden_states_nop = layer_outputs[0] - embed_pos
            # output from HF is (bsz, n_heads, seqlen, seqlen), take sum over all heads and all query tokens except the final padding
            layer_attns = layer_outputs[1].transpose(0,1).flatten(1,2)
            if contracting: # exclude final padding
                #layer_attns = layer_attns[:,:-padding_needed,:-padding_needed]
                hidden_states_nop = hidden_states_nop.flatten(0,1)[:-padding_needed]
                up_to_last_attns = layer_outputs[1][:-1].transpose(1,3).flatten(0,1).sum(2).sum(1)
                last_attns = layer_outputs[1][-1,:,:-padding_needed,:-padding_needed].sum(axis=1).sum(axis=0)
                layer_attns = torch.cat([up_to_last_attns, last_attns])
                assert torch.allclose(layer_attns[133], layer_outputs[1][0,:,:,133].sum())
                assert torch.allclose(layer_attns[757], layer_outputs[1][0,:,:,757].sum())
                assert torch.allclose(layer_attns[22], layer_outputs[1][0,:,:,22].sum())
            else:
                layer_attns = layer_attns.sum([0,1])
            layer_attns = layer_attns[1:-1] # cut off bos and eos

            print(f'sum of trees lengths: {running_span_sizes[-1]}\t hidden_states_nop length: {hidden_states_nop.shape}\tpadding needed: {padding_needed}\tattns: {layer_attns.shape}\tcontracting: {contracting}')
            if not ( ( running_span_sizes[-1] == len(layer_attns))):
                breakpoint()
            if len(hidden_states_nop) > 1024:
                assert contracting
                attn_chunks = [layer_attns[running_span_sizes[i]:running_span_sizes[i+1]] for i in range(len(trees))]
                options = []
                tok_idx = 0
                for i, (t,ac) in enumerate(zip(trees, attn_chunks)):
                    if not ( t.span_size == ac.shape[0]):
                        breakpoint()
                    new_drop_options, new_merge_options = t.determine_drops_and_merges(ac)
                    options += [(tok_idx+d.offset, 'drop', d, dcost, i) for d,dcost in new_drop_options]
                    options += [(tok_idx+m.offset, 'merge', m, mcost, i) for m,mcost in new_merge_options]
                    tok_idx += t.span_size

                assert set([x[2].span_size for x in options if x[1]=='drop']) == set([1])
                n_to_drop = min(n_to_drop_at_each_layer, len(hidden_states_nop)-1022)
                reductions = sorted(options, key=lambda x:x[3])[:n_to_drop] # hard-coding 300 for now, could do some fancy sampling thing
                reductions = sorted(reductions, key=lambda x:x[0])
                reduction_idxs = [x[0] for x in reductions]
                reduced_hiddens = [hidden_states_nop[0]] # start with just bos, will add others in for loop
                print(f'Layer: {idx}, Size: {hidden_states_nop.shape}')
                hidden_states_nop_inner = hidden_states_nop[1:-1] # cut off bos and eos
                n_to_exclude_after = 0
                option_idx = 0
                ts = lambda: sum(t.span_size for t in trees)
                prev = ts()
                assert prev==running_span_sizes[-1]
                prev_tok_idx = -1
                for i, hs in enumerate(hidden_states_nop_inner):
                    print(i)
                    if i in reduction_idxs:
                        tok_idx, red_type, node, _, tidx = reductions[option_idx]
                        #print(f'tok idx: {tok_idx}\ttree idx: {tidx}\toption idx: {option_idx}')
                        while option_idx<len(reductions) and reduction_idxs[option_idx]==i:
                            option_idx+=1
                        if prev_tok_idx==tok_idx:
                            breakpoint()
                        prev_tok_idx = tok_idx
                        if n_to_exclude_after>0:
                            n_to_exclude_after -= 1
                            print(f'merged {i} now skipping {red_type} because ntea is {n_to_exclude_after}')
                            continue
                        if not ( node in trees[tidx].descendents):
                            breakpoint()
                        assert tok_idx==i
                        if red_type == 'drop':
                            what_red_should_be = 1
                            if node.is_root:
                                print('dropping rootleaf')
                                assert trees[tidx].root==node
                                trees[tidx].root = 'DROPPED'
                            elif len(node.parent.children)==2: # replace the parent with the sibling
                                sibling = [n for n in node.parent.children if n!=node][0]
                                if node.parent.is_root:
                                    if not ( node.parent==trees[tidx].root):
                                        breakpoint()
                                    trees[tidx].root = sibling
                                    sibling.parent = trees[tidx]
                                    assert sibling.is_root
                                else:
                                    breakpoint()
                                    node.parent.parent.children = [sibling if c==node.parent else c for c in node.parent.parent.children]
                            else:
                                node.drop_non_root()

                        else:
                            n_to_exclude_after = node.span_size-1
                            what_red_should_be = n_to_exclude_after
                            print(f'merging node of size {node.span_size}')
                            node.merge()
                            merged = (layer_attns[i:i+n_to_exclude_after].unsqueeze(1)*hidden_states_nop_inner[i:i+n_to_exclude_after]).sum(0)/layer_attns[i:i+n_to_exclude_after].sum()
                            reduced_hiddens.append(merged)

                        if not ( ts() == prev-what_red_should_be):

                            breakpoint()
                        prev = ts()
                    else:
                        if n_to_exclude_after>0:
                            print(f'excluding because ntea is {n_to_exclude_after}')
                            n_to_exclude_after -= 1
                        else:
                            reduced_hiddens.append(hs)
                            print(f'appending, rh now of length {len(reduced_hiddens)}')

                reduced_hiddens.append(hidden_states_nop[-1]) # add eos
                hidden_states_nop = torch.stack(reduced_hiddens)
            else:
                assert not contracting

        return hidden_states_nop

class RootParseNode():
    def __init__(self, sent, word_toks):
        #word_tokens = [tokenizer.decode(t) for t in tokenizer(sent.text)['input_ids']]
        #prev_anchor_idx = 0
        #self.tok_split_idxs = []
        #for i,w in enumerate(word_tokens):
        #    if w.startswith(' ') or w in ['<s>','\n','[',']',':',';',]: # at an anchor
        #        if prev_anchor_idx < i-1: # something was split
        #            self.tok_split_idxs.append((prev_anchor_idx,i-1))
        #        prev_anchor_idx = i
        self.parse = sent.constituency
        assert self.parse.label=='ROOT'
        assert len(self.parse.children)==1
        #self.split_idxs = []
        #self.tokenizer = tokenizer
        #self.root = ParseNode(self.parse, self, 0, word_toks)
        self.root = ParseNode(self.parse, self, word_toks, 0)

    @property
    def span_size(self):
        if self.root=='DROPPED':
            return 0
        return self.root.span_size

    @property
    def descendents(self):
        if self.root=='DROPPED':
            return []
        return self.root.descendents

    def list_mergables(self):
        return [n for n in self.descendents if not n.is_leaf and n.child_idx==0 and all(c.is_leaf for c in n.children)]

    def list_droppables(self):
        if self.root=='DROPPED':
            return []
        return [c for c in self.root.children if c.is_leaf]

    def determine_drops_and_merges(self, attention_weights):
        try:
            possible_drops_costs = [(d, attention_weights[d.offset]) for d in self.list_droppables()]
        except:
            breakpoint()
        possible_merges_costs = [(m, merge_cost(attention_weights[m.offset:m.offset+m.span_size])) for m in self.list_mergables()]
        #overlap = [n[0] for n in possible_drops_costs if n[0] in [m[0] for m in possible_merges_costs]]
        #if len(overlap)>0:
            #breakpoint()
        return possible_drops_costs, possible_merges_costs

def merge_cost(merge_attns): # may also want to consider attention the mergees have for each other
    return merge_attns @ (1-merge_attns/merge_attns.sum())


def text_from_node_or_str(x):
    if isinstance(x, str):
        return x
    else:
        assert isinstance(x, Tree)
        joined_terminals = ' '.join([str(n)[1:-1].split()[1] for n in x.yield_preterminals()])
        return joined_terminals.replace(' , ',', ').replace(' ; ','; ').replace(' . ','. ')


class ParseNode():
    def __init__(self, stanza_node_or_text, parent, word_toks, child_idx):
        self.parent = parent
        self.toks = word_toks
        self.child_idx = child_idx
        #self.offset = offset
        self.is_orig_leaf = False
        if isinstance(stanza_node_or_text, str):
            self.text = stanza_node_or_text
            self.parse = None
            _children = self.toks if len(self.toks)>1 else []
        else:
            self.parse = stanza_node_or_text
            self.text = text_from_node_or_str(self.parse)
            n = self.parse

            while True:
                if len(n.children) == 0:
                    _children = [] if len(self.toks) == 1 else self.toks
                    break
                elif len(n.children) == 1: # ignore internal nodes with just one child
                    n = n.children[0]
                else:
                    _children = n.children
                    break

        cleaned_children = []
        num_to_exclude_after = 0
        for i,child in enumerate(_children):
            if num_to_exclude_after > 0:
                num_to_exclude_after -= 1
                continue
            combined_with_next_one = ''.join(text_from_node_or_str(c) for c in _children[i:i+2])
            if combined_with_next_one=='cannot': # treat separately because don't want to make 'can not', even though stanza parses it so
                cleaned_children.append(combined_with_next_one)
                num_to_exclude_after = 1
                continue
            cleaned_children.append(child)

        remaining_word_toks = list(self.toks) # make copy
        child_words = [text_from_node_or_str(c) for c in cleaned_children]
        child_toks = []
        for i,cw in enumerate(child_words):
            word_toks_to_match = ''
            matching_word_toks = []
            #while word_toks_to_match.lstrip(' ')!=cw.lstrip(' ') and word_toks_to_match.lstrip()!=cw.lstrip(' '): # need to lstrip cw because it can have leading spaceif this node is stanza leaf and had token children
            while not equals_mod_whitespace(word_toks_to_match, cw):
                new = remaining_word_toks.pop(0).replace('Ġ',' ').replace('Ċ','\n')
                word_toks_to_match += new
                matching_word_toks.append(new)
                #print(word_toks_to_match, cw, len(word_toks_to_match), len(cw))
                if len(word_toks_to_match.lstrip()) > len(cw.lstrip()):
                    breakpoint()
            if not ( matching_word_toks != []):
                breakpoint()
            child_toks.append(matching_word_toks)

        self.children = [ParseNode(c, self, cts, i) for i,(c,cts) in enumerate(zip(cleaned_children,child_toks))]
        if len(_children) == 0:
            self.children = []
            #self.is_leaf = True
            self.is_orig_leaf = True
            self.span_size = 1
        else:
            self.span_size = sum(c.span_size for c in self.children)
        if not ( strip_equals(''.join(self.toks), self.text)):
            breakpoint()
        assert self.span_size == len(self.toks)

    @property
    def is_leaf(self):
        return len(self.children)==0

    @property
    def is_root(self):
        return isinstance(self.parent, RootParseNode)

    def recompute_sibling_child_idxs(self):
        if self.is_root:
            return
        for s in self.parent.children[self.child_idx:]:
            s.child_idx -= 1
        assert self not in self.parent.children # should only be called after dropping
        #return self.parent.children.index(self)

    @property
    def offset(self):
        if self.is_root:
            return 0
        return self.parent.offset + sum(c.span_size for c in self.parent.children[:self.child_idx])

    @property
    def descendents(self):
        return [self] + sum([c.descendents for c in self.children],[])

    def merge(self):
        self.children = []
        reduction = self.span_size-1
        self.propagte_minus_span_size(reduction)

    def propagte_minus_span_size(self, reduction):
        p = self
        while isinstance(p, ParseNode):
            p.span_size -= reduction
            #p.child_idx = p.computed_child_idx()
            p = p.parent
        assert isinstance(p, RootParseNode)

    def drop_non_root(self):
        assert not self.is_root
        assert self.span_size == 1
        assert self.is_leaf
        assert len(self.parent.children)!=2
        self.parent.children.remove(self)
        self.parent.propagte_minus_span_size(1)
        self.recompute_sibling_child_idxs()

    #def __eq__(self, other):
        #return self.text==other.text and self.span_size==other.span_size and len(self.children)==len(other.children)

    def __repr__(self):
        toprint = ''
        for field in ('offset', 'span_size', 'text', 'child_idx'):
            if hasattr(self,field):
                try:
                    value = getattr(self, field)
                except Exception as e:
                    value = str(e) + str(type(e))
                toprint += f'{field}: {value}\n'
        if hasattr(self,'children'):
            toprint += f'num children: {len(self.children)}\t'
        return toprint
        #f'Offset: {self.offset}\nSpan size: {self.span_size}\n Num children: {len(self.children)} Child idx: {self.child_idx}\n' + \
        f'Preterminals: {"None" if self.parse is None else list(self.parse.yield_preterminals())}\nText: {self.text}\t'


if __name__ == '__main__':
    epname_list = ['oltl-10-18-10']
    tokenizer  = AutoTokenizer.from_pretrained('kabita-choudhary/finetuned-bart-for-conversation-summary')
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
    for en in epname_list:
        with open(f'../amazon_video/SummScreen/transcripts/{en}.json') as f:
            trans = json.load(f)['Transcript']

        all_anchors = []
        for tline in trans[106:]:
            if bool(maybe_match:=re.match(r'\w+:', tline)):
                start_of_speech = maybe_match.span()[1] + 1
                inner = tline[start_of_speech:]
            else:
                assert tline.startswith('[') and inner.endswith(']')
                inner = tline[1:-1]
            doc = nlp(inner)
            trees = [RootParseNode(sent, tokenizer) for sent in doc.sentences]
            for sent in doc.sentences:
                t = RootParseNode(sent, tokenizer)
                print(t.root.children[1].offset)
                drop_options, merge_options = t.determine_drops_and_merges(torch.rand(len(sent.tokens)))
                costs = torch.tensor([x[1] for x in drop_options]+[x[1] for x in merge_options])
                to_reduce = costs.topk(2).indices

                breakpoint()
