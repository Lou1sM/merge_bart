from transformers.models.bart.modeling_bart import BartEncoder
from utils import text_from_node_or_str, equals_mod_whitespace, TreeError
import re
import math
import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa
import nltk
#import re
nltk.download('punkt')



class SyntacticPoolingEncoder(BartEncoder):
    def __init__(self, cfg, n_contract, disallow_drops, verbose):
        super().__init__(cfg)
        self.nz = self.config.d_model
        self.seq_len = self.config.max_position_embeddings
        self.n_contract = n_contract
        self.disallow_drops = disallow_drops
        self.verbose = verbose

    def batchify(self, unbatched_hidden_states_nop):
        self.contracting = len(unbatched_hidden_states_nop) > self.seq_len
        if (self.contracting):
            self.pseudo_batch_size = math.ceil(unbatched_hidden_states_nop.shape[0]/self.seq_len) # assuming hidden_states shape is (n_tokens, emb_size)
            self.chunk_size = math.ceil(len(unbatched_hidden_states_nop)/ self.pseudo_batch_size)
            self.padding_needed = self.pseudo_batch_size*self.chunk_size - len(unbatched_hidden_states_nop)
            padded = torch.cat([unbatched_hidden_states_nop, torch.zeros(self.padding_needed, self.nz).to(unbatched_hidden_states_nop.device)])
            self.hidden_states_nop = padded.reshape(self.pseudo_batch_size,self.chunk_size,self.nz)
            attention_mask = torch.ones_like(self.hidden_states_nop[:,:,0])
            attention_mask[-1,-self.padding_needed:] = 0
            self.attention_mask = _prepare_4d_attention_mask_for_sdpa(attention_mask, torch.float32)
        else:
            self.pseudo_batch_size = 1
            self.chunk_size = len(unbatched_hidden_states_nop)
            self.attention_mask = None
            self.hidden_states_nop = unbatched_hidden_states_nop.unsqueeze(0)
            self.padding_needed = 0

    def forward(self, input_ids, trees, **kwargs):
        if input_ids.ndim == 2: # can't do batching atm
            assert input_ids.shape[0]==1 # there could be a dummy batch dim
            input_ids = input_ids.squeeze(0)
        else:
            assert input_ids.ndim==1

        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        unbatched_hidden_states_nop = inputs_embeds #'nop' means no pos embeds, for now
        unbatched_hidden_states_nop = self.layernorm_embedding(unbatched_hidden_states_nop)
        unbatched_hidden_states_nop = nn.functional.dropout(unbatched_hidden_states_nop, p=self.dropout, training=self.training)

        for idx, encoder_layer in enumerate(self.layers):
            running_span_sizes = [sum(t.span_size for t in trees[:i]) for i in range(len(trees)+1)]
            assert unbatched_hidden_states_nop.ndim == 2
            n_toks = unbatched_hidden_states_nop.shape[0]
            if not ( ( running_span_sizes[-1] == n_toks - 2)):
                breakpoint()
            self.batchify(unbatched_hidden_states_nop) # sets self.hidden_states_nop
            embed_pos = self.embed_positions(self.hidden_states_nop[:,:,-1])
            hidden_states = self.hidden_states_nop + embed_pos
            layer_outputs = encoder_layer(hidden_states, attention_mask=self.attention_mask, layer_head_mask=None, output_attentions=True)
            self.hidden_states_nop = layer_outputs[0] - embed_pos
            # output from HF is (bsz, n_heads, seqlen, seqlen), take sum over all heads and all query tokens except the final padding
            layer_attns = layer_outputs[1].transpose(0,1).flatten(1,2)
            if self.contracting: # exclude final padding
                unbatched_hidden_states_nop = self.hidden_states_nop.flatten(0,1)
                up_to_last_attns = layer_outputs[1][:-1].transpose(1,3).flatten(0,1).sum(2).sum(1)
                last_attns = layer_outputs[1][-1]
                if self.padding_needed>0:
                    unbatched_hidden_states_nop = unbatched_hidden_states_nop[:-self.padding_needed]
                    last_attns = last_attns[:,:-self.padding_needed,:-self.padding_needed]
                last_attns = last_attns.sum(axis=1).sum(axis=0)
                layer_attns = torch.cat([up_to_last_attns, last_attns])
                for tn in (22,133,757):
                    if tn >= self.chunk_size:
                        break
                    assert torch.allclose(layer_attns[tn], layer_outputs[1][0,:,:,tn].sum())
            else:
                layer_attns = layer_attns.sum([0,1])
                unbatched_hidden_states_nop = self.hidden_states_nop.squeeze(0)
            layer_attns = layer_attns[1:-1] # cut off bos and eos

            if self.verbose:
                print(f'layer {idx}--sum trees: {running_span_sizes[-1]}\t hiddens: {list(unbatched_hidden_states_nop.shape)}\tneeded pad: {self.padding_needed}\tattns: {list(layer_attns.shape)}\tpseudo-batchsize: {self.pseudo_batch_size}\tchunksize: {self.chunk_size}\tcontracting: {self.contracting}')
            assert ( ( running_span_sizes[-1] == len(layer_attns)))
            if len(unbatched_hidden_states_nop) > self.seq_len:
                assert self.contracting
                attn_chunks = [layer_attns[running_span_sizes[i]:running_span_sizes[i+1]] for i in range(len(trees))]
                options = []
                tok_idx = 0
                for i, (t,ac) in enumerate(zip(trees, attn_chunks)):
                    if not ( t.span_size == ac.shape[0]):
                        breakpoint()
                    new_drop_options, new_merge_options = t.determine_drops_and_merges(ac)
                    options += [(tok_idx+d.offset, 'drop', d, dcost, i) for d,dcost in new_drop_options]
                    if not self.disallow_drops:
                        options += [(tok_idx+m.offset, 'merge', m, mcost, i) for m,mcost in new_merge_options]
                    tok_idx += t.span_size

                assert self.disallow_drops or (set([x[2].span_size for x in options if x[1]=='drop']) == set([1]))
                n_to_drop = min(self.n_contract, len(unbatched_hidden_states_nop)-self.seq_len+2)
                reductions = sorted(options, key=lambda x:x[3])[:n_to_drop]
                reductions = sorted(reductions, key=lambda x:x[0])
                reduction_idxs = [x[0] for x in reductions]
                reduced_hiddens = [unbatched_hidden_states_nop[0]] # start with just bos, will add others in for loop
                hidden_states_nop_inner = unbatched_hidden_states_nop[1:-1] # drop bos/eos
                n_to_exclude_after = 0
                option_idx = 0
                ts = lambda: sum(t.span_size for t in trees)
                prev = ts()
                assert prev==running_span_sizes[-1]
                prev_tok_idx = -1
                for i, hs in enumerate(hidden_states_nop_inner):
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
                            continue
                        if not ( node in trees[tidx].descendents):
                            breakpoint()
                        assert tok_idx==i
                        if red_type == 'drop':
                            what_red_should_be = 1
                            if node.is_root:
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

                            node.merge()
                            merged = (layer_attns[i:i+n_to_exclude_after].unsqueeze(1)*hidden_states_nop_inner[i:i+n_to_exclude_after]).sum(0)/layer_attns[i:i+n_to_exclude_after].sum()
                            reduced_hiddens.append(merged)

                        assert ( ts() == prev-what_red_should_be)
                        prev = ts()
                    else:
                        if n_to_exclude_after>0:
                            n_to_exclude_after -= 1
                        else:
                            reduced_hiddens.append(hs)

                reduced_hiddens.append(unbatched_hidden_states_nop[-1]) # add eos
                unbatched_hidden_states_nop = torch.stack(reduced_hiddens)
            else:
                assert not self.contracting

        final_hidden_states = self.hidden_states_nop.unsqueeze(0)
        embed_pos = self.embed_positions(final_hidden_states[:,-1])
        final_hidden_states = final_hidden_states + embed_pos
        return BaseModelOutput(last_hidden_state=final_hidden_states)

class RootParseNode():
    def __init__(self, sent, word_toks):
        self.parse = sent.constituency
        assert self.parse.label=='ROOT'
        assert len(self.parse.children)==1
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
        #return [n for n in self.descendents if not n.is_leaf and n.child_idx==0 and all(c.is_leaf for c in n.children)]
        return [n for n in self.descendents if not n.is_leaf and all(c.is_leaf for c in n.children)]

    def list_droppables(self):
        if self.root=='DROPPED':
            return []
        return [c for c in self.root.children+[self.root] if c.is_leaf]

    def determine_drops_and_merges(self, attention_weights):
        possible_drops_costs = [(d, attention_weights[d.offset]) for d in self.list_droppables()]
        possible_merges_costs = [(m, self.merge_cost(attention_weights[m.offset:m.offset+m.span_size])) for m in self.list_mergables()]
        return possible_drops_costs, possible_merges_costs

    def merge_cost(self, merge_attns): #may also want to consider attn mergees have for each other
        return merge_attns @ (1-merge_attns/merge_attns.sum())

class FlatRootParseNode(RootParseNode):
    def __init__(self, sent, word_toks):
        self.parse = 'flatroothasnoparse'
        self.root = ParseNode(sent.text, self, word_toks, 0)
        assert len(self.root.children) == len(word_toks)

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

        if len(self.toks) == 1:
            self.text = self.toks[0]
            self.children = []
            self.is_orig_leaf = True
            self.span_size = 1
            return

        # else, set children by matching parse words and tokens
        cleaned_children = _children
        parse_words = [text_from_node_or_str(c) for c in cleaned_children]
        child_toks = []
        pw_start = 0; tok_start = 0
        cm_children = []
        toks_to_iter = [] if len(self.toks)==1 else self.toks
        while True:
            pw_span = 1
            tok_span = 1
            while True:
                pw_chunk = parse_words[pw_start:pw_start+pw_span]
                tok_chunk = toks_to_iter[tok_start:tok_start+tok_span]
                #print(f'pw: {"".join(pw_chunk)}\ttoks: {"".join(tok_chunk)}\tpwspan: {pw_span}\ttokspan: {tok_span}\tpwstart: {pw_start}\ttokstart: {tok_start}')
                if equals_mod_whitespace(''.join(pw_chunk), ''.join(tok_chunk)):
                    break
                elif len(re.sub(r'[\n\r\t ]', '', ''.join(pw_chunk))) > len(re.sub(r'[\n\r\t ]', '', ''.join(tok_chunk))): #normal way
                    tok_span += 1
                elif len(re.sub(r'[\n\r\t ]', '', ''.join(tok_chunk))) > len(re.sub(r'[\n\r\t ]', '', ''.join(pw_chunk))): #weird way
                    pw_span += 1
                else: # shouldn't happen
                    print('equal lens but no match for', ' '.join(parse_words))
                    print(' '.join(parse_words))
                    with open('mistakes.txt', 'a') as f:
                        f.write(' '.join(parse_words) + '\n')
                    raise TreeError

                #if max(pw_span,tok_span)>300:
                if pw_span > len(parse_words) or tok_span > len(toks_to_iter):
                    print('didn\'t find a match for', ' '.join(parse_words))
                    with open('mistakes.txt', 'a') as f:
                        f.write(' '.join(parse_words) + '\n')
                    raise TreeError
            if pw_span > tok_span:
                print(f'matching multiple syntax nodes to a single token, {tok_chunk} matched to {parse_words[pw_start:pw_start+pw_span]}')
            child_toks.append(tok_chunk)
            cm_children.append(''.join(pw_chunk))
            pw_start += pw_span
            tok_start += tok_span
            assert (pw_start>=len(parse_words)) == (tok_start>=len(toks_to_iter))
            if pw_start>len(parse_words):
                assert parse_words==toks_to_iter==[]
            if pw_start>=len(parse_words):
                break

        self.children = [ParseNode(c, self, cts, i) for i,(c,cts) in enumerate(zip(cm_children,child_toks))]
        self.text = ' '.join(c.text for c in self.children)
        self.span_size = sum(c.span_size for c in self.children)
        assert self.span_size == len(self.toks)
        assert equals_mod_whitespace(self.text, ''.join(self.toks))

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
        f'Preterminals: {"None" if self.parse is None else list(self.parse.yield_preterminals())}\nText: {self.text}\t'
