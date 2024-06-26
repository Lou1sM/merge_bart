from utils import text_from_node_or_str, equals_mod_whitespace, TreeError
import re

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
            while len(stanza_node_or_text.children)==1:
                stanza_node_or_text = stanza_node_or_text.children[0]
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
            #if pw_span > tok_span:
                #print(f'matching multiple syntax nodes to a single token, {tok_chunk} matched to {parse_words[pw_start:pw_start+pw_span]}')
            child_toks.append(tok_chunk)
            cm_children.append(''.join(pw_chunk))
            pw_start += pw_span
            tok_start += tok_span
            assert (pw_start>=len(parse_words)) == (tok_start>=len(toks_to_iter))
            if pw_start>len(parse_words):
                assert parse_words==toks_to_iter==[]
            if pw_start>=len(parse_words):
                break

        if len(cm_children)==1: # just make flat, could probs do sth better but fine for now
            self.children = [ParseNode(t, self, [t], i) for i,t in enumerate(self.toks)]
        else:
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
