from transformers.models.bart.modeling_bart import BartEncoder
import math
import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa



class SyntacticPoolingEncoder(BartEncoder):
    def __init__(self, cfg, n_contract, buffer_layers, verbose, run_checks):
        super().__init__(cfg)
        self.nz = self.config.d_model
        self.context_size = self.config.max_position_embeddings
        self.n_contract = n_contract
        self.buffer_layers = buffer_layers
        self.verbose = verbose
        self.run_checks = run_checks

    def batchify(self, x, attn_mask):
        self.n_toks = x.shape[1]
        self.contracting = self.n_toks > self.context_size
        if (self.contracting):
            self.pseudo_bs = math.ceil(self.n_toks/self.context_size) # assuming hiddens shape is (n_tokens, emb_size)
            self.chunk_size = math.ceil(self.n_toks / self.pseudo_bs)
            self.padding_needed = self.pseudo_bs*self.chunk_size - self.n_toks
            padding = torch.zeros(self.bs, self.padding_needed, self.nz).to(x.device)
            padded = torch.cat([x, padding], dim=1)
            padded_amask = torch.cat([attn_mask, padding[:,:,0].int()], dim=1)
            assert padded_amask.float().mean() <= attn_mask.float().mean()
            self.hiddens_nop = padded.reshape(self.bs*self.pseudo_bs,self.chunk_size,self.nz)
            self.layer_attn_mask = padded_amask.reshape(self.bs*self.pseudo_bs,self.chunk_size)
        else:
            self.pseudo_bs = 1
            self.chunk_size = self.n_toks
            self.layer_attn_mask = attn_mask
            self.hiddens_nop = x#.unsqueeze(0)
            self.padding_needed = 0

    def forward(self, input_ids, attn_mask, trees, **kwargs):
        inputs_embeds = self.embed_tokens(input_ids)# * self.embed_scale
        #print('inputs embeds:', inputs_embeds)
        self.bs = inputs_embeds.shape[0]

        unchunked_hiddens_nop = inputs_embeds #'nop' means no pos embeds, for now
        #print('after layer norm:', unchunked_hiddens_nop)

        #all_hiddens = []
        for idx, encoder_layer in enumerate(self.layers):
            self.batchify(unchunked_hiddens_nop, attn_mask)
            embed_pos = self.embed_positions(self.hiddens_nop)
            hiddens = self.hiddens_nop + embed_pos
            if idx==0:
                hiddens = self.layernorm_embedding(hiddens)
                hiddens = nn.functional.dropout(hiddens, p=self.dropout, training=self.training)
            #all_hiddens.append(hiddens)
            attn_mask4d = _prepare_4d_attention_mask_for_sdpa(self.layer_attn_mask, torch.float32)
            layer_outputs = encoder_layer(hiddens, attention_mask=attn_mask4d, layer_head_mask=None, output_attentions=True, doprint=idx==0)
            self.hiddens_nop = layer_outputs[0] - embed_pos # (bsz, n_heads, seqlen, seqlen)
            unchunked_hiddens_nop = self.hiddens_nop.reshape(self.bs,self.pseudo_bs*self.chunk_size,self.nz)
            if self.padding_needed>0:
                unchunked_hiddens_nop = unchunked_hiddens_nop[:,:-self.padding_needed]
            if self.contracting and idx>=self.buffer_layers: # exclude final padding
                attns = layer_outputs[1].detach()
                attns = attns.sum(1) # sum over heads
                attns = attns * self.layer_attn_mask.unsqueeze(2) # mask out attn from padding
                attns = attns.sum(1) # sum over keys
                attns = attns.reshape(self.bs, self.pseudo_bs*self.chunk_size)
                if self.padding_needed>0:
                    attns = attns[:,:-self.padding_needed] # cut the padding added for chunking
                if self.run_checks:
                    for tn in (22,133,757):
                        if tn >= self.chunk_size:
                            break
                        for i in range(self.bs):
                            if (x:=(attns[i,tn] - layer_outputs[1][i*self.pseudo_bs,:,:,tn].sum()).mean()) > 1e-5:
                                print(f'warning: values from two ways of computing attns differ by {x} when avg attn is {attns[tn].mean()}')
                assert attns.shape[1] == self.n_toks
                assert unchunked_hiddens_nop.shape[1] == self.n_toks
                attns = attns[:,1:-1] # cut off bos and eos
                n_to_drop = min(self.n_contract, self.n_toks-self.context_size+2)
                if trees is None:
                    reduction_idxs = (-attns).topk(self.n_toks-2 - n_to_drop).indices
                    unchunked_hiddens_nop = self.idx_inner(unchunked_hiddens_nop, reduction_idxs)
                    attn_mask = self.idx_inner(attn_mask, reduction_idxs)
                    assert unchunked_hiddens_nop.shape[1] == self.n_toks - n_to_drop
                    assert attn_mask.shape[1] == self.n_toks - n_to_drop
                else:
                    unchunked_hiddens_nop = self.reduce_using_trees(unchunked_hiddens_nop, attns, trees, n_to_drop)
                if self.verbose:
                    print(f'layer {idx}: ntoks: {self.n_toks}, drop: {n_to_drop}')
            elif not self.contracting:
                #unchunked_hiddens_nop = self.hiddens_nop#.squeeze(0)
                assert layer_outputs[1].shape[2] == self.n_toks

        #final_hiddens = self.hiddens_nop.unsqueeze(0)
        self.batchify(unchunked_hiddens_nop, attn_mask)
        embed_pos = self.embed_positions(self.hiddens_nop)
        final_hiddens = self.hiddens_nop + embed_pos
        final_hiddens = final_hiddens.reshape(self.bs, self.pseudo_bs*self.chunk_size, self.nz)
        if self.padding_needed > 0:
            final_hiddens = final_hiddens[:,:-self.padding_needed]
        #return BaseModelOutput(last_hidden_state=final_hiddens, attention_mask=attn_mask, hidden_states=all_hiddens)
        return final_hiddens, attn_mask#, all_hiddens

    def reduce_using_trees(self, unchunked_hiddens_nop, layer_attns, trees, n_to_drop):
        if trees is not None:
            running_span_sizes = [sum(t.span_size for t in trees[:i]) for i in range(len(trees)+1)]
            assert ( ( running_span_sizes[-1] == unchunked_hiddens_nop.shape[1] - 2))
        else:
            running_span_sizes = None
        attn_chunks = [layer_attns[running_span_sizes[i]:running_span_sizes[i+1]] for i in range(len(trees))]
        options = []
        tok_idx = 0
        for i, (t,ac) in enumerate(zip(trees, attn_chunks)):
            assert ( t.span_size == ac.shape[0])
            new_drop_options, new_merge_options = t.determine_drops_and_merges(ac)
            options += [(tok_idx+d.offset, 'drop', d, dcost, i) for d,dcost in new_drop_options]
            if not self.disallow_drops:
                options += [(tok_idx+m.offset, 'merge', m, mcost, i) for m,mcost in new_merge_options]
            tok_idx += t.span_size

        assert self.disallow_drops or (set([x[2].span_size for x in options if x[1]=='drop']) == set([1]))
        reductions = sorted(options, key=lambda x:x[3])[:n_to_drop]
        reductions = sorted(reductions, key=lambda x:x[0])
        reduction_idxs = [x[0] for x in reductions]
        reduced_hiddens = [unchunked_hiddens_nop[0]] # start with just bos, will add others in for loop
        hiddens_nop_inner = unchunked_hiddens_nop[1:-1] # drop bos/eos
        n_to_exclude_after = 0
        option_idx = 0
        ts = lambda: sum(t.span_size for t in trees)
        prev = ts()
        assert prev==running_span_sizes[-1]
        prev_tok_idx = -1
        for i, hs in enumerate(hiddens_nop_inner):
            if i in reduction_idxs:
                tok_idx, red_type, node, _, tidx = reductions[option_idx]
                #print(f'tok idx: {tok_idx}\ttree idx: {tidx}\toption idx: {option_idx}')
                while option_idx<len(reductions) and reduction_idxs[option_idx]==i:
                    option_idx+=1
                assert prev_tok_idx!=tok_idx
                prev_tok_idx = tok_idx
                if n_to_exclude_after>0:
                    n_to_exclude_after -= 1
                    continue
                assert ( node in trees[tidx].descendents)
                assert tok_idx==i
                if red_type == 'drop':
                    what_red_should_be = 1
                    if node.is_root:
                        assert trees[tidx].root==node
                        trees[tidx].root = 'DROPPED'
                    elif len(node.parent.children)==2: # replace the parent with the sibling
                        sibling = [n for n in node.parent.children if n!=node][0]
                        assert node.parent.is_root
                        assert ( node.parent==trees[tidx].root)
                        trees[tidx].root = sibling
                        sibling.parent = trees[tidx]
                        assert sibling.is_root
                    else:
                        node.drop_non_root()

                else:
                    n_to_exclude_after = node.span_size-1
                    what_red_should_be = n_to_exclude_after

                    node.merge()
                    merged = (layer_attns[i:i+n_to_exclude_after].unsqueeze(1)*hiddens_nop_inner[i:i+n_to_exclude_after]).sum(0)/layer_attns[i:i+n_to_exclude_after].sum()
                    reduced_hiddens.append(merged)

                assert ( ts() == prev-what_red_should_be)
                prev = ts()
            else:
                if n_to_exclude_after>0:
                    n_to_exclude_after -= 1
                else:
                    reduced_hiddens.append(hs)
            #if any(reduced_hiddens[-1].isnan().any()):

        reduced_hiddens.append(unchunked_hiddens_nop[-1]) # add eos
        return torch.stack(reduced_hiddens)

    def idx_inner(self, x, idx):
        inner = x[:,1:-1]
        assert idx.ndim == 2 # indexing that follows expects certain shapes
        assert x.ndim in (2,3)
        if x.ndim == 3:
            selected = inner.gather(1, idx.unsqueeze(2).tile(1,1,inner.shape[2]))
        else:
            selected =  inner.gather(1, idx)
        if self.run_checks:
            ref = torch.stack([torch.stack([inner[j,ix] for ix in idx[j]]) for j in range(self.bs)])
            assert (ref==selected).all()
        return torch.cat([x[:,:1], selected, x[:,-1:]], dim=1)
