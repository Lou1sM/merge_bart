from stanza.models.constituency.parse_tree import Tree
import re
import torch


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

def text_from_node_or_str(x):
    if isinstance(x, str):
        return x
    else:
        assert isinstance(x, Tree)
        joined_terminals = ' '.join([str(n)[1:-1].split()[1] for n in x.yield_preterminals()])
        text = joined_terminals.replace(' , ',', ').replace(' ; ','; ').replace(' . ','. ').replace(' " ','" ')
        text = re.sub(r'\" (\w+) \"',r'"\1"',joined_terminals) # remove space around "
        return text

def match_word_toks_to_sents(word_toks, sents):
    remaining_word_toks = list(word_toks) # make copy
    for i,sent in enumerate(sents):
        word_toks_to_match = ''
        matching_word_toks = []
        while word_toks_to_match!=sent:
            new = remaining_word_toks.pop(0).replace('Ġ',' ').replace('Ċ','\n')
            word_toks_to_match += new
            matching_word_toks.append(new)
            print(word_toks_to_match)

def strip_equals(s1,s2):
    return s1.replace(' ','').replace('\n','') == s2.replace(' ','').replace('\n','')

def equals_mod_whitespace(s_pred, s_target):
    s_pred = s_pred.replace(' ','')
    s_target = s_target.replace(' ','')
    if s_target.replace('\n','') == '' or s_pred.replace('\n','') == '': # don't match pred '' with target '\n\n'
        return s_pred == s_target
    else:
        return s_pred.replace('\n','') == s_target.replace('\n','')
