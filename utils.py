

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
