from stanza.models.constituency.parse_tree import Tree
import re
import torch
import rouge


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

rouge_eval = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                                                 max_n=2,
                                                 limit_length=False,
                                                 apply_avg=True,
                                                 apply_best=False,
                                                 alpha=0.5, # Default F1_score
                                                 stemming=False)

def display_rouges(r):
    return list(zip(['r1','r2','rL','rLsum'],r))

def rouge_preprocess(text):
    text = text.replace('...',' <eplipsis> ')
    text = rouge.Rouge.REMOVE_CHAR_PATTERN.sub(' ', text.lower()).strip()
    tokens = rouge.Rouge.tokenize_text(rouge.Rouge.KEEP_CANNOT_IN_ONE_WORD.sub('_cannot_', text))
    rouge.Rouge.stem_tokens(tokens)
    preprocessed_text = rouge.Rouge.KEEP_CANNOT_IN_ONE_WORD_REVERSED.sub('cannot', ' '.join(tokens))
    return preprocessed_text

def nelly_rouge(pred_,gt_):
    pred_sum_sents = [rouge_preprocess(p) for p in split_summ(pred_)]
    gt_sum_sents = [rouge_preprocess(g) for g in split_summ(gt_)]
    pred = '\n'.join(pred_sum_sents)
    gt = '\n'.join(gt_sum_sents)

    scores = rouge_eval.get_scores(pred, gt)

    pred_old = [rouge_preprocess(pred_)]
    gt_old = [rouge_preprocess(gt_)]
    old_scores = rouge_eval.get_scores(pred_old, gt_old)
    scores['rouge-lsum'] = scores['rouge-l']
    scores['rouge-l'] = old_scores['rouge-l']
    return scores

def old_nelly_rouge(pred,gt):
    if not isinstance(pred,list):
        pred = [pred]
    if not isinstance(gt,list):
        gt = [gt]
    pred_sums = [rouge_preprocess(pred) for pred in pred]
    gt_sums = [rouge_preprocess(g) for g in gt]
    scores = rouge_eval.get_scores(pred_sums, gt_sums)
    return scores

def split_summ(s):
    return s.replace('. ','.\n').split('\n')

def extract_main_rouges(scores):
    rouge1 = scores['rouge-1']['f'] * 100
    rouge2 = scores['rouge-2']['f'] * 100
    rougel = scores['rouge-l']['f'] * 100
    rougelsum = scores['rouge-lsum']['f'] * 100
    return rouge1, rouge2, rougel, rougelsum

def rouge_from_multiple_refs(pred, references, return_full, benchmark_rl):
    benchmark = -1
    for possible_gt in references:
        new_rouge = nelly_rouge(pred, possible_gt)
        maybe_new_benchmark = new_rouge['rouge-l']['f'] if benchmark_rl else new_rouge['rouge-2']['f']
        if maybe_new_benchmark > benchmark:
            benchmark = maybe_new_benchmark
            best_rouge = new_rouge
    if benchmark == 0:
        if not all([gt is None for gt in references]):
            print('rouge is zero')
    return best_rouge if return_full else extract_main_rouges(best_rouge)

def chunkify(text,max_chunk_size):
    if len(text.split())*4/3 < max_chunk_size:
        to_return = [text]
    else:
        first_chunk, second_chunk = split_text_by_sth(text)
        to_return = chunkify(first_chunk,max_chunk_size) + chunkify(second_chunk,max_chunk_size)
    if not all(len(x) <= max_chunk_size for sl in to_return for x in sl):
        breakpoint()
    return to_return

def split_text_by_sth(text):
    for sep in ('\n', '. ', ', ', ' '):
        if sep in text.strip():
            return split_text_by_sep(text.strip(),sep)
    return text[:len(text)//2], text[len(text)//2:]

def convert_script_to_prose(script_line):
    if maybe_speaker_name:=re.match(r'\w+: ', script_line):
        speaker_name = script_line[:maybe_speaker_name.span()[1]-2]
        speech = script_line[maybe_speaker_name.span()[1]:]
        return f'{speaker_name} said "{speech}"'
    elif stage_direction := re.match(r'(?<=\[ )[A-Z -]+(?= \])', script_line):
        return stage_direction
    else:
        return script_line

def split_text_by_sep(text,sep):
    lines = text.split(sep)
    N = len(text.split())
    first_chunk = ''
    for i,l in enumerate(lines):
        if abs(len((first_chunk+l).split()) - N/2) > abs(len(first_chunk.split())-N/2):
            break # get as close to halfway as possible
        if first_chunk=='':
            first_chunk = l+sep
        else:
            first_chunk += l+sep
        if not text.startswith(first_chunk):
            breakpoint()
    second_chunk = text[len(first_chunk):]
    assert first_chunk+second_chunk == text
    return first_chunk, second_chunk

