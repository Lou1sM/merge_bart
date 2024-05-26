from transformers import BartForConditionalGeneration, BartConfig
from transformers.models.auto.configuration_auto import AutoConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from .syntactic_pooling_encoder import SyntacticPoolingEncoder
import torch
from torch.nn import CrossEntropyLoss
from copy import deepcopy


class MergeBart(BartForConditionalGeneration):
    def __init__(self, load_from, n_contract=1000, buffer_layers=0, verbose=False, run_checks=False):
        cfg = AutoConfig.from_pretrained(load_from)
        self.tokenizer = AutoTokenizer.from_pretrained(load_from)
        loaded_state_dict = AutoModelForSeq2SeqLM.from_pretrained(load_from).state_dict()
        super().__init__(cfg)
        if run_checks:
            vanilla_enc = deepcopy(self.model.encoder)
        self.model.encoder = SyntacticPoolingEncoder(cfg, n_contract, buffer_layers, verbose, run_checks)
        self.load_state_dict(loaded_state_dict)
        if run_checks:
            self.vanilla_enc = vanilla_enc
        self.run_checks = run_checks
        self.loss_fct = CrossEntropyLoss(reduction='none')

    def forward(self, input_ids, labels=None, attention_mask=None, decoder_attention_mask=None, trees=None, encoder_outputs=None, past_key_values=None, use_cache=False):
        input_ids is not None or encoder_outputs is not None

        if encoder_outputs is None:
            encoder_outputs = self.model.encoder(input_ids, attention_mask, trees=trees)
        if labels is not None:
            labs = labels[:,:self.tokenizer.model_max_length+1]
            # if e.g. context-size==512, then the 512th prediction will predict the
            # 513th token, don't mask this, so need up to 513 tokens but only 512 of mask
            decoder_input_ids = labs[:,:-1]
            targets = labs[:,1:]
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask[:,:decoder_input_ids.shape[1]]
        else:
            decoder_input_ids = None
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=encoder_outputs[1],
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        if labels is not None:
            lm_logits = self.lm_head(decoder_outputs[0])
            lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)
            per_tok_loss = self.loss_fct(lm_logits.reshape(-1, self.config.vocab_size), targets.flatten())
            if decoder_attention_mask is not None:
                per_tok_loss = (decoder_attention_mask.flatten()*per_tok_loss)
                loss = per_tok_loss.sum()/decoder_attention_mask.sum()
            else:
                loss = per_tok_loss.mean()
        else:
            loss = torch.zeros(1).to(input_ids.device)
        return loss

    def generate(self, input_ids, attn_mask, trees=None, min_len=-1, max_len=-1):
        max_len = max(max_len, min_len)
        if self.run_checks:
            orig_dropout = self.model.encoder.dropout
            self.model.encoder.dropout=0
            for l in self.model.encoder.layers:
                l.dropout=0
            self.model.encoder.eval()
            encoder_outputs = self.model.encoder(input_ids[:,:512], attn_mask[:,:512], trees)
            self.model.encoder.train()
            self.vanilla_enc.dropout=0.0
            for l in self.vanilla_enc.layers:
                l.dropout=0
            self.vanilla_enc.load_state_dict(self.model.encoder.state_dict())
            self.vanilla_enc.eval()
            out2 = self.vanilla_enc(input_ids[:,:512], attn_mask[:,:512], output_hidden_states=True)
            if not ( (diff:=(encoder_outputs[0]-out2.last_hidden_state).max()) < 1e-5):
                print(f'warning, differs by {diff} from vanilla enc hiddens')
                breakpoint()
            encoder_outputs = self.model.encoder(input_ids[:,:512], attn_mask[:,:512], trees)
            self.model.encoder.dropout = orig_dropout
            for l in self.model.encoder.layers:
                l.dropout=orig_dropout
        encoder_outputs = self.model.encoder(input_ids, attn_mask, trees)
        past_key_values = None
        genned_ids = [last_genned:=torch.tensor(self.tokenizer.bos_token_id).cuda()]
        for i in range(max_len+1): # +1 for bos-tok
            decoder_outputs = self.model.decoder(
                input_ids=last_genned[None,None], # make have shape (1,1)
                encoder_hidden_states=encoder_outputs[0],
                past_key_values=past_key_values,
                use_cache=True,
            )

            lm_logits = self.lm_head(decoder_outputs[0])
            lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

            past_key_values = decoder_outputs.past_key_values
            last_genned = lm_logits.argmax()
            genned_ids.append(last_genned)
            if last_genned==self.tokenizer.eos_token_id and len(genned_ids)>min_len: break

        return torch.stack(genned_ids), encoder_outputs
