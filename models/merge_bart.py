from transformers import BartForConditionalGeneration, BartConfig
from transformers.models.auto.configuration_auto import AutoConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from .syntactic_pooling_encoder import SyntacticPoolingEncoder
import torch
from torch.nn import CrossEntropyLoss


class MergeBart(BartForConditionalGeneration):
    def __init__(self, load_from, n_contract=1000, disallow_drops=False, verbose=False, run_checks=False):
        cfg = AutoConfig.from_pretrained(load_from)
        self.tokenizer = AutoTokenizer.from_pretrained(load_from)
        loaded_state_dict = AutoModelForSeq2SeqLM.from_pretrained(load_from).state_dict()
        super().__init__(cfg)
        self.model.encoder = SyntacticPoolingEncoder(cfg, n_contract, disallow_drops, verbose, run_checks)
        self.load_state_dict(loaded_state_dict)
        self.loss_fct = CrossEntropyLoss(reduction='none')

    def forward(self, input_ids, labels, attn_mask, loss_mask, trees=None):
        encoder_outputs = self.model.encoder(input_ids, attn_mask, trees=trees)
        loss = 0
        labs = labels[:,:self.tokenizer.model_max_length+1]
        # if e.g. context-size==512, then the 512th prediction will predict the
        # 513th token, don't mask this, so need up to 513 tokens but only 512 of mask
        decoder_input_ids = labs[:,:-1]
        loss_mask = loss_mask[:,:decoder_input_ids.shape[1]]
        targets = labs[:,1:]
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs[0],
        )

        lm_logits = self.lm_head(decoder_outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        per_tok_loss = self.loss_fct(lm_logits.reshape(-1, self.config.vocab_size), targets.flatten())
        loss = (loss_mask.flatten()*per_tok_loss).mean()

        return loss


    def generate(self, input_ids, attn_mask, trees=None, min_len=-1, max_len=-1):
        max_len = max(max_len, min_len)
        encoder_outputs = self.model.encoder(input_ids, attn_mask, trees)
        past_key_values = None
        genned_ids = [last_genned:=torch.tensor(self.tokenizer.bos_token_id).cuda()]
        while True:
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
            if len(genned_ids)==max_len+1: break # +1 for bos-tok
            if last_genned==self.tokenizer.eos_token_id and len(genned_ids)>min_len: break

        return torch.stack(genned_ids)
