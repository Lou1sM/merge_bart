from transformers import BartForConditionalGeneration, BartConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import Seq2SeqLMOutput
from .pooling_encoder import PoolingEncoder
from .syntactic_pooling_encoder import SyntacticPoolingEncoder
import torch
from torch.nn import CrossEntropyLoss


class MergeBart(BartForConditionalGeneration):
    def __init__(self, encoder_type, load_from, n_contract=1000):
        default_bart_config = BartConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(load_from)
        loaded_state_dict = AutoModelForSeq2SeqLM.from_pretrained(load_from).state_dict()
        default_bart_config.vocab_size = loaded_state_dict['lm_head.weight'].shape[0]
        super().__init__(default_bart_config)
        if encoder_type == 'syn-pool':
            self.model.encoder = SyntacticPoolingEncoder(default_bart_config,n_contract)
        elif encoder_type == 'pool':
            self.model.encoder = PoolingEncoder()
        self.load_state_dict(loaded_state_dict)
        self.loss_fct = CrossEntropyLoss()

    def forward(self, input_ids, trees, labels):
        encoder_outputs = self.model.encoder(input_ids, trees)
        decoder_input_ids = labels[:-1]
        labels = labels[1:]
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs[0],
        )

        lm_logits = self.lm_head(decoder_outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            masked_lm_loss = self.loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        return Seq2SeqLMOutput(loss=masked_lm_loss,logits=lm_logits)


    def generate(self, input_ids, trees, min_len=-1, max_len=-1):
        max_len = max(max_len, min_len)
        encoder_outputs = self.model.encoder(input_ids, trees)
        past_key_values = None
        genned_ids = [last_genned:=torch.tensor(self.tokenizer.bos_token_id)]
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
