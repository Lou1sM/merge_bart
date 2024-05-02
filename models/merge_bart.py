from transformers import BartModel, BartConfig
from .pooling_encoder import PoolingEncoder
from .syntactic_pooling_encoder import SyntacticPoolingEncoder


class MergeBart(BartModel):
    def __init__(self, encoder_type):
        default_bart_config = BartConfig()
        super().__init__(default_bart_config)
        if encoder_type == 'syn-pool':
            self.encoder = SyntacticPoolingEncoder()
        elif encoder_type == 'pool':
            self.encoder = PoolingEncoder()

    def forward(self, input_ids, trees):
        encoder_hiddens = self.encoder(input_ids, trees)
        return self.decoder(input_ids=input_ids, encoder_hidden_states=encoder_hiddens)
