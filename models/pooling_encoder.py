from transformers.models.bart.modeling_bart import BartEncoder


class PoolingEncoder(BartEncoder):
    def __init__(self):
        pass

    def forward(self):
        raise NotImplementedError
