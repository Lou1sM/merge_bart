from transformers.models.bart.modeling_bart import BartEncoder


class SyntacticPoolingEncoder(BartEncoder):
    def __init__(self):
        pass

    def forward(self):
        raise NotImplementedError
