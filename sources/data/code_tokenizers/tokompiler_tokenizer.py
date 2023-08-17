from .tokompiler.tokenizer import Tokompiler

class TokompilerTokenizerWrapper(object):
    def __init__(self, vocab_path):
        name = 'Tokompiler'
        super().__init__(name)
        self.tokenizer = Tokompiler(vocab_path)

    @property
    def vocab_size(self):
        return len(self.tokenizer.encoder)

    @property
    def vocab(self):
        """Dictionary from vocab text token to id token."""
        return self.tokenizer.encoder

    @property
    def inv_vocab(self):
        """Dictionary from vocab id token to text token."""
        return self.tokenizer.decoder

    def encode(self, sequence=sequence, is_pretokenized=is_pre_tokenized):
        """
        Encode a sequence to corresponding ids.

        Args:
            sequence (Union[str, List[str]]): Sequence to be encoded,
                when is_pre_tokenized is False, the type should be str,
                when is_pre_tokenized is True, the type should be List[str]
            is_pre_tokenized (bool): Whether the input is already pre-tokenized

        Returns:
            list[int], list[int]: indices and mask for sequence

        """
    
    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)
    
    def token_to_id(self, token):
        return self.tokenizer.encoder[token]
    
    def id_to_token(self, id):
        return self.tokenizer.decoder[id]
    
    @property
    def cls(self):
        return self.tokenizer.encoder['[CLS]']

    @property
    def sep(self):
        return self.tokenizer.encoder['[SEP]']

    @property
    def pad(self):
        return self.tokenizer.encoder['[PAD]']

    @property
    def eod(self):
        return self.tokenizer.encoder['[EOS]']

    @property
    def mask(self):
        return self.tokenizer.encoder['[MSK]']
