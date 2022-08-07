import os
import sentencepiece
import collections
import torch
from torch.serialization import default_restore_location

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
        for token in tokens:
            token = token.rstrip().split("\t")[0].strip()
            if not token:
                token = "[Occupy]"
            vocab[token] = index
            index += 1
    print("we totally load " + str(len(vocab)) + " tokens")
    return vocab


def read_all_files(data_dir):
    all_files = []
    if os.path.isfile(data_dir):
        all_files.append(data_dir)
    else:
        for root, dirs, files in os.walk(data_dir):
            for cur_file in files:
                all_files.append(os.path.join(root, cur_file))
    return all_files


def read_all_data(data_dir):
    all_data = []
    all_files = read_all_files(data_dir)
    for cur_file in all_files:
        with open(cur_file, "r", encoding="utf-8", errors='ignore') as f_read:
            all_data.extend(f_read.readlines())
    return all_data


def load_model_for_inference(model, prompt_model_path):
    loaded_prompt_model = torch.load(prompt_model_path, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    if "state_dict" in loaded_prompt_model:
        loaded_prompt_model = loaded_prompt_model["state_dict"]
    prompt_state_dict = {}
    for (key, value) in loaded_prompt_model.items():
        if key.startswith('model.prompt_embed_tokens.'):
            key = key[26:]
        prompt_state_dict[key] = value
    missing_keys, extra_keys = model.prompt_embed_tokens.load_state_dict(prompt_state_dict, strict=False)
    print("prompt_missing_keys")
    print(missing_keys)
    print("prompt_extra_keys")
    print(extra_keys)
    return model


def _truncate_seq(tokens, max_length, fixlen=False):
    while True:
        if len(tokens) <= max_length:
            break
        tokens.pop()
    if fixlen:
        while True:
            if len(tokens) >= max_length:
                break
            tokens.append("[PAD]")
    return tokens


class SPETokenizer(object):
    """Runs end-to-end tokenization: punctuation splitting + wordpiece"""

    def __init__(self, vocab_file, bpe_model_file, max_len=None):
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file))
        # with open(vocab_file,'r') as f_vocab:
        #     self.vocab = json.loads(f_vocab.readline().strip())
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.sp_tokenizer = sentencepiece.SentencePieceProcessor()
        self.sp_tokenizer.Load(bpe_model_file)
        self.max_len = max_len if max_len is not None else int(1e12)

    def vocab_size(self):
        return len(self.vocab)

    def _tokenize(self, text, adversarial=False):
        split_tokens = ["<unk>" if x not in self.vocab else x for x in self.sp_tokenizer.EncodeAsPieces(text)]
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        return ids

    def convert_token_to_id(self, token):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        ids.append(self.vocab[token])
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens

    @classmethod
    def from_pretrained(cls, pretrained_model_path):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file.
        Download and cache the pre-trained model file if needed.
        """
        vocab_file = os.path.join(pretrained_model_path, "vocab.txt")
        bpe_file = os.path.join(pretrained_model_path, "sentencepiece.bpe.model")
        tokenizer = cls(vocab_file, bpe_file)
        return tokenizer
