import os
import sentencepiece
import collections
import re

class Patterns:
    SELF_BREAK_TOKEN = r'<selfbr>'
    SELF_BREAK_RGX = re.compile(SELF_BREAK_TOKEN)

    GET_SUBMISSION_SELF_TEXT_RGX = re.compile(
      r'(?<=%s).*$' % SELF_BREAK_TOKEN, re.DOTALL)

    BOT_BODY_RGX = re.compile(
      r"""^i am a bot|^i\'m a bot|^bleep.*?bloop|^beep.*?boop|i am a bot[^a-zA-Z]*$
      |^i\'m a bot[^a-zA-Z]*$|bleep.*?bloop[^a-zA-Z]*$|beep.*?boop[^a-zA-Z]*$'""",
      re.I)
    BOT_BUTTON_RGX = re.compile(r'\^\|\s*\^\[')
    BOT_AUTHOR_PATTERNS = [
      r'^imgur',
      r'^linkfixer',
      r'bots?[^a-zA-Z]*$',
      r'tips?$',
      r'quotes$',
      r'transcriber$',
      r'watch$',
      r'breaker$',
      r'fixer$',
    ]
    BOT_AUTHOR_RGX = re.compile('|'.join(BOT_AUTHOR_PATTERNS), re.I)

    # Markdown tables: https://www.markdownguide.org/extended-syntax/#tables
    DETECT_MARKDOWN_TABLE_RGX = re.compile(r'(\|\s*:?--*:?\s*\|)|(\+----*)')
    ALPHANUM_RGX = re.compile(r'[a-zA-Z0-9]')
    ANY_UPPERCASE_RGX = re.compile(r'[A-Z]')

    BZ2_EXT_RGX = re.compile(r'\.(bz2|bzip2)(\-luigi\-tmp\-\d*)?$')
    XZ_EXT_RGX = re.compile(r'\.xz(\-luigi\-tmp\-\d*)?$')
    GZ_EXT_RGX = re.compile(r'\.(gz|gzip)(\-luigi\-tmp\-\d*)?$')
    TXT_TSV_EXT_RGX = re.compile(r'\.(txt|tsv)(\-luigi\-tmp\-\d*)?$')

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
