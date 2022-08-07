import sys
import json
import re
from util import SPETokenizer

tokenizer = SPETokenizer(vocab_file="tokenizer_model/vocab.txt", bpe_model_file="tokenizer_model/sentencepiece.bpe.model")

SELF_BREAK_TOKEN = r'<selfbr>'

class Patterns:
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

def prep(s):
    s = s.strip()
    s = s.replace('\n', ' ')
    s = s.replace('\r', ' ')
    s = re.sub('[ ]+', ' ', s)
    return s

def filter(x):
    if x['score'] < 1:
        return False
    if x['body'] == '[deleted]' or x['body'] == '[removed]':
        return False
    if Patterns.BOT_AUTHOR_RGX.search(x['author'].lower()):
        return False
    if Patterns.BOT_BODY_RGX.search(x['body']):
        return False
    if Patterns.BOT_BUTTON_RGX.search(x['body']):
        return False
    if Patterns.DETECT_MARKDOWN_TABLE_RGX.search(x['body']):
        return False
    if not Patterns.ALPHANUM_RGX.search(x['body']):
        return False
    if re.search(r"http\S+",x['body']) != None:
        return False
    if len(tokenizer._tokenize(x['body'])) <= 3 or len(tokenizer._tokenize(x['body'])) >= 32:
        return False
    return True

idx=0
with open(sys.argv[1],encoding='utf-8',errors='ignore') as f_read, open(sys.argv[2],"w") as f_write:
    for item in f_read:
        infos=json.loads(item.strip())
        if filter(infos) == False:
            continue
        id=infos["id"].strip()
        parent_id=infos["parent_id"].strip()
        content=prep(infos["body"].strip())
        subreddit=infos["subreddit"].lower().strip()
        if id=="" or parent_id=="" or content=="" or subreddit=="":
            continue
        f_write.write("\t".join([id,parent_id,content,subreddit]).strip()+"\n")
        idx+=1
        if idx%100000==0:
            print(idx)

