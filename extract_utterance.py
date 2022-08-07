import sys
import json
import re
from tqdm import tqdm
from util import SPETokenizer, Patterns
from multiprocessing import Pool

tokenizer = SPETokenizer(vocab_file="tokenizer_model/vocab.txt", bpe_model_file="tokenizer_model/sentencepiece.bpe.model")

def prep(s):
    s = s.strip()
    s = s.replace('\n', ' ')
    s = s.replace('\r', ' ')
    s = re.sub('[ ]+', ' ', s)
    return s

def filter(content):
    if content == '[deleted]' or content == '[removed]':
        return False
    if len(tokenizer._tokenize(content)) <= 3 or len(tokenizer._tokenize(content)) >= 32:
        return False
    if re.search(r"http\S+",content) != None:
        return False
    if Patterns.BOT_BUTTON_RGX.search(content):
        return False
    if not Patterns.ALPHANUM_RGX.search(content):
        return False
    if Patterns.DETECT_MARKDOWN_TABLE_RGX.search(content):
        return False
    if Patterns.BOT_BODY_RGX.search(content):
        return False
    return True

def read_all_data(input_file_path, number_workers):
    total_idx=0
    all_data = [[] for i in range(number_workers)]
    with open(input_file_path,encoding='utf-8',errors='ignore') as f_read:
        for item in f_read:
            infos=json.loads(item.strip())
            id=infos["id"].strip()
            parent_id=infos["parent_id"].strip()
            content=infos["body"].strip()
            subreddit=infos["subreddit"].lower().strip()
            if id=="" or parent_id=="" or content=="" or subreddit=="" or infos['score'] < 1:
                continue
            total_idx += 1
            cur_bucket=total_idx%number_workers
            all_data[cur_bucket].append((id,parent_id,content,subreddit))
            if total_idx%10000==0:
                print("Current Status: "+str(total_idx))
    return all_data

def filter_data(input_data):
    filtered_data=[]
    for item in input_data:
        id = item[0].strip()
        parent_id = item[1].strip()
        content = item[2].strip()
        subreddit = item[3].lower().strip()
        content = prep(content)
        if filter(content) == True:
            filtered_data.append((id,parent_id,content,subreddit))
    return filtered_data

if __name__ == '__main__':
    n_workers = 200
    all_data = read_all_data(sys.argv[1], n_workers)
    with Pool(n_workers) as pool:
        all_result = list(
            tqdm(
                    pool.imap(filter_data, all_data),
                    total=n_workers
            )
        )
    with open(sys.argv[2],"w") as f_write:
        for cur_result in all_result:
            for item in cur_result:
                f_write.write("\t".join(item)+"\n")