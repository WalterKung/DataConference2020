import os
import random
import pickle
import numpy as np
import torch 
import scipy.io
import pandas as pd
from pathlib import Path
import nltk; 
from nltk.corpus import stopwords
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from collections import Counter
import random
  
def get_ids_vocab_clients(clientkeywords, id2word): 
    ids_vocab_clients = []
    for e in clientkeywords:
        try:
            ids_vocab_clients.append(id2word.token2id[e])
        except:
            pass
    return ids_vocab_clients
    
def get_rand_index(num_docs, batch_size):
    indices = torch.randperm(num_docs)
    indices = torch.split(indices, batch_size)
    numBatch = len(indices)
    return numBatch, indices      

flatten = lambda l: [item for sublist in l for item in sublist]

def list2Tokenscount(li_words,id2word):
    '''
    function to convert list of words to tokens and count
    '''
    corpus = [id2word.doc2bow(txt) for txt in li_words]
    doc_tokens = []; doc_counts = [];
    for doc in corpus:
        _tokens = []; _counts = [];
        for word in doc:
            _tokens.append(word[0]); _counts.append(word[1]);
        doc_tokens.append(np.array([_tokens])); doc_counts.append(np.array([_counts]));
    return doc_tokens, doc_counts

def getVocab_id2word(li_words):
    '''
    getvocab and id2word
    '''
    id2word = corpora.Dictionary(li_words)       
    vocab = [id2word[i] for i in range(len(id2word))]
    vocab_size = len(vocab)
    return id2word, vocab, vocab_size

def composeStopWords(stopwords_ext, lang = 'english'):
    '''
    generate stopword
    '''
    stop_words = stopwords.words('english')
    sw = list(stop_words) + stopwords_ext
    sw = list(set(sw))
    return {key: True for key in sw}

def create_white_words_list(li_doc, stop_words, list_must_included, min_term_freq=25, max_term_potion=0.4):
    '''
    create a white list of keywords
    '''
    remove_stopwords = lambda texts: [[w for w in simple_preprocess(str(words), deacc=True) if not stop_words.get(w,False)] for words in texts]
    data_words = remove_stopwords(li_doc)
    data_unique = [list(set(words)) for words in data_words]
    flattened_data_unique = flatten(data_unique)
    counts_unique = Counter(flattened_data_unique)
    counts = Counter(flatten(data_words))
    whitelist = [w for w in list(set(flattened_data_unique)) if (counts[w]>min_term_freq and counts_unique[w]<(len(li_doc)*max_term_potion))] + list_must_included
    whitelist = list(set(whitelist))
    return {key: True for key in whitelist}

def get_data_words(li_doc, white_list):
    '''
    Using get() key in dictionary. Return False if it is not found... much faster than in list.
    '''
    return [[w for w in simple_preprocess(str(words), deacc=True) if white_list.get(w, False)] for words in li_doc]

def doc2bow_li(data_words, id2word, vocab_size, device):
    '''
    convert list of words to bow
    '''
    tokens, counts = list2Tokenscount(data_words,id2word)
    li_size = len(tokens)
    data_batch = np.zeros((li_size, vocab_size))
    for words_id in range(li_size):
        keys = tokens[words_id]; freqs = counts[words_id]
        if len(keys.shape) == 1:
            keys = [keys]; count = [count]
        if tokens[words_id].shape[1]>0:
            for j, key in enumerate(keys):
                data_batch[words_id, key] = freqs[j]
    data_batch = torch.from_numpy(data_batch).float().to(device)  
    return data_batch

def sample_Text_With_WordCount(text, length):
    words = text.split(" ")
    if len(words) <= length: ret = text
    else:
        start =  random.randint(0,len(words)-length)
        ret = " ".join(words[start: start+length])
    return ret

def interleave_Text(text, startPos=0, wd_count = None):
    ws = text.split(" ")
    if wd_count is None: stop_n = len(ws)
    elif len(ws) < wd_count: stop_n = len(ws)
    else: stop_n = wd_count
    return " ".join([ws[i] for i in range(stop_n-1) if ((i+startPos)%2) == 0])

def calcIDF(data_batch):
    db = data_batch
    db = (db > 1).float()
    return (db.shape[1]/(db.sum(0)+1)).log()

class databunch():    
    def __init__(self, name, batch_size = 100, batch_wd_count=300, fp_kw = ""):
        # data attributes        

        self.clientkeywords = []
        if len(fp_kw) > 0:
            df_kws = pd.read_csv(fp_kw)
            keys = " ".join([e for e in df_kws.Topics.str.replace('(', ' ').str.replace(')', ' ').str.replace('/', ' ')]).split(" ")
            self.clientkeywords = list(set([e.lower() for e in keys if len(e) > 1]))
        
        self.batch_size = {"train": batch_size, "test": int(batch_size/5), "valid": int(batch_size/5)}
        self.batch_wd_count = batch_wd_count
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_term_freq=40         
        self.max_term_potion=0.4
        self.min_doc_length=200
        self.clientVocabWT = 3; 
        self.isTf=True
        self.isIDF=True

    def import_data(self, datadir, fname, col_name,  col_author, min_term_freq=None, max_term_potion=None, min_doc_length=None, stopword_ext=[]):                   
        if min_term_freq is None:
            min_term_freq = self.min_term_freq         
        if max_term_potion is None:
            max_term_potion = self.max_term_potion
        if min_doc_length is None:
            min_doc_length = self.min_doc_length
            
        path = Path(datadir)
        df = pd.read_csv(path.joinpath(fname))    
        df = df.sample(frac=1).reset_index(drop=True)            

        texts = df[col_name].values.tolist()   
        len_filter = [len(e.split(' ')) > min_doc_length for e in texts]
        df = df[len_filter]
        
        authors = df[col_author].astype(str).tolist()
        texts = df[col_name].values.tolist()

        print("# of docs: " + str(len(texts)))
        del df
        #        
        self.stopword = composeStopWords(stopword_ext, lang = 'english')             
        self.whitelist = create_white_words_list(texts, self.stopword, self.clientkeywords, min_term_freq, max_term_potion)
        self.id2word, self.vocab, self.vocab_size = getVocab_id2word(get_data_words(texts, self.whitelist))
        self.clienttoken = get_ids_vocab_clients(self.clientkeywords, self.id2word)
        mask = (np.zeros([1,self.vocab_size]) + 1)
        mask[0,self.clienttoken] = self.clientVocabWT
        self.mask = torch.from_numpy(mask).float().to(self.device)               
        self.data_batch= doc2bow_li(get_data_words(texts, self.whitelist), self.id2word, self.vocab_size, self.device)
        self.idf=calcIDF(self.data_batch)
        
        # separate train, valid, test        
        idx_tmp = np.sort(np.random.choice(range(len(texts)), int(len(texts) * 0.2), replace=False))
        idx_train = np.sort([idx for idx in range(len(texts)) if idx not in idx_tmp])
        idx_test = np.sort(np.random.choice(idx_tmp, int(len(idx_tmp) * 0.5), replace=False))
        idx_valid = np.sort([idx for idx in idx_tmp if idx not in idx_test])
        #
        texts_train = [texts[idx] for idx in idx_train]
        texts_test = [texts[idx] for idx in idx_test]
        texts_valid = [sample_Text_With_WordCount(texts[idx], self.batch_wd_count * 2) for idx in idx_valid]
        self.texts = {"train": texts_train, "test": texts_test, "valid": texts_valid}
        #
        authors_train = [authors[idx] for idx in idx_train]
        authors_test = [authors[idx] for idx in idx_test]
        authors_valid = [authors[idx] for idx in idx_valid]
        self.authors = {"train": authors_train, "test": authors_test, "valid": authors_valid}

        #
        self.num_docs = {"train": len(self.texts["train"]), "test": len(self.texts["test"]), "valid": len(self.texts["valid"])}
        nBatch_train, indices_train = get_rand_index(self.num_docs["train"], self.batch_size["train"])
        nBatch_test, indices_test = get_rand_index(self.num_docs["test"], self.batch_size["test"])
        nBatch_valid, indices_valid = get_rand_index(self.num_docs["valid"], self.batch_size["valid"])
        self.nBatch = {"train": nBatch_train, "test": nBatch_test, "valid": nBatch_valid}
        self.ind_Batch = {"train": indices_train, "test": indices_test, "valid": indices_valid}
        return self.vocab_size, self.num_docs
    #
    def get_batch(self, seq_Batch, datasrc="train", device = None, batch_wd_count = None, retText = False):
        """fetch input data by batch."""
        if device is None: device = self.device
        if batch_wd_count is None: batch_wd_count = self.batch_wd_count
        src_key = ''.join(c for c in datasrc if c.isalpha())
        src_num = ''.join(c for c in datasrc if not c.isalpha())
        ind = self.ind_Batch[src_key]
        batch_size = self.batch_size[src_key]
        data_batch = np.zeros((batch_size, self.vocab_size))   
        texts = []   

        for i, doc_id in enumerate(ind[seq_Batch]):
            if src_num == "1":
                partText = interleave_Text(self.texts[src_key][doc_id], startPos=0) 
            elif src_num == "2":
                partText = interleave_Text(self.texts[src_key][doc_id], startPos=1)
            else:
                partText = sample_Text_With_WordCount(self.texts[src_key][doc_id], batch_wd_count)
            texts.append(partText)
        data_batch = self.text2DataBatch(texts)

        if retText:
            return data_batch, texts
        else:
            return data_batch
        
    # to adjust for the client vocab
    def text2DataBatch(self, texts):
        data_words = get_data_words(texts, self.whitelist)
        db =  doc2bow_li(data_words, self.id2word, self.vocab_size, self.device).float()
        if self.clientVocabWT > 1:
            db = db * self.mask 
        if self.isTf:
            db = torch.transpose(torch.transpose(db,0,1) / (db.sum(1)+0.000000001), 0, 1) * 70 
        if self.isIDF:
            idf = calcIDF(db)
            db = db * idf / 10
        return db

    # to get the author2doc dictionary
    def getAuthor2doc(self):
        authors_train = pd.DataFrame({"author": self.authors["train"]})
        authors_train = authors_train.reset_index(name="doc_idx")
        df1 = authors_train.groupby('author')['doc_idx'].apply(list)
        author2doc = df1.to_dict()
        return author2doc
    
 