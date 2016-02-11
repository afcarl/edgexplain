'''
Created on 8 Feb 2016

@author: af
'''
import codecs
import numpy as np
import pdb
def load_embeddings(filename, n_dimensions=50):
    with codecs.open(filename, 'r', encoding='utf-8') as inf:
        lines = inf.readlines()
    fields = [l.strip().split(' ') for l in lines]
    embeddings = np.array([f[1:] for f in fields], dtype=float)
    vocab = [f[0] for f in fields]
    del fields
    vocab_index = {v:i for i, v in enumerate(vocab)}
    index_vocab = {i:v for i, v in enumerate(vocab)}
    return embeddings, vocab_index, index_vocab
    
def dump_embeddings(output_filename, embeddings, index_vocab):
    with codecs.open(output_filename, 'w', encoding='utf-8') as outf:
        for i in range(embeddings.shape[0]):
            fields = embeddings[i].tolist()
            line = index_vocab[i] + ' ' + ' '.join([str(f) for f in fields]) + '\n'
            outf.write(line)
            
def load_lexical_relations(filename, vocab_index):
    with codecs.open(filename, 'r', encoding='utf-8') as inf:
        lines = inf.readlines()
    lines = [l.strip() for l in lines]
    fields = [l.split() for l in lines]
    vocabset = set(vocab_index.keys())
    adj_mat = np.zeros((len(vocabset), len(vocabset)), dtype=np.int)
    relations = [(f[0], f[i]) for f in fields for i in range(1, len(f)) if f[0] in vocabset and f[i] in vocabset]
    for rel in relations:
        n1, n2 = rel
        n1_idx = vocab_index[n1]
        n2_idx = vocab_index[n2]
        adj_mat[n1_idx][n2_idx] = 1
        adj_mat[n2_idx][n1_idx] = 1
    return adj_mat
    
    
if __name__ == '__main__':
    embeddings, vocab_index, index_vocab = load_embeddings(filename='/lt/work/arahimi/wordvectors/glove/glove.6B.50d.txt', n_dimensions=50)
    load_lexical_relations(filename='/home/arahimi/git/retrofitting/lexicons/wordnet-synonyms+.txt', vocab_index=vocab_index)
    dump_embeddings(output_filename='/lt/work/arahimi/wordvectors/glove/glove.6B.50d_converted.txt', embeddings=embeddings, index_vocab=index_vocab)