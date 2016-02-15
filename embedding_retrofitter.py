'''
Created on 8 Feb 2016

@author: af
'''
import codecs
import numpy as np
import scipy.sparse as sp
import pdb
from scipy.sparse import csr_matrix, lil_matrix
from subprocess import call
from embedding_optimiser import edgexplain_retrofitting, iterative_edgexplain_retrofitting
import sys
from pandas import DataFrame
def load_embeddings(filename, n_dimensions=50, vocab=set()):
    with codecs.open(filename, 'r', encoding='utf-8') as inf:
        lines = inf.readlines()
    fields = [l.strip().split(' ') for l in lines]
    embeddings = np.array([f[1:] for f in fields if f[0] in vocab], dtype=float)
    vocab = [f[0] for f in fields if f[0] in vocab]
    del fields
    vocab_index = {v:i for i, v in enumerate(vocab)}
    index_vocab = {i:v for i, v in enumerate(vocab)}
    return embeddings, vocab_index, index_vocab
    
def dump_embeddings(output_filename, embeddings, index_vocab):
    #embeddings = np.nan_to_num(embeddings)
    with codecs.open(output_filename, 'w', encoding='utf-8') as outf:
        for i in range(embeddings.shape[0]):
            fields = embeddings[i].tolist()
            line = index_vocab[i] + ' ' + ' '.join([str(f) for f in fields]) + '\n'
            outf.write(line)
            
def load_lexical_relations(filename, vocab_index, sparse_output=True):
    with codecs.open(filename, 'r', encoding='utf-8') as inf:
        lines = inf.readlines()
    lines = [l.strip() for l in lines]
    fields = [l.split() for l in lines]
    vocabset = set(vocab_index.keys())
    adj_mat = lil_matrix((len(vocabset), len(vocabset)), dtype=np.int)
    relations = [(f[0], f[i]) for f in fields for i in range(1, len(f)) if f[0] in vocabset and f[i] in vocabset]
    for rel in relations:
        n1, n2 = rel
        n1_idx = vocab_index[n1]
        n2_idx = vocab_index[n2]
        adj_mat[n1_idx, n2_idx] = 1
        adj_mat[n2_idx, n1_idx] = 1
    if sparse_output:
        return adj_mat.tocsr()
    else:
        return adj_mat.toarray()

def toy_example():
    # labels            ccc,cc',ccc'',hh,hh'    the first three are ccc', hh the next three are ccc'', hh the last three are ccc, hh'
    # so the majority are hh but to explain all the links the cc of the 10th row should be ccc not ccc' or ccc''.
    # we expect the last row to become 1,0,0,1,0 (ccc, hh).
    embeddings = np.array([
                          [0,1,0,1,0],
                          [0,1,0,1,0],
                          [0,1,0,1,0],
                          [0,0,1,1,0],
                          [0,0,1,1,0],
                          [0,0,1,1,0],
                          [1,0,0,0,1],
                          [1,0,0,0,1],
                          [1,0,0,0,1],
                          [0,0,0,0,0]
                          ]
                          )
    #the first 3, the second 3 and the third 3 are connected to each other. the 10th is connected to all.
    A = np.array(
                 [
                 [0,1,1,0,0,0,0,0,0,1],
                 [1,0,1,0,0,0,0,0,0,1],
                 [1,1,0,0,0,0,0,0,0,1],
                 [0,0,0,0,1,1,0,0,0,1],
                 [0,0,0,1,0,1,0,0,0,1],
                 [0,0,0,1,1,0,0,0,0,1],
                 [0,0,0,0,0,0,0,1,1,1],
                 [0,0,0,0,0,0,1,0,1,1],
                 [0,0,0,0,0,0,1,1,0,1],
                 [1,1,1,1,1,1,1,1,1,0]
                 ]
                 )
    
    for edgexplain_coef in [0.001]:
        for alpha in [10]:
            for c in [0]:
                print edgexplain_coef, alpha, c
                new_embeddings = edgexplain_retrofitting(embeddings, csr_matrix(A), iterations=100, learning_rate=0.01, alpha=alpha, c=c, lambda1=edgexplain_coef)
                print DataFrame(new_embeddings)
    new_embeddings = iterative_edgexplain_retrofitting(embeddings, csr_matrix(A), iterations=10, c_k=None, alpha=10, c=1)
    print DataFrame(new_embeddings)
    sys.exit()
if __name__ == '__main__':
    toy_example()
    print 'limit vocab...'
    with codecs.open('wordsim.txt', 'r', encoding='utf-8') as inf:
        text = inf.read()
    words = text.split()
    words = set([w.strip().lower() for w in words])
    print 'num vocab', len(words)
        
    print 'loading embeddings...'
    embeddings, vocab_index, index_vocab = load_embeddings(filename='/lt/work/arahimi/wordvectors/glove/glove.6B.50d.txt', n_dimensions=50, vocab=words)
    print 'embedding size', embeddings.shape
    print 'loading lexicon...'
    A = load_lexical_relations(filename='/home/arahimi/git/retrofitting/lexicons/wordnet-synonyms+.txt', vocab_index=vocab_index, sparse_output=True)
    print 'relations size', A.shape
    
    for edgexplain_coef in [0.001, 0.01, 0.1, 1]:
        for alpha in [10, 1]:
            for c in [0, 10]:
                print 'hopefully improving embeddings by lexicon and edgexplain... with coef=', edgexplain_coef, alpha, c
                new_embeddings = edgexplain_retrofitting(embeddings, A, iterations=5, learning_rate=0.1, alpha=alpha, c=c, lambda1=edgexplain_coef)
                print 'dumping the hopefully improved embeddings...'
                dump_embeddings(output_filename='/lt/work/arahimi/wordvectors/glove/glove.6B.50d_edgexplain_' + str(edgexplain_coef) + '_' + str(alpha) +'_' + str(c) + '.txt', embeddings=new_embeddings, index_vocab=index_vocab)
                #print 'original embedding evaluation...'
                #call("python /home/arahimi/git/embedding-evaluation/wordsim/wordsim.py -l en -v /lt/work/arahimi/wordvectors/glove/glove.6B.50d.txt")
                #print 'explained embedding evaluation...'
                #call("python /home/arahimi/git/embedding-evaluation/wordsim/wordsim.py -l en -v /lt/work/arahimi/wordvectors/glove/glove.6B.50d_converted.txt")
            