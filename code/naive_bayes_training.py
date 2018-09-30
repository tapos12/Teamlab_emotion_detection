#!/usr/bin/python3

"""
Usage: naive.py <pred> <train>
"""
import random
import docopt
import re
import sys
import math
import time
from collections import Counter, defaultdict, namedtuple
from lib.reader import Corpus
from lib.naive_bayes import NaiveBayes
from lib.eval import pformat

def main(pred_path, gold_path):

    pred_corpus = Corpus(pred_path)
    train_corpus = Corpus(gold_path)

    Nb = NaiveBayes(train_corpus, pred_corpus)
    train_start_timestamp = time.time()
    Nb.train(train_corpus)

    train_elapsed = time.time()-train_start_timestamp

    i = 0
    for tweet in pred_corpus:
        if i%1000 == 0 and not i==0:
            print('\r%i' %i, file=sys.stderr, end='')
        i+=1
        tweet.prediction = Nb.predict(tweet)

    eval_results = pred_corpus.evaluate()

    print('\r'+pformat(eval_results)+'\n')
    print(' == Macro Avg F_Score: %.2f ==\n' %(eval_results['macro_avg_f_score']))
    print(' == Eval Accuracy:     %.2f ==\n' %(eval_results['accuracy']))
    print(' %s\n' %('-'*78))


if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    pred_path = args['<pred>']
    gold_path = args['<train>']
    main(pred_path, gold_path)
