from collections import Counter, defaultdict, namedtuple
from functools import reduce
from operator import mul
import pickle
import random
import re
import math
import sys

class NaiveBayes:
    def __init__(self, *corpora):
        self.priors = defaultdict(int)
        self.likelihood = defaultdict(lambda: defaultdict(int))
        self.features = set()
        self._establish_features(*corpora)

    def _establish_features(self, *corpora):
        for corpus in corpora:
            for tweet in corpus:
                previous_word = None
                for word in tweet.content:
                    word = word.lower()
                    self.features.add(word)
                    if previous_word is not None:
                        self.features.add(' '.join([previous_word, word]))
                        previous_word = word

    def train(self, tweets):
        prior_freqs = defaultdict(int)
        raw_likelihood = defaultdict(lambda: defaultdict(int))
        for i, tweet in enumerate(tweets, 1):
            prior_freqs[tweet.emotion] += 1

            if i%1000 == 0 and not i==0:
                print('\r%i' %i, file=sys.stderr, end='')

            previous_word = None
            for word in tweet.content:
                word = word.lower()
                raw_likelihood[tweet.emotion][word] += 1
                if previous_word is not None:
                    raw_likelihood[tweet.emotion][' '.join([previous_word, word])] += 1
                    previous_word = word

        for emotion, freq in prior_freqs.items():
            self.priors[emotion] = freq/i

        alpha = .06
        smoothing_divisor = alpha*len(self.features)
        for emotion in prior_freqs:
            emotion_freq = prior_freqs[emotion]
            for word in self.features:
                self.likelihood[emotion][word] = (raw_likelihood[emotion][word]+alpha)/(emotion_freq+smoothing_divisor)

    def predict(self, tweet):
        """ Return maximies the posterior """
        prediction = []
        for c in self.priors.keys():
            class_likelihood = self.likelihood[c]
            p = self.priors[c]
            previous_word = None
            features = set()
            for word in tweet.content:
                word = word.lower()
                features.add(word)
                if previous_word is not None:
                    features.add(' '.join([previous_word, word]))
                    previous_word = word
            l = reduce(mul, (class_likelihood[feature] for feature in features), 1)
            prediction.append((p*l, c))

        return max(prediction)[1]

    def save(self, path):
        with open(path, 'wb') as h:
            pickle.dump((self.priors, self.likelihood), h)
