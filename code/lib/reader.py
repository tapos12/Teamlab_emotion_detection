"""
Provides data structures which read and represent Twitter text corpora.
"""

# standard library
from collections import defaultdict
from functools import partial
import io
import logging
from multiprocessing.pool import ThreadPool, Pool
import numpy as np
import os
import pickle
import random
import re
import subprocess
import sys

from PIL import Image
from scipy.ndimage import imread


def load_image(tweet, image_res):
    path = tweet.image_path
    try:
        buf = io.BytesIO(subprocess.check_output(['convert', '-scale', '%ix%i!' %image_res, path, 'png:-'], stderr=subprocess.DEVNULL))
    except:
        return
    else:
        try:
            image = imread(buf)
            no_pixel = image.shape[2]
            # Repeat first channel if image is grayscale
            if no_pixel < 3:
                image[:,:,1] = image[:,:,2] = image[:,:,0]
            # Delete alpha channel
            elif no_pixel == 4:
                image = image[:,:,:3]
        except:
            image = None
        else:
            shape = image.shape
            # Repeat first channel if image is grayscale
            if len(shape) == 2:
                new_image = np.empty((shape[0], shape[1], 3))
                new_image[:,:,0] = new_image[:,:,1] = new_image[:,:,2] = image
                image = new_image
            elif shape[2] == 2:
                new_image = np.empty((shape[0], shape[1], 3))
                new_image[:,:,0] = new_image[:,:,1] = new_image[:,:,2] = image[:,:,0]
                image = new_image
            # Delete alpha channel
            elif shape[2] == 4:
                image = image[:,:,:3]
        return image


class EmptyCorpus(Exception):
    pass

class IncompletePrediction(Exception):
    pass


def _realabspath(path):
    if path.startswith('~'):
        path = os.path.expanduser('~')+path[1:]
    return os.path.realpath(path)


def clean_tweet_content(content):
    return tuple(re.sub('^@[^[]+', '@USERNAME', token) for token in content)
    clean_content = []
    for token in content:
        for new_token in re.split('\[NEWLINE\]', token):
            if new_token:
                new_token = re.sub('^@.*', '@USERNAME', new_token.strip())
                clean_content.append(new_token)
    return tuple(clean_content)

def index_image_dir(image_path):
    lookup = {}
    for p, ds, fs in os.walk(_realabspath(image_path), topdown=True):
        for f in fs:
            lookup[f] = os.path.join(p, f)
        ds[:] = filter(lambda x: x!='dirother', ds)
    return lookup

class Tweet:
    def __init__(self, tweet_id, emotion, content, image_path=None, prediction=None):
        self.tweet_id = tweet_id
        self.emotion = emotion
        # self.content = clean_tweet_content(content)
        self.content = content
        self.image_path = image_path
        self.feature_vector = None
        self.prediction = prediction

#     # Don't keep a reference to the actual data so it can be garbage collected
#     @property
#     def image(self):
#         try:
#             buf = io.BytesIO(subprocess.check_output(['convert', '-scale', '12x12!', self._image_path, 'png:-']))
#         except subprocess.CalledProcessError:
#             return
#         else:
#             return imread(buf)
#         return Image.open(self._image_path)

    def __str__(self):
        if self.prediction is not None:
            return "gold: %s prediction: %s content: %s" %(
                self.emotion, self.prediction, self.content)
        else:
            return "gold: %s content: %s" %(
                self.emotion, self.content)

class Corpus:
    def __init__(self, path, image_path=None, image_index_path=None, image_res=(12, 12)):
        self.logger = logging.getLogger(__name__+'.Corpus')
        self.logger.info('logging started')

        self.path = path
        self.image_path = image_path
        self.image_index_path = image_index_path
        self.image_res = image_res
        self.tweets = []
        self._indices = []
        self._tweet_id_map = {}
        self._read_tweets()

    def __iter__(self):
        indices = self._indices[:]
        for i in indices:
            yield self.tweets[i]

    def __len__(self):
        return len(self.tweets)

    def _read_tweets(self):
        if self.image_path is not None:
            try:
                with open(self.image_index_path, 'rb') as handle:
                    image_lookup = pickle.load(handle)
                self.logger.info('loaded image index from disk')
            except:
                self.logger.info('indexing image directory, this might take some time')
                image_lookup = index_image_dir(self.image_path)
                with open(self.image_index_path, 'wb') as handle:
                    pickle.dump(image_lookup, handle)

        with open(self.path) as handle:
            for line in handle:
                tweet_fields = line.strip('\n').split('\t')
                emotion, _, _, tweet_id, _, _, _, _, content, *_ = tweet_fields
                if self.image_path:
                    try:
                        image_path = image_lookup[tweet_id]
                    except KeyError:
                        pass
                    else:
                        tweet = Tweet(tweet_id, emotion, tuple(content.split(' ')), image_path)
                        self.tweets.append(tweet)
                        self._tweet_id_map[tweet_id] = len(self._indices)
                        self._indices.append(len(self._indices))
                else:
                    tweet = Tweet(tweet_id, emotion, tuple(content.split(' ')))
                    self.tweets.append(tweet)
                    self._tweet_id_map[tweet_id] = len(self._indices)
                    self._indices.append(len(self._indices))

        if self.image_path is not None:
            pool = Pool()
            for i, result in enumerate(pool.imap(partial(load_image, image_res=self.image_res), self.tweets, chunksize=100)):
                self.tweets[i].image = result
            pool.close()
            pool.join()

        self.logger.info('read %i tweets from %s' %(len(self.tweets), self.path))

    def shuffle(self):
        random.shuffle(self._indices)

    def get_tweet(self, tweet_id):
        return self.tweets[self._tweet_id_map[tweet_id]]

    def evaluate(self):
        if not self.tweets:
            raise EmptyCorpus

        emotions = set()
        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)

        global_tp = 0
        global_fp = 0

        for tweet in self.tweets:
            if tweet.prediction is None:
                continue
                raise IncompletePrediction
            emotions.add(tweet.emotion)
            emotions.add(tweet.prediction)
            if tweet.emotion == tweet.prediction:
                tp[tweet.prediction] += 1
                global_tp += 1

            else:
                fp[tweet.prediction] += 1
                fn[tweet.emotion] += 1
                global_fp += 1

        eval_results = {}
        eval_results['accuracy'] = 100*(global_tp/(global_tp+global_fp))
        eval_results['emotions'] = {}

        for emotion in sorted(emotions):
            eval_results['emotions'][emotion] = {}
            try:
                precision = 100*(tp[emotion]/(tp[emotion]+fp[emotion]))
            except ZeroDivisionError:
                precision = 0
            try:
                recall = 100*(tp[emotion]/(tp[emotion]+fn[emotion]))
            except ZeroDivisionError:
                recall = 0
            try:
                f_score = 2*((precision*recall)/(precision+recall))
            except ZeroDivisionError:
                f_score = 0

            eval_results['emotions'][emotion]['precision'] = precision
            eval_results['emotions'][emotion]['recall'] = recall
            eval_results['emotions'][emotion]['f_score'] = f_score

        eval_results['macro_avg_f_score'] = sum(eval_results['emotions'][emotion]['f_score'] for emotion in emotions)/len(emotions)

        return eval_results

    def __str__(self):
        string = ""
        for tweet in self.tweets:
            string += str(tweet) + '\n'
        return string
