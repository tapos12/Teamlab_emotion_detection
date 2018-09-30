from collections import defaultdict
import colorsys
from functools import partial
import logging
import math
from multiprocessing.pool import Pool
import numpy as np
import sys

from PIL import ImageStat
import numpy as np

class TextualFeatureExtractor:
    def __init__(self, corpora, cache=True, **kwargs):
        self.logger = logging.getLogger(__name__+'.TextualFeatureExtractor')
        self.logger.info('logging started')
        if cache:
            self.logger.info('caching enabled')

        self.cache = {} if cache else None
        self.classes = set()
        self.features = {}
        self._establish_feats(corpora)

        self.logger.info('found %i classes and %i features'
                         %(len(self.classes), len(self.features)))

    def _establish_feats(self, corpora):
        """Iterates over the given corpus to populate sets of all emotions and
        features.

        Args:
            tweets: An iterable of code.lib.reader.Tweet objects.
        """
        for corpus in corpora:

            i = 0
            pool = Pool()
            for tweet, feature_vector in pool.imap_unordered(partial(get_text_feature_vector, features=None, init=True), corpus, chunksize=1000):
                i += 1
                if i%100 == 0:
                    print('\r%i' %i, file=sys.stderr, end='')
                if self.cache is not None:
                    self.cache[tweet.tweet_id] = feature_vector
                self.features.update(**feature_vector)
                self.classes.add(tweet.emotion)
            pool.close()
            pool.join()

    def get_feature_vector(self, tweet):
        """Given a tweet, returns the corresponding feature vector from cache
        if available or calls _generate_feature_vector() to retrieve it.

        Args:
            tweet: A code.lib.reader.Tweet object.
        """
        if self.cache is not None:
            try:
                return self.cache[tweet.tweet_id]
            except KeyError:
                feature_vector = self._generate_feature_vector(tweet)
                self.cache[tweet.tweet_id] = feature_vector
                return feature_vector
        else:
            return self._generate_feature_vector(tweet)



class ImageFeatureExtractor:
    def __init__(self, corpora, predictions=[], cache=True, resolution=(12, 12), num_bins=3, **kwargs):
        self.logger = logging.getLogger(__name__+'.ImageFeatureExtractor')
        self.logger.info('logging started')
        self.predictions = predictions
        if cache:
            self.logger.info('caching enabled')

        self.cache = {} if cache else None
        self.resolution = resolution
        self.num_bins = num_bins
        # self.bin_width = 256/num_bins
        self.classes = set()
        self.features = {}
        self._establish_feats(corpora)

        self.logger.info('found %i classes and %i features'
                         %(len(self.classes), len(self.features)))

    def _establish_feats(self, corpora):
        """Iterates over the given corpus to populate sets of all emotions and
        features.

        Args:
            tweets: An iterable of code.lib.reader.Tweet objects.
        """
        for corpus in corpora:

            i = 0
            pool = Pool()
            for tweet, feature_vector in pool.imap_unordered(partial(get_image_feature_vector, predictions=self.predictions), corpus, chunksize=1000):
                i += 1
                if i%100 == 0:
                    print('\r%i' %i, file=sys.stderr, end='')
                if self.cache is not None:
                    self.cache[tweet.tweet_id] = feature_vector
                self.features.update(**feature_vector)
                self.classes.add(tweet.emotion)
            pool.close()
            pool.join()

    def get_feature_vector(self, tweet):
        """Given a tweet, returns the corresponding feature vector from cache
        if available or calls _generate_feature_vector() to retrieve it.

        Args:
            tweet: A code.lib.reader.Tweet object.
        """
        if self.cache is not None:
            try:
                return self.cache[tweet.tweet_id]
            except KeyError:
                try:
                    tweet, feature_vector = get_image_feature_vector(tweet, self.predictions)
                except ZeroDivisionError:
                    feature_vector = {'__bias__': 1.0}
                self.cache[tweet.tweet_id] = feature_vector
                return feature_vector
        else:
            return get_image_feature_vector(tweet, self.predictions)


def get_text_feature_vector(tweet, features, init=False):
    """Calculates a feature vector given a tweet and returns it as set.
    """
    feature_vector = {'__bias__': 1}
    previous_word = None
    for word in tweet.content:
        feature_vector['word=%s' %word] = 1
        feature_vector['lower=%s' %word.lower()] = 1
        feature_vector['is_alpha=%s' %word.isalpha()] = 1
        feature_vector['is_digit=%s' %word.isdigit()] = 1
        feature_vector['is_lower=%s' %word.islower()] = 1
        feature_vector['is_upper=%s' %word.isupper()] = 1
        feature_vector['is_title=%s' %word.istitle()] = 1
        feature_vector['first2=%s' %word[:2]] = 1
        feature_vector['last2=%s' %word[-2:]] = 1
        feature_vector['first3=%s' %word[:3]] = 1
        feature_vector['last3=%s' %word[-3:]] = 1
        feature_vector['first4=%s' %word[:4]] = 1
        feature_vector['last4=%s' %word[-4:]] = 1
        if not init and word not in features:
            word = '__UNKNOWN__'
        if previous_word is not None:
            feature_vector['word-1_word=%s' %' '.join([previous_word, word])] = 1
        previous_word = word

    return tweet, feature_vector


def get_image_feature_vector(tweet, predictions=[]):
    image = tweet.image
    if image is None:
        return tweet, {'__bias__': 1.0}
    norm = image.shape[0] * image.shape[1]

    medians = []
    brightnesses = []
    sat = []
    hue_all = []
    for row in image:
        for rgb in row:
            if issubclass(np.uint8, type(rgb)):
               rgb = (rgb, rgb, rgb)
            elif len(rgb) == 2:
               rgb = (rgb[0], rgb[0], rgb[0])

            brightnesses.append(sum(rgb)/3)
            hsv = colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])
            sat.append(hsv[1]*255)
            r, g, b = colorsys.hsv_to_rgb(hsv[0], 1, 255)
            hue_all.append(hsv[0]*360)

    feature_vector = defaultdict(float)
    feature_array = [brightnesses, sat, hue_all]

    for i, features in enumerate(feature_array):
        if i==0:
            feature_name = 'brightness'
        elif i==1:
            feature_name = 'saturation'
        elif i==2:
            feature_name = 'hue'

        if i == 2:
            num_bins = 9
            bin_width = 360/num_bins
            for feature in features:
                location, remainder = divmod(feature, bin_width)
                location = location % num_bins
                if remainder == 0:
                    feature_vector['%s_bin_%i' %(feature_name, location)] += 1.0
                else:
                    lower_bin = math.floor(location)
                    upper_bin = (lower_bin + 1) % num_bins
                    feature_vector['%s_bin_%i' %(feature_name, lower_bin)] += (bin_width-remainder)/bin_width
                    feature_vector['%s_bin_%i' %(feature_name, upper_bin)] += remainder/bin_width

        else:
            num_bins = 8
            bin_width = 256/num_bins
            for feature in features:
                if feature <= bin_width/2:
                    feature_vector[feature_name+'_bin_%i' %(0)] += 1.0
                elif feature >= 256-bin_width/2:
                    feature_vector[feature_name+'_bin_%i' %(num_bins-1)] += 1.0
                else:
                    location, remainder = divmod(feature, bin_width)
                    if remainder == 0:
                        feature_vector[feature_name+'_bin_%i' %(location)] += 1.0
                    else:
                        lower_bin = math.floor(location)
                        upper_bin = lower_bin + 1
                        feature_vector[feature_name+'_bin_%i' %(lower_bin)] += (bin_width-remainder)/bin_width
                        feature_vector[feature_name+'_bin_%i' %(upper_bin)] += remainder/bin_width

    for feature in feature_vector:
        feature_vector[feature] = feature_vector[feature]/norm

    for prediction in predictions:
        for emotion, value in prediction[tweet.tweet_id].items():
            feature_vector['bow_%s' %emotion] = sig(value)

    feature_vector['__bias__'] = 1.0

    return tweet, feature_vector


def sig(x):
    try:
        value = 1/(1+math.exp(-x))
    except OverflowError:
        value = 0
    return value
