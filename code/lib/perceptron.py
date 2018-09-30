from collections import defaultdict
import logging
from operator import itemgetter
import random
import sys

import numpy as np

class WeightVector(dict):
    """Almost a defaultdict, but missing keys are not added automatically
    """
    def __missing__(self, k):
        return random.random()-0.5

class Perceptron:
    """Implements a perceptron classifier as introduced by Rosenblatt in 1958.
    """
    def __init__(self, classes):
        self.logger = logging.getLogger(__name__+'.Perceptron')
        self.logger.info('logging started')

        self.class_map = {c:i for i, c in enumerate(classes)}

        self.w = defaultdict(WeightVector)
        # Second weight matrix for accumulating the telescoping sum and update
        # index for averaging
        self.v = defaultdict(WeightVector)
        self.i = 1

        # This will hold the averaged weight vector
        self.w_average = None
        # The average will be recalculated on prediction if this is True
        self.average_outdated = True

    def predict(self, feature_vector):
        """Given a tweet, predicts an emotion class based on the current
        model.

        Args:
            tweet: code.lib.reader.Tweet object
        """
        if self.average_outdated:
            self.logger.info('averaging weight vector')
            self.w_average = defaultdict(WeightVector)
            for cls in self.class_map:
                for weight, value in self.w[cls].items():
                    avg_value = self.v[cls][weight]
                    self.w_average[cls][weight] = value-avg_value/self.i
            self.average_outdated = False
            self.logger.info('averaged weight vector')
        return self._predict(feature_vector, self.w_average)

    def _predict(self, feature_vector, weight_vector):
        """Given a feature_vector, predicts an emotion class based on the
        current model.

        Args:
            feature_vector: set of features that hold in the training instance.
        """
        prediction = {}
        for cls in self.class_map:
            class_weight_vector = weight_vector[cls]
            prediction[cls] = sum(class_weight_vector[f]*v for f, v in feature_vector.items())

        return prediction
        # max_class = max(prediction.items(), key=itemgetter(1))[0]
        # return max_class

    def train(self, feature_vector, gold_class):
        """Given a tweet, predicts an emotion class based on the current
        model and updates the model if predicted and correct emotion class
        differ.

        Args:
            tweet: code.lib.reader.Tweet object.
        """
        prediction = self._predict(feature_vector, self.w)
        pred_class = max(prediction.items(), key=itemgetter(1))[0]

        # Update if predicted and gold emotion differ
        if pred_class != gold_class:
            for f in feature_vector:
                self.w[gold_class][f] += feature_vector[f]
                self.w[pred_class][f] -= feature_vector[f]
                self.v[gold_class][f] += self.i
                self.v[pred_class][f] -= self.i
            self.i += 1
            self.average_outdated = True

        return prediction
