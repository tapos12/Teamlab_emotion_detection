#!/usr/bin/python3

"""
Usage: eval <prediction_path> <gold_path>
"""

# standard library
from collections import defaultdict

# third-party
import docopt

# package-level
from lib.reader import Corpus

def main(pred_path, gold_path):
    """Evaluates prediction results against a gold standard and prints the
    resulting precision, recall and F-Score to stdout.

    Args:
        pred_path: Path to a file with prediction results.
        gold_path: Path to a file with the corresponding gold standard to
            compare against.
    """
    corpus = Corpus(gold_path, pred_path)
    eval_results = corpus.evaluate()

    print(pformat(eval_results)+'\n')

    print(' == Accuracy: %.2f ==' % (eval_results['accuracy']))

def pformat(eval_results):
    string = '\r'
    eval_width = 10
    emotion_width = len(max(eval_results['emotions'], key=lambda x: len(x)))

    string += ' '*eval_width
    for emotion in sorted(eval_results['emotions']):
        string += ' %s ' %emotion.rjust(emotion_width)

    string += '\n%s' %'Precision'.rjust(eval_width)
    for emotion in sorted(eval_results['emotions']):
        precision = '%.2f' %eval_results['emotions'][emotion]['precision']
        string += ' %s ' %precision.rjust(emotion_width)

    string += '\n%s' %'Recall'.rjust(eval_width)
    for emotion in sorted(eval_results['emotions']):
        recall = '%.2f' %eval_results['emotions'][emotion]['recall']
        string += ' %s ' %recall.rjust(emotion_width)

    string += '\n%s' %'F-Score'.rjust(eval_width)
    for emotion in sorted(eval_results['emotions']):
        f_score = '%.2f' %eval_results['emotions'][emotion]['f_score']
        string += ' %s ' %f_score.rjust(emotion_width)

    return string


if __name__ == '__main__':
    args = docopt.docopt(__doc__)

    pred_path = args['<prediction_path>']
    gold_path = args['<gold_path>']

    main(pred_path, gold_path)
