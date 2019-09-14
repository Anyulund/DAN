# neural_sentiment_classifier.py

import argparse
import sys
import time
from models import *
from sentiment_data import *
from typing import List

####################################################
#  #
####################################################
"""
Command to run the DAN on movie dataset:
python3 neural_sentiment_classifier.py --model DAN --word_vecs_path glove.6B.300d-relativized.txt --train_path train.txt --dev_path dev.txt --blind_test_path test.txt
"""

def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--model', type=str, default='DAN', help='model to run (TRIVIAL or DAN)')
    parser.add_argument('--train_path', type=str, default='train.txt', help='path to train set (you should not need to modify)')
    parser.add_argument('--dev_path', type=str, default='dev.txt', help='path to dev set (you should not need to modify)')
    parser.add_argument('--blind_test_path', type=str, default='test.txt', help='path to blind test set (you should not need to modify)')
    parser.add_argument('--test_output_path', type=str, default='test.output.txt', help='output path for test predictions')
    parser.add_argument('--no_run_on_test', dest='run_on_test', default=True, action='store_false', help='skip printing output on the test set')
    parser.add_argument('--word_vecs_path', type=str, default='glove.6B.300d-relativized.txt', help='path to word embeddings to use')
    parser.add_argument('--method', type=str, help='path to word embeddings to use')
    parser.add_argument('--func', type=str, help='path to word embeddings to use')
    # Some common args have been pre-populated for you. Again, you can add more during development, but your code needs
    # to run with the default neural_sentiment_classifier for submission.
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--hidden_size', type=int, default=100, help='hidden layer size')
    parser.add_argument('--batch_size', type=int, default=1, help='training batch size; 1 by default and you do not need to batch unless you want to')
    args = parser.parse_args()
    return args


def evaluate(classifier, exs):
    """
    Evaluates a given classifier on the given examples
    :param classifier: classifier to evaluate
    :param exs: the list of SentimentExamples to evaluate on
    :return: None (but prints output)
    """
    print_evaluation([ex.label for ex in exs], [classifier.predict(ex) for ex in exs])


def print_evaluation(golds: List[SentimentExample], predictions: List[SentimentExample]):
    """
    Prints evaluation statistics comparing golds and predictions, each of which is a sequence of 0/1 labels.
    Prints accuracy as well as precision/recall/F1 of the positive class, which can sometimes be informative if either
    the golds or predictions are highly biased.

    :param golds: gold SentimentExample objects
    :param predictions: pred SentimentExample objects
    :return:
    """
    num_correct = 0
    num_pos_correct = 0
    num_pred = 0
    num_gold = 0
    num_total = 0
    if len(golds) != len(predictions):
        raise Exception("Mismatched gold/pred lengths: %i / %i" % (len(golds), len(predictions)))
    for idx in range(0, len(golds)):
        gold = golds[idx]
        prediction = predictions[idx]
        if prediction == gold:
            num_correct += 1
        if prediction == 1:
            num_pred += 1
        if gold == 1:
            num_gold += 1
        if prediction == 1 and gold == 1:
            num_pos_correct += 1
        num_total += 1
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
    prec = float(num_pos_correct) / num_pred if num_pred > 0 else 0.0
    rec = float(num_pos_correct) / num_gold if num_gold > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec > 0 and rec > 0 else 0.0
    print("Precision: %i / %i = %f" % (num_pos_correct, num_pred, prec))
    print("Recall: %i / %i = %f" % (num_pos_correct, num_gold, rec))
    print("F1: %f" % f1)


if __name__ == '__main__':
    args = _parse_args()
    print(args)

    # Load train, dev, and test exs and index the words.
    train_exs = read_sentiment_examples(args.train_path)
    dev_exs = read_sentiment_examples(args.dev_path)
    test_exs = read_sentiment_examples(args.blind_test_path)
    print(repr(len(train_exs)) + " / " + repr(len(dev_exs)) + " / " + repr(len(test_exs)) + " train/dev/test examples")

    word_embeddings = read_word_embeddings(args.word_vecs_path)

    # Train and evaluate
    start_time = time.time()
    if args.model == "DAN":
        model = train_deep_averaging_network(args, train_exs, dev_exs, word_embeddings)
    else:
        model = TrivialSentimentClassifier()
    print("=====Train Accuracy=====")
    evaluate(model, train_exs)
    print("=====Dev Accuracy=====")
    evaluate(model, dev_exs)
    print("Time for training and evaluation: %.2f seconds" % (time.time() - start_time))

    # Write the test set output
    if args.run_on_test:
        test_exs_predicted = [SentimentExample(ex.words, model.predict(ex)) for ex in test_exs]
        write_sentiment_examples(test_exs_predicted, args.test_output_path)
