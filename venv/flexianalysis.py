import json
import os
import tiktoken
import together
import wandb
import pprint
import datasets
import sklearn
import evaluate
import Levenshtein as lev
from Levenshtein import distance
import random
import typing
from typing import Callable
import time
import requests
import matplotlib
from matplotlib import pyplot as plt


with open('arcT_eval_acc_70b', 'r') as f:
    acc_results = json.load(f)
with open('arcT_eval_lev_70b', 'r') as f:
    lev_results = json.load(f)
with open('ARC_train_docs.json', 'r') as f:
    train_docs = json.load(f)

pp = pprint.PrettyPrinter(indent=4)


def get_sample_idxs(answers):
    idx = []
    for key in answers:
        idx.append(key)

    return idx


def find_correct(results, idxs):
    corr_is = []
    for i in range(len(results)):
        if results[i] == 1:
            corr_is.append(idxs[i])

    return corr_is


def find_close(results, idxs):
    close_idxs = []
    for i in range(len(results)):
        if .984 <= results[i] <= 1:
            close_idxs.append(idxs[i])

    return close_idxs


def list_correct(corr_idxs, model_answers, train_docs):
    for idx in corr_idxs:
        print(model_answers[idx])
        print('/')
        print(train_docs[idx]['label'])
        if distance(model_answers[idx], train_docs[idx]['label']) > 0:
            print('DIFF:', distance(model_answers[idx], train_docs[idx]['label']))
        print('--')


def examine_high_scores(det_res, threshold):
    counter = 0
    for key in det_res:
        score = det_res[key]['score']
        if threshold <= score <= 1:
            print(det_res[key]['model_answer'])
            print('/')
            print(det_res[key]['label'])
            print('DIFF:', distance(det_res[key]['model_answer'], det_res[key]['label']))
            print('---')
            counter += 1

    print('> > For threshold', threshold, 'found this many high scores total:', counter)


def plot_score_dist(det_res):
    score_list = []
    for key in det_res:
        score_list.append(det_res[key]['score'])

    plt.hist(score_list, bins=100)
    plt.title('Levenshtein Distance Accuracy of Model Predictions from Ground Truth Answers')
    plt.ylabel('Number of predictions')
    plt.xlabel('Levenshtein Distance Accuracy (1 - LD(pred, label) / len(label))')
    plt.show()


def not_main():
    idxs_acc = get_sample_idxs(acc_results['model_answers'])
    idxs_lev = get_sample_idxs(lev_results['model_answers'])

    acc_corr = find_correct(acc_results['sample_scores'], idxs_acc)
    lev_corr = find_close(lev_results['sample_scores'], idxs_lev)

    print(len(acc_corr), len(lev_corr))

    list_correct(acc_corr, acc_results['model_answers'], train_docs)
    print('###')
    list_correct(lev_corr, lev_results['model_answers'], train_docs)

    print('acc', acc_results['total_score'], acc_results['total_samples'], acc_results['total_score'] / acc_results['total_samples'])
    print('lev', lev_results['total_score'], lev_results['total_samples'], lev_results['total_score'] / lev_results['total_samples'])


def main():
    det_res_acc = acc_results['detailed_score']
    det_res_lev = lev_results['detailed_score']

    # pp.pprint(det_res_acc)

    examine_high_scores(det_res_lev, .98)

    plot_score_dist(det_res_lev)


if __name__ == "__main__":
    main()

