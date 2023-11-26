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

pp = pprint.PrettyPrinter(indent=4)

with open('arcT_eval_acc_70b', 'r') as f:
    acc70b_results = json.load(f)
with open('arcT_eval_lev_70b', 'r') as f:
    lev70b_results = json.load(f)
with open('arcT_eval_acc_13b2', 'r') as f:
    acc13b_results = json.load(f)
with open('arcT_eval_lev_13b2', 'r') as f:
    lev13b_results = json.load(f)
with open('arcT_eval_acc_70b_pe', 'r') as f:
    acc70A_results = json.load(f)
with open('arcT_eval_lev_70b_pe', 'r') as f:
    lev70A_results = json.load(f)

with open('ARC_train_docs.json', 'r') as f:
    train_docs = json.load(f)


def plot_score_dist(det_res, name):
    score_list = []
    for key in det_res:
        score_list.append(det_res[key]['score'])

    plt.hist(score_list, bins=100)
    plt.title('Levenshtein Dist. Accuracy of Model Predictions, ' + name)
    plt.ylabel('Number of predictions')
    plt.xlabel('Levenshtein Dist. Accuracy (1 - LD(pred, label) / len(label))')
    plt.savefig(name + '_scoredist.png', dpi=300)
    plt.show()


def main():
    # Table values
    print('13B, acc:',
          acc13b_results['total_score']/acc13b_results['total_samples'],
          ' | lev:',
          lev13b_results['total_score']/lev13b_results['total_samples'])
    print('70B, acc:',
          acc70b_results['total_score'] / acc70b_results['total_samples'],
          ' | lev:',
          lev70b_results['total_score'] / lev70b_results['total_samples'])
    print('70B PE, acc:',
          acc70A_results['total_score'] / acc70A_results['total_samples'],
          ' | lev:',
          lev70A_results['total_score'] / lev70A_results['total_samples'])

    # Score dists
    plot_score_dist(lev13b_results['detailed_score'], 'LLaMA2 13B')
    plot_score_dist(lev70b_results['detailed_score'], 'LLaMA2 70B')
    plot_score_dist(lev70A_results['detailed_score'], 'LLaMA2 70B, Ctx. Aug.')


if __name__ == "__main__":
    main()


