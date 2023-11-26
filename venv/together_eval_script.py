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


SEP = ','
WANDB_API_KEY = "3ba2e8bcbdd054c8b4d1d239a4a95d16bf3005c1"
TOGETHER_API_KEY = "146706bab46d0101232eb1519664fb6e53c5be853f85aae3cfa9abd266e9434d"
together.api_key = TOGETHER_API_KEY

LLAMA2_7B_RAW = 'togethercomputer/llama-2-7b'
LLAMA2_13B_RAW = 'togethercomputer/llama-2-13b'
LLAMA2_70B_RAW = 'togethercomputer/llama-2-70b'
LLAMA2_34B_P = 'togethercomputer/CodeLlama-34b-Python'

EVAL_MODEL = LLAMA2_13B_RAW
EVAL_CONFIG = {
    'data': 'ARC eval',
    'portion': 1,
    'default model': EVAL_MODEL
}

pp = pprint.PrettyPrinter(indent=4)


def model_completion(prompt: str, model: str = EVAL_MODEL, max_tokens: int = 1000) -> str:
    response = together.Complete.create(prompt=prompt,
                                        model=model,
                                        max_tokens=max_tokens,
                                        stop=['\n\n'],
                                        temperature=0.0,
                                        top_p=0.7,
                                        top_k=50,
                                        repetition_penalty=0.0,
                                        logprobs=None,
                                        api_key=TOGETHER_API_KEY,
                                        cast=False,
                                        )

    return response['output']['choices'][0]['text']


def eval_lev(ref: str, pred: str) -> float:
    return 1.0 - (distance(pred, ref) / len(ref))


def eval_lev_strict(ref: str, pred: str) -> float:
    ref1 = ref.replace(',', '')
    pred1 = pred.replace(',', '')

    return 1.0 - (distance(pred1, ref1) / len(ref1))


def eval_acc(ref: str, pred: str) -> float:
    return int(ref == pred)


def save_eval_results(total_score, succ_eval, str_metric, data_source, model, detailed_score, save_path):
    results = {
        'total_score': total_score,
        'total_samples': succ_eval,
        'metric': str_metric,
        'data_source': data_source,
        'model': model,
        'detailed_score': detailed_score
    }

    with open(save_path, 'w') as f:
        json.dump(results, f)


def model_eval(metric: Callable[[str, str], float],
               data_source: str = 'ARC_train_docs.json',
               save_path: str = 'ARC_eval_results.json',
               model: str = EVAL_MODEL,
               num_samples: int = None) -> dict:

    with open(data_source, 'r') as f:
        eval_docs = json.load(f)
    total_score = 0
    succ_eval = 0
    detailed_score = {}

    for doc in eval_docs:
        prompt = eval_docs[doc]['input']
        label = eval_docs[doc]['label']

        try:
            pred = model_completion(prompt, model=model, max_tokens=len(label)*2)
            score = metric(label, pred[:-2])        # pred[:-2] to get rid of stopping token \n\n

            print(doc, succ_eval, len(label), len(pred), score)

            total_score += score
            detailed_score[doc] = {'model_answer': pred[:-2],
                                   'label': label,
                                   'score': score}
            succ_eval += 1

            if num_samples is not None and succ_eval == num_samples:
                break

            if succ_eval % 50 == 0:     # Basically check pointing eval results
                save_eval_results(total_score,
                                  succ_eval,
                                  str(metric),
                                  data_source,
                                  model,
                                  detailed_score,
                                  save_path)

        except requests.exceptions.HTTPError:
            print(doc, 'failed, specifically 400 error prob')

    results = {
        'total_score': total_score,
        'total_samples': succ_eval,
        'metric': str(metric),
        'data_source': data_source,
        'model': model,
        'detailed_score': detailed_score,
    }

    with open(save_path, 'w') as f:
        json.dump(results, f)

    return results


def main():
    arc_train_docs = 'ARC_train_docs.json'
    # arc_eval_docs = 'ARC_eval_docs.json'

    # testresults1 = model_eval(eval_lev,
    #                       data_source=arc_train_docs,
    #                       model=LLAMA2_7B_RAW,
    #                       save_path='arcT_eval_lev_7b',)
    # testresults2 = model_eval(eval_acc,
    #                       data_source=arc_train_docs,
    #                       model=LLAMA2_7B_RAW,
    #                       save_path='arcT_eval_acc_7b',)

    # results1 = model_eval(eval_lev,
    #                       data_source=arc_train_docs,
    #                       model=LLAMA2_13B_RAW,
    #                       save_path='arcT_eval_lev_13b',)
    # results2 = model_eval(eval_acc,
    #                       data_source=arc_train_docs,
    #                       model=LLAMA2_13B_RAW,
    #                       save_path='arcT_eval_acc_13b',)
    #
    # pp.pprint(results1)
    # pp.pprint(results2)

    # results3 = model_eval(eval_lev,
    #                       data_source=arc_train_docs,
    #                       model=LLAMA2_70B_RAW,
    #                       save_path='arcT_eval_lev_70b',)
    # results4 = model_eval(eval_acc,
    #                       data_source=arc_train_docs,
    #                       model=LLAMA2_70B_RAW,
    #                       save_path='arcT_eval_acc_70b',)
    #
    # pp.pprint(results3)
    # pp.pprint(results4)

    # results5 = model_eval(eval_lev,
    #                       data_source=arc_train_docs,
    #                       model=LLAMA2_13B_RAW,
    #                       save_path='arcT_eval_lev_13b2', )
    # results6 = model_eval(eval_acc,
    #                       data_source=arc_train_docs,
    #                       model=LLAMA2_13B_RAW,
    #                       save_path='arcT_eval_acc_13b2', )
    #
    # pp.pprint(results5)
    # pp.pprint(results6)

    results7 = model_eval(eval_lev,
                          data_source=arc_train_docs,
                          model=LLAMA2_34B_P,
                          save_path='arcT_eval_lev_34bp', )
    results8 = model_eval(eval_acc,
                          data_source=arc_train_docs,
                          model=LLAMA2_34B_P,
                          save_path='arcT_eval_acc_34bp', )

    pp.pprint(results7)
    pp.pprint(results8)


if __name__ == "__main__":
    # wandb.login(key=WANDB_API_KEY)
    #
    # wandb.init(
    #     project='together-llama13Braw-ARC-ft',
    #     config=EVAL_CONFIG
    # )

    main()

    # wandb.finish()



#
# test_prompt = '''Input:
# 0,0,5
# 0,5,0
# 5,0,0
# Output:
# 3,3,3
# 4,4,4
# 2,2,2
#
# Input:
# 0,0,5
# 0,0,5
# 0,0,5
# Output:
# 3,3,3
# 3,3,3
# 3,3,3
#
# Input:
# 5,0,0
# 0,5,0
# 5,0,0
# Output:
# 2,2,2
# 4,4,4
# 2,2,2
#
# Input:
# 0,5,0
# 0,0,5
# 0,5,0
# Output:
# 4,4,4
# 3,3,3
# 4,4,4
#
# Input:
# 0,0,5
# 5,0,0
# 0,5,0
# Output:
# '''
#
#     label = '''3,3,3
# 2,2,2
# 4,4,4
#
# '''


# cut_target = 20
            # random_cut = random.randint(0, min(len(label) - 1, cut_target))
            # pred = label[:-random_cut] if random_cut > 0 else label
            # if pred == label:
            #     print('1!!!')
            # pred += '\n\n'

