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
LLAMA2_70B_CHAT = 'togethercomputer/llama-2-70b-chat'

EVAL_MODEL = LLAMA2_13B_RAW
EVAL_CONFIG = {
    'data': 'ARC eval',
    'portion': 1,
    'default model': EVAL_MODEL
}

PROMPT_PREFIXES = {
    1: """You are an intelligent STEM PhD student taking a test of abstract pattern recognition. Your goal is to get as many questions right as possible, which means correctly completing each sequence by understanding the abstract pattern the examples share and applying that pattern to the final input in the sequence to produce the correct corresponding output. You are a world expert in abstract pattern recognition and can correctly solve a wide range of pattern recognition tasks. Furthermore, you are determined to achieve your goal.

Now, you will analyze the following sequence, and correctly complete the pattern. You will respond with only the sequence representing your final answer.

""",
    2: """This is an example question from a test of abstract reasoning. Each input and output represents a colored grid. The input grid is transformed to the output grid based on an abstract pattern, which is the same for all the input-output pairs. Below is a correct example of abstract pattern completion for one of these questions. 

""",
    3: "",
    4: "",
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
                                        # api_key=TOGETHER_API_KEY,
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


def save_eval_results(total_score,
                      succ_eval,
                      str_metric,
                      data_source,
                      model,
                      detailed_score,
                      pr_prefix,
                      save_path):
    results = {
        'total_score': total_score,
        'total_samples': succ_eval,
        'metric': str_metric,
        'data_source': data_source,
        'model': model,
        'prompt_prefix': PROMPT_PREFIXES[pr_prefix],
        'detailed_score': detailed_score,
    }

    with open(save_path, 'w') as f:
        json.dump(results, f)


def format_prompt_for_chat(instruction, exs):
    return "[INST] " + instruction + exs + "[/INST]"


def format_prompt(instruction, exs):
    return instruction + exs


def model_eval(metric: Callable[[str, str], float],
               data_source: str = 'ARC_train_docs.json',
               save_path: str = 'ARC_eval_results.json',
               model: str = EVAL_MODEL,
               num_samples: int = None,
               pr_prefix: int = None) -> dict:

    with open(data_source, 'r') as f:
        eval_docs = json.load(f)
    total_score = 0
    succ_eval = 0
    detailed_score = {}

    for doc in eval_docs:
        prompt = eval_docs[doc]['input']
        if pr_prefix is not None:
            # prompt = format_prompt_for_chat(PROMPT_PREFIXES[pr_prefix], prompt)
            prompt = format_prompt(PROMPT_PREFIXES[pr_prefix], prompt)

        label = eval_docs[doc]['label']

        try:
            pred = model_completion(prompt, model=model, max_tokens=len(label)*2)
            score = metric(label, pred[:-2])        # pred[:-2] to get rid of stopping token \n\n

            print(prompt)
            print('#' + pred + '#')
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
                                  pr_prefix,
                                  save_path)

        except requests.exceptions.HTTPError:
            print(doc, 'failed, specifically 400 error prob')

    results = {
        'total_score': total_score,
        'total_samples': succ_eval,
        'metric': str(metric),
        'data_source': data_source,
        'model': model,
        'prompt_prefix': PROMPT_PREFIXES[pr_prefix],
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
                          model=LLAMA2_70B_RAW,
                          save_path='arcT_eval_lev_70b_pe',
                          # num_samples=10,
                          pr_prefix=2,
                          )
    results8 = model_eval(eval_acc,
                          data_source=arc_train_docs,
                          model=LLAMA2_70B_RAW,
                          save_path='arcT_eval_acc_70b_pe',
                          # num_samples=5,
                          pr_prefix=2,
                          )

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

