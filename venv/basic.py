import json
import os
import tiktoken
import together

SEP = ','
WANDB_API_KEY = None
TOGETHER_API_KEY = "146706bab46d0101232eb1519664fb6e53c5be853f85aae3cfa9abd266e9434d"
together.api_key = TOGETHER_API_KEY
LLAMA2_13B_RAW = 'togethercomputer/llama-2-13b'


def file_to_grid(filename: str, type: str='train') -> (str, str, str):
    train_exs = ""

    if type == 'train':
        grid_dict = json.load(open('ARC_train/' + filename))
    elif type == 'eval':
        grid_dict = json.load(open('ARC_eval/' + filename))
    else:
        raise ValueError('type argument invalid. Got:', type, ', expected either: train, eval')

    for ex in grid_dict['train']:
        train_exs += 'Input:\n'

        for row in ex['input']:
            for num in row:
                train_exs += str(num) + SEP
            train_exs = train_exs[:-len(SEP)]
            train_exs += '\n'

        train_exs += 'Output:\n'

        for row in ex['output']:
            for num in row:
                train_exs += str(num) + SEP
            train_exs = train_exs[:-len(SEP)]
            train_exs += '\n'

        train_exs += '\n'

    train_input = "Input:\n"
    for row in grid_dict['test'][0]['input']:
        for num in row:
            train_input += str(num) + SEP
        train_input = train_input[:-len(SEP)]
        train_input += '\n'
    train_input += "Output:\n"

    train_output = ""
    for row in grid_dict['test'][0]['output']:
        for num in row:
            train_output += str(num) + SEP
        train_output = train_output[:-len(SEP)]
        train_output += '\n'

    return train_exs[:-2], train_input[:-1], train_output[:-1]


def get_train_filenames():
    return [filename for filename in os.listdir("ARC_train")]


def get_eval_filenames():
    return [filename for filename in os.listdir("ARC_eval")]


def cache_train_docs_and_text():
    all_filenames = get_train_filenames()

    training_docs = {}
    for i in range(len(all_filenames)):
        exs, input, output = file_to_grid(all_filenames[i])
        training_docs[i] = {'input': exs + '\n\n' + input + '\n', 'label': output}

    with open('ARC_train_docs.json', 'w') as f:
        json.dump(training_docs, f)

    training_text = ""
    for doc in training_docs:
        # TODO: Consider what to use to delineate moving to a new pattern. Current: "New sample:" = [3648, 6205, 512], only 512 shared with ":\n" after input and output tags
        training_text += 'New sample:\n' + training_docs[doc]['input'] + '\n' + training_docs[doc]['label'] + '\n\n'

    with open('ARC_train_text.txt', 'w') as f:
        f.write(training_text)


def cache_eval_docs_and_text():
    all_filenames = get_eval_filenames()

    eval_docs = {}
    for i in range(len(all_filenames)):
        exs, input, output = file_to_grid(all_filenames[i], type='eval')
        eval_docs[i] = {'input': exs + '\n\n' + input + '\n', 'label': output}

    with open('ARC_eval_docs.json', 'w') as f:
        json.dump(eval_docs, f)

    eval_text = ""
    for doc in eval_docs:
        # TODO: Consider what to use to delineate moving to a new pattern. Current: "New sample:" = [3648, 6205, 512], only 512 shared with ":\n" after input and output tags
        eval_text += 'New sample:\n' + eval_docs[doc]['input'] + '\n' + eval_docs[doc]['label'] + '\n\n'

    with open('ARC_eval_text.txt', 'w') as f:
        f.write(eval_text)


def determine_data_token_length():
    cache_eval_docs_and_text()

    with open('ARC_eval_text.txt', 'r') as f:
        eval_text = f.read()

    encoding = tiktoken.get_encoding('cl100k_base')
    eval_tokenized = encoding.encode(eval_text)

    print('Length of eval data:', len(eval_tokenized))
    # Training corpus is 889,968 tokens ~= 900k tokens. Eval corpus is 1,487,224 tokens ~= 1.5M tokens.


def get_num_docs(filepath):
    with open(filepath, 'r') as f:
        train_docs = json.load(f)
    # with open('ARC_processed/ARC_p_eval/ARC_eval_docs.json', 'r') as f:
    #     eval_docs = json.load(f)

    # encoding = tiktoken.get_encoding('cl100k_base')
    # eval_tokenized = encoding.encode(eval_text)

    print('Length of train docs:', len(train_docs))
    # print('Length of eval docs:', len(eval_docs))

    train_files = get_train_filenames()
    eval_files = get_eval_filenames()
    print('double checking num docs/overall examples (train / eval)', len(train_files), ' / ', len(eval_files))


def create_llama2_ft_data(num_docs: int = None):
    with open('ARC_train_docs.json', 'r') as f:
        train_docs = json.load(f)

    ft_data = []

    seen = []
    counter = 0
    for doc in train_docs:
        doc_dict = train_docs[doc]
        ft_data.append({'text': doc_dict['input'] + doc_dict['label'] + '\n\n'})

        seen.append(doc)
        counter += 1
        if num_docs is not None and counter == num_docs:
            print('Seen docs:', seen)
            print('In total, cached this many docs', counter)
            break

    together.Files.save_jsonl(ft_data, "ARC_train_llama_raw_half.jsonl")
    print(together.Files.check(file='ARC_train_llama_raw_half.jsonl'))


def main():
    # cache_train_docs_and_text()
    # cache_eval_docs_and_text()
    # get_num_docs('ARC_train_docs.json')
    create_llama2_ft_data(num_docs=200)

    print(together.Models.instances())


if __name__ == "__main__":
    main()

