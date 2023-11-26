import json
import os
import tiktoken
import together
import wandb

# FTing colab guide: https://colab.research.google.com/drive/11DwtftycpDSgp3Z1vnV-Cy68zvkGZL4K?usp=sharing#scrollTo=yufmqUI-azNy
SEP = ','
WANDB_API_KEY = "3ba2e8bcbdd054c8b4d1d239a4a95d16bf3005c1"
TOGETHER_API_KEY = "146706bab46d0101232eb1519664fb6e53c5be853f85aae3cfa9abd266e9434d"
LLAMA2_13B_RAW = 'togethercomputer/llama-2-13b'
ARC_TRAIN_LLAMA_RAW = 'ARC_train_llama_raw.jsonl'
ARC_TRAIN_LLAMA_RAW_HALF = 'ARC_train_llama_raw_half.jsonl'
FT_CONFIG = {
    'model': 'LLaMA 2 13B, Raw',
    'lr': 3e-5,
    'batch_size': 4,
    'epochs': 1
}


def upload_ft_data(filepath: str) -> dict:
    initial_check = together.Files.check(file=filepath)
    print('Initial check if given file is valid for FTing:', initial_check)

    print(filepath)
    official_check = together.Files.upload(file=filepath)
    print('Official check if given file is valid for FTing:', official_check)

    return official_check


def ft_model(model: str, file_id: str) -> dict:
    ft_resp = together.Finetune.create(
        training_file=file_id,
        model=model,
        n_epochs=3,
        n_checkpoints=3,
        batch_size=4,
        learning_rate=3e-5,
        suffix='ARC-train-half-ft-13b',
        # wandb_api_key=WANDB_API_KEY,
        estimate_price=True,
    )

    return ft_resp


def list_uploaded_files() -> None:
    files_list = together.Files.list()
    print(files_list['data'])


def list_available_models() -> None:
    print(together.Models.list())


def list_ft_job_key_detials(ft_details_file: str = 'ft_job_details.json') -> None:
    ft_details = json.load(ft_details_file)

    print(together.Finetune.get_job_status(fine_tune_id=ft_details['id']))  # pending, running, completed
    print(together.Finetune.is_final_model_available(fine_tune_id=ft_details['id']))  # True, False
    print(together.Finetune.get_checkpoints(fine_tune_id=ft_details['id']))  # list of checkpoints


def main():
    # Upload FT data
    off_resp = upload_ft_data(ARC_TRAIN_LLAMA_RAW_HALF)

    # Launch FTing job
    ft_details = ft_model(LLAMA2_13B_RAW, off_resp['id'])

    # Saving details
    with open('ft_job_details.json', 'w') as f:
        json.dump(ft_details, f)


if __name__ == "__main__":
    # wandb.login(key=WANDB_API_KEY)
    #
    # wandb.init(
    #     project='together-llama13Braw-ARC-ft',
    #     config=FT_CONFIG
    # )

    main()

    # wandb.finish()
