import os 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import random
from datasets import load_dataset
from tqdm.auto import tqdm
import argparse
from peft import PeftModel
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_P2P_DISABLE"] = "1"

# 시드 설정
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str, default="outputs/checkpoint-100")
    parser.add_argument("--pred_file_name", type=str, default="predict.txt")

    return parser.parse_args()

def main(args):
    text_to_sql = load_dataset('csv', data_files={
        'train': './data/train.csv',
        'test': './data/validation.csv'
    })
    test_data = text_to_sql["test"].to_pandas()

    model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None: # pad_token 설정이 되어있지 않는 경우가 존재합니다.
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda"
    )

    if args.lora_path is not None and Path(args.lora_path).exists():
        print(f"로라 모듈을 불러옵니다 :  {args.lora_path}")
        model = PeftModel.from_pretrained(model, args.lora_path).merge_and_unload()
    else:
        print(f"LoRA path {args.lora_path} does not exist. Using base model")

    system_prompt = """You are a text to SQL query translator. Users will ask you questions in English and you will generate a SQL query."""
    user_prompt = """Given the <USER_QUERY>, generate the corresponding SQL command to retrieve the desired data, considering the query's syntax, semantics, and schema constraints.

    <USER_QUERY>
    {question}
    </USER_QUERY>"""

    preds = []
    golds = []
    db_ids = []
    for idx in tqdm(range(len(test_data))):
        test_input = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt.format(question=test_data["question"][idx])},
        ]
        golds.append(" ".join(test_data["query"][idx].split("\n")) )
        db_ids.append(test_data["db_id"][idx])
        print("정답 :", test_data["query"][idx])

        inputs = tokenizer.apply_chat_template(test_input, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        input_prompt = tokenizer.apply_chat_template(test_input, add_generation_prompt=True, tokenize=False)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs.to("cuda"),
                max_new_tokens=512,
                do_sample=False,
                num_beams=1,
                stop_strings=["<|endofturn|>", "<|stop|>"],
                tokenizer=tokenizer
            )

        response = tokenizer.batch_decode(output_ids)[0][len(input_prompt) - 1:]
        response = response.replace("<|im_end|>", "").replace("<|endofturn|>", "").strip()
        response = response.replace("<SQL_COMMAND>", "").replace("</SQL_COMMAND>", "").strip()
        response = response.replace(";", "")
        response = " ".join(response.split("\n")) 
        preds.append(response)
        print("예측 :", response)
    
    with open(f"./test-suite-sql-eval-master/gold.txt", "w",encoding="utf-8") as f:
        # 각 SQL 쿼리와 db_id를 탭으로 구분하여 저장
        gold_lines = [f"{sql}\t{db_id}" for sql, db_id in zip(golds, db_ids)]
        f.write("\n".join(gold_lines))

    with open(f"./test-suite-sql-eval-master/{args.pred_file_name}", "w",encoding="utf-8") as f:
        # 예측 파일도 동일한 형식으로 저장 (db_id 포함)
        pred_lines = [f"{sql}\t{db_id}" for sql, db_id in zip(preds, db_ids)]
        f.write("\n".join(pred_lines))


if __name__ == "__main__":
    main(parse_args())