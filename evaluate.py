import json
from datasets import load_dataset
from sacrebleu.metrics import BLEU
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from tqdm import tqdm
import argparse
bleu = BLEU(tokenize="flores200")


def get_prompt(input: str):
    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n
### Instruction:
Translate message to Thai
### Input:
{input}
### Response:"""
    return prompt


def main(base_model: str, lora_path: str, eval_dataset):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    print(f"loaded: {base_model}")
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    model.to(device)
    ds = load_dataset("json", data_files={"test": eval_dataset}, split="test")
    results = []
    references = []
    inputs = []

    for row in tqdm(iter(ds), total=len(ds)):
        prompt = get_prompt(row["en"])
        references.append(row["th"])
        inputs.append(row['en'])
        input = tokenizer([prompt], return_tensors="pt").to(device)
        output = model.generate(**input, max_new_tokens=256)
        output = tokenizer.decode(output[0][input['input_ids'].shape[-1]:], skip_special_tokens=True)
        results.append(output)

    print(
        {
            "bleu": str(bleu.corpus_score(results, [references])),
        }
    )
    with open("eval_results.json", "w") as w:
        json.dump(
            [{"pred": pred, "ref": ref, "input": ip} for pred, ref, ip in zip(results, references, inputs)], w, ensure_ascii=False
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', type=str, default='scb10x/typhoon-7b')
    parser.add_argument('--lora-path', type=str)
    parser.add_argument ('--eval-dataset', type=str, default='scb_mt_enth_2020_wiki_1k_test.jsonl')
    args = parser.parse_args()
    main(
        args.base_model,
        lora_path=args.lora_path,
        eval_dataset=args.eval_dataset,
    )
