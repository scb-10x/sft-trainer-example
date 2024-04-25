from datasets import load_dataset
from sacrebleu.metrics import BLEU
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from tqdm import tqdm

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
    )
    print(f"loaded: {base_model}")
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    model.to(device)
    ds = load_dataset(eval_dataset)["test"]
    results = []
    references = []

    for row in tqdm(iter(ds)):
        prompt = get_prompt(row["input"])
        references.append(row["output"])
        inputs = tokenizer([prompt], return_tensors="pt").to(device)
        output = model.generate(**inputs)
        results.extend(output)

    print(
        {
            "bleu": str(bleu.corpus_score(results, [references])),
        }
    )
