import json
import os
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()


def call_openai(
    user_prompt: str,
    model="typhoon-instruct",
    max_tokens=1000,
    top_p=0.1,
    temperature=1.0,
):
    client = OpenAI(base_url=os.environ['OPENAI_BASE_URL'], api_key=os.environ['OPENAI_API_KEY'])
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_prompt},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        top_p=top_p,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    finish_reason = response.choices[0].finish_reason
    return content


def translate_task(text: str, outpath: str):
    assert isinstance(text, str)
    user_prompt = f"""
Translate message below to Thai:
---
{text}
---
Output only translation result
"""
    translate_resp = call_openai(user_prompt)
    row = {"en": text, "th": translate_resp}
    with open(outpath, "a") as w:
        w.write(f"{json.dumps(row, ensure_ascii=False)}\n")


def process_row(example, outpath):
    for conv in example["conversations"]:
        translate_task(conv['value'], outpath=outpath)


def main():
    ds = load_dataset("openaccess-ai-collective/oasst1-guanaco-extended-sharegpt", split="train")
    ds = ds.select(range(100))
    print(ds)
    for row in tqdm(iter(ds)):
        process_row(row, outpath="output.jsonl")


if __name__ == "__main__":
    main()
