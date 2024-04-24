import json
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()


def call_openai(
    user_prompt: str,
    model="gpt-3.5-turbo",
    max_tokens=1000,
    top_p=0.1,
    temperature=1.0,
):
    client = OpenAI()
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
    user_prompt = f"""
Translate message below to Thai:
---
{text}
---
Output only translation result
"""
    translate_resp = call_openai(user_prompt)
    row = {"input": text, "output": translate_resp}
    with open(outpath, "a") as w:
        w.write(f"{json.dumps(row, ensure_ascii=False)}\n")


def process_row(example, outpath):
    for conv in example["conversations"]:
        translate_task(conv, outpath=outpath)


def main():
    ds = load_dataset("GAIR/lima", split="train")
    ds = ds.select(range(100))
    print(ds)
    for row in tqdm(iter(ds)):
        process_row(row, outpath="output.jsonl")


if __name__ == "__main__":
    main()
