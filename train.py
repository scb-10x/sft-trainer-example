from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset
from peft import LoraConfig
import torch

@dataclass
class ScriptArguments:
    """
    Define the arguments used in this script.
    """

    model_name: Optional[str] = field(default="scb10x/typhoon-7b", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(default='ssmi153/Capybara-ShareGPT', metadata={"help": "the dataset name"})
    use_4_bit: Optional[bool] = field(default=True, metadata={"help": "use 4 bit precision"})
    batch_size: Optional[int] = field(default=4, metadata={"help": "input batch size"})
    gradient_accumulation_steps: Optional[int] = field(default=1, metadata={"help": "input grad accum step"})
    max_seq_length: Optional[int] = field(default=2048, metadata={"help": "max sequence length"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "ckpt output"})

def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    torch_dtype = torch.bfloat16
    # Load model and tokenizer
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
    ) 
    model = AutoModelForCausalLM.from_pretrained(args.model_name, quantization_config=bnb_config, torch_dtype=torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    tokenizer.pad_token_id = tokenizer.unk_token_id
    dataset = load_dataset(args.dataset_name, split="train")
    
    def formatting_prompts_func(examples):
        output_texts = []
        for i in range(len(examples['conversations'])):
            row = examples['conversations'][i]
            instruction = row[0]['value']
            response = row[1]['value']
            text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n
### Instruction:
{instruction}
### Response:
{response}'''
            output_texts.append(text)
        return output_texts

    # we need to make sure it 
    response_template = "\n### Response:"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:] 
    print(args)
    training_arguments = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.batch_size,
        report_to=['tensorboard'],
        optim='adamw_torch',
        logging_steps=1,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        output_dir=args.output_dir
    )

    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    trainer = SFTTrainer(
        model,
        args=training_arguments,
        data_collator=collator,
        tokenizer=tokenizer,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=args.max_seq_length,
        formatting_func=formatting_prompts_func
    )

    trainer.train()

if __name__ == '__main__':
    main()