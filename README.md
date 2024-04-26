```
conda create -n llm-trainer-env python=3.10
pip install -r requirements.txt
python train.py --dataset_name translation-dataset.jsonl --gradient_accumulation_steps 4
python evaluate.py --lora-path output/checkpoint-186 --eval-dataset translation-dataset-test.jsonl
```