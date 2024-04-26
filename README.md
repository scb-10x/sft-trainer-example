```
conda create -n llm-trainer-env python=3.10
pip install -r requirements.txt
python train.py --dataset_name scb10x/scb_mt_enth_2020_aqdf_1k --gradient_accumulation_steps 4
python evaluate.py --lora-path output/checkpoint-186 --eval-dataset scb10x/translation_val
```