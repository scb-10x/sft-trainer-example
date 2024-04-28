### Description
Simple script to fine-tune LLM using trl SFTTrainer

### Install dependency
```
eval "$(/home/user/miniconda3/bin/conda shell.zsh hook)"
conda create -n llm-trainer-env python=3.10
conda activate llm-trainer-env
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia # if cuda v11.8
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia # if cuda v12.1
pip install packaging ninja
pip install -r requirements.txt
pip install flash-attn # if doesn't work try find pre-build binary here https://github.com/Dao-AILab/flash-attention/releases
```

### Training
```
python train.py --dataset_name scb10x/scb_mt_enth_2020_aqdf_1k --gradient_accumulation_steps 4
```

### Eval
```
python evaluate.py --lora-path output/checkpoint-186 --eval-dataset scb10x/translation_val
```