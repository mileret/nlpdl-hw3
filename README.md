# nlpdl-hw3

# run train.py to fine-tune pre-trained models

```
python train.py --output_dir {\path\to\save\checkpoints} --seed {seed} --dataset_name {dataset_name} --model_name {model_name}
```

dataset_name in ['restaurant_sup', 'acl_sup', 'agnews_sup']
model_name in ['roberta-base', 'bert-base-uncased', 'allenai/scibert_scivocab_uncased']

# Apply PEFT on Llama3b

```
python peft_llama.py --output_dir {\path\to\save\checkpoints} --seed {seed} --use_lora {True or False} --use_adapter {True or False}
```
