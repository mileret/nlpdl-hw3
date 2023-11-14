# nlpdl-hw3

This is the code implementation of NLP-DL Assignment#3, 2023 Fall.

Official implementation can be found [here](https://github.com/linhaowei1/NLPDL/tree/assignment4/Assignment_4).

# Task2: Fine-tune pre-trained models

```
python train.py --output_dir {\path\to\save\checkpoints} --seed {seed} --dataset_name {dataset_name} --model_name {model_name}
```

`dataset_name` in `['restaurant_sup', 'acl_sup', 'agnews_sup']`

`model_name` in `['roberta-base', 'bert-base-uncased', 'allenai/scibert_scivocab_uncased']`

# Task3: Apply PEFT on Llama3b

```
python peft_llama.py --output_dir {\path\to\save\checkpoints} --seed {seed} --use_lora {True or False} --use_adapter {True or False}
```
