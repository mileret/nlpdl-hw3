import pdb
import wandb
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, Trainer, HfArgumentParser, TrainingArguments
from transformers.integrations import WandbCallback


from dataHelper import get_dataset

def train(args):

    # define training arguments
    training_args = TrainingArguments(
        output_dir='./checkpoints',     # output directory
        num_train_epochs=10,         # total number of training epochs
        per_device_train_batch_size=1,  # batch size per device during training
        per_device_eval_batch_size=1,   # batch size for evaluation
        warmup_steps=500,               # number of warmup steps for learning rate scheduler
        weight_decay=0.01,              # strength of weight decay
        logging_dir='./logs',           # directory for storing logs
        logging_steps=10,
        evaluation_strategy='steps',
        eval_steps=10,
        save_steps=10,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        disable_tqdm=True,
        report_to=['wandb']
    )

    # add args to training_args
    for arg in vars(args):
        if getattr(args, arg) is not None:
            setattr(training_args, arg, getattr(args, arg))

    # load pre-trained model and tokenizer
    model_name = "bert-base-uncased"
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    datasetDict = get_dataset(['restaurant_sup'], tokenizer.sep_token)

    # tokenize
    def tokenize(examples):
        tokenized = tokenizer(examples['text'], padding=True, truncation=True)
        tokenized['label'] = examples['label']
        return tokenized
    
    train_dataset = datasetDict['train'].map(tokenize, batched=True)
    eval_dataset = datasetDict['test'].map(tokenize, batched=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # fine-tune
    trainer.train()


if __name__ == '__main__':
    args = HfArgumentParser(TrainingArguments).parse_args()
    train(args)
