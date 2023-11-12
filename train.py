import pdb
import wandb
import os
import torch
from dataclasses import dataclass, field
import logging
import json

from transformers import (AutoTokenizer, 
                          AutoModelForSequenceClassification, 
                          AutoConfig, 
                          Trainer, 
                          HfArgumentParser, 
                          TrainingArguments, 
                          set_seed, 
                          DataCollatorWithPadding)
from transformers.integrations import WandbCallback

from dataHelper import get_dataset


@dataclass
class MyArguments(TrainingArguments):
    # overwrite default values
    output_dir : str = field(default='./checkpoints')
    num_train_epochs : int = field(default=30)
    per_device_train_batch_size : int = field(default=64)
    per_device_eval_batch_size : int = field(default=64)
    warmup_steps : int = field(default=0)
    weight_decay : int= field(default=0.1)
    logging_dir : str = field(default='./logs')
    logging_steps : int  = field(default=10)
    evaluation_strategy : str = field(default='steps')
    eval_steps : int = field(default=10)
    save_steps : int = field(default=10)
    save_total_limit : int = field(default=1)
    load_best_model_at_end : bool = field(default=True)
    metric_for_best_model : str = field(default='eval_loss')
    greater_is_better : bool = field(default=False)
    disable_tqdm : bool = field(default=True)
    report_to : str = field(default='wandb')
    run_name : str = field(default='test')
    seed : int = field(default=2022)
    learning_rate : float = field(default=1e-5)
    # add custom arguments
    dataset_name : str = field(default='restaurant_sup')
    model_name : str = field(default='bert-base-uncased')
    load_from_local : bool = field(default=True)
    local_config_dir : str = field(default='./config')
    local_model_dir : str = field(default='./model')
    local_tokenizer_dir : str = field(default='./tokenizer')
    save_pretrained : bool = field(default=False)

    num_labels : int = field(default=3)
    max_seq_length : int = field(default=128)

    eval_result_dir : str = field(default='./eval_results')
    

def process_args(args) -> MyArguments:
    dataset2label : dict = {'restaurant_sup': 3, 'acl_sup' : 6, 'agnews_sup' : 4}
    args.num_labels = dataset2label[args.dataset_name]

    args.run_name = f'{args.dataset_name}_{args.model_name}_seed{args.seed}'

    if args.load_from_local:
        args.local_config_dir = f'./pretrained/{args.model_name}_config'
        args.local_model_dir = f'./pretrained/{args.model_name}_model'
        args.local_tokenizer_dir = f'./pretrained/{args.model_name}_tokenizer'

    return args


def f1_score(y_true, y_pred, average='micro'):
    '''
    Compute f1 score.
    '''
    from sklearn.metrics import f1_score as sk_f1_score
    return sk_f1_score(y_true, y_pred, average=average)


def train(args):
    # set seed
    set_seed(args.seed)

    # logging
    logging.basicConfig(level=logging.INFO)

    # load pre-trained model and tokenizer

    config = AutoConfig.from_pretrained(args.model_name if not args.load_from_local else args.local_config_dir, 
                                        num_labels=args.num_labels)
    
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name if not args.load_from_local else args.local_model_dir,
                                                               config=config, ignore_mismatched_sizes=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name if not args.load_from_local else args.local_tokenizer_dir)

    if args.save_pretrained:
        model.save_pretrained(f'./pretrained/{args.model_name}_model')
        tokenizer.save_pretrained(f'./pretrained/{args.model_name}_tokenizer')
        config.save_pretrained(f'./pretrained/{args.model_name}_config')

    # load dataset
    datasetDict = get_dataset(args.dataset_name, tokenizer.sep_token)

    # tokenize
    def tokenize(examples):
        tokenized = tokenizer(examples['text'], padding=False, truncation=True, max_length=args.max_seq_length)
        tokenized['label'] = examples['label']
        return tokenized
    
    train_dataset = datasetDict['train'].map(tokenize, batched=True)
    eval_dataset = datasetDict['test'].map(tokenize, batched=True)


    # define metrics
    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
        refs = p.label_ids

        # compute acc, micro_f1, macro_f1
        acc_result = round(sum(preds == refs) / len(preds), 3)
        micro_f1_result = round(f1_score(y_true=refs, y_pred=preds, average='micro'), 3)
        macro_f1_result = round(f1_score(y_true=refs, y_pred=preds, average='macro'), 3)

        # Cannot use the following code with evaluate API, because Boya compute cluster cannot access to hugingface server

        # acc_result = round(metric_acc.compute(predictions=preds, references=refs)['accuracy'], 3)
        # micro_f1_result = round(metric_f1.compute(predictions=preds, references=refs, average='micro')['f1'], 3)
        # macro_f1_result = round(metric_f1.compute(predictions=preds, references=refs, average='macro')['f1'], 3)

        return {
            'accuracy': acc_result,
            'micro_f1': micro_f1_result,
            'macro_f1': macro_f1_result
        }
    

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer, padding=True, pad_to_multiple_of=8),
        compute_metrics=compute_metrics
    )

    # fine-tune
    trainer.train()

    # evaluate
    eval_results = trainer.evaluate() # dict

    # save results use json
    if not os.path.exists(args.eval_result_dir):
        os.makedirs(args.eval_result_dir)
    with open(os.path.join(args.eval_result_dir, f'{args.run_name}.json'), 'w') as f:
        json.dump(eval_results, f)

    # logging message
    logging.info(f'eval_results: {eval_results}')


if __name__ == '__main__':

    parser = HfArgumentParser((MyArguments, ))
    args = parser.parse_args_into_dataclasses()[0]
    args = process_args(args)

    train(args)
