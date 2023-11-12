import os
import json
import pdb
import jsonlines

from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict, ClassLabel, Value, load_from_disk



def get_dataset(dataset_name, sep_token):
    '''
    Get dataset from HuggingFace datasets library.
    Supported datasets: restaurant_sup and laptop_sup dataset for Aspect Based Sentiment Analysis(ABSA).

    Parameters
    ----------
    dataset_name: str or list of str
        Name of the dataset to load.
    sep_token: str
        The sep_token used by tokenizer(e.g. '<sep>') to separate the aspect and the sentence.

    Returns
    -------
    datasetDict: HuggingFace dataset object
        The dataset object containing the dataset with 'train' and 'test' splits.
        Each split is a dictionary with 'text' and 'label' keys.
    '''

    assert type(dataset_name) == str or type(dataset_name) == list, "dataset_name must be a string or a list of strings."
    assert type(sep_token) == str, "sep_token must be a string."

    if isinstance(dataset_name, str):
        dataset_name_list = [dataset_name]
    else:
        dataset_name_list = dataset_name
    
    datasetDictList = []

    for dataset_name in dataset_name_list:
        # ABSA datasets
        if dataset_name.startswith('restaurant') or dataset_name.startswith('laptop'):
            datasetDict = load_res_laptop_dataset(dataset_name, sep_token)
                
        # acl_arc dataset
        elif dataset_name.startswith('acl'):
            datasetDict = load_aclarc_dataset(dataset_name)

        # agnews dataset
        elif dataset_name.startswith('agnews'):
            datasetDict = load_agnews_dataset(dataset_name)
        
        else:
            raise NotImplementedError(f"Dataset {dataset_name} is not supported yet.")
        
        # few shot version, 32 samples for training
        if dataset_name.endswith('_fs'):
            train_dataset = datasetDict['train'].shuffle().select(range(32))
            datasetDict = DatasetDict({'train': train_dataset, 'test': datasetDict['test']})

        datasetDictList.append(datasetDict)


    # aggregate all datasets
    if len(dataset_name_list) > 1:
        datasetDict = aggregate_datasets(datasetDictList, dataset_name_list)
    else:
        datasetDict = datasetDictList[0]

    return datasetDict


def aggregate_datasets(datasetDictList, dataset_name_list):
    '''
    Aggregate multiple datasets into one dataset.
    
    Parameters
    ----------
    datasetDictList: list of HuggingFace datasetDict object
        The list of datasetDict objects containing the dataset with 'train' and 'test' splits.
        Each split is a dictionary with 'text' and 'label' keys.
    dataset_name_list: list of str
    
    Returns
    -------
    datasetDict: HuggingFace dataset object
        The dataset object containing the dataset with 'train' and 'test' splits.
        Each split is a dictionary with 'text' and 'label' keys.
    '''

    label_offset_dict = {'restaurant_sup': 3, 'laptop_sup': 3, 'acl_sup': 6, 'agnews_sup': 4,
                         'restaurant_fs': 3, 'laptop_fs': 3, 'acl_fs': 6, 'agnews_fs': 4}
    label_offset = 0
    aggregated_train_dataset, aggregated_test_dataset = None, None
    
    for i, datasetDict in enumerate(datasetDictList):
        dataset_name = dataset_name_list[i]
        datasetDict = datasetDict.map(lambda example: {'label': example['label'] + label_offset}, remove_columns=['label'])
        label_offset += label_offset_dict[dataset_name]

        train_dataset = datasetDict['train']
        test_dataset = datasetDict['test']

        # concatenate the datasets
        if aggregated_train_dataset is None:
            aggregated_train_dataset = train_dataset
            aggregated_test_dataset = test_dataset
        else:
            aggregated_train_dataset = concatenate_datasets([aggregated_train_dataset, train_dataset])
            aggregated_test_dataset = concatenate_datasets([aggregated_test_dataset, test_dataset])

    aggregated_datasetDict = DatasetDict({'train': aggregated_train_dataset, 'test': aggregated_test_dataset})

    return aggregated_datasetDict



def load_res_laptop_dataset(dataset_name, sep_token):
    '''
    Load restaurant_sup or laptop_sup dataset from local directory.
    '''

    # load dataset from local dir
    if dataset_name.startswith('restaurant'):
        data_dir = f"./dataset/SemEval14-res"
    elif dataset_name.startswith('laptop'):
        data_dir = f"./dataset/SemEval14-laptop"

    label2idx = {'positive': 0, 'neutral': 1, 'negative': 2}

    datasetDict = {}
    # json format processing script
    for split in ['train', 'test']:
        with open(f'{data_dir}/{split}.json', 'r') as f:
            data = json.load(f)
        
        datasetDict[split] = {}
        datasetDict[split] = {'text': [data[k]['term'] + ' ' + sep_token + data[k]['sentence'] for k in data.keys()], 
                              'label': [data[k]['polarity'] for k in data.keys()]}
        
        # convert label from string to int
        datasetDict[split]['label'] = [label2idx[label] for label in datasetDict[split]['label']]

        datasetDict[split] = Dataset.from_dict(datasetDict[split])

    datasetDict = DatasetDict(datasetDict)

    return datasetDict


def load_aclarc_dataset(dataset_name):
    '''
    Load acl_arc dataset from local directory.
    '''

    label2idx = {'Uses': 0, 'Future': 1, 'CompareOrContrast': 2, 'Motivation': 3, 'Extends': 4, 'Background': 5}
    # classLabel = ClassLabel(num_classes=6, names=['Uses', 'Future', 'CompareOrContrast', 'Motivation', 'Extends', 'Background'])
    new_data = {}

    for ds in ['train', 'test']:
        ds = ds
        new_data[ds] = {}
        new_data[ds]['text'] = []
        new_data[ds]['label'] = []
        with open(f'./dataset/data/citation_intent/{ds}.jsonl', 'r+') as f:
            for item in jsonlines.Reader(f):
                new_data[ds]['text'].append(item['text'])
                new_data[ds]['label'].append(label2idx[item['label']])

    # we may re-partitial, by classes
    train_ratio = 0.9
    num_label = 6
    total_num = len(new_data['train']['label'])

    for label in range(num_label):

        num_takeout = int((total_num * (1-train_ratio)) // num_label)
        label_pos = [lab_id for lab_id,lab in enumerate(new_data['train']['label']) if lab == label][:num_takeout]

        label_takeout = [lab for lab_id,lab in enumerate(new_data['train']['label']) if lab_id in label_pos]
        text_takeout = [lab for lab_id,lab in enumerate(new_data['train']['text']) if lab_id in label_pos]

        new_data['test']['label'] += label_takeout
        new_data['test']['text'] += text_takeout

        new_data['train']['label'] = [lab for lab_id,lab in enumerate(new_data['train']['label']) if lab_id not in label_pos]
        new_data['train']['text'] = [lab for lab_id,lab in enumerate(new_data['train']['text']) if lab_id not in label_pos]


    datasetDict = DatasetDict({'train': Dataset.from_dict(new_data['train']), 
                               'test': Dataset.from_dict(new_data['test'])})

    return datasetDict


def load_agnews_dataset(dataset_name):
    '''
    Load agnews dataset from hugingface datasets library.
    '''
    load_from_local = True

    if load_from_local:
        datasetDict = load_from_disk('dataset/agnews')

    else:
        # only use the test set
        dataset = load_dataset("ag_news", split="test")

        # convert dataset label from classLabel to int
        new_feature = dataset.features.copy()
        new_feature['label'] = Value('int64')
        dataset = dataset.cast(new_feature)

        # random split the dataset into training and test set
        datasetDict = dataset.train_test_split(test_size=0.1, seed=2022)

    return datasetDict



# debug
if __name__ == '__main__':
    datasetDict = get_dataset(['agnews_sup'], '<sep>')
    pdb.set_trace()
    print(datasetDict['train'][0])
    print(datasetDict['test'][0])
