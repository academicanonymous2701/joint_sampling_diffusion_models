import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk, Dataset, concatenate_datasets
import transformers
from typing import Any, Dict, List, Optional, Union, Sequence
import copy
from dataclasses import dataclass
import json 
from tqdm import tqdm
import os
from jinja2 import Template
from transformers import AutoTokenizer
import hashlib
import numpy as np

IGNORE_INDEX = -100


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
        )
        for text in strings
    ]
    return [tokenized.input_ids[0] for tokenized in tokenized_list]

def preprocess(examples, tokenizer, query, response, noised_response, max_response_length) -> Dict:
    """Preprocess the data by tokenizing."""
    responses_tokenized, queries_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples[response], examples[query])]
    noised_responses_tokenized = [_tokenize_fn(string, tokenizer) for string in examples[noised_response]]
    input_ids = [torch.cat([s,t[:max_response_length],torch.zeros(max(max_response_length-len(t),0))+tokenizer.pad_token_id],dim=0).long() for s,t in zip(queries_tokenized,responses_tokenized)]
    noised_input_ids = [[torch.cat([s,t_[:max_response_length],torch.zeros(max(max_response_length-len(t_),0))+tokenizer.pad_token_id],dim=0).long() for t_ in t] for s,t in zip(queries_tokenized,noised_responses_tokenized)]
    labels = [copy.deepcopy(input_id) for input_id in input_ids]
    for i in range(len(input_ids)):
        labels[i][:len(queries_tokenized[i])] = IGNORE_INDEX
        labels[i][len(queries_tokenized[i])+len(responses_tokenized[i]):] = IGNORE_INDEX
    input_ids = [input_id[-tokenizer.model_max_length:] for input_id in input_ids]
    noised_input_ids = [[noised_input_id_[-tokenizer.model_max_length:] for noised_input_id_ in noised_input_id] for noised_input_id in noised_input_ids]
    labels = [label[-tokenizer.model_max_length:] for label in labels]

    data_dict = dict(input_ids=input_ids, labels=labels, noised_input_ids=noised_input_ids)
    return data_dict

def new_preprocess(data, tokenizer):
    def filter_history(his):
        first_pad_his = np.where(np.array(his) == tokenizer.pad_token_id)[0]
        if (first_pad_his.size>0):
            first_pad_his = first_pad_his[0]
            return his[:first_pad_his]
        else : 
            return his
    history_filtered = [filter_history(x) for x in data['history']]
    noised_input_ids = [[pr+his_ for his_ in his[i:i+9]] for pr,his in zip(data['prompt'],history_filtered)  for i in range(len(his)-9)]
    # def pad(x):
    #     max_len = max([len(y) for y in x])
    #     return [y+[tokenizer.pad_token_id for _ in range(max_len-len(y))] for y in x]
    # noised_input_ids = [pad(x) for x in noised_input_ids]
    # noised_lens = [len(x[0]) for x in noised_input_ids]
    input_ids = [(pr+resp) for (pr,resp,his) in zip(data["prompt"], data["response"], history_filtered) for _ in range(len(his)-9)]
    labels = [([IGNORE_INDEX]*len(pr)+resp) for (pr,resp,his) in zip(data["prompt"], data["response"], history_filtered) for _ in range(len(his)-9)]
    return dict(input_ids=input_ids, labels=labels, noised_input_ids=noised_input_ids)

def load_and_preprocess(tokenizer, path, data_format='lmeval_gsm8k', max_response_length=128, cache_dir=None):
    key_str = f"{path}-{tokenizer.__class__.__name__}-{max_response_length}-{data_format}"
    key_hash = hashlib.md5(key_str.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, key_hash)

    if (not ('tokenized' in data_format)):
        # If cached dataset exists, just load it
        if os.path.exists(cache_path):
            print(f"[CACHE HIT] Loading preprocessed dataset from {cache_path}")
            return load_from_disk(cache_path)

        print(f"[CACHE MISS] Processing dataset {path}, caching to {cache_path}")
        os.makedirs(cache_dir, exist_ok=True)

    split_range = None
    if "[" in path and path.endswith("]"):
        main_path, range_str = path.split("[", 1)
        range_str = range_str.rstrip("]")  # Remove the closing bracket
        path = main_path  # Update path to the main dataset path

        if range_str.startswith(":"):
            start = 0
            end = int(range_str[1:])
            split_range = (start, end)
        elif range_str.endswith(":"):
            start = int(range_str[:-1])
            split_range = (start, None)  # None indicates till the end
        elif ":" in range_str:
            start, end = range_str.split(":")
            start = int(start) if start else 0
            end = int(end) if end else None
            split_range = (start, end)

    if (data_format == 'lmeval_gsm8k'):
        template = Template(tokenizer.chat_template)
        def _format(question,answer):
            # query_response = template.render(messages=[{"role": "user", "content": question},{"role": "assistant", "content": answer}],bos_token=tokenizer.bos_token,add_generation_prompt=False)
            # query = template.render(messages=[{"role": "user", "content": question}],bos_token=tokenizer.bos_token,add_generation_prompt=True)
            # response = query_response[len(query):]
            # return dict({'query': query, 'response':response})
            return dict({'query': tokenizer.bos_token+question, 'response':answer})

        with open(path, "r") as f:
            data = [json.loads(line) for line in tqdm(f)]
        data = [_format(x["arguments"]['gen_args_0']['arg_0'], x["resps"][0][0]) for x in data]

    elif (data_format == 'nonlmeval'):
        with open(path, "r") as f:
            data = json.load(f)

        data = [dict({'query': x["prompt"], 'response':x["response"]}) for x in data]

    elif (data_format == 'history_based'):
        with open(path, "r") as f:
            data = json.load(f)
        print(f"Done JSON loading dataset file")

        if (end is not None):
            data = data[:end]
        data = [dict({'query': x["prompt"], 'response':x["response"], "noised_response":x['history'][i:i+8]}) for x in data for i in range(len(x['history'])-7)]

    elif (data_format == 'history_based_tokenized'):
        data = Dataset.load_from_disk(path)
        # start, end = split_range
        # if (end is not None):
        #     data = data.select(range(0, end))
        data = data.map(
            new_preprocess,
            batched=True,
            batch_size=300,
            num_proc=8,
            remove_columns=data.column_names,
            desc="Expanding train dataset",
            fn_kwargs={"tokenizer": tokenizer},
        )
        train_dataset = data
    else : 
        raise NotImplementedError
        # try :
        #     raw_train_datasets = load_dataset(path, split=args.dataset_split)
        # except Exception as e :
        #     raw_train_datasets = load_from_disk(path)

    if (not ('tokenized' in data_format)):
        raw_train_datasets = Dataset.from_list(data)

        if split_range is not None:
            start, end = split_range
            if end is None:
                raw_train_datasets = raw_train_datasets.select(range(start, len(raw_train_datasets)))
            else:
                raw_train_datasets = raw_train_datasets.select(range(start, end))

        num_proc = 8 if len(raw_train_datasets)>10000 else 1
        
        train_dataset = raw_train_datasets.map(
            preprocess,
            batched=True,
            batch_size=3000,
            num_proc=num_proc,
            remove_columns=raw_train_datasets.column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on train dataset",
            fn_kwargs={"tokenizer": tokenizer, "max_response_length":max_response_length, "query": 'query', "response": 'response', "noised_response": 'noised_response'},
        )

    if (not ('tokenized' in data_format)):
        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            train_dataset.save_to_disk(cache_path)
    return train_dataset

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def pad_to_left(self, tensor_list, pad_token):
        if (tensor_list[0].ndim == 1):
            max_len = max([len(x) for x in tensor_list])
            tensor_list = [torch.cat([torch.zeros(max_len-len(x),dtype=x.dtype)+pad_token, x],dim=0) for x in tensor_list]
        else : 
            max_len = max([len(x[0]) for x in tensor_list])
            tensor_list = [torch.cat([torch.zeros((len(x),max_len-len(x[0])),dtype=x.dtype)+pad_token, x],dim=1) for x in tensor_list]
        return tensor_list

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, noised_input_ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "noised_input_ids"))
        input_ids = [torch.tensor(x) for x in input_ids]
        # min_length = min([len(x) for x in input_ids])
        input_ids = self.pad_to_left(input_ids, self.tokenizer.pad_token_id)
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        # input_ids = torch.stack([x[:min_length] for x in input_ids], dim=0)
        labels = [torch.tensor(x) for x in labels]
        labels = self.pad_to_left(labels, IGNORE_INDEX)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        noised_input_ids = [torch.tensor(x) for x in noised_input_ids]
        noised_input_ids = self.pad_to_left(noised_input_ids, self.tokenizer.pad_token_id)
        noised_input_ids = torch.nn.utils.rnn.pad_sequence(noised_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        # labels = torch.stack([x[:min_length] for x in labels], dim=0)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        attention_mask = torch.ones_like(attention_mask)
        tok_idx = attention_mask.long().cumsum(-1) - 1
        tok_idx.masked_fill_(attention_mask == 0, 1)
        attention_mask = torch.logical_and(
            attention_mask.unsqueeze(1).unsqueeze(-2),
            attention_mask.unsqueeze(1).unsqueeze(-1),
        )

        return dict(
            input_ids=input_ids,
            noised_input_ids=noised_input_ids,
            labels=labels,
            attention_mask=attention_mask,
            tok_idx=tok_idx,
        )

def get_dataloaders(train_set, config, shuffle, data_collator, mode):
    if (mode == 'train'):
        bs = config.loader.batch_size
    elif (mode == 'val'):
        bs = config.loader.eval_batch_size
    else : 
        raise Exception("unknown mode %s"%mode)

    train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=bs,
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
            shuffle=shuffle,
            collate_fn=data_collator, 
            persistent_workers=False)
    return train_loader





@dataclass
class DataCollatorForDataGeneration(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def pad_to_left(self, tensor_list, pad_token):
        if (tensor_list[0].ndim == 1):
            max_len = max([len(x) for x in tensor_list])
            tensor_list = [torch.cat([torch.zeros(max_len-len(x),dtype=x.dtype)+pad_token, x],dim=0) for x in tensor_list]
        else : 
            max_len = max([len(x[0]) for x in tensor_list])
            tensor_list = [torch.cat([torch.zeros((len(x),max_len-len(x[0])),dtype=x.dtype)+pad_token, x],dim=0) for x in tensor_list]
        return tensor_list

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.LongTensor(y['prompt_ids']) for y in instances]
        input_ids = self.pad_to_left(input_ids, self.tokenizer.pad_token_id)
        attn_mask = [torch.LongTensor(y['attn_mask']) for y in instances]
        attn_mask = self.pad_to_left(attn_mask, 0)
        dict_ = dict({
        "prompt_ids" : torch.stack(input_ids, dim=0),
        "attn_mask" : torch.stack(attn_mask, dim=0),
        })
        return dict_

def get_context_free_loader(prompt_ids, tokenizer, config): #
    data = [{
        "prompt_ids" : prompt_ids[i],
        "attn_mask" : [1 for _ in range(len(prompt_ids[i]))],
    } for i in range(len(prompt_ids))]
    ds = Dataset.from_list(data)
    
    # collate_fn = lambda x : dict({
    #     "prompt_ids" : torch.stack([torch.LongTensor(y['prompt_ids']) for y in x], dim=0),
    #     "attn_mask" : torch.stack([torch.LongTensor(y['attn_mask']) for y in x], dim=0),
    # })
    collate_fn = DataCollatorForDataGeneration(tokenizer)
    data = get_dataloaders(ds, config, False, collate_fn, 'val')
    return data


def merge_datasets(files, outpath):
    data = []
    for file in files: 
        with open(file,'r') as f : 
            data.extend(json.load(f))
    with open(outpath, 'w') as f:
        json.dump(data, f, indent=4) # Save as a JSON list of dictionaries
    return 

def merge_datasets_numpy(files, outpath):
    data = []
    for file in files: 
        data.append(Dataset.load_from_disk(file))
    ds = concatenate_datasets(data)
    ds.save_to_disk(outpath)
    return


if __name__ == "__main__":
    
    # CODE to merge json files of generated samples from different seeds into one file for easier loading in the dataloader.

    #final merged file for json format
    out_path = '<path_to_save_data>' # $SCRATCH/diffusion/eagle/uncond/09.18_instruct_entropy_metamath128_100/uncond_generation/tokenized_generated_samples.pt'

    #smaller files to be merged
    file_template1 = '<path_to_load_data>' # $SCRATCH/diffusion/eagle/uncond/09.18_instruct_entropy_metamath128_100/uncond_generation/0918.154617_%d/tokenized_generated_samples.pt'
    files = [file_template1%seed for seed in range(4)]
    merge_datasets_numpy(files, out_path)
