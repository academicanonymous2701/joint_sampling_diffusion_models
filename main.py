import os
import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch
import math
import utils
import json
import wandb
from tqdm import tqdm
from datetime import datetime
import re
import numpy as np
from datasets import load_dataset, Dataset
import data_utils
import jointsampler
from omegaconf import DictConfig, OmegaConf
import torch.distributed as dist
import time 
import mauve 

from transformers import AutoTokenizer, AutoModel

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)
omegaconf.OmegaConf.register_new_resolver(
  'node_count', lambda: int(os.environ.get("SLURM_JOB_NUM_NODES", 1)))



@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True,
  file_path: str = None) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if file_path is None : 
    file_path = config.checkpointing.save_dir
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        file_path), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    print(f'First {k} tokens:', tokenizer.decode(first))
    print('ids:', first)
    print(f'Last {k} tokens:', tokenizer.decode(last))
    print('ids:', last)

def freeze_params(model):
  for param in model.parameters():
    param.requires_grad_(False)
    

def get_trainer(config, training=True):
  wandb_logger = None
  if (config.get('wandb', None) is not None) and (training):
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)

  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  strategy = hydra.utils.instantiate(config.strategy)
    
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=strategy,
    logger=wandb_logger)

  return trainer

def _load_from_checkpoint(path, config):
  base_model = AutoModel.from_pretrained(config.base_model, trust_remote_code=True).eval()
  tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True) # 
  if (utils.fsspec_exists(path)):
    # model = jointsampler.JointSampler(config, base_model, tokenizer)
    model = jointsampler.JointSampler.load_from_checkpoint(checkpoint_path=path, config=config, base_model=base_model, tokenizer=tokenizer)
    # state_dict = torch.load(path,weights_only=False, map_location='cpu')['state_dict']
    # state_dict = {k:v for k,v in state_dict.items() if k.startswith('drafter.')}
    # model = jointsampler.JointSampler(config=config, base_model=base_model, tokenizer=tokenizer).cuda()
    # model.load_drafter_weights(state_dict)
  else : 
    raise Exception("Model doesn't exist")
    # logger.info('ckpt doesnt exist. loading a fresh model')
    # model = jointsampler.JointSampler(config=config, base_model=base_model, tokenizer=tokenizer)
  return model


@torch.no_grad()
def lm_eval(config, logger):
  # logger.info('Starting PPL Evaluation')
  # trainer = get_trainer(config, training=False)
  # L.seed_everything(config.seed+trainer.global_rank)

  # base_model = AutoModel.from_pretrained(config.base_model, trust_remote_code=True)
  # tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
  # model = jointsampler.JointSampler(config, base_model, tokenizer)

  model = _load_from_checkpoint(config.lm_eval.ckpt_path, config).cuda()
  model.evaluate_lm()


@torch.no_grad()
def generative_ppl_evaluation(config, logger):
  logger.info('Starting PPL Evaluation')
  trainer = get_trainer(config, training=False)
  L.seed_everything(config.seed+trainer.global_rank)

  base_model = AutoModel.from_pretrained(config.base_model, trust_remote_code=True)
  tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
  model = jointsampler.JointSampler(config, base_model, tokenizer)

  prompt_template = tokenizer.bos_token if config.sampling.add_bos_token else ''
  prompts = [prompt_template for _ in range(config.uncond_generation.num_samples)]
  prompts_ids = tokenizer(prompts, return_tensors="pt", padding=True).input_ids
  
  data = data_utils.get_context_free_loader(prompts_ids, tokenizer, config)

  print ("num_sentences ", len(data))
  model.forward_pass_tracker.reset()
  generation_ids = trainer.predict(model, data, return_predictions=True, ckpt_path=config.lm_eval.ckpt_path)
  generation_stats = model.forward_pass_tracker.all_stats()
  print ("here are the generation_stats : ", generation_stats)

  text_samples, log_samples = [], []
  for generation_id in generation_ids:
    bs = len(generation_id.sequences)
    for batch_idx in range(bs): 
      history = [y[batch_idx] for y in generation_id.history]
      g = generation_id.sequences[batch_idx]
      try : 
        text_samples.append(tokenizer.decode(g.tolist(), skip_special_tokens=True).strip())
        history_text = [tokenizer.decode(y.tolist(), skip_special_tokens=True).strip() for y in history]
        encoded = [' '.join([str(y_) for y_ in y.tolist()]) for y in history]
        log_samples.append({'response': text_samples[-1],'history':history_text, 'encoded': encoded})
      except Exception as e : 
        print ("Exception ", e, "generation ", g)
  
  text_samples = model.gather_lists_across_rank(text_samples)
  generation_stats = model.gather_metrics_across_ranks(generation_stats)

  if (trainer.global_rank == 0):
    gen_ppl_metric = jointsampler.compute_generative_perplexity(text_samples, config)
    if (len(text_samples)>=500):
      if (not utils.fsspec_exists(config.lm_eval.mauve_ref_path)):
        print('mauve ref path at %s doesnt exist. skipping mauve calc'%config.lm_eval.mauve_ref_path)
      else : 
        with open(config.lm_eval.mauve_ref_path, 'r') as file:
          list_of_dicts = json.load(file)
        ref_text_samples = [x['response'] for x in list_of_dicts]
        results = mauve.compute_mauve(p_text=ref_text_samples[:len(text_samples)], q_text=text_samples, device_id=0, max_text_length=1024, verbose=False)
        gen_ppl_metric['mauve'] = results.mauve
    gen_ppl_metric.update(generation_stats)  
    gen_ppl_metric.update(OmegaConf.to_container(config.sampling, resolve=True))
    gen_ppl_metric.update(OmegaConf.to_container(config.drafting_params, resolve=True))
    gen_ppl_metric.update(OmegaConf.to_container(config.uncond_generation, resolve=True))
    output_path = "%s/gen_ppl_eval/%s"%(config.lm_eval.output_path, datetime.now().strftime("%m%d.%H%M%S"))
    os.makedirs(output_path, exist_ok=True)
    _print_config(config, resolve=True, save_cfg=True, file_path=output_path)
    samples_path = os.path.join(output_path, "results.json")
    log_samples_path = os.path.join(output_path, "samples.json")
    with open(samples_path, 'w') as f:
      json.dump(gen_ppl_metric, f, indent=4) # Save as a JSON list of dictionaries
    with open(log_samples_path, 'w') as f:
      json.dump(log_samples, f, indent=4) # Save as a JSON list of dictionaries
  
  dist.barrier()
  return

@torch.no_grad()
def uncond_generation(config, logger):
  logger.info('Starting Unconditional Generation')
  trainer = get_trainer(config, training=False)
  L.seed_everything(config.seed+trainer.global_rank)

  base_model = AutoModel.from_pretrained(config.base_model, trust_remote_code=True)
  tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
  model = jointsampler.JointSampler(config, base_model, tokenizer)

  output_path = "%s/uncond_generation/%s_%d"%(config.lm_eval.output_path, datetime.now().strftime("%m%d.%H%M%S"), trainer.global_rank)
  os.makedirs(output_path, exist_ok=True)
  _print_config(config, resolve=True, save_cfg=True, file_path=output_path)
  
  if (config.uncond_generation.mode == 'uncond'):
    prompt_template = tokenizer.bos_token if config.sampling.add_bos_token else ''
    prompts = [prompt_template for _ in range(config.uncond_generation.num_samples)]
    prompts_ids = tokenizer(prompts, return_tensors="pt", padding=True).input_ids
  elif (config.uncond_generation.mode == 'metamath'):
    queries = load_dataset('meta-math/MetaMathQA')['train']['query']
    unique_queries = list(set(queries))
    import random
    random.shuffle(unique_queries)
    unique_queries = unique_queries[:int(config.uncond_generation.num_samples)]
    unique_messages = [[{"role": "user", "content": x}] for x in unique_queries]
    prompts_ids = tokenizer.apply_chat_template(unique_messages, add_generation_prompt=True)

  data = data_utils.get_context_free_loader(prompts_ids, tokenizer, config)

  generation_ids = trainer.predict(model, data, return_predictions=True)

  all_generations_data = []
  all_generations_data_numpy = []
  # decode
  # sequences = [g for generation_id in generation_ids for g in generation_id.sequences]
  # for g in sequences:
  #   # Decode the generated sequence, starting after the initial prompt
  #   try : 
  #     response_text = tokenizer.decode(g.tolist()).split(tokenizer.eos_token)[0].strip()
  #     all_generations_data.append({"prompt": prompt_template, "response": response_text[len(prompt_template):]})
  #   except Exception as e : 
  #     print ("Exception ", e, "generation ", g)

  for generation_id in generation_ids:
    bs = len(generation_id.sequences)
    for batch_idx in range(bs): 
      history = [y[batch_idx] for y in generation_id.history]
      prompt = generation_id.prompt[batch_idx]
      g = generation_id.sequences[batch_idx]
      # Decode the generated sequence, starting after the initial prompt
      try : 
        no_mask_prompt = [x for x in prompt.tolist() if (x!= tokenizer.mask_token_id)]
        response = g.tolist()[len(no_mask_prompt):]
        history = [y.tolist()[len(no_mask_prompt):] for y in history]
        prompt = [x for x in prompt.tolist() if (x!=tokenizer.eos_token_id and x!= tokenizer.mask_token_id)]
        prompt_text = tokenizer.decode(prompt).strip()
        response_text = tokenizer.decode(response).strip()
        history_text = [tokenizer.decode(y).strip() for y in history]
        all_generations_data.append({"prompt": prompt_text, "response": response_text, "history": history_text})
        all_generations_data_numpy.append({"prompt": prompt, "response": response, "history": history})
        # response_text = tokenizer.decode(g.tolist()).split(tokenizer.eos_token)[0].strip()[len(prompt_template):]
        # history_text = [tokenizer.decode().split(tokenizer.eos_token)[0].strip()[len(prompt_template):] for y in history]
        # all_generations_data.append({"prompt": prompt_template, "response": response_text, "history": history_text})
        # all_generations_data_numpy.append({"prompt": prompt_template_id, "response": [len(prompt_template_id):], "history": [y.tolist()[len(prompt_template_id):] for y in history]})
      except Exception as e : 
        print ("Exception ", e, "generation ", g)

  all_generations_data_numpy = Dataset.from_list(all_generations_data_numpy)
  samples_path = os.path.join(output_path, "generated_samples_with_prompts.json")
  numpy_path = os.path.join(output_path, "tokenized_generated_samples.pt")
  all_generations_data_numpy.save_to_disk(numpy_path)

  with open(samples_path, 'w') as f:
    json.dump(all_generations_data, f, indent=4) # Save as a JSON list of dictionaries
  return all_generations_data

def _train(config, logger):
  _print_config(config, resolve=True, save_cfg=True)
  logger.info('Starting Training.')

  base_model = AutoModel.from_pretrained(config.base_model, trust_remote_code=True)
  tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)

  freeze_params(base_model)

  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
  else:
    ckpt_path = None

  # Lightning callbacks
  datasets = data_utils.load_and_preprocess(tokenizer, config.data.data_path, data_format=config.data.data_format, max_response_length=config.data.max_response_length, cache_dir=config.data.cache_dir)
  # datasets = datasets.train_test_split(test_size=0.001, shuffle=False)
  datasets = datasets.train_test_split(test_size=0.01, shuffle=False)
  train_set = datasets['train']
  val_set = datasets['test']
  data_collator = data_utils.DataCollatorForSupervisedDataset(tokenizer=tokenizer)
  train_ds = data_utils.get_dataloaders(train_set, config, True, data_collator, 'train')
  valid_ds = data_utils.get_dataloaders(val_set, config, False, data_collator, 'val')

  _print_batch(train_ds, valid_ds, tokenizer)

  trainer = get_trainer(config)

  L.seed_everything(config.seed+trainer.global_rank)
  
  model = jointsampler.JointSampler(config, base_model, tokenizer)

  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
    """Main entry point for training."""
    L.seed_everything(config.seed)
    
    logger = utils.get_logger(__name__)
    if (config.mode == 'train'):
      _train(config, logger)
    elif (config.mode == 'eval'):
      lm_eval(config, logger)
    elif (config.mode == 'uncond_gen'):
      uncond_generation(config, logger)
    elif (config.mode == 'ppl_eval'):
      generative_ppl_evaluation(config,logger)


if __name__ == '__main__':
    main()
    