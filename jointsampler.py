import itertools
import math
import os
import typing
from dataclasses import dataclass,field

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import transformers
from torch import Tensor
from torchmetrics.aggregation import MeanMetric
from datetime import datetime

import models
import utils
from torch.distributions import Beta
import wandb 
import torch.distributed as dist
import time 

import re
import json
import fnmatch
import jsonlines
import argparse
import logging
from pathlib import Path
from torch.distributions import Categorical

from lm_eval import evaluator, utils
# from lm_eval.api.registry import ALL_TASKS
# from lm_eval.evaluator import request_caching_arg_to_dict
# from lm_eval.loggers import EvaluationTracker, WandbLogger
from lm_eval.tasks import TaskManager
from lm_eval.utils import handle_non_serializable, make_table, simple_parse_args_string
from eval import Dream

from omegaconf import DictConfig, OmegaConf
import datasets

import warnings
import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, List
import data_utils

import torch
import torch.distributions as dists
from transformers.generation.configuration_utils import (
    GenerationConfig
)
from transformers.utils import (
    ModelOutput,
    is_torchdynamo_compiling,
    logging,
)

logger = logging.get_logger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

LOG2 = math.log(2)
IGNORE_INDEX=-100

is_power_of_2 = lambda n : ((n & (n-1) == 0) and n != 0)

@torch.no_grad()
def compute_generative_perplexity(text_samples, config):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    supported_models = ['q7b', 'd7b']
    gen_ppl_models = config.training.generative_ppl_models.strip().split(',')
    metric_dict = dict({})
    for gen_ppl_model in gen_ppl_models:
        assert gen_ppl_model in supported_models
        if (gen_ppl_model == 'q7b'):
            logger.info('loading qwen')
            eval_model = transformers.AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B').eval().cuda()
            tokenizer = transformers.AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B')
            logger.info('done loading qwen')
        elif (gen_ppl_model == 'd7b'):
            tokenizer = transformers.AutoTokenizer.from_pretrained('Dream-org/Dream-v0-Base-7B')

        bs = config.loader.eval_batch_size
        num_batches = (len(text_samples) + bs - 1) // bs
        
        total_nlls = []
        if (gen_ppl_model == 'q7b'):
            logger.info('inside if')
            for i in range(num_batches):
                logger.info('doing batch %d out of %d'%(i,num_batches))
                tokenized_batch = tokenizer(text_samples[i*bs:(i+1)*bs], return_tensors='pt',padding=True)
                tokenized_batch = dict({
                    "input_ids" : tokenized_batch["input_ids"].cuda(),
                    "attention_mask" : tokenized_batch["attention_mask"].cuda()
                })
                logits = eval_model(tokenized_batch["input_ids"], attention_mask=tokenized_batch["attention_mask"]).logits
                logits = logits.transpose(-1, -2)
                nlls = F.cross_entropy(logits[..., :-1],tokenized_batch["input_ids"][..., 1:],reduction='none')
                token_mask = (tokenized_batch["input_ids"][..., 1:] != tokenizer.eos_token_id)
                total_nlls.extend(nlls[token_mask].view(-1).cpu().detach().tolist())

            logger.info('starting del')
            del eval_model
            torch.cuda.empty_cache()
            logger.info('done del')
        
        elif  (gen_ppl_model == 'd7b'):
            raise NotImplementedError
            # bs = self.config.loader.eval_batch_size
            # for i in range(len(text_samples)):
            #     input_ids = tokenizer(text_samples[i], return_tensors='pt',padding=True)['input_ids']
            #     input_ids = input_ids.repeat(bs,1).to(self.device)
            #     masked_input_ids = self.noising_process(input_ids, input_ids)
            #     _, logits = self.get_base_features(masked_input_ids)
            #     nlls = F.cross_entropy(logits.transpose(-1, -2),input_ids,reduction='none')
            #     token_mask = (masked_input_ids == tokenizer.mask_token_id)
            #     total_nlls.extend(nlls[token_mask].view(-1).cpu().detach().tolist())

        gen_nll = float(sum(total_nlls)/len(total_nlls))
        metric_dict['val/gen_ppl_%s'%gen_ppl_model] = gen_nll
    return metric_dict


@dataclass
class ForwardPassTracker(object):
    """Collate examples for supervised fine-tuning."""

    generated_tokens: int = 0
    num_sequences: int = 0
    num_specs: int = 256
    spec_lens: List[int] = field(init=False)
    total_time: float = 0.0
    num_to_log: float = 8

    def __post_init__(self):
        # Initialize spec_lens dynamically based on num_specs
        self.spec_lens = [0 for _ in range(self.num_specs)]
    
    def update(self, generated_tokens_per_generation, num_sequences):
        for g in generated_tokens_per_generation.tolist() : 
            if (g>0):
                self.spec_lens[int(g)-1] += 1

        self.generated_tokens += generated_tokens_per_generation.sum()
        self.num_sequences += num_sequences
    
    def reset(self,):
        self.generated_tokens = self.num_sequences = 0
        self.spec_lens = [0 for _ in range(self.num_specs)]
        self.total_time = 0.0

    def update_time(self, num_secs):
        self.total_time += num_secs

    def all_stats (self):
        num_secs = self.total_time
        metrics = dict({"avg_tokens":(float(self.generated_tokens)/max(1,float(self.num_sequences)))})
        metrics.update({"tokens_per_sec":(float(self.generated_tokens)/(num_secs+1e-6))})
        spec_len_stats = [i/sum(self.spec_lens) for i in self.spec_lens]
        metrics.update({"spec_len@%d"%i:spec_len_stats[i] for i in range(self.num_to_log)})
        return metrics

class Metric:
    def __init__(self, keys, num_heads):
        self.keys = keys
        self.num_heads = num_heads
        self.reset()
    def reset(self):
        self.metrics = [{key : [] for key in self.keys} for _ in range(self.num_heads)]
    def update(self, loss, key, head_id):
        self.metrics[head_id][key].append(loss)
    def compute(self, prefix='val/'):
        summary = dict({})
        for head_id in range(self.num_heads):
            summary.update({
                '%s%s@%d'%(prefix,key,head_id): sum(self.metrics[head_id][key])/(len(self.metrics[head_id][key])+1e-5)
                for key in self.keys
            })
        return summary 

def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits

def sample_gumbel(logits, temperature, top_p, noise):

    if temperature > 0:
        logits = logits / temperature
        y = (logits + noise).argmax(-1)
    else : 
        y = (logits).argmax(-1)
    return y

def ebm_sample_tokens(logits, temperature=0.0, noise=None):

    if temperature > 0:
        logits = logits / temperature
        y = (logits + noise).argmax(-1)
    else : 
        y = (logits).argmax(-1)
    probs = torch.softmax(logits, dim=-1)
    epsilon = 1e-10
    log_probs = torch.log(probs + epsilon)
    confidence = torch.sum(probs * log_probs, dim=-1)
    return confidence, y


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):

    if temperature > 0:
        logits = logits / temperature
    
    if margin_confidence:
        raise NotImplementedError


    # if (not only_entropy):
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)
    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    # else : 
    #     x0 = None
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    return confidence, x0


@dataclass
class DreamModelOutput(ModelOutput):
    prompt: torch.LongTensor = None
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None


class JointSampler(L.LightningModule):
    def __init__(
        self,
        config,
        base_model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer):

        super().__init__()
        self.save_hyperparameters('config')
        self.config = config

        self.tokenizer = tokenizer
        self.base_model = base_model
        self.vocab_size = self.tokenizer.vocab_size

        self.mask_token_id = self.tokenizer.mask_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

        self.drafter = models.dream.DrafterModel(self.config)
        self.smoothl1loss = torch.nn.SmoothL1Loss(reduction="none")
        if (self.config.sampling.decoding_strategy in ['adaptive_small','ebm_small']): # , 'adaptive'
            self.strict_loading = False
            self.ebm_eval_model = transformers.AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B').eval().requires_grad_(False)
        elif (self.config.sampling.decoding_strategy in ['ebm_big', 'adaptive_big']):
            self.strict_loading = False
            self.ebm_eval_model = transformers.AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B').eval().requires_grad_(False)

        spec_weight = float(self.config.training.sampled_spec_idx_weight) if self.config.training.sampled_spec_idx else 1
        spec_weighing = torch.Tensor([spec_weight**i for i in range(self.config.model.speculation_heads)])
        self.set_reweighing(spec_weighing)
        
        # assert is_power_of_2(int(self.config.model.speculation_heads+1))

        self.headwise_valid_metrics = Metric(["kl", "reg", "reg_withbase", "top1agree", "loss"],self.config.model.speculation_heads)
        self.headwise_naive_valid_metrics = Metric(["kl", "reg", "top1agree"],self.config.model.speculation_heads)

        self.forward_pass_tracker = ForwardPassTracker()
        # metrics are automatically reset at end of epoch

    def load_drafter_weights(self, state_dict):
        old_state_dict = self.drafter.state_dict()
        for k,v in state_dict.items():
            assert k[:8] == 'drafter.'
            old_state_dict[k[8:]] = v.to(self.device)
        self.drafter.load_state_dict(old_state_dict)

    def set_reweighing(self,losses):
        assert losses.ndim==1 and (losses.shape[0] == self.config.model.speculation_heads)
        self.unroll_probs = F.softmax(losses, dim=-1)
        # self.unroll_reweighing = 1/(self.unroll_probs*self.config.model.speculation_heads)
        return 

    # BASE MODEL FUNCTIONS
    @torch.no_grad()
    def embed_indices(self, input_ids):
        return self.base_model.model.embed_tokens(input_ids).detach()

    def get_logits(self, hidden_states):
        return F.log_softmax(self.base_model.lm_head(hidden_states),dim=-1)

    def get_base_features(self, input_ids, attention_mask=None, tok_idx=None):
        with torch.amp.autocast('cuda',dtype=torch.bfloat16):
            self.base_model.eval()
            # if (self.trainer.global_rank == 0):
            #     print ("input attention mask", attention_mask)
            # if isinstance(attention_mask, torch.Tensor):
            #     attention_mask = torch.where(attention_mask == 0, float('-inf'), 1e-6)
            # if (self.trainer.global_rank == 0):
            #     print ("out attention mask", attention_mask)
            attention_mask, tok_idx = "full", None
            with torch.no_grad():
                outputs = self.base_model(input_ids, attention_mask, tok_idx, output_hidden_states=True)
            if self.config.training.layerwise_feats:
                layers = 28
                list_of_hidden_states = [outputs.hidden_states[idx] for idx in range(len(outputs.hidden_states)) if (idx==(layers-1) or idx==(layers-3) or (idx==layers//2) or idx==2)]
                base_hidden_states = self.drafter.features_linear_layer(torch.cat(list_of_hidden_states,dim=-1).detach())
            else : 
                base_hidden_states = outputs.hidden_states[-1].detach()
            base_hidden_states = torch.cat([base_hidden_states[:,:1], base_hidden_states[:, :-1]], dim=1)
            base_logits = F.log_softmax(outputs.logits.detach(),dim=-1)
            base_logits = torch.cat([base_logits[:,:1], base_logits[:, :-1]], dim=1)
            return base_hidden_states, base_logits

    def get_LMEval_model(self, config):
        model_args = OmegaConf.to_container(config.sampling, resolve=True)
        model_args.update({"batch_size": config.loader.eval_batch_size})
        lm = Dream(self, self.tokenizer, **model_args)
        return lm, model_args

    # TRAINING FUNCTIONS
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.drafter.parameters(),
        lr=self.config.optim.lr,
        betas=(self.config.optim.beta1,
                self.config.optim.beta2),
        eps=self.config.optim.eps,
        weight_decay=self.config.optim.weight_decay)

        scheduler = hydra.utils.instantiate(
        self.config.lr_scheduler, optimizer=optimizer)
        scheduler_dict = {
        'scheduler': scheduler,
        'interval': 'step',
        'monitor': 'val/loss',
        'name': 'trainer/lr',
        }
        return [optimizer], [scheduler_dict]

    def training_step(self, batch, batch_idx):
        return self.forward_pass_diffusion(batch, prefix='train')

    def validation_step(self, batch, batch_idx):
        return self.forward_pass_diffusion(batch, prefix='val')

    @torch.no_grad()
    def on_validation_epoch_start(self):
        self.base_model.eval()
        self.drafter.eval()

    def on_train_epoch_start(self):
        self.base_model.train()
        self.drafter.train()

    @torch.no_grad()
    def generate_contextfree_samples(self):
        prompt_template = self.tokenizer.bos_token if self.config.sampling.add_bos_token else ''
        data = data_utils.get_context_free_loader(prompt_template, self.config.training.num_gen_sample, self.tokenizer, self.config)

        text_samples = []
        for batch in data:
            generation_id = self.predict_step(batch, 0)
            for g in generation_id.sequences:
                try : 
                    text_samples.append(self.tokenizer.decode(g.tolist()))
                except Exception as e : 
                    continue
        return text_samples

    def gather_lists_across_rank(self, text_samples):
        if dist.is_initialized() and dist.get_world_size() > 1:
            all_node_data = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(all_node_data, text_samples)
        else : 
            all_node_data = [text_samples]
        text_samples = [x for y in all_node_data for x in y]
        return text_samples

    def gather_metrics_across_ranks(self, local_data):
        if dist.is_initialized() and dist.get_world_size() > 1:
            all_node_data = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(all_node_data, local_data)
        else : 
            all_node_data = [local_data]
        gathered_summarized_metrics = {k: sum(d[k] for d in all_node_data) / len(all_node_data) for k in all_node_data[0]}
        return gathered_summarized_metrics

    @torch.no_grad()
    def on_validation_epoch_end(self):
        local_data = dict({})
        
        if self.config.training.compute_generative_perplexity:
            text_samples = self.generate_contextfree_samples()
            gen_ppl_metric = compute_generative_perplexity(text_samples, self.config)
            local_data.update(gen_ppl_metric)

        local_data.update(self.headwise_valid_metrics.compute(prefix='val/'))
        local_data.update(self.headwise_naive_valid_metrics.compute(prefix='val/naive_'))
            
        self.headwise_valid_metrics.reset()
        self.headwise_naive_valid_metrics.reset()

        log_metrics = self.gather_metrics_across_ranks(local_data)
        log_metrics['trainer/global_step'] = self.global_step
        for k in log_metrics.keys():
            if ('gen_ppl_' in k):
                log_metrics[k] = math.exp(log_metrics[k])

        # if self.config.training.unroll_adaptive_sampling != 'none':
        #     losses_per_head = [log_metrics['val/loss@%d'%i]/self.config.training.unroll_adaptive_sampling_temperature for i in range(self.config.model.speculation_heads)]
        #     if (all([x>0 for x in losses_per_head])):
        #         self.update_reweighing(torch.tensor(losses_per_head))

        if self.trainer.global_rank == 0:
            self.trainer.logger.experiment.log({
                "val/reweighing":wandb.Histogram(np_histogram = (self.unroll_probs.cpu().numpy(),np.arange(self.config.model.speculation_heads+1))),
                "trainer/global_step": self.global_step,
                })
            # self.log_dict(log_metrics, on_epoch=True,on_step=False,rank_zero_only=True)
            if (log_metrics['val/loss@0']>0):
                self.trainer.logger.experiment.log(log_metrics)

            if hasattr(self.trainer.logger, 'log_table') and self.config.training.compute_generative_perplexity:
                # Log the last generated samples
                self.trainer.logger.log_table(
                    key=f'samples@global_step{self.global_step}',
                    columns=['Generated Samples'],
                    data=[[s] for s in text_samples[:2]])
            

    def noising_process(self, input_ids, labels):
        B, S = input_ids.shape

        if (self.config.drafting_params.noising == 'random'):
            # Only mask positions where labels == input_ids
            valid_mask = (labels == input_ids).float()  # (B, S)
            t = torch.rand(B, device=self.device)# *0.1 + 0.9
            mask_indices = (torch.bernoulli(t.unsqueeze(-1).expand(-1,S))*valid_mask).bool()
            new_input_ids = torch.where(mask_indices, self.mask_token_id, input_ids)
            
            return new_input_ids
        else : 
            raise Exception("Unknown noising : %s"%self.config.drafting_params.noising)

    def single_unmask_batch_process(self, noised_input_ids, filled_input_ids, base_logits):
        B, S = noised_input_ids.shape

        num_speculate = self.config.drafting_params.speculation_len+1
        new_labels = filled_input_ids.unsqueeze(0).repeat(num_speculate,1,1)

        new_input_ids_list, sampled_positions = [], []
        new_input_ids = noised_input_ids.clone()

        for _ in range(num_speculate):
            for _ in range(self.config.drafting_params.unmask_per_unroll):
                mask_pos = torch.logical_and((new_input_ids==self.mask_token_id), (filled_input_ids!=self.mask_token_id))
                # mask_pos = (new_input_ids==self.mask_token_id)
                if (self.config.drafting_params.unmasking == 'first'):
                    sampled_pos = torch.argmax(mask_pos.int(), dim=-1)
                elif (self.config.drafting_params.unmasking == 'random'):
                    sampled_pos = []
                    for batch_idx in range(mask_pos.shape[0]):
                        masked_indices = torch.where(mask_pos[batch_idx])[0]
                        num_masked_indices = len(masked_indices)
                        if (num_masked_indices>0):
                            sampled_pos.append(masked_indices[torch.randint(num_masked_indices, (1,))])
                        else : 
                            sampled_pos.append(torch.zeros(1,).long().to(masked_indices.device))
                    sampled_pos = torch.cat(sampled_pos,dim=0)  # shape [B]
                elif (self.config.drafting_params.unmasking == 'entropy'):
                    entropy = -(base_logits * base_logits.exp()).sum(dim=-1)
                    entropy[torch.logical_not(mask_pos)] = float('inf')
                    sampled_pos = torch.argmin(entropy, dim=-1)
                    
                new_input_ids[torch.arange(B), sampled_pos] = filled_input_ids[torch.arange(B), sampled_pos]

            new_input_ids_list.append(new_input_ids.clone())
            sampled_positions.append(sampled_pos)

        new_input_ids = torch.stack(new_input_ids_list,dim=0)
        sampled_positions = torch.stack(sampled_positions,dim=1) # shape (B, num_spec)

        new_labels = torch.where(new_input_ids==self.mask_token_id, new_labels, -100)
        new_labels = torch.where(new_labels == self.mask_token_id, -100, new_labels)
        
        return new_input_ids.long(), new_labels.long(), sampled_positions


    def _compute_loss(self, hidden_states, logits, labels, target_hidden_states, target_logits, base_hidden_states, base_logits, prefix='train', spec_idx=0):
        loss_indices = (labels != -100)
        # metric_idx = int(math.log2(spec_idx+2))-1 if ((prefix == 'val') and is_power_of_2(int(spec_idx+2))) else None
        metric_idx = spec_idx if (prefix == 'val') else None
        
        if (self.config.training.soft_logits):
            if (self.config.training.reg_weight > 0.0):
                if (self.config.training.reg_type == 'mse'):
                    reg_loss  = (hidden_states-target_hidden_states)[loss_indices].pow(2).mean(dim=-1)
                    reg_loss_with_base  = (hidden_states-base_hidden_states)[loss_indices].pow(2).mean(dim=-1)
                elif (self.config.training.reg_type == 'l1'):
                    reg_loss = self.smoothl1loss(hidden_states[loss_indices], target_hidden_states[loss_indices]).mean(dim=-1)
                    reg_loss_with_base = self.smoothl1loss(hidden_states[loss_indices], base_hidden_states[loss_indices]).mean(dim=-1)
                if metric_idx is not None : 
                    if (self.config.training.reg_type == 'mse'):
                        naive_reg = (base_hidden_states-target_hidden_states)[loss_indices].pow(2).mean().detach().clone()
                    elif (self.config.training.reg_type == 'l1'):
                        naive_reg = self.smoothl1loss(target_hidden_states[loss_indices], base_hidden_states[loss_indices]).mean(dim=-1).mean().detach().clone()

                    self.headwise_valid_metrics.update(reg_loss.detach().clone().mean().cpu(), 'reg', metric_idx)
                    self.headwise_valid_metrics.update(reg_loss_with_base.detach().clone().mean().cpu(), 'reg_withbase', metric_idx)
                    self.headwise_naive_valid_metrics.update(naive_reg.cpu(), 'reg', metric_idx)

            else :
                reg_loss = 0.0
                reg_loss_with_base = 0.0

            if (self.config.training.kl_weight > 0.0):
                # if (self.trainer.global_rank == 0) :
                #     print ("target logits : ", target_logits[loss_indices])
                #     print ("input logits : ", logits[loss_indices])
                #     print ("loss indices : ", loss_indices)
                kl = F.kl_div(logits[loss_indices], target_logits[loss_indices], log_target=True, reduction='none').sum(dim=-1)
                if metric_idx is not None : 
                    naive_kl = F.kl_div(base_logits[loss_indices], target_logits[loss_indices], log_target=True, reduction='none').sum(dim=-1).mean().detach().clone()
                    self.headwise_valid_metrics.update(kl.mean().detach().clone().cpu(), 'kl', metric_idx)
                    self.headwise_naive_valid_metrics.update(naive_kl.cpu(), 'kl', metric_idx)
            else : 
                kl = 0.0

            loss = (self.config.training.reg_weight*reg_loss + self.config.training.kl_weight*kl + self.config.training.reg_with_base_weight*reg_loss_with_base)
            if (self.config.training.weighing_factor == 'none'):
                loss = loss.mean()
            elif (self.config.training.weighing_factor == 'median'):
                median = torch.median(loss).detach().clone()
                weight = torch.exp(loss/median).detach().clone()
                weight = weight/weight.mean()
                loss = (loss*weight).mean()
            else : 
                raise Exception("unknown weighing factor %s"%self.config.training.weighing_factor)

            if metric_idx is not None : 
                target_argmax_probs, target_argmax = target_logits[loss_indices].max(dim=-1)
                naive_argmax = base_logits[loss_indices].argmax(dim=-1)
                our_argmax = logits[loss_indices].argmax(dim=-1)
                self.headwise_valid_metrics.update((our_argmax==target_argmax).float().mean().cpu(), 'top1agree', metric_idx)
                self.headwise_naive_valid_metrics.update((naive_argmax==target_argmax).float().mean().cpu(), 'top1agree', metric_idx)
        else : 
            raise Exception("Didn't implement metrics for this")
            ######### CODE below is fine, just that metrics is not implemented #################
            # ce_loss = F.cross_entropy(logits, labels)
            # loss = ce_loss
            # if (prefix == 'val'):
            #     naive_argmax = base_logits[loss_indices].argmax(dim=-1)
            #     our_argmax = logits[loss_indices].argmax(dim=-1)
            #     metrics = dict({
            #         '%s/ce_loss@%d'%(prefix,spec_idx): ce_loss,
            #         '%s/naive_ce_loss@%d'%(prefix,spec_idx):  F.cross_entropy(base_logits, labels).detach(),
            #         '%s/acc@%d'%(prefix,spec_idx): (our_argmax==labels[loss_indices]).detach().float().mean(),
            #         '%s/naive_acc@%d'%(prefix,spec_idx):  (naive_argmax==labels[loss_indices]).detach().float().mean(),
            #     })
            #     self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
            
        if metric_idx is not None : 
            self.headwise_valid_metrics.update(loss.clone().detach().float().cpu(), 'loss', metric_idx)
        return loss
    
    def forward_pass_drafter(self, tokens, attention_mask, tok_idx, prior_hidden_states, spec_idx):
        input_embeds = self.embed_indices(tokens)
        if (not self.config.drafting_features.use_base_features):
            prior_hidden_states = torch.zeros_like(prior_hidden_states, dtype=prior_hidden_states.dtype).to(self.device)
        attention_mask, tok_idx = "full", None
        # if isinstance(attention_mask, torch.Tensor):
        #     attention_mask = torch.where(attention_mask == 0, float('-inf'), 1e-6)
        hidden_states = self.drafter(input_embeds, prior_hidden_states, attention_mask, tok_idx, spec_idx)
        logits = self.get_logits(hidden_states)
        return hidden_states, logits

    def forward_pass_diffusion(self, batch, prefix='train'):
        filled_input_ids = batch['input_ids']
        if (batch.get('noised_input_ids') is None):
            noised_input_ids = self.noising_process(filled_input_ids, batch['labels'])
        else : 
            noised_input_ids  = batch['noised_input_ids'][:,0]

        base_hidden_states, base_logits = self.get_base_features(noised_input_ids, batch['attention_mask'], batch['tok_idx'])
        # print ("base_logits", base_logits.shape)
        # print ("noised_input_ids", noised_input_ids.shape)
        # print ("base_hidden_states", base_hidden_states)
        # print ("batch attention mask", batch['attention_mask'])
        # print ("batch['tok_idx']", batch['tok_idx'])

        if (batch.get('noised_input_ids') is None):
            input_ids, labels, sampled_positions = self.single_unmask_batch_process(noised_input_ids, filled_input_ids, base_logits)
        else : 
            num_speculate = self.config.drafting_params.speculation_len+1
            input_ids = batch['noised_input_ids'][:, 1:1+num_speculate].transpose(0,1)
            new_labels = filled_input_ids.unsqueeze(0).repeat(num_speculate,1,1)
            new_labels = torch.where(torch.logical_or((new_labels == self.mask_token_id), (new_labels == self.tokenizer.pad_token_id)), -100, new_labels)
            labels = torch.where(input_ids==self.mask_token_id, new_labels, -100).contiguous()
            sampled_positions = []
            for seq_idx in range(num_speculate):
                diff_positions = torch.where(batch['noised_input_ids'][:,seq_idx]!=batch['noised_input_ids'][:,seq_idx+1])[1]
                if (self.trainer.global_rank == 0):
                    if (len(diff_positions)!=batch['noised_input_ids'].shape[0]):
                        print ("seq_idx", seq_idx)
                        print (torch.where(batch['noised_input_ids'][:,seq_idx]!=batch['noised_input_ids'][:,seq_idx+1])[0])
                        print ("diff positions", diff_positions)
                        print ("noised_input_ids 3", batch['noised_input_ids'][3,seq_idx])
                        print ("noised_input_ids +1 3", batch['noised_input_ids'][3,seq_idx+1])
                        print ("noised_input_ids 6", batch['noised_input_ids'][6,seq_idx])
                        print ("noised_input_ids +1 6", batch['noised_input_ids'][6,seq_idx+1])
                sampled_positions.append(diff_positions)
            sampled_positions = torch.stack(sampled_positions,dim=1)
        # if (self.trainer.global_rank == 0):
        #     print ("filled input ids ", filled_input_ids[0])
        #     print ("noised input ids ", noised_input_ids[0])
        #     print ("additional input ids ", input_ids[:,0])

        if (self.config.training.short_inputs):
            base_hidden_states = torch.gather(base_hidden_states,1,sampled_positions.unsqueeze(-1).expand(-1,-1,base_hidden_states.shape[-1]))
            base_logits = torch.gather(base_logits,1,sampled_positions.unsqueeze(-1).expand(-1,-1,base_logits.shape[-1]))

        prior_hidden_states = base_hidden_states.clone()

        total_loss = None
        num_to_speculate = self.config.drafting_params.speculation_len
        reweighing = 1
        if (self.config.training.sampled_spec_idx):
            # num_to_speculate = np.random.randint(1,num_to_speculate+1)
            num_to_speculate = int(Categorical(self.unroll_probs).sample())+1
            # if (self.config.training.unroll_adaptive_sampling == 'constant'):
            #     reweighing = self.unroll_reweighing[num_to_speculate-1]

        for spec_idx in range(num_to_speculate):
            if (self.config.drafting_features.tokens == 'sampled'):
                tokens = input_ids[spec_idx]
            elif (self.config.drafting_features.tokens == 'complete'):
                tokens = filled_input_ids
            elif (self.config.drafting_features.tokens == 'base'):
                tokens = noised_input_ids
            elif (self.config.drafting_features.tokens == 'repeated'):
                assert self.config.drafting_params.unmask_per_unroll == 1
                sampled_tokens = filled_input_ids[torch.arange(input_ids.shape[1]), sampled_positions[spec_idx]]
                tokens = sampled_tokens.unsqueeze(-1).repeat(1, input_ids.shape[-1])
            elif (self.config.drafting_features.tokens == 'just_sampled'):
                assert self.config.drafting_params.unmask_per_unroll == 1
                tokens = torch.zeros_like(input_ids[spec_idx]).long().to(input_ids.device)+self.mask_token_id
                tokens[torch.arange(input_ids.shape[1]), sampled_positions[spec_idx]] = filled_input_ids[torch.arange(input_ids.shape[1]), sampled_positions[spec_idx]]
            
            target_labels = labels[spec_idx]
            if (self.config.training.short_inputs):
                tokens = torch.gather(tokens,1,sampled_positions)
                target_labels = torch.gather(target_labels,1,sampled_positions)
                target_labels[:, spec_idx+2:] = IGNORE_INDEX
                tok_idx = torch.gather(batch['tok_idx'],1,sampled_positions)
                hidden_states, logits = self.forward_pass_drafter(tokens, "full", tok_idx, prior_hidden_states, spec_idx)
                # print ('logits', logits)
                # print ('prior_hidden_states', prior_hidden_states)
                # print ('tokens', tokens)
            else : 
                hidden_states, logits = self.forward_pass_drafter(tokens, batch['attention_mask'], batch['tok_idx'], prior_hidden_states, spec_idx)
            prior_hidden_states = hidden_states
            
            if (self.config.training.sampled_spec_idx and spec_idx != num_to_speculate-1):
                continue

            target_hidden_states, target_logits = None, None
                
            if (self.config.training.soft_logits):
                target_hidden_states, target_logits = self.get_base_features(input_ids[spec_idx], batch['attention_mask'], batch['tok_idx'])
                if (self.config.training.short_inputs):
                    target_hidden_states = torch.gather(target_hidden_states,1,sampled_positions.unsqueeze(-1).expand(-1,-1,target_hidden_states.shape[-1]))
                    target_logits = torch.gather(target_logits,1,sampled_positions.unsqueeze(-1).expand(-1,-1,target_logits.shape[-1]))
                target_hidden_states = target_hidden_states.view(-1,target_hidden_states.shape[-1])
                target_logits = target_logits.view(-1,target_logits.shape[-1])


            loss = self._compute_loss(
                hidden_states = hidden_states.view(-1,hidden_states.shape[-1]), 
                logits = logits.view(-1,logits.shape[-1]), 
                labels = target_labels.view(-1), 
                target_hidden_states = target_hidden_states, 
                target_logits = target_logits,
                base_hidden_states = base_hidden_states.view(-1,base_hidden_states.shape[-1]), 
                base_logits = base_logits.view(-1,base_logits.shape[-1]),
                prefix = prefix,
                spec_idx = spec_idx,
                )
            
            if (self.trainer.global_rank == 0) and torch.isnan(loss):
            # if (self.trainer.global_rank == 0):
                print ("spec_idx : ", spec_idx)
                print ("target labels : ", target_labels)
                print ("additional input_ids : ", torch.gather(input_ids[spec_idx], 1, sampled_positions))
                print ("filled input ids : ",torch.gather(filled_input_ids,1,sampled_positions))
                print ("noised input ids : ",torch.gather(noised_input_ids,1,sampled_positions))
                print ("incured loss : ", loss)
                print ("filled input ids all: ",filled_input_ids[0])
                print ("noised input ids all: ",noised_input_ids[0])
                print ("additional input_ids all: ",input_ids[spec_idx][0])
            
            total_loss = total_loss + loss if total_loss else loss

        total_loss *= reweighing
        self.log('%s/loss'%prefix, total_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)

        return total_loss 

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    # Generation functions 

    @torch.no_grad()
    def diffusion_generate(self, inputs = None, generation_config = None, **kwargs, ):
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        generation_config = self.base_model._prepare_generation_config(generation_config, **kwargs)
        generation_tokens_hook_func = kwargs.pop("generation_tokens_hook_func", lambda step, x, logits: x)
        generation_logits_hook_func = kwargs.pop("generation_logits_hook_func", lambda step, x, logits: logits)

        # 2. Define model inputs
        assert inputs is not None
        input_ids = inputs
        device = self.device
        attention_mask = kwargs.pop("attention_mask", None)
        self.base_model._prepare_special_tokens(generation_config, device=device)

        # 3. Prepare `max_length`.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config = self.base_model._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )

        self.base_model._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
        
        # 4. Check input_ids
        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )
        if (
            hasattr(generation_config, "pad_token_id") and
            torch.any(input_ids == generation_config.pad_token_id) and 
            attention_mask is None
        ):
            warnings.warn(
                "Padding was detected but no attention mask is passed here. For correct "
                "generation results, please set `attention_mask` when batch-padding inputs.",
                UserWarning,
            )

        input_ids, attention_mask = self.base_model._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask 
        )

        result = self._sample(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            generation_tokens_hook_func=generation_tokens_hook_func,
            generation_logits_hook_func=generation_logits_hook_func
        )
        return result

    @torch.no_grad()    
    def predict_step(self, batch, batch_idx):
        generation_ids = self.diffusion_generate(
            batch['prompt_ids'].to(self.device),
            attention_mask=batch['attn_mask'].to(self.device),
            max_new_tokens=self.config.sampling.max_new_tokens,
            output_history=True,
            return_dict_in_generate=True,
            steps=self.config.sampling.diffusion_steps,
            temperature=self.config.sampling.temperature,
            top_p=self.config.sampling.top_p,
            top_k=None,
            alg=self.config.sampling.alg,
            alg_temp=self.config.sampling.alg_temp,
        )
        return generation_ids

    def clipped_probs(self, orig_logits, pred_logits, path_probs):
        orig_probs, pred_probs = orig_logits.exp(), torch.softmax(pred_logits,dim=-1)
        path_probs = path_probs.unsqueeze(-1)
        min_probs, max_probs = (orig_probs-1+path_probs)/path_probs, orig_probs/path_probs
        pred_probs = pred_probs.clamp(min=min_probs, max=max_probs)
        pred_probs = pred_probs/pred_probs.sum(dim=-1, keepdim=True)
        return torch.log(pred_probs)


    def _sample_postprocess(self, x, mask_index, mask_token_id, eos_token_id, fill_eos=False):
        # ###### code below fills sentences with eos_token #########
        # ###### removed eos from args. wont work for now #########
        if (fill_eos):
            for batch_idx in range(x.shape[0]):
                last_nonmasked = (x[batch_idx] != mask_token_id).nonzero(as_tuple=True)[0][-1]
                if (x[batch_idx][last_nonmasked] == eos_token_id):
                    x[batch_idx][last_nonmasked:] = eos_token_id

        # ForwardPassTracker changes
        # logger.info('inside post process')
        num_unmasked = (mask_index) & (~((x == mask_token_id) | (x == eos_token_id)))
        # num_unmasked = (mask_index) & (~(x == mask_token_id))
        num_unmasked_per_generation = num_unmasked.float().sum(dim=-1)
        num_sentences = mask_index.any(dim=-1).float().sum()
        # logger.info(f'num_unmasked {num_unmasked_per_generation}, num_sentences {num_sentences}')
        if getattr(self, "forward_pass_tracker", None):
            tracker = self.forward_pass_tracker
        else : 
            raise Exception("forward_pass_tracker not found")
        tracker.update(num_unmasked_per_generation, num_sentences)

        return x

    @torch.no_grad()
    def _sample(self, input_ids, attention_mask, generation_config, generation_tokens_hook_func, generation_logits_hook_func):
        start_time = time.time()
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        eos_token_id = generation_config.eos_token_id
        pad_token_id = generation_config.pad_token_id
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k
        # these parameters are not in Dreams Generation config. Hence accessing them directly. Hacky but okay. 
        # Need to think about this in context of LMEval
        confidence_thres=self.config.sampling.confidence_thres
        decoding_strategy=self.config.sampling.decoding_strategy
        clipped_probs=self.config.sampling.clipped_probs
        adaptive_mixture=self.config.sampling.adaptive_mixture
        num_ebm_samples = 2

        histories = [] if (return_dict_in_generate and output_history) else None

        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
        # logger.info(f"num masks {max_length - input_ids.shape[1]}")
        input_prompt = x.clone()
        
        # if (self.trainer.global_rank == 0):
        #     print ('x', x )
        #     print ('attention_mask', attention_mask )

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        # if (self.trainer.global_rank == 0):
        #     print ('tok_idx', tok_idx )
        
        assert decoding_strategy in ['fixed', 'ebm_small', 'ebm_big', 'adaptive_small', 'adaptive_big'], "Wrong decoding strategy %s"%decoding_strategy # , 'thresholding'
        assert alg in ['entropy'], "Wrong alg %s"%alg
        assert (alg_temp is None) or (alg_temp == 0)

        if (decoding_strategy in ['thresholding', 'adaptive_small', 'adaptive_big']):
            logger.info(f"Ignoring steps={steps} as decoding_strategy is {decoding_strategy}. Setting steps to max_length")
            steps = max_length

        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)



        # this allows user-defined token control of the intermediate steps
        x = generation_tokens_hook_func(None, x, None)
        noise = None
        for i in range(steps):
            step_mask_index = (x == mask_token_id)
            if (step_mask_index.any(dim=-1).float().sum() <= 0):
                break
            # logger.info(f"i {i}, num_mask {step_mask_index.float().sum()}")

            hidden_states, logits = self.get_base_features(x, attention_mask, tok_idx)

            # this allows user-defined logits control of the intermediate steps
            logits = generation_logits_hook_func(i, x, logits)
            # logits = F.log_softmax(logits,dim=-1)

            logits[:,:,mask_token_id] = float('-inf')

            if (noise is None) and (decoding_strategy.startswith('adaptive') or decoding_strategy.startswith('ebm')):
                if (decoding_strategy.startswith('adaptive')):
                    noise = (torch.distributions.Gumbel(0, 1).sample(logits.shape).to(logits.device))
                if (decoding_strategy.startswith('ebm')):
                    noise = [(torch.distributions.Gumbel(0, 1).sample(logits.shape).to(logits.device)) for _ in range(num_ebm_samples)]


            if (self.config.sampling.mask_eos):
                logits[:,:,pad_token_id] = float('-inf')
                logits[:,:,self.tokenizer.bos_token_id] = float('-inf')

            # mask_logits[:,pad_token_id] += eos_penalty * torch.log(1-t+eps)
            if (clipped_probs):
                orig_logits = F.log_softmax(logits.clone().detach(),dim=-1)
                sampled_path_prob = torch.ones((x.size(0),)).to(orig_logits.device)

            t = timesteps[i]
            s = timesteps[i + 1]

            if (decoding_strategy == 'thresholding'):
                number_transfer_tokens = self.config.drafting_params.speculation_len+1
            else:
                num_mask_token = step_mask_index.sum() / step_mask_index.shape[0]
                number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else int(num_mask_token)
            
            active_indices = torch.arange(x.size(0), device=self.device).long()

            # logger.info(f'before starting {i}, decoded {self.tokenizer.decode(x[0])}')
            
            if (decoding_strategy.startswith('adaptive')):
                orig_logits = logits.clone()
                spec_len = self.config.drafting_params.speculation_len+1
                mask_index = (x == mask_token_id)
                if (not self.config.sampling.use_marginals):
                    raise NotImplementedError
                    x0_marginal = x.clone()
                    local_mask_index = (x0_marginal==mask_token_id)
                    for num_transfered_ in range(spec_len):
                        x0 = sample_gumbel(logits[local_mask_index], temperature, top_p, noise[local_mask_index])
                        x0_marginal[local_mask_index] = x0
                        if (num_transfered_ == spec_len-1):
                            break
                        for btch_idx in range(x.shape[0]):
                            first_mask_token = (x == mask_token_id)[btch_idx].nonzero()
                            if (first_mask_token.shape[0]>0):
                                x0_marginal[btch_idx,first_mask_token[0]+1:] = mask_token_id
                        with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                            hidden_states, logits_ = self.forward_pass_drafter(x0_marginal, "full", None, hidden_states, num_transfered_)
                            local_mask_index = (x0_marginal==mask_token_id)
                            logits[local_mask_index.unsqueeze(-1).expand(-1,-1,logits.shape[-1])] = logits_[local_mask_index.unsqueeze(-1).expand(-1,-1,logits.shape[-1])].clone()
                else : 
                    x0 = sample_gumbel(logits[mask_index], temperature, top_p, noise[mask_index])
                    x0_marginal = x.clone()
                    x0_marginal[mask_index] = x0
                # logger.info(f'x0_marginal {i}, decoded {self.tokenizer.decode(x0_marginal[0])}')
                x0_marginal = x0_marginal.clamp(min=0,max=151935)
                with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                    ebm_logits = self.ebm_eval_model(x0_marginal).logits
                    ebm_logits = torch.cat([ebm_logits[:, :1,:], ebm_logits[:, :-1,:]],dim=1)
                    ebm_logits_ = torch.zeros_like(logits).to(logits.device)
                    ebm_logits_[:,:,:ebm_logits.shape[-1]] = ebm_logits
                    # ebm_logits = F.log_softmax(ebm_logits_,dim=-1)
                    ebm_logits = ebm_logits_

                adaptive_logits = adaptive_mixture*logits + (1-adaptive_mixture)*ebm_logits
                # adaptive_logits = 0.34*orig_logits + 0.33*logits + 0.33*ebm_logits
                # adaptive_logits = 0.5*orig_logits + 0.5*ebm_logits
                x0 = sample_gumbel(adaptive_logits[mask_index], temperature, top_p, noise[mask_index])
                x0_joint = x.clone()
                x0_joint[mask_index] = x0
                # logger.info(f'x0_joint {i}, decoded {self.tokenizer.decode(x0_joint[0])}')
                for btch_idx in range(x.shape[0]):
                    matching_indices = (x0_marginal==x0_joint)[btch_idx].float()
                    unmasked_indices = (x0_marginal==x)[btch_idx].float()
                    first_mask_token = (unmasked_indices==0).nonzero()
                    if (first_mask_token.shape[0]>0):
                        first_mask_token = first_mask_token[0]
                        matching_indices[:first_mask_token+1] = 1.0
                        matching_indices = matching_indices.cumprod(dim=-1)
                        idx = (matching_indices==0).nonzero()
                        if (idx.shape[0]>0):
                            # idx = min(idx[0], first_mask_token+spec_len)
                            idx = idx[0]
                            x[btch_idx,:idx] = x0_marginal[btch_idx,:idx].clone()
                        else : 
                            idx = first_mask_token+spec_len
                            x[btch_idx,:idx] = x0_marginal[btch_idx,:idx].clone()
                    else : 
                        continue
                    # if ('John drives' in self.tokenizer.decode(x[btch_idx])):
                    #     logger.info(f'decoded {self.tokenizer.decode(x[btch_idx])}')

            elif (decoding_strategy.startswith('ebm')):
                num_ebm_samples = 2
                mask_index = (x == mask_token_id)
                sampled_x0s = []
                for sample_idx in range(num_ebm_samples):
                    # confidence, x0 = sample_tokens(logits[mask_index], temperature, top_p=None, top_k=None, neg_entropy=True)
                    confidence, x0 = ebm_sample_tokens(logits[mask_index], temperature, noise[sample_idx][mask_index])
                    x0_ = x.clone()
                    x0_[mask_index] = x0.clone()
                    sampled_x0s.append(x0_)
                full_confidence = torch.full_like(x, -torch.inf, device=self.device, dtype=logits.dtype)
                full_confidence[mask_index] = confidence

                _, transfer_index = torch.topk(full_confidence, number_transfer_tokens) # replaced number_of_transfer_tokens by 1

                sampled_x0s = torch.cat(sampled_x0s,dim=0)
                sampled_x0s = sampled_x0s.clamp(min=0,max=151935)
                with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                    ebm_logits = self.ebm_eval_model(sampled_x0s).logits
                # try : 
                # except : 
                #     for j in range(8):
                #         logger.info (f"on global rank {self.trainer.global_rank} {sampled_x0s[j]}")
                    
                ebm_logits = ebm_logits.transpose(-1, -2)
                nlls = F.cross_entropy(ebm_logits[..., :-1],sampled_x0s[..., 1:],reduction='none')
                token_mask = (sampled_x0s[..., 1:] != self.tokenizer.eos_token_id)
                nlls = torch.where(sampled_x0s[..., 1:]==self.tokenizer.eos_token_id, 0, nlls)
                nlls = nlls.sum(dim=-1)/token_mask.float().sum(dim=-1)
                # nlls = nlls.cpu().detach()

                sampled_nlls = torch.stack([nlls[i*x.shape[0]:(i+1)*x.shape[0]] for i in range(num_ebm_samples)],dim=0)
                x0_final = torch.stack([sampled_x0s[j*x.shape[0] + btch_idx] for btch_idx, j in enumerate(sampled_nlls.argmin(dim=0))],dim=0).contiguous()
                row_indices = torch.arange(x.shape[0]).unsqueeze(1).expand_as(transfer_index)
                x[row_indices,transfer_index] = x0_final[row_indices,transfer_index].clone()

            else : 
                for num_transfered_ in range(number_transfer_tokens):
                    if (len(active_indices)==0):
                        break
                    mask_index = (x == mask_token_id)

                    confidence, x0 = sample_tokens(logits[mask_index], temperature, top_p=top_p, top_k=top_k, neg_entropy=True)

                    full_confidence = torch.full_like(x, -torch.inf, device=self.device, dtype=logits.dtype)
                    full_confidence[mask_index] = confidence

                    _, transfer_index = torch.topk(full_confidence, 1) # replaced number_of_transfer_tokens by 1
                    active_indices = torch.where((x == mask_token_id).float().sum(dim=-1)>0)[0]
                    
                    if ((num_transfered_>0) and (decoding_strategy == 'thresholding')):
                        ## here i assume that transfer index is of shape (bs,1)
                        active_indices = torch.where(full_confidence[torch.arange(x.size(0), device=self.device),transfer_index[:,0]]>confidence_thres)[0]
                        # if (self.trainer.global_rank == 0):
                        #     print ("active indices at step %d after num_transfered %d"%(i,num_transfered_), active_indices)
                        #     print ("transfer_index at step %d after num_transfered %d"%(i,num_transfered_), transfer_index)
                        #     print ("x0 at step %d after num_transfered %d"%(i,num_transfered_), x[:,0])

                    transfer_index = transfer_index[active_indices]
                    row_indices = active_indices.unsqueeze(1).expand_as(transfer_index)

                    sampling_logits = F.log_softmax(logits[row_indices,transfer_index],dim=-1)
                    if (clipped_probs):
                        sampling_logits = self.clipped_probs(orig_logits[row_indices,transfer_index], sampling_logits, sampled_path_prob[row_indices])
                        
                    sampled_confidence, x_ = sample_tokens(sampling_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=False, margin_confidence=False)
                    x[row_indices,transfer_index] = x_.clone()
                    if (clipped_probs):
                        sampled_path_prob[row_indices] *= orig_logits[row_indices,transfer_index,x_].exp()

                    if ((not self.config.sampling.use_marginals) and (len(active_indices)>0)):
                        with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                            if (tok_idx is None):
                                hidden_states_, logits_ = self.forward_pass_drafter(x[active_indices], attention_mask, tok_idx, hidden_states[active_indices], num_transfered_)
                            else : 
                                hidden_states_, logits_ = self.forward_pass_drafter(x[active_indices], attention_mask[active_indices], tok_idx[active_indices], hidden_states[active_indices], num_transfered_)
                            hidden_states[active_indices] = hidden_states_
                            logits[active_indices] = logits_


            x = self._sample_postprocess(x, step_mask_index, mask_token_id, eos_token_id, decoding_strategy.startswith('adaptive'))

            if histories is not None:
                histories.append(x.clone())
        
        end_time = time.time()
        self.forward_pass_tracker.update_time(end_time-start_time)

        if return_dict_in_generate:
            return DreamModelOutput(
                prompt = input_prompt,
                sequences=x,
                history=histories,
            )
        else:
            return x

    # @torch.no_grad()
    # def evaluate_lm(self):
    #     return 
    #     assert self.config.sampling.mask_eos == False, "running with sampling.mask_eos which is not well defined for LM_Eval. dont do this."
    #     eval_logger = utils.eval_logger
    #     task_manager = TaskManager("INFO")
    #     task_list = self.config.lm_eval.tasks.split(",")
    #     task_names = task_manager.match_tasks(task_list)
    #     for task in [task for task in task_list if task not in task_names]:
    #         if os.path.isfile(task):
    #             config = utils.load_yaml_config(task)
    #             task_names.append(config)
    #     task_missing = [
    #         task for task in task_list if task not in task_names and "*" not in task
    #     ]  # we don't want errors if a wildcard ("*") task name was used

    #     if task_missing != []:
    #         missing = ", ".join(task_missing)
    #         eval_logger.error(
    #             f"Tasks were not found: {missing}\n"
    #             f"{utils.SPACING}Try `lm-eval -h` for list of available tasks",
    #         )
    #         raise ValueError(f"Tasks {missing} were not found.")

    #     if self.config.lm_eval.output_path:
    #         path = Path(self.config.lm_eval.output_path)
    #         # check if file or 'dir/results.json' exists
    #         if path.is_file() or Path(self.config.lm_eval.output_path).joinpath("results.json").is_file():
    #             eval_logger.warning(
    #                 f"File already exists at {path}. Results will be overwritten."
    #             )
    #             output_path_file = path.joinpath("results.json")
    #             assert not path.is_file(), "File already exists"
    #         # if path json then get parent dir
    #         elif path.suffix in (".json", ".jsonl"):
    #             output_path_file = path
    #             path.parent.mkdir(parents=True, exist_ok=True)
    #             path = path.parent
    #         else:
    #             path.mkdir(parents=True, exist_ok=True)
    #             output_path_file = path.joinpath("results.json")
    #     elif self.config.lm_eval.log_samples and not self.config.lm_eval.output_path:
    #         assert self.config.lm_eval.output_path, "Specify --output_path"

    #     eval_logger.info(f"Selected Tasks: {task_names}")
        
    #     lm, model_args = self.get_LMEval_model(config)
    #     model_args_str = json.dumps(model_args)
        
    #     results = evaluator.simple_evaluate(
    #         model=lm,
    #         tasks=task_names,
    #         batch_size=self.config.loader.eval_batch_size,
    #         device=self.device,
    #         limit=self.config.lm_eval.limit,
    #         log_samples=self.config.lm_eval.log_samples,
    #         task_manager=task_manager,
    #         verbosity="INFO",
    #         confirm_run_unsafe_code=True,
    #     )

    #     if results is not None:
    #         dump_results = dict({"results": results["results"]})
    #         dump_results.update(OmegaConf.to_container(self.config.sampling, resolve=True))
    #         dump_results.update(OmegaConf.to_container(self.config.lm_eval, resolve=True))
    #         dump_results.update(OmegaConf.to_container(self.config.drafting_features, resolve=True))
    #         dump_results.update(OmegaConf.to_container(self.config.drafting_params, resolve=True))

    #         if self.config.lm_eval.log_samples:
    #             samples = results.pop("samples")
    #         dumped = json.dumps(dump_results, indent=2, default=lambda o: str(o))

    #         batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

    #         if self.config.lm_eval.output_path:
    #             output_path_file.open("w").write(dumped)

    #             if self.config.lm_eval.log_samples:
    #                 for task_name, config in results["configs"].items():
    #                     output_name = "{}_{}".format(
    #                         re.sub("/|=", "__", model_args_str), task_name
    #                     )
    #                     filename = path.joinpath(f"{output_name}.jsonl")

    #                     with jsonlines.open(filename, "w") as f:
    #                         f.write_all(samples[task_name])

    #         print(
    #             f"{self.config.lm_eval.ckpt_path} ({model_args_str}), limit: {self.config.lm_eval.limit}, "
    #             f"batch_size: {self.config.loader.eval_batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
    #         )

    #         print(make_table(results))
    #         if "groups" in results:
    #             print(make_table(results, "groups"))
    
    @torch.no_grad()
    def evaluate_lm(self):
        assert self.config.sampling.mask_eos == False, "running with sampling.mask_eos which is not well defined for LM_Eval. dont do this."
        eval_logger = utils.eval_logger
        eval_logger.setLevel(getattr(logging, "INFO"))
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        task_manager = TaskManager("INFO")


        task_list = self.config.lm_eval.tasks.split(",")
        task_names = task_manager.match_tasks(task_list)
        for task in [task for task in task_list if task not in task_names]:
            if os.path.isfile(task):
                config = utils.load_yaml_config(task)
                task_names.append(config)
        task_missing = [
            task for task in task_list if task not in task_names and "*" not in task
        ]  # we don't want errors if a wildcard ("*") task name was used

        if task_missing != []:
            missing = ", ".join(task_missing)
            eval_logger.error(
                f"Tasks were not found: {missing}\n"
                f"{utils.SPACING}Try `lm-eval --tasks list` for list of available tasks",
            )
            raise ValueError(
                f"Tasks not found: {missing}. Try `lm-eval --tasks {{list_groups,list_subtasks,list_tags,list}}` to list out all available names for task groupings; only (sub)tasks; tags; or all of the above, or pass '--verbosity DEBUG' to troubleshoot task registration issues."
            )

        path = Path(os.path.join(self.config.lm_eval.output_path, 'lm_evals/%s_%s'%(datetime.now().strftime("%m%d.%H%M%S"), self.config.lm_eval.gpu_id)))
        os.makedirs(path, exist_ok=True)
        path.mkdir(parents=True, exist_ok=True)
        output_path_file = path.joinpath("results.json")

        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
        eval_logger.info(f"Selected Tasks: {task_names}")

        lm, model_args = self.get_LMEval_model(self.config)
        # model_args_str = json.dumps(model_args)
        model_args_str = '' 

        self.forward_pass_tracker.reset()
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=task_names,
            batch_size=self.config.loader.eval_batch_size,
            device=self.device,
            limit=self.config.lm_eval.limit,
            log_samples=self.config.lm_eval.log_samples,
            num_fewshot=self.config.lm_eval.num_fewshot,
            apply_chat_template=self.config.lm_eval.apply_chat_template,
            # evaluation_tracker=evaluation_tracker,
            task_manager=task_manager,
            verbosity="INFO",
        )
        generation_stats = self.forward_pass_tracker.all_stats()

        if results is not None:
            if self.config.lm_eval.log_samples:
                samples = results.pop("samples")
            
            results_to_dump = dict({'results' : results['results']})
            results_to_dump.update(generation_stats)
            results_to_dump.update(OmegaConf.to_container(self.config.sampling, resolve=True))
            results_to_dump.update(OmegaConf.to_container(self.config.lm_eval, resolve=True))
            results_to_dump.update(OmegaConf.to_container(self.config.drafting_params, resolve=True))
            dumped = json.dumps(results_to_dump, indent=2, default=handle_non_serializable, ensure_ascii=False)

            batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

            if self.config.lm_eval.output_path:
                
                output_path_file.open("w").write(dumped)

                if self.config.lm_eval.log_samples:
                    for task_name, config in results["configs"].items():
                        output_name = "{}_{}".format(
                            re.sub("/|=", "__", model_args_str), task_name
                        )
                        filename = path.joinpath(f"{output_name}.jsonl")

                        with jsonlines.open(filename, "w") as f:
                            f.write_all(samples[task_name])


            print(
                f"{self.config.lm_eval.ckpt_path} ({model_args_str}), limit: {self.config.lm_eval.limit}, "
                f"batch_size: {self.config.loader.eval_batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
            )

            print(make_table(results))
            if "groups" in results:
                print(make_table(results, "groups"))


