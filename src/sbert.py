from sentence_transformers import SentenceTransformer
import transformers
import torch 
import json
from collections import OrderedDict
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable
from torch import nn, Tensor, device
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from tqdm.autonotebook import trange
import math
import queue
import tempfile
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.model_card_templates import ModelCardTemplate
from sentence_transformers.util import import_from_string, batch_to_device, fullname, snapshot_download
import logging

class SentenceTransformerA(SentenceTransformer):
        
    def fit(self,
        train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
        evaluator: SentenceEvaluator = None,
        epochs: int = 1,
        steps_per_epoch = None,
        scheduler: str = 'WarmupLinear',
        warmup_steps: int = 10000,
        optimizer_class: Type[Optimizer] = transformers.AdamW,
        optimizer_params : Dict[str, object]= {'lr': 2e-5},
        weight_decay: float = 0.01,
        evaluation_steps: int = 0,
        output_path: str = None,
        save_best_model: bool = True,
        max_grad_norm: float = 1,
        use_amp: bool = False,
        callback: Callable[[float, int, int], None] = None,
        show_progress_bar: bool = True,
        checkpoint_path: str = None,
        checkpoint_save_steps: int = 500,
        checkpoint_save_total_limit: int = 0,
        accum_iter: int = 1
        ):
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.
        :param train_objectives: Tuples of (DataLoader, LossFunction). Pass more than one for multi-task learning
        :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
        :param epochs: Number of epochs for training
        :param steps_per_epoch: Number of training steps per epoch. If set to None (default), one epoch is equal the DataLoader size from train_objectives.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param optimizer_class: Optimizer
        :param optimizer_params: Optimizer parameters
        :param weight_decay: Weight decay for model parameters
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluation files
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param callback: Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        :param show_progress_bar: If True, output a tqdm progress bar
        :param checkpoint_path: Folder to save checkpoints during training
        :param checkpoint_save_steps: Will save a checkpoint after so many steps
        :param checkpoint_save_total_limit: Total number of checkpoints to store
        : param accum_iter: Number of batches for gradient accumulation 
        """

        ##Add info to model card
        #info_loss_functions = "\n".join(["- {} with {} training examples".format(str(loss), len(dataloader)) for dataloader, loss in train_objectives])
        info_loss_functions =  []
        for dataloader, loss in train_objectives:
            info_loss_functions.extend(ModelCardTemplate.get_train_objective_info(dataloader, loss))
        info_loss_functions = "\n\n".join([text for text in info_loss_functions])

        info_fit_parameters = json.dumps({"evaluator": fullname(evaluator), "epochs": epochs, "steps_per_epoch": steps_per_epoch, "scheduler": scheduler, "warmup_steps": warmup_steps, "optimizer_class": str(optimizer_class), "weight_decay": weight_decay, "evaluation_steps": evaluation_steps, "max_grad_norm": max_grad_norm }, indent=4, sort_keys=True) #"optimizer_params": optimizer_params, 
        logging.info(info_fit_parameters)
        self._model_card_text = None
        self._model_card_vars['{TRAINING_SECTION}'] = ModelCardTemplate.__TRAINING_SECTION__.replace("{LOSS_FUNCTIONS}", info_loss_functions).replace("{FIT_PARAMETERS}", info_fit_parameters)


        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.to(self._target_device)

        dataloaders = [dataloader for dataloader, _ in train_objectives]

        # Use smart batching
        for dataloader in dataloaders:
            dataloader.collate_fn = self.smart_batching_collate

        loss_models = [loss for _, loss in train_objectives]
        for loss_model in loss_models:
            loss_model.to(self._target_device)

        self.best_score = -9999999

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        num_train_steps = int(steps_per_epoch * epochs)

        # Prepare optimizers
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = list(optimizer_params['params']) if 'params' in optimizer_params else list(loss_model.named_parameters())

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            del optimizer_params['params']

            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
            scheduler_obj = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)


        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        num_train_objectives = len(train_objectives)
        skip_scheduler = False
        
        loss_accum = 0
        for epoch in trange(num_train_steps, desc="Epoch", disable=not show_progress_bar):

            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

            for train_idx in range(num_train_objectives):
                loss_model = loss_models[train_idx]
                optimizer = optimizers[train_idx]
                scheduler = schedulers[train_idx]
                data_iterator = data_iterators[train_idx]

                try:
                    data = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(dataloaders[train_idx])
                    data_iterators[train_idx] = data_iterator
                    data = next(data_iterator)


                features, labels = data


                if use_amp:
                    with autocast():
                        loss_value = loss_model(features, labels)
                        
                    scale_before_step = scaler.get_scale()
                    scaler.scale(loss_value/accum_iter).backward()
                    
                    if (epoch + 1) % accum_iter == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()

                        skip_scheduler = scaler.get_scale() != scale_before_step
                        optimizer.zero_grad()

                        
                else:
                    loss_value = loss_model(features, labels)/accum_iter
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                    
                    if (epoch + 1) % accum_iter == 0:
                        optimizer.step()
                        optimizer.zero_grad()

            
            if (epoch + 1) % accum_iter == 0:    
                if not skip_scheduler:
                    scheduler.step()
                global_step += 1
                #print(f"update grad at epoch {epoch}, global epoch {global_step}...")

            if global_step == 1 or global_step % evaluation_steps == 0:
                logging.info(f"global step {global_step}")
                self._eval_during_training(evaluator, output_path, save_best_model, epoch, global_step, callback)

                for loss_model in loss_models:
                    loss_model.zero_grad()
                    loss_model.train()

            if checkpoint_path is not None and checkpoint_save_steps is not None and checkpoint_save_steps > 0 and global_step % checkpoint_save_steps == 0 and (epoch + 1) % accum_iter == 0:
                print("save at global step ", global_step)
                self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)


        if evaluator is None and output_path is not None:   #No evaluator, but output path: save final model version
            self.save(output_path)

        if checkpoint_path is not None:
            self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)
