import importlib
import time
import torch
import torch.amp as amp
from contextlib import nullcontext
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from muon import MuonWithAuxAdam

import os
import random
import numpy as np
import cv2
import pyiqa
import torch.nn.functional as F
from functools import partial

from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
from basicsr.utils.dist_util import master_only


class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1,1)).item()
    
        r_index = torch.randperm(target.size(0)).to(self.device)
    
        target = lam * target + (1-lam) * target[r_index, :]
        input_ = lam * input_ + (1-lam) * input_[r_index, :]
    
        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments)-1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_
    
@MODEL_REGISTRY.register()
class ATDUNetModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ATDUNetModel, self).__init__(opt)

        # define network

        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        metric_kwargs = {}
        test_y_channel = opt['val']['metrics']['psnr'].get('test_y_channel', False)
        metric_kwargs['test_y_channel'] = test_y_channel
        if test_y_channel: 
            metric_kwargs['color_space'] = 'ycbcr'
            
        self.psnr = pyiqa.create_metric('psnr', device=self.device, **metric_kwargs)
        self.ssim = pyiqa.create_metric('ssim', device=self.device, **metric_kwargs)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
            if self.mixing_flag:
                mixup_beta       = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
                use_identity     = self.opt['train']['mixing_augs'].get('use_identity', False)
                self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)
                
            self.init_training_settings()

            self.amp_scaler = amp.GradScaler() if self.opt['train'].get('use_fp16', False) else None

            self.iters = self.opt['datasets']['train'].get('iters')
            self.batch_size = self.opt['datasets']['train'].get('batch_size_per_gpu')
            self.mini_batch_sizes = self.opt['datasets']['train'].get('mini_batch_sizes')
            self.gt_size = self.opt['datasets']['train'].get('gt_size')
            self.mini_gt_sizes = self.opt['datasets']['train'].get('gt_sizes')
            self.groups = np.array([sum(self.iters[0:i + 1]) for i in range(0, len(self.iters))])
            self.logger_j = [True] * len(self.groups)

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(
                self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            raise ValueError('pixel loss are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        optim_type = train_opt['optim_g'].pop('type')

        if optim_type == 'MUON':
            optim_params_hidden = []
            optim_params_nonhidden = []
            for k, v in self.net_g.named_parameters():
                if v.requires_grad:
                    if 'first' not in k:
                        optim_params_hidden.append(v)
                    else:
                        optim_params_nonhidden.append(v)
                else:
                    logger = get_root_logger()
                    logger.warning(f'Params {k} will not be optimized.')
            optim_params = [optim_params_hidden, optim_params_nonhidden]
        else:
            for k, v in self.net_g.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger = get_root_logger()
                    logger.warning(f'Params {k} will not be optimized.')

        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        elif optim_type == 'MUON':
            hidden_weights = [p for p in optim_params[0] if p.ndim >= 2]
            hidden_gains_biases = [p for p in optim_params[0] if p.ndim < 2]
            nonhidden_params = optim_params[1]
            param_groups = [
                dict(params=hidden_weights, use_muon=True,
                    lr=train_opt['optim_g']['lr'][0], weight_decay=0.01),
                dict(params=hidden_gains_biases+nonhidden_params, use_muon=False,
                    lr=train_opt['optim_g']['lr'][1], betas=(0.9, 0.95), weight_decay=0.01),
            ]
            self.optimizer_g = MuonWithAuxAdam(param_groups)
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):

        j = ((current_iter>self.groups) !=True).nonzero()[0]
        if len(j) == 0:
            bs_j = len(self.groups) - 1
        else:
            bs_j = j[0]

        mini_gt_size = self.mini_gt_sizes[bs_j]
        mini_batch_size = self.mini_batch_sizes[bs_j]
        if self.logger_j[bs_j]:
            logger = get_root_logger()
            logger.info('\n Updating Patch_Size to {} and Batch_Size to {} \n'.format(mini_gt_size, mini_batch_size*torch.cuda.device_count())) 
            self.logger_j[bs_j] = False
            
        if mini_batch_size < self.batch_size:
            indices = random.sample(range(0, self.batch_size), k=mini_batch_size)
            self.lq = self.lq[indices]
            self.gt = self.gt[indices]

        if mini_gt_size < self.gt_size:
            x0 = int((self.gt_size - mini_gt_size) * random.random())
            y0 = int((self.gt_size - mini_gt_size) * random.random())
            x1 = x0 + mini_gt_size
            y1 = y0 + mini_gt_size
            self.lq = self.lq[:,:,x0:x1,y0:y1]
            self.gt = self.gt[:,:,x0:x1,y0:y1]

        self.optimizer_g.zero_grad()

        with amp.autocast(device_type="cuda") if self.opt['train'].get('use_fp16', False) else nullcontext():
            preds = self.net_g(self.lq)
            if not isinstance(preds, list):
                preds = [preds]

            self.output = preds[-1]

            loss_dict = OrderedDict()
            # pixel loss
            l_pix = 0.
            for pred in preds:
                l_pix += self.cri_pix(pred, self.gt)

            loss_dict['l_pix'] = l_pix
        
        if self.amp_scaler is None:
            l_pix.backward()
            if self.opt['train']['use_grad_clip']:
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
            self.optimizer_g.step()
        else:
            self.amp_scaler.scale(l_pix).backward()
            if self.opt['train']['use_grad_clip']:
                self.amp_scaler.unscale_(self.optimizer_g)
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
            self.amp_scaler.step(self.optimizer_g)
            self.amp_scaler.update()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def pad_test(self, window_size):        
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq      
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        num_img = 0

        metric_data = dict()
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data)
            test()
            num_img += self.lq.shape[0]

            metric_data['img'] = self.output.clamp(0, 1)
            metric_data['img2'] = self.gt

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    # self.metric_results[name] += calculate_metric(metric_data, opt_).sum().item()
                    if 'psnr' in name:
                        self.metric_results[name] += self.psnr(metric_data['img'], metric_data['img2']).sum().item()
                    elif 'ssim' in name:
                        self.metric_results[name] += self.ssim(metric_data['img'], metric_data['img2']).sum().item()

            visuals = self.get_current_visuals()
            sr_img = [tensor2img(visuals['result'][ii], rgb2bgr=rgb2bgr) for ii in range(self.output.shape[0])]
            if 'gt' in visuals:
                # gt_img = tensor2img([visuals['gt'][0]], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            for ii in range(len(val_data['lq_path'])):
                if save_img:
                    img_name = osp.splitext(osp.basename(val_data['lq_path'][ii]))[0]

                    if self.opt['is_train']:
                        save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}.png')
                    else:
                        if self.opt['val']['suffix']:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                    f'{img_name}_{self.opt["val"]["suffix"]}.png')
                        else:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                    f'{img_name}_{self.opt["name"]}.png')
                            
                    imwrite(sr_img[ii], save_img_path)

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()
        current_metric = 0.

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= num_img
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

        return current_metric


    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

    @master_only
    def save_training_state(self, epoch, current_iter):
        """Save training states during training, which will be used for
        resuming.

        Args:
            epoch (int): Current epoch.
            current_iter (int): Current iteration.
        """
        if current_iter != -1:
            state = {'epoch': epoch, 'iter': current_iter, 'optimizers': [], 'schedulers': []}
            for o in self.optimizers:
                state['optimizers'].append(o.state_dict())
            for s in self.schedulers:
                state['schedulers'].append(s.state_dict())
            save_filename = f'{current_iter}.state'
            save_path = os.path.join(self.opt['path']['training_states'], save_filename)
            if self.amp_scaler is not None:
                state['amp_scaler'] = self.amp_scaler.state_dict()

            # avoid occasional writing errors
            retry = 3
            while retry > 0:
                try:
                    torch.save(state, save_path)
                except Exception as e:
                    logger = get_root_logger()
                    logger.warning(f'Save training state error: {e}, remaining retry times: {retry - 1}')
                    time.sleep(1)
                else:
                    break
                finally:
                    retry -= 1
            if retry == 0:
                logger.warning(f'Still cannot save {save_path}. Just ignore it.')
                # raise IOError(f'Cannot save {save_path}.')

    def resume_training(self, resume_state):
        """Reload the optimizers and schedulers for resumed training.

        Args:
            resume_state (dict): Resume state.
        """
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)
        
        if self.amp_scaler is not None:
            if 'amp_scaler' in resume_state:
                self.amp_scaler.load_state_dict(resume_state['amp_scaler'])
                if self.opt['rank'] == 0:
                    logger = get_root_logger()
                    logger.info("Loading scaler from resumed state...")