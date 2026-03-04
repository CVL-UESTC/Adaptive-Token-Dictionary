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

import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial

from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
from .sr_model import SRModel


@MODEL_REGISTRY.register()
class ATDModel(SRModel):

    def __init__(self, opt):
        super(ATDModel, self).__init__(opt)
        if self.opt.get('train', False):
            self.amp_scaler = amp.GradScaler() if self.opt['train'].get('use_fp16', False) else None

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        context = amp.autocast if self.opt['train'].get('use_fp16', False) else nullcontext
        with context(device_type="cuda"):
            self.output = self.net_g(self.lq)

            l_total = 0
            loss_dict = OrderedDict()
            # pixel loss
            if self.cri_pix:
                l_pix = self.cri_pix(self.output, self.gt)
                l_total += l_pix
                loss_dict['l_pix'] = l_pix

        if self.amp_scaler is None:
            l_total.backward()
            self.optimizer_g.step()
        else:
            self.amp_scaler.scale(l_total).backward()
            self.amp_scaler.step(self.optimizer_g)
            self.amp_scaler.update()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()
