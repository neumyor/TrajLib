# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# https://github.com/facebookresearch/moco

# Modified by: yanchuan

import torch
import torch.nn as nn
from fedtraj.config import *

class Projector(nn.Module):
    def __init__(self, nin, nout):
        super(Projector, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(nin, nin), nn.ReLU(), nn.Linear(nin, nout))
        self.reset_parameter()

    def forward(self, x):
        return self.mlp(x)

    def reset_parameter(self):
        def _weights_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain=1.414)
                torch.nn.init.zeros_(m.bias)

        self.mlp.apply(_weights_init)

class BaseMoCo(nn.Module):
    def __init__(
        self, encoder_q, encoder_k, nemb, nout, queue_size, mmt=0.999, temperature=0.07, use_global_queue=False
    ):
        super(BaseMoCo, self).__init__()

        self.queue_size = queue_size
        self.mmt = mmt
        self.temperature = temperature
        self.use_global_queue = use_global_queue

        self.criterion = nn.CrossEntropyLoss()

        # create the encoders
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k

        self.mlp_q = Projector(nemb, nout)
        self.mlp_k = Projector(nemb, nout)

        self._init_encoder_params()

        # create the queue
        self.register_buffer("queue", torch.randn(nout, queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        if use_global_queue:
            self.register_buffer(
                "global_queue", torch.randn(nout, Config.cls_num * queue_size)
            )
            self.global_queue = nn.functional.normalize(self.global_queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def _init_encoder_params(self):
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.mlp_q.parameters(), self.mlp_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.mmt + param_q.data * (1.0 - self.mmt)

        for param_q, param_k in zip(self.mlp_q.parameters(), self.mlp_k.parameters()):
            param_k.data = param_k.data * self.mmt + param_q.data * (1.0 - self.mmt)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr : ptr + batch_size] = keys.T
        else:
            self.queue[:, ptr : self.queue_size] = keys.T[:, 0 : self.queue_size - ptr]
            self.queue[:, 0 : batch_size - self.queue_size + ptr] = keys.T[
                :, self.queue_size - ptr :
            ]

        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def fcl_dequeue_and_enqueue(self, ori_keys):
        batch_size = ori_keys.shape[0]
        this_size = batch_size * 3 // 4

        tmp_keys = ori_keys.T
        random_indices = torch.randperm(tmp_keys.size(1))
        tmp_keys = tmp_keys[:, random_indices[:this_size]]

        global_keys = self.global_queue.cuda()
        random_indices = torch.randperm(global_keys.size(1))
        global_keys = global_keys[:, random_indices[: batch_size - this_size]]

        keys = torch.cat((tmp_keys, global_keys), dim=1)
        keys = keys.T

        self._dequeue_and_enqueue(keys)

    def forward(self, kwargs_q, kwargs_k):
        # compute query features
        q = self.mlp_q(self.encoder_q(**kwargs_q))  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()  # update the key encoder
            k = self.mlp_k(self.encoder_k(**kwargs_k))  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

        # compute logits
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temperature

        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        if self.use_global_queue:
            self.fcl_dequeue_and_enqueue(k)
        else:
            self._dequeue_and_enqueue(k)

        return logits, labels

    def loss(self, logit, target):
        return self.criterion(logit, target)

class MoCo(BaseMoCo):
    def __init__(
        self, encoder_q, encoder_k, nemb, nout, queue_size, mmt=0.999, temperature=0.07
    ):
        super(MoCo, self).__init__(encoder_q, encoder_k, nemb, nout, queue_size, mmt, temperature, use_global_queue=False)

class MoCo_with_Buffer(BaseMoCo):
    def __init__(
        self, encoder_q, encoder_k, nemb, nout, queue_size, mmt=0.999, temperature=0.07
    ):
        super(MoCo_with_Buffer, self).__init__(encoder_q, encoder_k, nemb, nout, queue_size, mmt, temperature, use_global_queue=True)
