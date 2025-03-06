import torch
import torch.nn as nn

from fedtraj.model.moco import MoCo, MoCo_with_Buffer
from .layers.dual_attention import DualSTB
from fedtraj.utils.traj import *


class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()

    def parameters_to_tensor(self):
        params = []
        for param in self.parameters():
            params.append(param.view(-1))
        params = torch.cat(params, 0)
        return params

    def tensor_to_parameters(self, tensor):
        p = 0
        for param in self.parameters():
            shape = param.shape
            delta = 1
            for item in shape:
                delta *= item
            param.data = tensor[p : p + delta].view(shape).detach().clone()
            p += delta

    def load_parameters(self, new_params):
        with torch.no_grad():
            for param, new_param in zip(self.parameters(), new_params):
                new_param = (
                    np.array(new_param)
                    if not isinstance(new_param, np.ndarray)
                    else new_param
                )
                param.copy_(torch.from_numpy(new_param))


class BaseTrajCL(BaseModule):
    def __init__(self, moco_class):
        super(BaseTrajCL, self).__init__()
        encoder_q = DualSTB(
            Config.seq_embedding_dim,
            Config.trans_hidden_dim,
            Config.trans_attention_head,
            Config.trans_attention_layer,
            Config.trans_attention_dropout,
            Config.trans_pos_encoder_dropout,
        )
        encoder_k = DualSTB(
            Config.seq_embedding_dim,
            Config.trans_hidden_dim,
            Config.trans_attention_head,
            Config.trans_attention_layer,
            Config.trans_attention_dropout,
            Config.trans_pos_encoder_dropout,
        )

        self.clmodel = moco_class(
            encoder_q,
            encoder_k,
            Config.seq_embedding_dim,
            Config.moco_proj_dim,
            Config.moco_nqueue,
            temperature=Config.moco_temperature,
        )

    def _create_padding_mask(self, trajs_len):
        max_len = trajs_len.max().item()
        return (
            torch.arange(max_len, device=Config.device)[None, :] >= trajs_len[:, None]
        )

    def forward(
        self, trajs1_emb, trajs1_emb_p, trajs1_len, trajs2_emb, trajs2_emb_p, trajs2_len
    ):
        src_padding_mask1 = self._create_padding_mask(trajs1_len)
        src_padding_mask2 = self._create_padding_mask(trajs2_len)

        logits, targets = self.clmodel(
            {
                "src": trajs1_emb,
                "attn_mask": None,
                "src_padding_mask": src_padding_mask1,
                "src_len": trajs1_len,
                "srcspatial": trajs1_emb_p,
            },
            {
                "src": trajs2_emb,
                "attn_mask": None,
                "src_padding_mask": src_padding_mask2,
                "src_len": trajs2_len,
                "srcspatial": trajs2_emb_p,
            },
        )
        return logits, targets

    def interpret(self, trajs1_emb, trajs1_emb_p, trajs1_len):
        src_padding_mask1 = self._create_padding_mask(trajs1_len)
        traj_embs = self.clmodel.encoder_q(
            **{
                "src": trajs1_emb,
                "attn_mask": None,
                "src_padding_mask": src_padding_mask1,
                "src_len": trajs1_len,
                "srcspatial": trajs1_emb_p,
            }
        )
        return traj_embs

    def loss(self, logits, targets):
        return self.clmodel.loss(logits, targets)


class TrajCL_with_Buffer(BaseTrajCL):
    def __init__(self):
        super(TrajCL_with_Buffer, self).__init__(MoCo_with_Buffer)

    def load_checkpoint(self):
        checkpoint_file = "{}/{}_TrajCL_best{}.pt".format(
            Config.checkpoint_dir, Config.dataset_prefix, Config.dumpfile_uniqueid
        )
        checkpoint = torch.load(checkpoint_file)
        self.load_state_dict(checkpoint["model_state_dict"])
        return self


class TrajCL(BaseTrajCL):
    def __init__(self):
        super(TrajCL, self).__init__(MoCo)

    def load_checkpoint(self):
        checkpoint_file = "{}/{}_TrajCL_best{}.pt".format(
            Config.checkpoint_dir, Config.dataset_prefix, ""
        )
        checkpoint = torch.load(checkpoint_file)
        self.load_state_dict(checkpoint["model_state_dict"])
        return self
