"""Decoder module library."""

# FIXME

import functools
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union
from pyparsing import alphas
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from savi.lib import utils
from savi.lib.utils import init_fn

Shape = Tuple[int]

DType = Any
Array = torch.Tensor
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]]  # pytype: disable=not-supported-yet
ProcessorState = ArrayTree
PRNGKey = Array
NestedDict = Dict[str, Any]



class SpatialBroadcastDecoder(nn.Module):
	"""Spatial broadcast decoder for as set of slots (per frame)."""

	def __init__(self,
				 resolution: Sequence[int],
				 backbone: nn.Module,
				 pos_emb: nn.Module,
				 target_readout: nn.Module = None,
				 weight_init = None
				):
		super().__init__()

		self.resolution = resolution
		self.backbone = backbone
		self.pos_emb = pos_emb
		self.target_readout = target_readout
		self.weight_init = weight_init

		# submodules
		self.mask_pred = nn.Linear(self.backbone.features[-1], 8)
		# nn.init.xavier_uniform_(self.mask_pred.weight)
		init_fn[weight_init['linear_w']](self.mask_pred.weight)
		init_fn[weight_init['linear_b']](self.mask_pred.bias)

	def forward(self, slots: Array) -> Array:

		#batch_size, n_slots, n_features = slots.shape
		BT, H, W, C = slots.shape
		x = slots
		'''
		# Fold slot dim into batch dim.
		x = slots.reshape(shape=(batch_size * n_slots, n_features))

		# Spatial broadcast with position embedding.
		x = utils.spatial_broadcast(x, self.resolution)
		'''
		
		x = self.pos_emb(x)

		# bb_features.shape = (batch_size * n_slots, h, w, c)
		bb_features = self.backbone(x, channels_last=True)
		spatial_dims = bb_features.shape[-3:-1]
		

		alpha_logits = self.mask_pred( # take each feature separately
			bb_features.reshape(shape=(-1, bb_features.shape[-1])))
		#alpha_logits = alpha_logits.reshape(
			#shape=(batch_size, n_slots, *spatial_dims, -1)) # (B O H W 1)
		alpha_logits = alpha_logits.reshape(
			shape=(BT, -1, *spatial_dims, 1)) # (BT, 8, H, W, 1)
		
		
		alpha_mask = alpha_logits.softmax(dim=1)
		log_alpha_mask = F.log_softmax(alpha_logits,dim=1)


		preds_dict = dict()
		
		flow = np.zeros([BT, 512, 640, 3])
		flow = torch.from_numpy(flow)
		flow = flow.cuda()
		preds_dict["flow"] = flow
		
		preds_dict["alpha_mask"] = alpha_mask
		#print(alpha_mask.requires_grad)
		#print(alpha_mask.shape)[32,8,64,64,1]
		preds_dict["log_alpha_mask"] = log_alpha_mask
		preds_dict["segmentations"] = alpha_logits.argmax(dim=1)
		

		return preds_dict
