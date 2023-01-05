"""Miscellaneous modules."""

# FIXME

from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import sys

import savi.lib.metrics as metrics
import savi.lib.metrics_jax as metrics_jax
import savi.modules.evaluator as evaluator
from savi.lib import utils
from savi.lib.utils import init_fn

#from LibMTL.weighting import MGDA

DType = Any
Array = torch.Tensor
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]]  # pytype: disable=not-supported-yet
ProcessorState = ArrayTree
PRNGKey = Array
NestedDict = Dict[str, Any]

# class Identity(nn.Module):
#     """Module that applies the identity function, ignoring any additional args."""

#     def __init__(self):
#         super().__init__()

#     def forward(self, inputs: Array, **args) -> Array:
#         return inputs


class Readout(nn.Module):
	"""Module for reading out multiple targets from an embedding."""

	def __init__(self,
				 keys: Sequence[str],
				 readout_modules: nn.ModuleList,
				 stop_gradient: Optional[Sequence[bool]] = None
				):
		super().__init__()

		self.keys = keys
		self.readout_modules = readout_modules
		self.stop_gradient = stop_gradient

	def forward(self, inputs: Array) -> ArrayTree:
		num_targets = len(self.keys)
		assert num_targets >= 1, "Need to have at least one target."
		assert len(self.readout_modules) == num_targets, (
			f"len(modules):({len(self.readout_modules)}) and len(keys):({len(self.keys)}) must match.")
		if self.stop_gradient is not None:
			assert len(self.stop_gradient) == num_targets, (
			f"len(stop_gradient):({len(self.stop_gradient)}) and len(keys):({len(self.keys)}) must match.")
		outputs = {}
		modules_iter = iter(self.readout_modules)
		for i in range(num_targets):
			if self.stop_gradient is not None and self.stop_gradient[i]:
				x = x.detach() # FIXME
			else:
				x = inputs
			outputs[self.keys[i]] = next(modules_iter)(x)
		return outputs

class DummyReadout(nn.Module):

	def forward(self, inputs: Array) -> ArrayTree:
		return {}

class MLP(nn.Module):
	"""Simple MLP with one hidden layer and optional pre-/post-layernorm."""

	def __init__(self,
				 input_size: int, # FIXME: added because or else can't instantiate submodules
				 hidden_size: int,
				 output_size: int, # if not given, should be inputs.shape[-1] at forward
				 num_hidden_layers: int = 1,
				 activation_fn: nn.Module = nn.ReLU,
				 layernorm: Optional[str] = None,
				 activate_output: bool = False,
				 residual: bool = False,
				 weight_init = None
				):
		super().__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.num_hidden_layers = num_hidden_layers
		self.activation_fn = activation_fn
		self.layernorm = layernorm
		self.activate_output = activate_output
		self.residual = residual
		self.weight_init = weight_init

		# submodules
		## layernorm
		if self.layernorm == "pre":
			self.layernorm_module = nn.LayerNorm(input_size, eps=1e-6)
		elif self.layernorm == "post":
			self.layernorm_module = nn.LayerNorm(output_size, eps=1e-6)
		## mlp
		self.model = nn.ModuleList()
		self.model.add_module("dense_mlp_0", nn.Linear(self.input_size, self.hidden_size))
		self.model.add_module("dense_mlp_0_act", self.activation_fn())
		for i in range(1, self.num_hidden_layers):
			self.model.add_module(f"den_mlp_{i}", nn.Linear(self.hidden_size, self.hidden_size))
			self.model.add_module(f"dense_mlp_{i}_act", self.activation_fn())
		self.model.add_module(f"dense_mlp_{self.num_hidden_layers}", nn.Linear(self.hidden_size, self.output_size))
		if self.activate_output:
			self.model.add_module(f"dense_mlp_{self.num_hidden_layers}_act", self.activation_fn())
		for name, module in self.model.named_children():
			if 'act' not in name:
				# nn.init.xavier_uniform_(module.weight)
				init_fn[weight_init['linear_w']](module.weight)
				init_fn[weight_init['linear_b']](module.bias)

	def forward(self, inputs: Array, train: bool = False) -> Array:
		del train # Unused

		x = inputs
		if self.layernorm == "pre":
			x = self.layernorm_module(x)
		for layer in self.model:
			x = layer(x)
		if self.residual:
			x = x + inputs
		if self.layernorm == "post":
			x = self.layernorm_module(x)
		return x

class myGRUCell(nn.Module):
	"""GRU cell as nn.Module

	Added because nn.GRUCell doesn't match up with jax's GRUCell...
	This one is designed to match ! (almost; output returns only once)

	The mathematical definition of the cell is as follows

  	.. math::

		\begin{array}{ll}
		r = \sigma(W_{ir} x + W_{hr} h + b_{hr}) \\
		z = \sigma(W_{iz} x + W_{hz} h + b_{hz}) \\
		n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
		h' = (1 - z) * n + z * h \\
		\end{array}
	"""

	def __init__(self,
				 input_size: int,
				 hidden_size: int,
				 gate_fn = torch.sigmoid,
				 activation_fn = torch.tanh,
				 weight_init = None
				):
		super().__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.gate_fn = gate_fn
		self.activation_fn = activation_fn
		self.weight_init = weight_init

		# submodules
		self.dense_ir = nn.Linear(input_size, hidden_size)
		self.dense_iz = nn.Linear(input_size, hidden_size)
		self.dense_in = nn.Linear(input_size, hidden_size)
		self.dense_hr = nn.Linear(hidden_size, hidden_size, bias=False)
		self.dense_hz = nn.Linear(hidden_size, hidden_size, bias=False)
		self.dense_hn = nn.Linear(hidden_size, hidden_size)
		self.reset_parameters()

	def reset_parameters(self) -> None:
		recurrent_weight_init = nn.init.orthogonal_
		if self.weight_init is not None:
			weight_init = init_fn[self.weight_init['linear_w']]
			bias_init = init_fn[self.weight_init['linear_b']]
		else:
			weight_init = nn.init.xavier_normal_
			bias_init = nn.init.zeros_
		# input weights
		weight_init(self.dense_ir.weight)
		bias_init(self.dense_ir.bias)
		weight_init(self.dense_iz.weight)
		bias_init(self.dense_iz.bias)
		weight_init(self.dense_in.weight)
		bias_init(self.dense_in.bias)
		# hidden weights
		recurrent_weight_init(self.dense_hr.weight)
		recurrent_weight_init(self.dense_hz.weight)
		recurrent_weight_init(self.dense_hn.weight)
		bias_init(self.dense_hn.bias)
	
	def forward(self, inputs, carry):
		h = carry
		# input and recurrent layeres are summed so only one needs a bias
		r = self.gate_fn(self.dense_ir(inputs) + self.dense_hr(h))
		z = self.gate_fn(self.dense_iz(inputs) + self.dense_hz(h))
		# add bias because the linear transformations aren't directly summed
		n = self.activation_fn(self.dense_in(inputs) +
							   r * self.dense_hn(h))
		new_h = (1. - z) * n + z * h
		return new_h

# class GRU(nn.Module):
#     """GRU cell as nn.Module."""

#     def __init__(self,
#                  input_size: int, # FIXME: added for submodules
#                  hidden_size: int, # FIXME: added for submodules
#                 ):
#         super().__init__()

#         # submodules
#         self.gru = nn.GRUCell(input_size, hidden_size)
	
#     def forward(self, carry: Array, inputs: Array,
#                 train: bool = False) -> Array:
#         del train # unused

#         carry = self.gru(inputs, carry)
#         return carry


# class Dense(nn.Module):
#     """Dense layer as nn.Module accepting "train" flag. """

#     def __init__(self,
#                  input_shape: int, # FIXME: added for submodules
#                  features: int,
#                  use_bias: bool = True
#                 ):
#         super().__init__()
		
#         # submodules
#         self.dense = nn.Linear(input_shape, features, use_bias)

#     def forward(self, inputs: Array, train: bool = False) -> Array:
#         del train # Unused.
#         return self.dense(inputs)


class PositionEmbedding(nn.Module):
	"""A module for applying N-dimensional position embedding.
	
	Attr:
		embedding_type: A string defining the type of position embedding to use.
			One of ["linear", "discrete_1d", "fourier", "gaussian_fourier"].
		update_type: A string defining how the input is updated with the position embedding.
			One of ["proj_add", "concat"].
		num_fourier_bases: The number of Fourier bases to use. For embedding_type == "fourier",
			the embedding dimensionality is 2 x number of position dimensions x num_fourier_bases. 
			For embedding_type == "gaussian_fourier", the embedding dimensionality is
			2 x num_fourier_bases. For embedding_type == "linear", this parameter is ignored.
		gaussian_sigma: Standard deviation of sampled Gaussians.
		pos_transform: Optional transform for the embedding.
		output_transform: Optional transform for the combined input and embedding.
		trainable_pos_embedding: Boolean flag for allowing gradients to flow into the position
			embedding, so that the optimizer can update it.
	"""

	def __init__(self,
				 input_shape: Tuple[int], # FIXME: added for submodules.
				 embedding_type: str,
				 update_type: str,
				 num_fourier_bases: int = 0,
				 gaussian_sigma: float = 1.0,
				 pos_transform: nn.Module = nn.Identity(),
				 output_transform: nn.Module = nn.Identity(),
				 trainable_pos_embedding: bool = False,
				 weight_init = None
				):
		super().__init__()

		self.input_shape = input_shape
		self.embedding_type = embedding_type
		self.update_type = update_type
		self.num_fourier_bases = num_fourier_bases
		self.gaussian_sigma = gaussian_sigma
		self.pos_transform = pos_transform
		self.output_transform = output_transform
		self.trainable_pos_embedding = trainable_pos_embedding
		self.weight_init = weight_init

		# submodules defined in module.
		self.pos_embedding = nn.Parameter(self._make_pos_embedding_tensor(input_shape),
										  requires_grad=self.trainable_pos_embedding)
		if self.update_type == "project_add":
			self.project_add_dense = nn.Linear(self.pos_embedding.shape[-1], input_shape[-1])
			# nn.init.xavier_uniform_(self.project_add_dense.weight)
			init_fn[weight_init['linear_w']](self.project_add_dense.weight)
			init_fn[weight_init['linear_b']](self.project_add_dense.bias)

	# TODO: validate
	def _make_pos_embedding_tensor(self, input_shape):
		if self.embedding_type == "discrete_1d":
			# An integer tensor in [0, input_shape[-2]-1] reflecting
			# 1D discrete position encoding (encode the second-to-last axis).
			pos_embedding = np.broadcast_to(
				np.arange(input_shape[-2]), input_shape[1:-1])
		else:
			# A tensor grid in [-1, +1] for each input dimension.
			pos_embedding = utils.create_gradient_grid(input_shape[1:-1], [-1.0, 1.0])

		if self.embedding_type == "linear":
			pos_embedding = torch.from_numpy(pos_embedding)
		elif self.embedding_type == "discrete_1d":
			pos_embedding = F.one_hot(torch.from_numpy(pos_embedding), input_shape[-2])
		elif self.embedding_type == "fourier":
			# NeRF-style Fourier/sinusoidal position encoding.
			pos_embedding = utils.convert_to_fourier_features(
				pos_embedding * np.pi, basis_degree=self.num_fourier_bases)
			pos_embedding = torch.from_numpy(pos_embedding)
		elif self.embedding_type == "gaussian_fourier":
			# Gaussian Fourier features. Reference: https://arxiv.org/abs/2006.10739
			num_dims = pos_embedding.shape[-1]
			projection = np.random.normal(
				size=[num_dims, self.num_fourier_bases]) * self.gaussian_sigma
			pos_embedding = np.pi * pos_embedding.dot(projection)
			# A slightly faster implementation of sin and cos.
			pos_embedding = np.sin(
				np.concatenate([pos_embedding, pos_embedding + 0.5 * np.pi], axis=-1))
			pos_embedding = torch.from_numpy(pos_embedding)
		else:
			raise ValueError("Invalid embedding type provided.")
		
		# Add batch dimension.
		pos_embedding = pos_embedding.unsqueeze(0)
		pos_embedding = pos_embedding.float()

		return pos_embedding
	
	def forward(self, inputs: Array) -> Array:

		# Apply optional transformation on the position embedding.
		pos_embedding = self.pos_transform(self.pos_embedding).to(inputs.get_device())

		# Apply position encoding to inputs.
		if self.update_type == "project_add":
			# Here, we project the position encodings to the same dimensionality as
			# the inputs and add them to the inputs (broadcast along batch dimension).
			# This is roughly equivalent to concatenation of position encodings to the
			# inputs (if followed by a Dense layer), but is slightly more efficient.
			x = inputs + self.project_add_dense(pos_embedding)
		elif self.update_type == "concat":
			# Repeat the position embedding along the first (batch) dimension.
			pos_embedding = torch.broadcast_to(
				pos_embedding, inputs.shape[:-1] + pos_embedding.shape[-1:])
			# concatenate along the channel dimension.
			x = torch.concat((inputs, pos_embedding), dim=-1)
		else:
			raise ValueError("Invalid update type provided.")
		
		# Apply optional output transformation.
		x = self.output_transform(x)
		return x


#####################################################
# Losses

class ReconLoss(nn.Module):
	"""L2 loss."""
	
	def __init__(self, l2_weight=1, reduction="none"):
		super().__init__()

		self.l2 = nn.MSELoss(reduction=reduction)
		self.l2_weight = l2_weight
	
	
	def gen_masked_flow(self, flow, seg, label):
		
		B, S = label.size(0), label.size(1)
		
		flow = flow.view(-1, flow.size(2), flow.size(3), flow.size(4))
		seg = seg.view(-1, seg.size(2), seg.size(3))
		label = label.view(-1)
		
		flow = np.asarray(flow.cpu())
		seg = np.asarray(seg.cpu())
		label = np.asarray(label.cpu())
		
		pos_seg = (seg!=0).astype(np.int)
		neg_seg = (seg==0).astype(np.int)
		
		pos_seg = pos_seg.reshape(pos_seg.shape[0],pos_seg.shape[1],pos_seg.shape[2],1)
		neg_seg = neg_seg.reshape(neg_seg.shape[0],neg_seg.shape[1],neg_seg.shape[2],1)
		pos_seg = np.repeat(pos_seg, 3, axis=3)
		neg_seg = np.repeat(neg_seg, 3, axis=3)
		
		for i in range(len(label)):
			if label[i] == 0: #unlabeled
				flow[i] = flow[i] * pos_seg[i]
				flow[i] = flow[i] + neg_seg[i]		
						
		flow = torch.from_numpy(flow)
		flow = flow.view(B, S, flow.size(1), flow.size(2), flow.size(3))
		flow = flow.cuda()
		return flow
		


	def forward(self, model_outputs, batch, label):
		if isinstance(model_outputs, dict):
			pred_flow = model_outputs["outputs"]["flow"]
			#[B,S,H,W,1]
			pred_seg = model_outputs["outputs"]["segmentations"]
		else:
			pred_flow = model_outputs[1]
			#[B,S,H,W,1]
			pred_seg = model_outputs[0]
			
		pred_flow = torch.squeeze(pred_flow, 4)#[B,S,H,W,3]
		pred_seg = torch.squeeze(pred_seg, 4)#[B,S,H,W]
		
		video, boxes, segmentations, gt_flow, padding_mask, mask = batch
		
		#對於unlabeled frame使用pred_seg對gt_flow進行mask
		
		#gt_flow = self.gen_masked_flow(gt_flow, pred_seg, label)
		
		# l2 loss between images and predicted images
		loss = self.l2_weight * self.l2(pred_flow, gt_flow)
		
		# sum over elements, leaving [B, -1]
		#return loss.reshape(loss.shape[0], -1).sum(-1)
		
		# sum over elements, leaving [B, S, -1]
		return loss.reshape(loss.shape[0], loss.shape[1], -1).sum(-1)

class FocalLoss(nn.Module):
	def __init__(self, gamma=6, alpha=None, size_average=True):
		super().__init__()
		self.gamma = gamma
		self.alpha = alpha
		if isinstance(alpha,(float,int)):
			self.alpha = torch.Tensor([alpha,1-alpha])
		if isinstance(alpha,list):
			self.alpha = torch.Tensor(alpha)
		self.size_average = size_average

	def forward(self, model_outputs, batch):
		video, boxes, segmentations, gt_flow, padding_mask, mask = batch
		#change to [B,S,C,H,W]
		if isinstance(model_outputs, dict):
			#[B,S,C,H,W,1]
			out_seg = model_outputs["outputs"]["log_alpha_mask"]
			#[B*S,C,H,W] with softmax and log
		else:
			#[B,S,C,H,W,1]
			out_seg = model_outputs[3]
		
		out_seg = torch.squeeze(out_seg, 5)
		out_seg = out_seg.view(-1,out_seg.size(2),out_seg.size(3),out_seg.size(4))
		onehot_seg = out_seg	
			
		
		#[B,S,H,W]
		B = segmentations.shape[0]
		S = segmentations.shape[1]
		gt_seg = segmentations
		gt_seg = gt_seg.view(-1,gt_seg.size(2),gt_seg.size(3))
		
		#gt_seg [B*S,H,W]
		#onehot_seg [B*S,C,H,W]
		
		#print(onehot_seg.shape)
		input = onehot_seg #S,C,H,W
		target = gt_seg #S,H,W
		target = target.type(torch.int64)

		input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
		input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
		input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
		target = target.view(-1,1)

		logpt = input
		logpt = logpt.gather(1,target)
		logpt = logpt.view(-1)
		pt = Variable(logpt.data.exp())

		loss = -1 * (1-pt)**self.gamma * logpt
			
		return loss.reshape(B, S, -1).sum(-1)
		
class FocalLoss_semi(nn.Module):
	def __init__(self, gamma=6, alpha=None, size_average=True):
		super().__init__()
		self.gamma = gamma
		self.alpha = alpha
		if isinstance(alpha,(float,int)):
			self.alpha = torch.Tensor([alpha,1-alpha])
		if isinstance(alpha,list):
			self.alpha = torch.Tensor(alpha)
		self.size_average = size_average

	def forward(self, model_outputs, batch, label):
		video, boxes, segmentations, gt_flow, padding_mask, mask = batch
		#change to [B,S,C,H,W]
		if isinstance(model_outputs, dict):
			#[B,S,C,H,W,1]with softmax and log
			out_seg = model_outputs["outputs"]["log_alpha_mask"]
		else:
			#[B,S,C,H,W,1]
			out_seg = model_outputs[3]
		
		out_seg = torch.squeeze(out_seg, 5)#[B,S,C,H,W]
		out_seg = out_seg.view(-1,out_seg.size(2),out_seg.size(3),out_seg.size(4))
		onehot_seg = out_seg#[B*S,C,H,W]	
		
		#[B,S,H,W]
		gt_seg = segmentations
		gt_seg = gt_seg.view(-1,gt_seg.size(2),gt_seg.size(3))
		
		#gt_seg [B*S,H,W]
		#onehot_seg [B*S,C,H,W]
		label = label.view(-1) #[B*S]
		sum = torch.zeros([1]).cuda()
		cnt = 0
		for i in range(len(label)):
			if label[i] != 0:
				sum += self.focal1(onehot_seg[i], gt_seg[i])
				cnt += 1
		return sum/cnt		
				
	def focal1(self, onehot_seg, gt_seg):
		
		input = onehot_seg #C,H,W
		target = gt_seg #H,W
		target = target.type(torch.int64)

		input = input.view(input.size(0),-1)  # C,H,W => C,H*W
		input = input.transpose(0,1)    # C,H*W => H*W,C
		target = target.view(-1,1) #H*W,1

		logpt = input
		logpt = logpt.gather(1,target)
		logpt = logpt.view(-1)
		pt = Variable(logpt.data.exp())

		loss = -1 * (1-pt)**self.gamma * logpt
		return loss.sum(-1)
	
class SemiLoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.reconloss = ReconLoss()
		self.focalloss = FocalLoss_semi()

	def forward(self, model_outputs, batch):
	#recon, focal, label = [B,S]
		video, boxes, segmentations, gt_flow, padding_mask, mask, label = batch
		batch = (video, boxes, segmentations, gt_flow, padding_mask, mask)
				
		#recon = self.reconloss(model_outputs, batch, label)
		#recon = recon.mean()
		
		focal = self.focalloss(model_outputs, batch, label)
			
		#loss = recon*0.3 + focal*0.7
		
		recon = 0
		recon = np.array(recon)
		recon = torch.from_numpy(recon)
		recon = recon.cuda()
		loss = focal
		
		return loss, recon, focal



#######################################################
# Eval Metrics

class ARI(nn.Module):
	"""ARI."""

	def forward(self, model_outputs, batch, args):
		video, boxes, segmentations, flow, padding_mask, mask = batch

		pr_seg = model_outputs[0].squeeze(-1).int().cpu().numpy()
		# pr_seg = model_outputs["outputs"]["segmentations"][:, 1:].squeeze(-1).int().cpu().numpy()
		gt_seg = segmentations.int().cpu().numpy()
		input_pad = padding_mask.cpu().numpy()
		mask = mask.cpu().numpy()
		
		mask = None
		
		# ari_bg = metrics.Ari.from_model_output(
		ari_bg = metrics_jax.Ari.from_model_output(
			predicted_segmentations=pr_seg, ground_truth_segmentations=gt_seg,
			padding_mask=input_pad,
			ground_truth_max_num_instances=args.max_instances + 1,
			predicted_max_num_instances=args.num_slots,
			ignore_background=False, mask=mask)
		# ari_nobg = metrics.Ari.from_model_output(
		ari_nobg = metrics_jax.Ari.from_model_output(
			predicted_segmentations=pr_seg, ground_truth_segmentations=gt_seg,
			padding_mask=input_pad, 
			ground_truth_max_num_instances=args.max_instances + 1,
			predicted_max_num_instances=args.num_slots,
			ignore_background=True, mask=mask)
		
		return ari_bg, ari_nobg

class mIOU(nn.Module):
	"""IOU."""
	def iou_binary_torch(self, y_true, y_pred):
		"""
		@param y_true: 4d tensor <b,s,h,w>
		@param y_pred: 4d tensor <b,s,h,w>
		@return output: int
		"""
		assert y_true.shape == y_pred.shape
		assert y_true.dim() == 4

		epsilon = 1e-15
		# sum for dim (h, w)
		intersection = (y_pred * y_true).sum(dim=[-4, -3, -2, -1])
		union = y_true.sum(dim=[-4, -3, -2, -1]) + y_pred.sum(dim=[-4, -3, -2, -1])

		return (intersection + epsilon) / (union - intersection + epsilon)

	def iou_multi_ts(self, y_true, y_pred):
		"""
		@param y_true: 4d tensor <b,s,h,w>
		@param y_pred: 4d tensor <b,s,h,w>
		@return output: int
		"""

		assert y_true.ndim == 4
		assert y_pred.ndim == 4

		result = {}

		# only calculate all labels preseneted in gt, ignore background
		temp_y_true = y_true.flatten()
		temp_y_true = temp_y_true.cpu()
		temp_y_true = np.asarray(temp_y_true)

		for instrument_id in set(temp_y_true):
			re = self.iou_binary_torch(y_true == torch.tensor(instrument_id), y_pred == torch.tensor(instrument_id))
		result[instrument_id] = re.item()
		# background with index 0 should not be counted
		if len(result.values()) != 1:
			result.pop(0, None)
		
		return sum(result.values()) / len(result.values())


	def forward(self, model_outputs, batch, args):
		video, boxes, segmentations, flow, padding_mask, mask = batch

		pr_seg = model_outputs[0].squeeze(-1)
		gt_seg = segmentations
		input_pad = padding_mask
		mask = None
		
		eval = self.iou_multi_ts(gt_seg, pr_seg)	
		
		return eval
