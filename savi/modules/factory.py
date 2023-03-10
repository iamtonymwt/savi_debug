"""Return model, loss, and eval metrics in 1 go 
for the SAVi model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from savi.lib.utils import init_fn

import savi.modules as modules
import savi.modules.misc as misc

from savi.modules.network import deeplabv3plus_mobilenet
from savi.modules.network import resnet34


def build_model(args):
	if args.model_size == "small":
		#slot_size = 128
		slot_size = 256
		num_slots = args.num_slots
		weight_init = args.weight_init
		'''
		encoder_backbone = modules.CNN2(
			conv_modules=nn.ModuleList([
				nn.Conv2d(3, 32, (5, 5), (1, 1), (2, 2)),
				nn.Conv2d(32, 32, (5, 5), (1, 1), (2, 2)),
				nn.Conv2d(32, 32, (5, 5), (1, 1), (2, 2)),
				nn.Conv2d(32, 32, (5, 5), (1, 1), (2, 2))]),
			weight_init=weight_init)
		encoder = modules.FrameEncoder(
			backbone=encoder_backbone,
			pos_emb=modules.PositionEmbedding(
				input_shape=(-1, 512, 640, 32),
				embedding_type="linear",
				update_type="project_add",
				output_transform=modules.MLP(
					input_size=32,
					hidden_size=64,
					output_size=32,
					layernorm="pre",
					weight_init=weight_init),
				weight_init=weight_init))
		'''
		#encoder_backbone = deeplabv3plus_mobilenet(num_classes=8)
		encoder_backbone = resnet34()
		encoder = modules.FrameEncoder(
			backbone=encoder_backbone,
			pos_emb=modules.PositionEmbedding(
				#input_shape=(-1, 64, 80, 32),
				input_shape=(-1, 64, 80, 512),
				embedding_type="linear",
				update_type="project_add",
				output_transform=modules.MLP(
					#input_size=32,
					input_size=512,
					hidden_size=64,
					#output_size=32,
					output_size=256,
					layernorm="pre",
					weight_init=weight_init),
				weight_init=weight_init))
		
		# Corrector
		corrector = modules.SlotAttention(
			#input_size=32, # TODO: validate, should be backbone output size
			input_size=256,
			#qkv_size=128,
			qkv_size=256,
			slot_size=slot_size,
			num_iterations=1,
			weight_init=weight_init)
		# Predictor
		predictor = modules.TransformerBlock(
			embed_dim=slot_size,
			num_heads=4,
			#qkv_size=128,
			qkv_size=256,
			#mlp_size=256,
			mlp_size=1024,
			weight_init=weight_init)
		# Initializer
		
		initializer = modules.CoordinateEncoderStateInit(
			embedding_transform=modules.MLP(
				input_size=4, # bounding boxes have feature size 4
				hidden_size=256,
				output_size=slot_size,
				layernorm=None,
				weight_init=weight_init),
			prepend_background=True,
			center_of_mass=False)
		
		#initializer = modules.GaussianStateInit(
			#shape=(num_slots, slot_size))
		#initializer = modules.ParamStateInit(
			#shape=(num_slots, slot_size),
			#init_fn="normal")
		# Decoder
		readout_modules = nn.ModuleList([
			nn.Linear(64, out_features) for out_features in args.targets.values()])
		for module in readout_modules.children():
			init_fn[weight_init['linear_w']](module.weight)
			init_fn[weight_init['linear_b']](module.bias)
		# decoder_backbone = modules.CNN(
		# 	features=[slot_size, 64, 64, 64, 64],
		# 	kernel_size=[(5, 5), (5, 5), (5, 5), (5, 5)],
		# 	strides=[(2, 2), (2, 2), (2, 2), (1, 1)],
		# 	padding=[2, 2, 2, "same"],
		# 	transpose_double=True,
		# 	layer_transpose=[True, True, True, False],
		# 	weight_init=weight_init)
		decoder_backbone = modules.CNN2(
			nn.ModuleList([
				nn.ConvTranspose2d(slot_size, 64, (5, 5), (2, 2), padding=(2, 2), output_padding=(1, 1)),
				nn.ConvTranspose2d(64, 64, (5, 5), (2, 2), padding=(2, 2), output_padding=(1, 1)),
				nn.ConvTranspose2d(64, 64, (5, 5), (2, 2), padding=(2, 2), output_padding=(1, 1)),
				nn.Conv2d(64, 64, (5, 5), (1, 1), (2,2))]),
				
			transpose_modules=[True, True, True, False],
			weight_init=weight_init)
		decoder = modules.SpatialBroadcastDecoder(
			resolution=(64,80), # Update if data resolution or strides change.
			backbone=decoder_backbone,
			pos_emb=modules.PositionEmbedding(
				input_shape=(-1, 64, 80, slot_size),
				embedding_type="linear",
				update_type="project_add",
				weight_init=weight_init),
			target_readout=modules.Readout(
				keys=list(args.targets),
				readout_modules=readout_modules),
			weight_init=weight_init)
		# SAVi Model
		model = modules.SAVi(
			encoder=encoder,
			decoder=decoder,
			corrector=corrector,
			predictor=predictor,
			initializer=initializer,
			decode_corrected=True,
			decode_predicted=False)
	elif args.model_size == "medium":
		slot_size = 128
		num_slots = args.num_slots
		weight_init = args.weight_init
		# Encoder
		# encoder_backbone = modules.CNN(
			# features=[3, 64, 64, 64, 64],
			# kernel_size=[(5, 5), (5, 5), (5, 5), (5, 5)],
			# strides=[(2, 2), (1, 1), (1, 1), (1, 1)],
			# padding=[(2, 2), "same", "same", "same"],
			# layer_transpose=[False, False, False, False],
			# weight_init=weight_init)
		encoder_backbone = modules.CNN2(
			conv_modules=nn.ModuleList([
				nn.Conv2d(3, 64, (5, 5), (2, 2), (2, 2)),
				nn.Conv2d(64, 64, (5, 5), (1, 1), (2, 2)),
				nn.Conv2d(64, 64, (5, 5), (1, 1), (2, 2)),
				nn.Conv2d(64, 64, (5, 5), (1, 1), (2, 2))]),
			weight_init=weight_init)
		encoder = modules.FrameEncoder(
			backbone=encoder_backbone,
			pos_emb=modules.PositionEmbedding(
				input_shape=(-1, 64, 64, 64),
				embedding_type="linear",
				update_type="project_add",
				output_transform=modules.MLP(
					input_size=64,
					hidden_size=64,
					output_size=64,
					layernorm="pre",
					weight_init=weight_init),
				weight_init=weight_init))
		# Corrector
		corrector = modules.SlotAttention(
			input_size=64, # TODO: validate, should be backbone output size
			qkv_size=128,
			slot_size=slot_size,
			num_iterations=1,
			weight_init=weight_init)
		# Predictor
		predictor = modules.TransformerBlock(
			embed_dim=slot_size,
			num_heads=4,
			qkv_size=128,
			mlp_size=256,
			weight_init=weight_init)
		# Initializer
		initializer = modules.CoordinateEncoderStateInit(
			embedding_transform=modules.MLP(
				input_size=4, # bounding boxes have feature size 4
				hidden_size=256,
				output_size=slot_size,
				layernorm=None,
				weight_init=weight_init),
			prepend_background=True,
			center_of_mass=False)
		# Decoder
		readout_modules = nn.ModuleList([
			nn.Linear(64, out_features) for out_features in args.targets.values()])
		for module in readout_modules.children():
			init_fn[weight_init['linear_w']](module.weight)
			init_fn[weight_init['linear_b']](module.bias)
		# decoder_backbone = modules.CNN(
		# 	features=[slot_size, 64, 64, 64, 64],
		# 	kernel_size=[(5, 5), (5, 5), (5, 5), (5, 5)],
		# 	strides=[(2, 2), (2, 2), (2, 2), (2, 2)],
		# 	padding=[2, 2, 2, 2],
		# 	transpose_double=True,
		# 	layer_transpose=[True, True, True, True],
		# 	weight_init=weight_init)
		decoder_backbone = modules.CNN2(
			nn.ModuleList([
				nn.ConvTranspose2d(slot_size, 64, (5, 5), (2, 2), padding=(2, 2), output_padding=(1, 1)),
				nn.ConvTranspose2d(64, 64, (5, 5), (2, 2), padding=(2, 2), output_padding=(1, 1)),
				nn.ConvTranspose2d(64, 64, (5, 5), (2, 2), padding=(2, 2), output_padding=(1, 1)),
				nn.ConvTranspose2d(64, 64, (5, 5), (2, 2), padding=(2, 2), output_padding=(1, 1))]),
			transpose_modules=[True, True, True, True],
			weight_init=weight_init)
		decoder = modules.SpatialBroadcastDecoder(
			resolution=(8,8), # Update if data resolution or strides change.
			backbone=decoder_backbone,
			pos_emb=modules.PositionEmbedding(
				input_shape=(-1, 8, 8, slot_size),
				embedding_type="linear",
				update_type="project_add",
				weight_init=weight_init),
			target_readout=modules.Readout(
				keys=list(args.targets),
				readout_modules=readout_modules),
			weight_init=weight_init)
		# SAVi Model
		model = modules.SAVi(
			encoder=encoder,
			decoder=decoder,
			corrector=corrector,
			predictor=predictor,
			initializer=initializer,
			decode_corrected=True,
			decode_predicted=False)
	else:
		raise NotImplementedError
	return model


def build_modules(args):
	"""Return the model and loss/eval processors."""
	model = build_model(args)	
	#loss = misc.ReconLoss()
	#loss = misc.FocalLoss()
	loss = misc.SemiLoss()
	#metrics = misc.ARI()
	metrics = misc.mIOU()

	return model, loss, metrics
