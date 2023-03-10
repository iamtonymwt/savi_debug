# Reference: MAE github https://github.com/facebookresearch/mae

# TODO

import jax
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import time
from typing import Iterable, Optional

import random
import math
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
import wandb

from savi.datasets.tfds import tfds_input_pipeline
from savi.datasets.tfds.tfds_dataset_wrapper import MOViData
from savi.datasets.dataloader import endovis2017
import savi.modules as modules
import savi.modules_flow as modules_flow
import savi.modules.evaluator

import savi.trainers.utils.misc as misc
import savi.trainers.utils.lr_sched as lr_sched
import savi.trainers.utils.lr_decay as lr_decay

processors_dict = {
	'savi': modules.savi_build_modules,
	'flow': modules_flow.flow_build_modules
}

def get_args():
	parser = argparse.ArgumentParser('TFDS dataset training for SAVi.')
	def adrg(name, default, type=str, help=None):
		"""ADd aRGuments to parser."""
		if help:
			parser.add_argument(name, default=default, type=type, help=help)
		else:
			parser.add_argument(name, default=default, type=type)
	
	# Training config
	adrg('--seed', 42, int)
	adrg('--epochs', 80, int)
	adrg('--num_train_steps', 53770, int)
	adrg('--batch_size', 2, int, help='Batch size')
	adrg('--seq', 3, int, help='sequence length')
	parser.add_argument('--accum_iter', default=1, type=int, help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)")
	parser.add_argument('--gpu', default='0', type=str, help='GPU id to use.')
	parser.add_argument('--slice_decode_inputs', action='store_true', help="decode in slices.")
	
	# Adam optimizer config
	adrg('--lr', 2e-4, float)
	adrg('--warmup_steps', 100, int)
	adrg('--max_grad_norm', 0.05, float)

	# Logging and Saving config
	adrg('--log_loss_every_step', 50, int)
	adrg('--eval_every_steps', 80000, int)
	adrg('--checkpoint_every_steps', 5000, int)
	# adrg('--output_dir', './output_dir', help="path where to save, empty for no saving.")
	# adrg('--log_dir', './output_dir', help="path where to log tensorboard log")
	adrg('--exp', 'test', help="experiment name")
	parser.add_argument('--no_snap', action='store_true', help="don't snapshot model")
	parser.add_argument('--wandb', action='store_true', help="wandb logging")
	adrg('--group', 'test', help="wandb logging group")

	# Loading model
	adrg('--resume_from', None, str, help="absolute path of experiment snapshot")
	adrg('--resume_encoder', None, str)
	adrg('--resume_attention', None, str)

	# Metrics Spec
	adrg('--metrics', 'loss,ari,ari_nobg')

	# Dataset
	#adrg('--tfds_name', "movi_a/128x128:1.0.0", help="Dataset for training/eval")
	#adrg('--data_dir', "/mnt/hd1/JinYueming/slot-attention-video/")
	# adrg('--shuffle_buffer_size', 64, help="should be batch_size")
	adrg('--fold', 0, int, help="fold used for validating")

	# Model
	adrg('--max_instances', 7, int, help="Number of slots") # For Movi-A,B,C, only up to 10. for MOVi-D,E, up to 23.
	adrg('--model_size', 'small', help="How to prepare data and model architecture.")
	adrg('--model_type', 'savi', help="model type")
	parser.add_argument('--init_weight', default='lecun_normal', help='weight init')
	parser.add_argument('--init_bias', default='zeros', help='bias init')

	# Evaluation
	adrg('--eval_slice_size', 6, int)
	# adrg('--eval_slice_keys', 'video,segmentations,flow,boxes')
	parser.add_argument('--eval', action='store_true', help="Perform evaluation only")
	parser.add_argument('--visual', action='store_true', help="visualize valid fold")

	args = parser.parse_args()
	# Weights
	args.weight_init = {
		'param': args.init_weight,
		'linear_w': args.init_weight,
		'linear_b': args.init_bias,
		'conv_w': args.init_weight,
		# convtranspose kernel shape requires special handling.
		'convtranspose_w': "lecun_normal_fan_out" if args.init_weight == 'lecun_normal' else args.init_weight,
		'conv_b': args.init_bias}
	# Training
	args.gpu = [int(i) for i in args.gpu.split(',')]
	# Metrics
	args.train_metrics_spec = {
		v: v for v in args.metrics.split(',')}
	args.eval_metrics_spec = {
		f"eval_{v}": v for v in args.metrics.split(',')}
	# Misc
	args.num_slots = args.max_instances + 1 # only used for metrics
	args.logging_min_n_colors = args.max_instances
	# args.eval_slice_keys = [v for v in args.eval_slice_keys.split(',')]
	args.shuffle_buffer_size = args.batch_size * 8
	# if not args.group:
	# 	args.group = f"{args.model_type}_{args.tfds_name.split('/')[0]}"
	kwargs = {}
	kwargs['slice_decode_inputs'] = True if args.slice_decode_inputs else False
	args.kwargs = kwargs

	# HARDCODED
	args.targets = {"flow": 3}
	args.losses = {f"recon_{target}": {"loss_type": "recon", "key": target}
		for target in args.targets}
	'''
	# Preprocessing
	if args.model_size == "small":
		args.preproc_train = [
			"video_from_tfds",
			f"sparse_to_dense_annotation(max_instances={args.max_instances})",
			"temporal_random_strided_window(length=6)",
			"resize_small(64)",
			"flow_to_rgb()"  # NOTE: This only uses the first two flow dimensions.
		]
		args.preproc_eval = [
			"video_from_tfds",
			f"sparse_to_dense_annotation(max_instances={args.max_instances})",
			"temporal_crop_or_pad(length=24)",
			"resize_small(64)",
			"flow_to_rgb()"  # NOTE: This only uses the first two flow dimensions.
		]
	elif args.model_size == "medium":
		args.preproc_train = [
			"video_from_tfds",
			f"sparse_to_dense_annotation(max_instances={args.max_instances})",
			"temporal_random_strided_window(length=6)",
			"resize_small(128)",
			"flow_to_rgb()"  # NOTE: This only uses the first two flow dimensions.
		]
		args.preproc_eval = [
			"video_from_tfds",
			f"sparse_to_dense_annotation(max_instances={args.max_instances})",
			"temporal_crop_or_pad(length=24)",
			"resize_small(128)",
			"flow_to_rgb()"  # NOTE: This only uses the first two flow dimensions.
		]
	'''
	return args


def build_datasets(args):
	#rng = jax.random.PRNGKey(args.seed)
	#train_ds, eval_ds = tfds_input_pipeline.create_datasets(args, rng)
	
	#traindata = MOViData(train_ds)
	#evaldata = MOViData(eval_ds)
	
	traindata = endovis2017(split='train', fold = args.fold, t = args.seq)
	evaldata = endovis2017(split='valid', fold = args.fold, t = args.seq)
	
	return traindata, evaldata

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
					data_loader: Iterable, optimizer: torch.optim.Optimizer,
					device: torch.device, epoch: int, global_step, start_time, 
					max_norm: Optional[float] = None, args=None, 
					val_loader=None, evaluator=None, scheduler=None):
	model.train(True)

	# TODO: this is needed ... cuz using hack tfds wrapper.
	dataset = data_loader.dataset
	#dataset.reset_itr()
	len_data = len(dataset)
	data_loader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=False, num_workers=3, pin_memory=True)

	# TODO: only first epoch has scheduler, and does step-wise scheduling
	#if epoch == 0:
	#	# scheduler = None
	#	scheduler = lr_sched.get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.num_train_steps, num_cycles=0.5, last_epoch=-1)
	#else:
	#	scheduler = None

	loss, flow_loss, seg_loss = None, None, None
	grad_accum = 0
	for data_iter_step, (video, boxes, segmentations, flow, padding_mask, mask, label) in enumerate(data_loader):
		# need to squeeze because of weird dataset wrapping ...
		video = video.squeeze(0).to(device, non_blocking=True) # [64, 6, 64, 64, 3]
		boxes = boxes.squeeze(0).to(device, non_blocking=True)
		flow = flow.squeeze(0).to(device, non_blocking=True)
		padding_mask = padding_mask.squeeze(0).to(device, non_blocking=True)
		mask = mask.squeeze(0).to(device, non_blocking=True) if len(mask) > 0 else None
		label = label.squeeze(0).to(device, non_blocking=True)
		segmentations = segmentations.squeeze(0).to(device, non_blocking=True)
		batch = (video, boxes, segmentations, flow, padding_mask, mask, label)

		conditioning = boxes # TODO: make this not hardcoded
		
		outputs = model(video=video, conditioning=conditioning, 
			padding_mask=padding_mask)
		itr_loss, itr_flow_loss, itr_seg_loss = criterion(outputs, batch)
		if loss == None:
			loss, flow_loss, seg_loss = itr_loss, itr_flow_loss, itr_seg_loss
		del outputs
		del batch

		grad_accum += 1
		if grad_accum != args.accum_iter:
			# accumulating gradients to reach effective batch size.
			# effective batch size is batch_size * accum_iter

			# since loss will be [loss(item) for item in batch], we can
			# update the loss by extending the losses (i think). need to check
			loss = torch.cat([loss, itr_loss], dim=0)
			flow_loss = torch.cat([flow_loss, itr_flow_loss], dim=0)
			seg_loss = torch.cat([seg_loss, itr_seg_loss], dim=0)	
		else:
			#loss = loss.mean() # sum over elements, mean over batch.
			loss_value = loss.item()
			flow_loss_value = flow_loss.item()
			seg_loss_value = seg_loss.item()

			print(f"step: {global_step+1} / {args.num_train_steps}, loss: {loss_value}, clock: {datetime.now()-start_time}")

			if not math.isfinite(loss_value):
				print("Loss is {}, stopping training".format(loss_value))
				sys.exit(1)

			optimizer.zero_grad(set_to_none=True)

			loss.backward()
			for name,weight in model.named_parameters():
				if weight.requires_grad and weight.grad != None:
					#print("name: ",name,"weight_grad: ",weight.grad)
					print("name: ",name)
			sys.exit(1)
			# clip grad norm
			# TODO: fix grad norm clipping, as it's making the loss NaN
			if max_norm is not None:
				torch.nn.utils.clip_grad_norm(model.parameters(), max_norm)
			optimizer.step()
			if scheduler is not None:
				scheduler.step()

			if args.wandb:
				wandb.log({'train/loss': loss_value})
				wandb.log({'train/flow_loss': flow_loss_value})
				wandb.log({'train/seg_loss': seg_loss_value})
				wandb.log({'train/lr': optimizer.param_groups[0]['lr']})
			
			#print('##################')
			#print(optimizer.param_groups[0]['lr'])
			#print(optimizer.param_groups[1]['lr'])
			#print('##################')
			

			# global stepper.
			global_step += 1
			# if global_step % args.log_loss_every_step == 0:
			# 	# TODO: log the loss (with tensorboard / csv)
			# 	if args.wandb:
			# 		wandb.log({'train/loss': loss_value})
			# 	print()
			# 	print()
			if global_step % args.eval_every_steps == 0:
				print()
				batch = (video, boxes, segmentations, flow, padding_mask, mask)
				evaluate(val_loader, model, criterion, evaluator, device, args, global_step)
			if not args.no_snap and global_step % args.checkpoint_every_steps == 0:
				misc.save_snapshot(args, model.module, optimizer, global_step, f'./experiments/{args.group}_{args.exp}/snapshots/{global_step}.pt')
			# SAVi doesn't train on epochs, just on steps.
			if global_step >= args.num_train_steps:
				# save before exit
				print('done training')
				misc.save_snapshot(args, model.module, optimizer, global_step, f'./experiments/{args.group}_{args.exp}/snapshots/{global_step}.pt')
				print('exiting')
				# if args.wandb:
				# 	wandb.alert(
				# 		title="End of Run",
				# 		text=f"Run {args.group}_{args.exp} ended after {datetime.now()-start_time} time")
				sys.exit(0)

			grad_accum = 0
			loss, flow_loss, seg_loss = None, None, None
	
	return global_step, loss


@torch.no_grad()
def evaluate(data_loader, model, criterion, evaluator, device, args, name="test"):

	# switch to evaluation mode
	model.eval()

	# TODO: this is needed ... cuz using hack tfds wrapper.
	dataset = data_loader.dataset
	#dataset.reset_itr()
	len_data = len(dataset)
	data_loader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=False,num_workers=3, pin_memory=True)

	loss_value = 1e12
	#ari_running = {'total': 0, 'count': 0}
	#ari_nobg_running = {'total': 0, 'count': 0}
	miou_list = []
	for i_batch, (video, boxes, segmentations, flow, padding_mask, mask, label) in enumerate(data_loader):
		# need to squeeze because of weird dataset wrapping ...
		#print(video.shape)
		#device = torch.device('cuda')
		video = video.squeeze(0).to(device, non_blocking=True) # [64, 6, 64, 64, 3]
		boxes = boxes.squeeze(0).to(device, non_blocking=True)
		flow = flow.squeeze(0).to(device, non_blocking=True)
		padding_mask = padding_mask.squeeze(0).to(device, non_blocking=True)
		mask = mask.squeeze(0).to(device, non_blocking=True) if len(mask) > 0 else None
		segmentations = segmentations.squeeze(0).to(device, non_blocking=True)
		label = label.squeeze(0).to(device, non_blocking=True)
		#batch = (video, boxes, segmentations, flow, padding_mask, mask, label)

		conditioning = boxes # TODO: don't hardcode
		
		
		batch = (video, boxes, segmentations, flow, padding_mask, mask)
		# compute output
		if args.model_type == "savi":
			outputs = savi.modules.evaluator.eval_step(model, batch, slice_size=args.eval_slice_size)
		else:
			outputs = model(video=video, conditioning=conditioning, 
				padding_mask=padding_mask, **args.kwargs)
		
		batch = (video, boxes, segmentations, flow, padding_mask, mask, label)
		loss, flow_loss, seg_loss = criterion(outputs, batch)
		#loss = loss.mean() # mean over devices
		loss_value = loss.item()
		flow_loss_value = flow_loss.item()
		seg_loss_value = seg_loss.item()

		batch = (video, boxes, segmentations, flow, padding_mask, mask)
		#ari_bg, ari_nobg = evaluator(outputs, batch, args)
		miou = evaluator(outputs, batch, args)
		miou_list.append(miou)


		#for k, v in ari_bg.items():
			#ari_running[k] += v.item()
		#for k, v in ari_nobg.items():
			#ari_nobg_running[k] += v.item()

		#print(f"{i_batch+1} / {len_data}, loss: {loss_value}, running_ari: {ari_running['total'] / ari_running['count']}, running_ari_fg: {ari_nobg_running['total'] / ari_nobg_running['count']}")
		print(f"{i_batch+1} / {len_data}, loss: {loss_value}, miou: {miou}")

		# visualize first 3 iterations
		
		if i_batch == 0:
			for i_sample in range(2):
				if args.model_type == "savi":
					B, T, H, W, _ = video.shape
					# attn = outputs['attention'][0].squeeze(1)
					#print(outputs[2])
					attn = outputs[2][i_sample].squeeze(1)
					#attn = attn.reshape(shape=(attn.shape[0], args.num_slots, int(attn.shape[-1] ** (1/2)), int(attn.shape[-1] ** (1/2))))
					attn = attn.reshape(shape=(attn.shape[0], args.num_slots, int(H/8), int(W/8)))
					attn = F.interpolate(attn, size=(H,W), mode='bilinear', align_corners=False)
					if attn.shape[-2:] != video.shape[-3:-1]:
						attn = F.interpolate(attn, size=video.shape[-3:-1], mode='bilinear', align_corners=True).view(attn.shape[0], args.num_slots, *video.shape[-3:-1])
					# pr_flow = outputs['outputs']['flow'][0]
					pr_flow = outputs[1][i_sample]
					# pr_seg = outputs['outputs']['segmentations'][0].squeeze(-1)
					pr_seg = outputs[0][i_sample].squeeze(-1)
				else:
					pr_flow = outputs[2][i_sample]
					B, T, H, W, _ = video.shape
					pr_flow = torch.cat([torch.zeros(1,H,W,2).to(pr_flow.get_device()), pr_flow], dim=0)
					pr_flow = torchvision.utils.flow_to_image(
						pr_flow.permute(0,3,1,2)).permute(0,2,3,1).reshape(shape=(T, H, W, 3))
					attn = outputs[4][i_sample]
					attn = attn.reshape(shape=(T, H, W, args.num_slots)).permute(0, 3, 1, 2)
					pr_seg = outputs[1][i_sample].squeeze(-1)
					pr_vid = outputs[0][i_sample]
					pr_vid = torch.clamp(pr_vid, 0.0, 1.0)
				pr_flow = torch.clamp(pr_flow, 0.0, 1.0)
				# visualize attention
				misc.viz_slots_flow(video[i_sample].cpu().numpy(),
					flow[i_sample].cpu().numpy(), pr_flow.cpu().numpy(), attn.cpu().numpy(),
					f"./experiments/{args.group}_{args.exp}/viz_slots_flow/{name}_{i_batch}.png",
					trunk=8, send_to_wandb=True if args.wandb else False)
				# visualize attention again
				if args.model_type == "flow":
					misc.viz_slots_frame_pred(video[i_sample].cpu().numpy(),
						pr_vid.cpu().numpy(), pr_flow.cpu().numpy(), attn.cpu().numpy(),
						f"./experiments/{args.group}_{args.exp}/viz_slots_frame_pred/{name}_{i_batch}.png",
						trunk=6, send_to_wandb=True if args.wandb else False)
				# visualize segmentation
				misc.viz_seg(video[i_sample].cpu().numpy(),
					segmentations[i_sample].int().cpu().numpy(),
					pr_seg.int().cpu().numpy(),
					f"./experiments/{args.group}_{args.exp}/viz_seg/{name}_{i_batch}.png",
					trunk=10, send_to_wandb=True if args.wandb else False)


	final_loss = loss_value
	#final_ari = ari_running['total'] / ari_running['count']
	#final_ari_nobg = ari_nobg_running['total'] / ari_nobg_running['count']
	final_miou = sum(miou_list)/len(miou_list)

	#print(f"{name}: loss: {final_loss}, ari_bg: {final_ari}, ari_fg: {final_ari_nobg}")
	print(f"{name}: loss: {final_loss}, miou: {final_miou}")

	# switch back to training
	model.train()

	# TODO: log (tensorboard or csv)
	if args.wandb:
		#wandb.log({'eval/loss': final_loss, 'eval/ari': final_ari, 'eval/ari_fg': final_ari_nobg})
		wandb.log({'eval/loss': final_loss, 'eval/iou': final_miou})

	#return final_loss, final_ari, final_ari_nobg
	return final_loss, final_miou



@torch.no_grad()
def visual(data_loader, model, criterion, evaluator, device, args, name="visual"):

	# switch to evaluation mode
	model.eval()

	# TODO: this is needed ... cuz using hack tfds wrapper.
	dataset = data_loader.dataset
	#dataset.reset_itr()
	len_data = len(dataset)
	data_loader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=False,num_workers=3, pin_memory=True)

	for i_batch, (video, boxes, segmentations, flow, padding_mask, mask, label) in enumerate(data_loader):
		batch = (video, boxes, segmentations, flow, padding_mask, mask)
		conditioning = boxes 

		# compute output
		if args.model_type == "savi":
			outputs = savi.modules.evaluator.eval_step(model, batch, slice_size=args.eval_slice_size)
		else:
			outputs = model(video=video, conditioning=conditioning, 
				padding_mask=padding_mask, **args.kwargs)

		# i_sample means the first & only seq
		i_sample = 0
		if args.model_type == "savi":
			B, T, H, W, _ = video.shape
			
			attn = outputs[2][i_sample].squeeze(1)
			#attn = attn.reshape(shape=(attn.shape[0], args.num_slots, int(attn.shape[-1] ** (1/2)), int(attn.shape[-1] ** (1/2))))
			attn = attn.reshape(shape=(attn.shape[0], args.num_slots, int(H/8), int(W/8)))
			attn = F.interpolate(attn, size=(H,W), mode='bilinear', align_corners=False)
			if attn.shape[-2:] != video.shape[-3:-1]:
				attn = F.interpolate(attn, size=video.shape[-3:-1], mode='bilinear', align_corners=True).view(attn.shape[0], args.num_slots, *video.shape[-3:-1])
			
			# pr_flow = outputs['outputs']['flow'][0]
			pr_flow = outputs[1][i_sample]
			pr_flow = torch.clamp(pr_flow, 0.0, 1.0)
			
			# pr_seg = outputs['outputs']['segmentations'][0].squeeze(-1)
			pr_seg = outputs[0][i_sample].squeeze(-1)
		
		# visualize segmentation
		misc.viz_seg(video[i_sample].cpu().numpy(),
			segmentations[i_sample].int().cpu().numpy(),
			pr_seg.int().cpu().numpy(),
			f"./experiments/{args.group}_{args.exp}/visual_seg/{name}_{i_batch}.png",
			trunk=10, send_to_wandb=True if args.wandb else False)
			
		misc.viz_slots_flow(video[i_sample].cpu().numpy(),
			flow[i_sample].cpu().numpy(), 
			pr_flow.cpu().numpy(), attn.cpu().numpy(),
			f"./experiments/{args.group}_{args.exp}/visual_slots_flow/{name}_{i_batch}.png",
			trunk=8, send_to_wandb=True if args.wandb else False)
			

#############################################################################################
def run(args):

	print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
	print("{}".format(args).replace(', ', ',\n'))

	if args.wandb:
		wandb.init(project="savi", name=args.exp, group=args.group)
	# TODO: tensorboard or csv

	device = torch.device(args.gpu[0])

	# fix the seed for reproducibility
	seed = args.seed
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)

	
	dataset_train, dataset_val = build_datasets(args)
	
	# Not using DistributedDataParallel ... only DataParallel
	# Need to set batch size to 1 because only passing through the torch dataset interface
	train_loader = torch.utils.data.DataLoader(dataset_train, args.batch_size, shuffle=True,num_workers=3, pin_memory=True)
	val_loader = torch.utils.data.DataLoader(dataset_val, args.batch_size, shuffle=True,num_workers=3, pin_memory=True)
	
	
	# Model setup
	model, criterion, evaluator = processors_dict[args.model_type](args)
	model = model.to(device)
	criterion = criterion.to(device)
	evaluator = evaluator.to(device)


	# build optimizer
	backbone_params = list(map(id, model.encoder.backbone.parameters()))
	base_params = filter(lambda p:id(p) not in backbone_params, model.parameters())
	#optimizer = torch.optim.Adam(
				#[{"params":model.encoder.backbone.parameters(),"lr":args.lr/10},
				#{"params":base_params}], 
				#lr=args.lr)
	optimizer = torch.optim.Adam(
				[{"params":model.encoder.backbone.parameters(),"lr":args.lr},
				{"params":base_params}], 
				lr=args.lr)
	
	resume_path = []
	resume_type = []
	if args.resume_from is not None:
		resume_type.append(0)
		resume_path.append(args.resume_from)
		print("___Fully__load__from___: " + args.resume_from)
	else:
		resume_path.append('None')
	if args.resume_encoder is not None:
		resume_type.append(1)
		resume_path.append(args.resume_encoder)
		print("___Partially__load__from___: " + args.resume_encoder)
	else:
		resume_path.append('None')
	if args.resume_attention is not None:
		resume_type.append(2)
		resume_path.append(args.resume_attention)
		print("___Partially__load__from___: " + args.resume_attention)
	else:
		resume_path.append('None')
	misc.load_snapshot(args, model, optimizer, device, resume_path, resume_type)
	
	

	n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
	
	print('number of params (M): %.2f' % (n_parameters / 1.e6))
	print("basic lr: %.2e" % args.lr)
	#print("backbone lr: %.2e" % (args.lr/10))
	print("backbone lr: %.2e" % (args.lr))
	print(f"effective batch size: {args.batch_size * args.accum_iter}")

	# Loss
	print("criterion = %s" % str(criterion))

	# make dataparallel
	model = nn.DataParallel(model, device_ids=args.gpu)
	criterion = nn.DataParallel(criterion, device_ids=args.gpu)

	print(f"Start training for {args.num_train_steps} steps.")
	start_time = datetime.now()
	#global_step = resume_step if args.resume_from is not None else 0
	global_step = 0
	
	scheduler = lr_sched.get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.num_train_steps, num_cycles=0.5, last_epoch=-1)

	# visual only
	if args.visual:
		visual(val_loader, model, criterion, evaluator, device, args, f"visual")
		sys.exit(1)
	
	# eval only
	if args.eval:
		# assert isinstance(args.resume_from, str), "no snapshot given."
		evaluate(val_loader, model, criterion, evaluator, device, args, f"eval")
		# evaluate(train_loader, model, criterion, evaluator, device, args, f"eval")
		sys.exit(1)

	for epoch in range(args.epochs):
		new_step, loss = train_one_epoch(
			model, criterion, train_loader,
			optimizer, device, epoch,
			global_step, start_time,
			args.max_grad_norm, args,
			val_loader, evaluator, scheduler = scheduler
		)
		global_step = new_step
		print(f"epoch: {epoch+1}, loss: {loss}, clock: {datetime.now()-start_time}")

		evaluate(val_loader, model,
			criterion, evaluator, device, args,
			f"epoch_{epoch+1}")

		if not args.no_snap:
			misc.save_snapshot(args, model.module, optimizer, global_step, f'./experiments/{args.exp}/snapshots/{epoch+1}.pt')

		# global stepper
		if global_step >= args.num_train_steps:
			break

def main():
	args = get_args()
	
	# if args.output_dir:
	# 	Path(args.output_dir).mkdir(parents=True, exist_ok=True)
	
	run(args)

def test():
	main()
	# import ipdb

	# args = get_args()

	# dataset_train, dataset_val = build_datasets(args)
	# dataloader = DataLoader(dataset_train, 1, shuffle=False)

	# for i, out in enumerate(dataloader):
	# 	print(i, [a.shape for a in out], end='\r')

	# ipdb.set_trace()

"""

python -m savi.main --gpu 1,2,3,4

"""
