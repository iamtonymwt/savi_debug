from email.mime import image
import cv2
import numpy as np
np.set_printoptions(edgeitems=1000)
import random
from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision.transforms.functional as TF
from torchvision import transforms

MEAN = [0.40789654, 0.44719302, 0.47026115]
STD = [0.28863828, 0.27408164, 0.27809835]

RMEAN = np.array(MEAN, dtype=np.float32)[None, None, :]
RSTD = np.array(STD, dtype=np.float32)[None, None, :]

Folds = [[1,3],[2,5],[4,8],[6,7]]

ins_types = ['Bipolar_Forceps','Prograsp_Forcep','Large_Needle_Driver','Vessel_Sealer', 'Grasping_Retractor',
            'Monopolar_Curved_Scissors','Other']
            
#PATH = '/mnt/hd1/JinYueming/SAVi-pytorch-2017/SAVi-pytorch-2017/savi/datasets/organized2017_bySeq/'
PATH = 'organized2017_bySeq/'
            
class endovis2017(data.Dataset):
    
    #初始化了images的列表，以[seq,num]的形式存储
    def __init__(self, split, t=1, fold=0):
        super(endovis2017, self).__init__()
        self.split = split
        self.mean = np.array(MEAN, dtype=np.float32)[None, None, :]
        self.std = np.array(STD, dtype=np.float32)[None, None, :]
        self.semi_percentage = 0.3333
        self.img_size = {'h': 1024, 'w': 1280}
        self.t = t
        self.class_num = 7
        
        self.images = []
        train_images = []
        valid_images = []
        for f in range(4):
            if f==fold:
                valid_images += [[j,i] for j in Folds[f] for i in range(224)]
            else:
                train_images += [[j,i] for j in Folds[f] for i in range(224)]
        
        #train_images += [[1,i] for i in range(224)]
        #valid_images += [[1,i] for i in range(224)]
        
        self.images = train_images if self.split=='train' else valid_images
        print('Loaded {}frames'.format(len(self.images)))
        self.num_samples = len(self.images)
        self.labels = self.mark_labels('interval')
        
    #把data以numpy形式load进来，该初始化的初始化
    def load_data(self, ins, frame, t):
        # (4, 1024, 1280, 3)
        # (4, 1024, 1280, 3)
        # (4, 1024, 1280)
        # (4, 1024, 1280)
        # (4, 7, 4)

        images = [] #t, h, w, 3
        flow = [] #t, h, w, 3
        segmentation = [] #t, h, w, 1
        boxes = [] #t, 7, 4    done
        padding_mask = [] #t, h, w, 1 全1
        label = [] #t, 
        
        #开头的几个
        begin, end, flag = 0, 0, 0
        if t > frame + 1:
            begin, end = frame+t-1, frame-1
            flag = -1
        else:
            begin, end = frame-t+1, frame+1
            flag = 1

        image_path = 'images'
        flow_path = 'flow'
        for i in range(begin, end, flag):
            if int(self.labels[i]) == 1:
                image_path = 'masked_' + image_path
                flow_path = 'masked_' + flow_path
            images.append(np.asarray(Image.open(PATH + 'seq{}/'.format(ins) + image_path + '/seq{}_frame{:03d}.png'.format(ins,i))))
            flow.append(np.asarray(Image.open(PATH + 'seq{}/'.format(ins) + flow_path + '/seq{}_frame{:03d}.png'.format(ins,i))))
            segmentation.append(np.array(cv2.imread(PATH + 'seq{}/segmentation/seq{}_frame{:03d}.png'.format(ins,ins,i), cv2.IMREAD_GRAYSCALE)))
            label.append(int(self.labels[i]))

        padding_mask = (np.ones((t, int(self.img_size['h']), int(self.img_size['w']))))
        label = np.asarray(label)
        images = np.asarray(images)
        flow = np.asarray(flow)
        segmentation = np.asarray(segmentation)

        return images, flow, segmentation, padding_mask, label

    def calBBox(self, segmentation):
        t = self.t
        boxes = [] #t, 7, 4    done
        boxes = np.zeros((t, self.class_num, 4))
        for i in range(t):
            seg = segmentation[i]
            for id in range(1, self.class_num + 1):
                xy = np.where(seg == id)
                x = list(xy)[1]
                y = list(xy)[0]
                if (len(x) == 0):
                    continue;
                #ymin, xmin, ymax, xmax
                boxes[i, id-1] = np.array([min(y)/float(self.img_size['h']),\
                                                min(x)/float(self.img_size['w']),\
                                                        max(y)/float(self.img_size['h']),\
                                                            max(x)/float(self.img_size['w'])])
        return boxes

    def mark_labels(self, method):
        '''
        mark labeles for dataset
        the first and last label should be 1
        e.g. semi_percentage = 0.3
        interval: labeled = 1,0,0,1,0,0,1......1
        random: randomly mark semi_percentage of data as labeled
        '''
        num_files = self.num_samples
        ratio = self.semi_percentage
        assert num_files > 2
        if method == 'interval':
            labeled = np.zeros((num_files,))
            sep = int(1 // ratio)
            labeled[::sep] = 1
            labeled[-1] = 1
            # print(sep)
            # print(labeled)
        elif method == 'random':
            labeled = np.random.choice([0,1], size=num_files, p=[1-ratio, ratio])
        else:
            raise NotImplementedError

        return labeled

    def transform(self, images, flow, segmentation, padding_mask):

        # Random Resize
        self.img_size['w'] = 64
        self.img_size['h'] = 64

        scale = random.random()*0.4+1
        width = int(self.img_size['w']*scale)
        height = int(self.img_size['h']*scale)

        images = list([cv2.resize(image,(width,height),interpolation=cv2.INTER_LINEAR) for image in images])
        flow = list([cv2.resize(fl,(width,height),interpolation=cv2.INTER_LINEAR) for fl in flow])
        segmentation = list([cv2.resize(seg,(width,height),interpolation=cv2.INTER_NEAREST) for seg in segmentation])
        padding_mask = list([cv2.resize(pad,(width,height),interpolation=cv2.INTER_NEAREST) for pad in padding_mask])

        images = [Image.fromarray(np.uint8(img)) for img in images]
        flow = [Image.fromarray(np.uint8(fl)) for fl in flow]
        segmentation = [Image.fromarray(np.uint8(seg)) for seg in segmentation]
        padding_mask = [Image.fromarray(np.uint8(pad)) for pad in padding_mask]

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            images[0], output_size=(self.img_size['w'], self.img_size['h']))
        images = list([TF.crop(image, i, j, h, w) for image in images])
        flow = list([TF.crop(fl, i, j, h, w) for fl in flow])
        segmentation = list([TF.crop(seg, i, j, h, w) for seg in segmentation])
        padding_mask = list([TF.crop(pad, i, j, h, w) for pad in padding_mask])

        # Random horizontal flipping
        if random.random() > 0.5:
            images = list([TF.hflip(image) for image in images])
            flow = list([TF.hflip(fl) for fl in flow])
            segmentation = list([TF.hflip(seg) for seg in segmentation])

        # Random vertical flipping
        if random.random() > 0.5:
            images = list([TF.vflip(image) for image in images])
            flow = list([TF.vflip(fl) for fl in flow])
            segmentation = list([TF.vflip(seg) for seg in segmentation])

        images = [np.asarray(np.uint8(img)) for img in images]
        flow = [np.asarray(np.uint8(fl)) for fl in flow]
        segmentation = [np.asarray(np.uint8(seg)) for seg in segmentation]
        padding_mask = [np.asarray(np.uint8(pad)) for pad in padding_mask]

        images = np.asarray(images)
        flow = np.asarray(flow)
        segmentation = np.asarray(segmentation)
        padding_mask = np.asarray(padding_mask)
        
        #images = images - np.min(images)
        #images = images / np.max(images)
        #images = (images - RMEAN) / RSTD
        
        images = images / 255
        images = (images - RMEAN) / RSTD
        flow = flow / 255

        return images, flow, segmentation, padding_mask
        
        
    def __getitem__(self, index):
        ins,frame = self.images[index]
        imgs, flow, segmentation, padding_mask, label = self.load_data(ins, frame, self.t)
        imgs, flow, segmentation, padding_mask = self.transform(imgs, flow, segmentation, padding_mask)
        boxes = self.calBBox(segmentation)

        imgs = torch.from_numpy(imgs)
        imgs = imgs.type(torch.float32)
        flow = torch.from_numpy(flow)
        flow = flow.type(torch.float32)
        segmentation = torch.from_numpy(segmentation)
        segmentation = segmentation.type(torch.float32)
        padding_mask = torch.from_numpy(padding_mask)
        padding_mask = padding_mask.type(torch.int32)
        boxes = torch.from_numpy(boxes)
        boxes = boxes.type(torch.float32)
        label = torch.from_numpy(label)
        label = label.type(torch.float32)
        
        mask = torch.empty(0, dtype=torch.bool)

        # (4, 512, 512, 3)
        # (4, 512, 512, 3)
        # (4, 512, 512)
        # (4, 7, 4)
        # (4, 512, 512)
        
        # print(imgs.shape)
        # print(boxes.shape)
        # print(segmentation.shape)
        # print(flow.shape)
        # print(padding_mask.shape)
        # print(label.shape)

        return imgs, boxes, segmentation, flow, padding_mask, mask, label


    def __len__(self):
        return self.num_samples

class ReconLoss(nn.Module):
	"""L2 loss."""
	
	def __init__(self, l2_weight=1, reduction="none"):
		super().__init__()

		self.l2 = nn.MSELoss(reduction=reduction)
		self.l2_weight = l2_weight

	def forward(self, model_outputs, batch):
		if isinstance(model_outputs, dict):
			pred_flow = model_outputs["outputs"]["flow"]
			#pred_flow = model_outputs["outputs"]["segmentations"]
		else:
			# TODO: need to clean all of this up
			pred_flow = model_outputs[1]
			#pred_flow = model_outputs[0]
		pred_flow = torch.squeeze(pred_flow, 4)
		
		video, boxes, segmentations, gt_flow, padding_mask, mask = batch

		# l2 loss between images and predicted images
		loss = self.l2_weight * self.l2(pred_flow, gt_flow)
		#loss = self.l2_weight * self.l2(pred_flow, segmentations)

		# sum over elements, leaving [B, -1]
		return loss.reshape(loss.shape[0], -1).sum(-1)

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
		video, boxes, segmentations, gt_flow, padding_mask, mask, _ = batch
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
		loss = loss.mean()
			
		return loss

if __name__ == '__main__':
    dataset = endovis2017(split='train', fold = 0, t = 2)
    #train mode, use Fold 0 to valid, t/seq=4
    train_loader = torch.utils.data.DataLoader(dataset, 2, shuffle=False)
    for d in train_loader:
        (imgs, boxes, segmentation, flow, padding_mask, mask, label) = d

        print(segmentation.shape)
        zero = torch.zeros([2,2,64,64])
        focalloss = FocalLoss()
        loss = focalloss.forward(segmentation, zero)
        print(loss)
        loss = loss.mean()
        loss_value = loss.item()
        print(loss_value)

