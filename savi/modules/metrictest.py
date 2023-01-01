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
        
        #开头的几个
        if frame < t:
            images += list([np.asarray(Image.open(PATH + 'seq{}/masked_image/seq{}_frame{:03d}.png'.format(ins,ins,i))) \
                for i in range(frame, frame+t)])
            images = np.asarray(images)
            
            flow += list([np.asarray(Image.open(PATH + 'seq{}/masked_flow/seq{}_frame{:03d}.png'.format(ins,ins,i))) \
                for i in range(frame, frame+t)])
            flow = np.asarray(flow)
            
            segmentation += list([np.array(cv2.imread(PATH + 'seq{}/segmentation/seq{}_frame{:03d}.png'.format(ins,ins,i), cv2.IMREAD_GRAYSCALE)) \
                for i in range(frame, frame+t)])
            segmentation = np.asarray(segmentation)

        #结尾的几个
        elif frame > 223-t+1:
            images += list([np.asarray(Image.open(PATH + 'seq{}/masked_image/seq{}_frame{:03d}.png'.format(ins,ins,i))) \
                for i in range(frame-t+1, frame+1)])
            images = np.asarray(images) 
                
            flow += list([np.asarray(Image.open(PATH + 'seq{}/masked_flow/seq{}_frame{:03d}.png'.format(ins,ins,i))) \
                for i in range(frame-t+1, frame+1)])
            flow = np.asarray(flow)
                
            segmentation += list([np.array(cv2.imread(PATH + 'seq{}/segmentation/seq{}_frame{:03d}.png'.format(ins,ins,i), cv2.IMREAD_GRAYSCALE)) \
                for i in range(frame-t+1, frame+1)])
            segmentation = np.asarray(segmentation)

        #正常的
        else:
            images += list([np.asarray(Image.open(PATH + 'seq{}/masked_image/seq{}_frame{:03d}.png'.format(ins,ins,i))) \
                for i in range(frame-2, frame+t-2)])
            images = np.asarray(images)    
                
            flow += list([np.asarray(Image.open(PATH + 'seq{}/masked_flow/seq{}_frame{:03d}.png'.format(ins,ins,i))) \
                for i in range(frame-2, frame+t-2)])
            flow = np.asarray(flow)    
                
            segmentation += list([np.array(cv2.imread(PATH + 'seq{}/segmentation/seq{}_frame{:03d}.png'.format(ins,ins,i), cv2.IMREAD_GRAYSCALE)) \
                for i in range(frame-2, frame+t-2)])
            segmentation = np.asarray(segmentation)

        padding_mask = (np.ones((t, int(self.img_size['h']), int(self.img_size['w']))))

        return images, flow, segmentation, padding_mask

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
        imgs, flow, segmentation, padding_mask = self.load_data(ins, frame, self.t)
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
        
        mask = torch.empty(0, dtype=torch.bool)

        # (4, 512, 512, 3)
        # (4, 512, 512, 3)
        # (4, 512, 512)
        # (4, 7, 4)
        # (4, 512, 512)
        

        return imgs, boxes, segmentation, flow, padding_mask, mask

    def __len__(self):
        return self.num_samples

def iou_binary_torch(y_true, y_pred):
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

def iou_multi_ts(y_true, y_pred):
    """
    @param y_true: 4d tensor <b,s,h,w>
    @param y_pred: 4d tensor <b,s,h,w>
    @return output: int
    """

    assert y_true.ndim == 4
    assert y_pred.ndim == 4

    result = {}

    # only calculate all labels preseneted in gt, ignore background
    temp_y_true = np.asarray(y_true.flatten())

    for instrument_id in set(temp_y_true):
        re = iou_binary_torch(y_true == torch.tensor(instrument_id), y_pred == torch.tensor(instrument_id))
        result[instrument_id] = re.item()
    # background with index 0 should not be counted
    result.pop(0, None)

    return sum(result.values()) / len(result.values())

if __name__ == '__main__':
    dataset = endovis2017(split='train', fold = 0, t = 2)
    #train mode, use Fold 0 to valid, t/seq=4
    train_loader = torch.utils.data.DataLoader(dataset, 2, shuffle=False)
    for d in train_loader:
        (imgs, boxes, segmentation, flow, padding_mask, mask) = d
        zero = torch.zeros([2,2,64,64])
        eval = iou_multi_ts(segmentation, segmentation)
        print(eval)
        break

