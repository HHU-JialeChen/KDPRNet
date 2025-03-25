import torch
from skimage import measure
import torch.nn.functional as F
import math
import torch.nn as nn
from torchvision.utils import save_image

def obj_loc(score, threshold):
    smax, sdis, sdim = 0, 0, score.size(0)
    minsize = int(math.ceil(sdim * 0.125))  #0.125
    snorm = (score - threshold).sign()
    snormdiff = (snorm[1:] - snorm[:-1]).abs()

    szero = (snormdiff==2).nonzero()
    if len(szero)==0:
       zmin, zmax = int(math.ceil(sdim*0.125)), int(math.ceil(sdim*0.875))
       return zmin, zmax

    if szero[0] > 0:
       lzmin, lzmax = 0, szero[0].item()
       lzdis = lzmax - lzmin
       lsmax, _ = score[lzmin:lzmax].max(0)
       if lsmax > smax:
          smax, zmin, zmax, sdis = lsmax, lzmin, lzmax, lzdis
       if lsmax == smax:
          if lzdis > sdis:
             smax, zmin, zmax, sdis = lsmax, lzmin, lzmax, lzdis

    if szero[-1] < sdim:
       lzmin, lzmax = szero[-1].item(), sdim
       lzdis = lzmax - lzmin
       lsmax, _ = score[lzmin:lzmax].max(0)
       if lsmax > smax:
          smax, zmin, zmax, sdis = lsmax, lzmin, lzmax, lzdis
       if lsmax == smax:
          if lzdis > sdis:
             smax, zmin, zmax, sdis = lsmax, lzmin, lzmax, lzdis

    if len(szero) >= 2:
       for i in range(len(szero)-1):
           lzmin, lzmax = szero[i].item(), szero[i+1].item()
           lzdis = lzmax - lzmin
           lsmax, _ = score[lzmin:lzmax].max(0)
           if lsmax > smax:
              smax, zmin, zmax, sdis = lsmax, lzmin, lzmax, lzdis
           if lsmax == smax:
              if lzdis > sdis:
                 smax, zmin, zmax, sdis = lsmax, lzmin, lzmax, lzdis

    if zmax - zmin <= minsize:
        pad = minsize-(zmax-zmin)
        if zmin > int(math.ceil(pad/2.0)) and sdim - zmax > pad:
            zmin = zmin - int(math.ceil(pad/2.0)) + 1
            zmax = zmax + int(math.ceil(pad/2.0))
        if zmin < int(math.ceil(pad/2.0)):
            zmin = 0
            zmax =  minsize
        if sdim - zmax < int(math.ceil(pad/2.0)):
            zmin = sdim - minsize + 1
            zmax = sdim

    return zmin, zmax
def myAOLM(feature_maps,num_train): 
    batch=feature_maps.size(0)
    width = feature_maps.size(-1)
    height = feature_maps.size(-2)
    A = torch.sum(feature_maps, dim=1, keepdim=True)#320*1*11*11
    
    camscore = F.interpolate(A, size=(84, 84), mode='bilinear', align_corners=True)#320*1*84*84
    #camscore = torch.sigmoid(camscore)#320*1*84*84
    wscore = F.max_pool2d(camscore, (84, 1)).squeeze(dim=2)  #320*1*84
    hscore = F.max_pool2d(camscore, (1, 84)).squeeze(dim=3)
    coordinates = []
    
    if (num_train==5):
        score=0.2
    else:
        score=0.4
    
    for i in range(batch):
        xs = wscore[i,0,:].squeeze()
        ys = hscore[i,0,:].squeeze()
        if xs.max() == xs.min():
           xs = xs/xs.max()
        else: 
           xs = (xs-xs.min())/(xs.max()-xs.min())
        if ys.max() == ys.min():
           ys = ys/ys.max()
        else:
           ys = (ys-ys.min())/(ys.max()-ys.min())
        x1, x2 = obj_loc(xs, score)
        y1, y2 = obj_loc(ys, score)
        coordinate = [y1, x1, y2, x2]
        coordinates.append(coordinate)
    return coordinates


def AOLM(feature_maps,num_train): #[320,512,11,11]   
    width = feature_maps.size(-1)
    height = feature_maps.size(-2)
    A = torch.sum(feature_maps, dim=1, keepdim=True)
    a = torch.mean(A, dim=[2, 3], keepdim=True)
    M = (A > a).float() #[320,1,11,11]


    coordinates = []
    for i, m in enumerate(M): # m [1,11,11]
        mask_np = m.cpu().numpy().reshape(height, width)
        component_labels = measure.label(mask_np)

        properties = measure.regionprops(component_labels)
        areas = []
        for prop in properties:
            areas.append(prop.area)
        if len(areas)==0:
            bbox = [0,0,height, width]
        else:

            max_idx = areas.index(max(areas))

            bbox = properties[max_idx].bbox

        temp = 84/width
        temp = math.floor(temp)
        x_lefttop = bbox[0] * temp - 1
        y_lefttop = bbox[1] * temp - 1
        x_rightlow = bbox[2] * temp- 1
        y_rightlow = bbox[3] * temp - 1
        x_lefttop = max(0,x_lefttop)
        y_lefttop = max(0,y_lefttop)

        coordinate = [x_lefttop, y_lefttop, x_rightlow, y_rightlow]
        coordinates.append(coordinate)
    return coordinates

def crop_featuremaps(raw_imgs, feature_maps,num_train): #[320,3,84,84] #[320,512,11,11]
    batch_size = feature_maps.size(0)
    coordinates = myAOLM(feature_maps.detach(),num_train)#my
    crop_imgs = torch.zeros([batch_size,3,84,84]).cuda()

    for i in range(batch_size):
        [x0, y0, x1, y1] = coordinates[i]
        crop_imgs[i:i+1] = F.interpolate(raw_imgs[i:i + 1, :, x0:(x1+1), y0:(y1+1)], size=(84, 84),mode='bilinear', align_corners=True)
    return crop_imgs
    

def drop_featuremaps(feature_maps):#[320,512,11,11] 
    #width = feature_maps.size(-1)
    #height = feature_maps.size(-2)
    A = torch.sum(feature_maps, dim=1, keepdim=True) #[320,1,11,11]
    a = torch.max(A,dim=3,keepdim=True)[0] #[320,1,11,1]
    a = torch.max(a,dim=2,keepdim=True)[0] #[320,1,1,1]
    threshold = 0.85
    M = (A<=threshold*a).float() #[320,1,11,11]
    fm_temp = feature_maps*M
    return fm_temp
