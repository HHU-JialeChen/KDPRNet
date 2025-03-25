import torch
import os 
import pickle  
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models.resnet12 import resnet12
from models.resnet18 import resnet18
from models.conv4 import ConvNet4
from .xcos import Xcos,cosine_similarity_weighted
from .BAS import crop_featuremaps, drop_featuremaps
from .IA_CLoss import SupCluLoss
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
color = ['red', 'blue', 'green', 'orange', 'purple']
#print_tsne= True
def print_fig(ftest,ytest,epoch,save_dir,types):
    tsne = TSNE(n_components=2)
    transformed_features = tsne.fit_transform(ftest)
    
    plt.figure()
    for i in range(75):
        if torch.all(ytest[i]==torch.tensor([1, 0, 0, 0, 0])):
            plt.scatter(transformed_features[i, 0], transformed_features[i, 1], c=color[0])
        elif torch.all(ytest[i]==torch.tensor([0, 1, 0, 0, 0])):
            plt.scatter(transformed_features[i, 0], transformed_features[i, 1], c=color[1])
        elif torch.all(ytest[i]==torch.tensor([0, 0, 1, 0, 0])):
            plt.scatter(transformed_features[i, 0], transformed_features[i, 1], c=color[2])
        elif torch.all(ytest[i]==torch.tensor([0, 0, 0, 1, 0])):
            plt.scatter(transformed_features[i, 0], transformed_features[i, 1], c=color[3])
        elif torch.all(ytest[i]==torch.tensor([0, 0, 0, 0, 1])):
            plt.scatter(transformed_features[i, 0], transformed_features[i, 1], c=color[4])
    plt.title('t-SNE Visualization')
    plt.savefig(save_dir+'/tsne_figs/{}/{}_t_sne_plot-ori.png'.format(types,epoch))
     
def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize
        z = F.normalize(z, dim=1) # l2-normalize
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception


class DynamicGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, num_nodes):
        super(DynamicGraphConvolution, self).__init__()

        self.static_adj = nn.Sequential(
            nn.Conv1d(num_nodes, num_nodes, 1, bias=False),
            nn.LeakyReLU(0.2))
        self.static_weight = nn.Sequential(
            nn.Conv1d(in_features, out_features, 1),
            nn.LeakyReLU(0.2))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv_global = nn.Conv1d(in_features, in_features, 1)
        self.bn_global = nn.BatchNorm1d(in_features)
        self.relu = nn.LeakyReLU(0.2)

        self.conv_create_co_mat = nn.Conv1d(in_features*2, num_nodes, 1)
        self.dynamic_weight = nn.Conv1d(in_features, out_features, 1)
        self.num_nodes= num_nodes

    def forward_static_gcn(self, x):#4,512,150
        x = self.static_adj(x.transpose(1, 2))
        x = self.static_weight(x.transpose(1, 2))
        return x

    def forward_construct_dynamic_graph(self, x):
        ### Model global representations ###
        x_glb = self.gap(x)
        x_glb = self.conv_global(x_glb)
        x_glb = self.bn_global(x_glb)
        x_glb = self.relu(x_glb)
        x_glb = x_glb.expand(x_glb.size(0), x_glb.size(1), x.size(2))
        
        ### Construct the dynamic correlation matrix ###
        x = torch.cat((x_glb, x), dim=1)
        dynamic_adj = self.conv_create_co_mat(x)
        dynamic_adj = torch.sigmoid(dynamic_adj)
        return dynamic_adj

    def forward_dynamic_gcn(self, x, dynamic_adj):
        x = torch.matmul(x, dynamic_adj)
        x = self.relu(x)
        x = self.dynamic_weight(x)
        x = self.relu(x)
        return x

    def forward(self, x, dynamic_adj):#4,150,512   4,150,150 

        dynamic_adj = dynamic_adj * 0.5 / (dynamic_adj.sum(2, keepdims=True) + 1e-6)#* 0.5
        identity_matrix = torch.eye(self.num_nodes).cuda()
        identity_matrix = identity_matrix.unsqueeze(0).expand_as(dynamic_adj)  
        dynamic_adj = dynamic_adj + identity_matrix*0.8 #4,80,80
        
        x=x.transpose(1, 2)#4,512,150
        out_static = self.forward_static_gcn(x)
        x = x + out_static # residual

        return x



class Model(nn.Module):
    def __init__(self, num_classes=64, backbone='C',save_dir='',is_tsne=False,num_sample=80):
        super(Model, self).__init__()
        self.backbone = backbone

        if self.backbone == 'R':
            print('Using ResNet12')
            self.base = resnet12()
            # self.width = 6
            self.in_channel = 512
            self.temp = 64
        elif self.backbone == 'C':
            print('Using Conv64')
            self.base = ConvNet4()
            # self.width = 5
            self.in_channel = 64
            self.temp = 8
        else:
            print('Using R18')
            self.base = resnet18()
            # self.width = 5
            self.in_channel = 512
            self.temp = 64
        self.nFeat = self.base.nFeat

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.softmax= nn.Softmax(dim=-1)
        
        self.criterion_sup = SupCluLoss()
        self.distance_sup = MaximizeDistanceLossWithRegularization()

        self.epoch=0
        self.save_dir=save_dir
        self.is_tsne= False
        self.has_printed= False
        if is_tsne:
            self.is_tsne= True
            self.has_printed= False
            os.makedirs(save_dir+'/tsne_figs/train')
            os.makedirs(save_dir+'/tsne_figs/test')
        self.gcn = DynamicGraphConvolution(self.in_channel, self.in_channel, num_sample)
        self.fc = nn.Conv2d(self.nFeat, 5, (1,1), bias=False)
        
        self.transform_matrix =  nn.Sequential(
            nn.Linear(self.in_channel, self.in_channel),  
            nn.BatchNorm1d(self.in_channel),  
            nn.ReLU(),
        ) 
        self.score= 0.02#2 # 51-0.02  55-0.016



    def forward(self, xtrain, xtest, ytrain, ytest,pids_1,epoch,s_wordvec):#[4,5,3,84,84]

        batch_size, num_train = xtrain.size(0), xtrain.size(1) 
        num_test = xtest.size(1)
        K = ytrain.size(2)
        ytrain = ytrain.transpose(1, 2)
        
        xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4)) #[20,3,84,84]
        xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4)) #[300,3,84,84]
        x_all = torch.cat((xtrain, xtest), 0) #[320,3,84,84]


        f = self.base(x_all) #[320,512,11,11]
        crop_imgs= crop_featuremaps(x_all, f,num_train)
        crop_f = self.base(crop_imgs) #[320,512,11,11]

        f = self.avgpool(f).squeeze()
        crop_f = self.avgpool(crop_f).squeeze()
                  
        # Getting Prototype ftrain and ftest
        ftrain = f[:batch_size * num_train]  # [20,512,11,11]
        ftrain = ftrain.view(batch_size, num_train, -1)  # [4,5,61952]
        ftest = f[batch_size * num_train:]
        ftest = ftest.view(batch_size, num_test, -1)  #[4,75,512,11,11]
        
        
        embeddings_raw = torch.cat([ftrain,ftest],dim=1)#4,80,512
        normalized_embeddings_raw = torch.nn.functional.normalize(embeddings_raw.detach(), p=2, dim=2)#
        similarity_scores_raw = torch.matmul(normalized_embeddings_raw.unsqueeze(2), normalized_embeddings_raw.unsqueeze(1).transpose(2, 3)).squeeze() 
        similarity_scores_raw=torch.where(similarity_scores_raw> 0.999, torch.zeros_like(similarity_scores_raw),similarity_scores_raw)
        ##top way
        topk_values_raw, topk_indices_raw = torch.topk(similarity_scores_raw, int(8+num_train/5), dim=2)    
        similarity_scores_raw = torch.zeros_like(similarity_scores_raw, dtype=torch.float32)
        similarity_scores_raw.scatter_(2, topk_indices_raw, 1)#1
        ##soft gate  way  
        #similarity_scores_raw = self.softmax(similarity_scores_raw)
        #similarity_scores_raw = torch.where(similarity_scores_raw> self.score, torch.ones_like(similarity_scores_raw), torch.zeros_like(similarity_scores_raw))

        #if num_train==25:
        data = ytrain.transpose(1, 2) # 4,25,5
        mask = torch.eye(data.shape[1], device=data.device).expand(data.shape[0], -1, -1).bool()
        expanded1 = data[:, :, None, :].expand(-1, -1, data.shape[1], -1)
        expanded2 = data[:, None, :, :].expand(-1, data.shape[1], -1, -1)
        small_tensor = ((expanded1 == expanded2).all(dim=3) & ~mask).float()        
        similarity_scores_raw[:, :small_tensor.shape[1], :small_tensor.shape[2]] = small_tensor#*2
        

        z_raw = self.gcn(embeddings_raw,similarity_scores_raw)#4,150,512   4,150,150
        z_raw = z_raw.transpose(1, 2)#4,80,512

        z_raw_train = z_raw[:,:num_train,:]
        z_raw_train = torch.bmm(ytrain, z_raw_train)  
        z_raw_train = z_raw_train.div(ytrain.sum(dim=2, keepdim=True).expand_as(z_raw_train))
        z_raw_train = z_raw_train.view(batch_size, -1, *f.size()[1:])  #[4,5,512,11,11]
        f1 = z_raw_train.unsqueeze(1).repeat(1, num_test, 1, 1)  # [4,75,5,512]

        z_raw_test = z_raw[:,num_train:,:]
        f2 = z_raw_test.unsqueeze(2).repeat(1, 1, K, 1) #[4,75,5,512,11,11]

        similar1 = Xcos(f1, f2)  # [300,5]
        
        if self.is_tsne and self.epoch == epoch:
            print_fig(ftest[0].detach().cpu(),ytest[0].detach().cpu(),epoch,self.save_dir,"train")
            self.epoch+=1
            self.has_printed=True
        if self.is_tsne and not self.training and self.has_printed:
            print_fig(ftest[0].detach().cpu(),ytest[0].detach().cpu(),epoch,self.save_dir,"test")
            self.has_printed=False
        
        # Getting Prototype ftrain_crop and ftest_crop
        ftrain_crop = crop_f[:batch_size * num_train]
        ftrain_crop = ftrain_crop.view(batch_size, num_train, -1)  # [4,5,61952]
        ftest_crop = crop_f[batch_size * num_train:]
        ftest_crop = ftest_crop.view(batch_size, num_test, -1)
        
        embeddings_crop = torch.cat([ftrain_crop,ftest_crop],dim=1)
        normalized_embeddings_crop = torch.nn.functional.normalize(embeddings_crop.detach(), p=2, dim=2)
        similarity_scores_crop = torch.matmul(normalized_embeddings_crop.unsqueeze(2), normalized_embeddings_crop.unsqueeze(1).transpose(2, 3)).squeeze() 
        similarity_scores_crop=torch.where(similarity_scores_crop> 0.999,  torch.zeros_like(similarity_scores_crop),similarity_scores_crop) 
        ## top way
        topk_values_crop, topk_indices_crop = torch.topk(similarity_scores_crop, int(8+num_train/5), dim=2)    
        similarity_scores_crop = torch.zeros_like(similarity_scores_crop, dtype=torch.float32)
        similarity_scores_crop.scatter_(2, topk_indices_crop, 1)#1
        ## soft gate way     
        #similarity_scores_crop = self.softmax(similarity_scores_crop)
        #similarity_scores_crop=torch.where(similarity_scores_crop> self.score,  torch.ones_like(similarity_scores_crop), torch.zeros_like(similarity_scores_crop))
               
        similarity_scores_crop[:, :small_tensor.shape[1], :small_tensor.shape[2]] = small_tensor#*2
        
        z_crop = self.gcn(embeddings_crop,similarity_scores_crop)#4,150,512   4,150,150
        z_crop = z_crop.transpose(1, 2)#4,150,512#embeddings_crop + 
        
        z_crop_train = z_crop[:,:num_train,:]
        z_crop_train = torch.bmm(ytrain, z_crop_train)  
        z_crop_train = z_crop_train.div(ytrain.sum(dim=2, keepdim=True).expand_as(z_crop_train))
        z_crop_train = z_crop_train.view(batch_size, -1, *f.size()[1:])  #[4,5,512]
        f1_crop = z_crop_train.unsqueeze(1).repeat(1, num_test, 1, 1)  # [4,75,5,512]

        z_crop_test = z_crop[:,num_train:,:]
        f2_crop = z_crop_test.unsqueeze(2).repeat(1, 1, K, 1) #[4,75,5,512,11,11]

        similar2 = Xcos(f1_crop,f2_crop) # [300,5]
        
        ################################################################################################################
        support_vec = torch.bmm(ytrain, s_wordvec)  #4,5,512
        support_vec = support_vec.div(ytrain.sum(dim=2, keepdim=True).expand_as(support_vec))
        
        ###########
        support_vec_t = support_vec.view(batch_size * 5, -1)
        support_vec_transform = self.transform_matrix(support_vec_t).view(batch_size, 5, -1)
        support_vec_t = 0.2*support_vec_transform+0.8*support_vec#4,5,512  
        ############
        
        
        support_vec_tr = support_vec_t.transpose(1, 2)#4,5,512
        outputs1 = torch.einsum('bij,bjk->bik', z_raw, support_vec_tr)#z_raw
        outputs2 = torch.einsum('bij,bjk->bik', z_crop, support_vec_tr)#z_crop
        #output = 0.5*outputs1 + 0.5*outputs2

        if not self.training:
            return similar1, similar2, outputs1, outputs2 #, kl_loss#, sup_loss,query_weights,query_pid
        
        orth_loss = self.distance_sup(support_vec_transform)

        semantic_matrx = torch.cat([z_raw_train,z_crop_train],dim=1)#4,15,512,
        semantic_matrx_logits = nn.functional.normalize(semantic_matrx, dim=-1)
        sup_loss = self.criterion_sup(semantic_matrx_logits)

        return similar1, similar2, outputs1, outputs2, sup_loss,orth_loss
        
        
        
class MaximizeDistanceLossWithRegularization(nn.Module):  
    def __init__(self, lambda_reg=0.001):  
        super(MaximizeDistanceLossWithRegularization, self).__init__()  
        self.lambda_reg = lambda_reg  
  
    def forward(self, embeddings):  #4,5,512

        embeddings = F.normalize(embeddings, p=2, dim=2)
        dot_products = torch.bmm(embeddings, embeddings.transpose(1, 2))  
        mask = torch.eye(5, 5).to(embeddings.device).unsqueeze(0).expand(2, -1, -1)  
        dot_products.masked_fill_(mask.bool(), 0)  
        loss = torch.sum(dot_products ** 2)  
        
        return loss.mean()          

        
class MarginLoss(nn.Module):  
    def __init__(self, margin=2.0):  
        super(MarginLoss, self).__init__()  
        self.margin = margin  
  
    def forward(self, embeddings):  
        num_batches, batch_size, embedding_dim = embeddings.size()  
 
        losses = []  

        for i in range(num_batches):  
 
            batch_embeddings = embeddings[i:i+1] 
  
            expanded_embeddings = batch_embeddings.squeeze(0)  
   
            dist_sq = torch.cdist(expanded_embeddings, expanded_embeddings, p=2).pow(2)  
  
            mask = torch.eye(batch_size, device=dist_sq.device).bool()  
            dist_sq[mask] = float('inf')  
 
            min_dist_sq, _ = torch.min(dist_sq, dim=1, keepdim=True)  

            loss = torch.clamp(self.margin**2 - min_dist_sq, min=0.0)  
            
            batch_loss = loss.mean()  
  
            losses.append(batch_loss)  

        total_loss = torch.stack(losses).mean()  

        return total_loss 
        
        
        
        
        
        
        
        
        





