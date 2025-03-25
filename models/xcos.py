import torch
import torch.nn as nn
import torch.nn.functional as F



cos = nn.CosineSimilarity(dim=1, eps=1e-6)

def cosine_similarity_weighted(x1, x2, eps=1e-8):
    B, n, C = x1.size() #4,75,5,512
    x1 = x1.view(-1, C) #[1500,512]
    x2 = x2.view(-1, C)
    similarity = cos(x1, x2)
    return similarity

def Xcos(ftrain, ftest):#[4,75,5,512] #[4,75,5,512]
    B, n2, n1, C = ftrain.size() #4,75,5,512

    ftrain = ftrain.view(-1, C) #[1500,512]
    ftest = ftest.view(-1, C) #[1500,512]
    cos_map = 10*cos(ftrain,ftest).view(B*n2, n1)#

    return cos_map#[300,5,121]

