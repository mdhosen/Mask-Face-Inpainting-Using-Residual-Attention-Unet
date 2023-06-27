#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import glob
from PIL import Image

torch.manual_seed(42)


# In[2]:


class ImageDataset(Dataset):
    def __init__(self,root_masked,root_binary,transform=None):
        self.transform = transform
        self.files_masked = sorted(glob.glob(root_masked+"/*.*"))
        self.files_binary = sorted(glob.glob(root_binary+"/*.*"))
        
    def __getitem__(self,index):
        item_masked = self.transform(Image.open(self.files_masked[index%len(self.files_masked)]))
        item_binary = self.transform(Image.open(self.files_binary[index%len(self.files_binary)]))
        return (item_masked-0.5)*2,(item_binary-0.5)*2
    
    def __len__(self):
        return min(len(self.files_masked),len(self.files_binary))


# In[3]:


def crop(image,new_shape):
    middle_height = image.shape[2]//2
    middle_width = image.shape[3]//2
    starting_height = middle_height-round(new_shape[2]/2)
    final_height = starting_height+new_shape[2]
    starting_width = middle_width-round(new_shape[3]/2)
    final_width = starting_width+new_shape[3]
    cropped_image = image[:,:,starting_height:final_height,starting_width:final_width]
    return cropped_image


# In[4]:


class ContractingBlock(nn.Module):
    def __init__(self,input_channels,use_dropout=False,use_in=True):
        super(ContractingBlock,self).__init__()
        self.conv = nn.Conv2d(input_channels,input_channels*2,kernel_size=3,padding=1)
        self.activation = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        if use_in:
            self.insnorm = nn.InstanceNorm2d(input_channels*2)
        self.use_in = use_in
        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout
    
    def forward(self,x):
        x = self.conv(x)
        if self.use_in:
            x = self.insnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.maxpool(x)
        return x
    
class ExpandingBlock(nn.Module):
    def __init__(self,input_channels,use_dropout=False,use_in=True):
        super(ExpandingBlock,self).__init__()
        self.tconv = nn.ConvTranspose2d(input_channels,input_channels//2,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.conv2 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=3, padding=1)
        if use_in:
            self.insnorm = nn.InstanceNorm2d(input_channels//2)
        self.use_in = use_in
        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self,x,skip_x):
        x = self.tconv(x)
        skip_x = crop(skip_x,x.shape)
        x = torch.cat([x,skip_x],axis=1)     #really need ???
        x = self.conv2(x)
        if self.use_in:
            x = self.insnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        return x

class FeatureMapBlock(nn.Module):
    def __init__(self,input_channels,output_channels):
        super(FeatureMapBlock,self).__init__()
        self.conv = nn.Conv2d(input_channels,output_channels,kernel_size=1)
    
    def forward(self,x):
        x = self.conv(x)
        return x
    
class UNet(nn.Module):
    def __init__(self,input_channels,output_channels,hidden_channels=32):
        super(UNet,self).__init__()
        self.upfeature = FeatureMapBlock(input_channels,hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels,use_in=False)
        self.contract2 = ContractingBlock(hidden_channels*2)
        self.contract3 = ContractingBlock(hidden_channels*4)
        self.contract4 = ContractingBlock(hidden_channels*8)
        self.contract5 = ContractingBlock(hidden_channels*16)
        self.expand0 = ExpandingBlock(hidden_channels*32)
        self.expand1 = ExpandingBlock(hidden_channels*16)
        self.expand2 = ExpandingBlock(hidden_channels*8)
        self.expand3 = ExpandingBlock(hidden_channels*4)
        self.expand4 = ExpandingBlock(hidden_channels*2)
        self.downfeature = FeatureMapBlock(hidden_channels,output_channels)
        self.tanh = torch.nn.Tanh()
    
    def forward(self,x):
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)    #x4:512
        x5 = self.contract5(x4)    #x5:1024
        x6 = self.expand0(x5,x4)
        x7 = self.expand1(x6,x3)
        x8 = self.expand2(x7,x2)
        x9 = self.expand3(x8,x1)
        x10 = self.expand4(x9,x0)
        xn = self.downfeature(x10)
        return self.tanh(xn)


# In[5]:


class Discriminator(nn.Module):
    def __init__(self,input_channels,hidden_channels=8):
        super(Discriminator,self).__init__()
        self.upfeature = FeatureMapBlock(input_channels,hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels,use_in=False)
        self.contract2 = ContractingBlock(hidden_channels*2)
        self.contract3 = ContractingBlock(hidden_channels*4)
        self.contract4 = ContractingBlock(hidden_channels*8)
        self.final = nn.Conv2d(hidden_channels*16,1,kernel_size=1)   #should change?
        
    def forward(self,x):      ##without concat with masked img
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        xn = self.final(x4)
        return xn


# In[6]:


import torch.nn.functional as F
adv_criterion = nn.BCEWithLogitsLoss()
recon_criterion = nn.L1Loss()
lambda_recon = 200

n_epochs=10
input_dim = 3
binary_dim = 1         # (-1,1,224,224)
display_step = 1000
batch_size = 4
lr = 0.0002
target_shape = 224
device = 'cpu'

transform = transforms.Compose([
    transforms.ToTensor()
])
#masked_root = "data/masked"
#binary_root = "data/binary"
#dataset = ImageDataset(masked_root,binary_root,transform=transform)


# In[7]:


gen = UNet(input_dim,binary_dim).to(device)

gen_opt = torch.optim.Adam(gen.parameters(),lr=lr)
disc = Discriminator(binary_dim).to(device)
disc_opt = torch.optim.Adam(disc.parameters(),lr=lr)

#def weights_init(m):
    #if isinstance(m,nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
        #torch.nn.init.normal_(m.weight,0.0,0.02)



loaded_state = torch.load("maskModel/Bi_UNet.pth",map_location=torch.device('cpu'))
gen.load_state_dict(loaded_state["gen"])
gen_opt.load_state_dict(loaded_state["gen_opt"])
disc.load_state_dict(loaded_state["disc"])
disc_opt.load_state_dict(loaded_state["disc_opt"])


# #### Testing

# In[8]:


import cv2
path = "input_4.png"
img = cv2.imread(path)
img = cv2.resize(img,(224,224))
saveDir = "resizeInput4.png"
cv2.imwrite(saveDir,img)

from torchvision.utils import save_image
img_path = saveDir
img = transform(Image.open(img_path))

img = img.detach().cpu().view(-1,*(img.shape))
pre = gen(img)
pre = (pre + 1) / 2
pre = pre.detach().cpu().view(-1, *(1,224,224))

maskDir= 'mask_binary4.jpg'

save_image(pre,maskDir)




