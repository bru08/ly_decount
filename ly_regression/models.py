# %%
import torch
import torchvision
from torchsummary import summary
from torch import nn

# %%
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        
        # get the pretrained VGG19 network
        self.vgg = torchvision.models.vgg19_bn(pretrained=True, progress=True)
        
        # disect the network to access its last convolutional layer
        self.features_conv = self.vgg.features[:-1]
        
        # get the max pool of the features stem
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        # get the classifier of the vgg19
        self.classifier = self.vgg.classifier
        in_features = self.classifier[-1].in_features
        self.classifier[-1] = nn.Linear(in_features, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv[:30](x)
        
        # register the hook
        if x.requires_grad:
          h = x.register_hook(self.activations_hook)

        x = self.features_conv[30:](x)
        
        # apply the remaining pooling
        x = self.max_pool(x)
        x = self.avg_pool(x)
 
        x = x.view((x.shape[0], -1))

        x = self.classifier(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv[:30](x)


class DenseMy(nn.Module):

    def __init__(self, model="dense121", img_size=299):
        super().__init__()
        if model == "dense121":
            self.feat_bb = torchvision.models.densenet121(pretrained=121, progress=True).features
        
        size_tst = torch.rand(1,3,img_size,img_size).float()
        ch = self.feat_bb(size_tst).detach().shape[1]


        self.loc_head = nn.Sequential(
            nn.ConvTranspose2d(in_channels=ch, out_channels=ch//2, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=ch//2, out_channels=ch//4, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=ch//4, out_channels=ch//8, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=ch//8, out_channels=1, kernel_size=5, stride=2)
        )


        self.count_head = nn.Sequential(
            nn.Linear(in_features=141*141, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=1)
        )

    def forward(self, x):
        conv_feats = self.feat_bb(x)
        loc_feats = self.loc_head(conv_feats)
        flat = loc_feats.view(loc_feats.shape[0],-1)
        counts = self.count_head(flat)
        return loc_feats, counts


class DensUNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.base_model = torchvision.models.densenet121(pretrained=True)
        self.base_layers = list(self.base_model.children())
