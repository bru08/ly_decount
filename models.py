import torch
import torchvision

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



class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        
        # get the pretrained VGG19 network
        self.dense = torchvision.models.densenet121(pretrained=True, progress=True)
        
        # disect the network to access its last convolutional layer
        self.features_conv = self.dense.features[:-1]
        
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


# %%
model = torchvision