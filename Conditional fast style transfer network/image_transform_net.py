import torch

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride):
        super(ConvLayer, self).__init__()
        self.reflect_pad = torch.nn.ReflectionPad2d(kernel_size // 2)
        self.conv = torch.nn.Conv2d(in_channel, out_channel, kernel_size, stride)
        
    def forward(self, x):
        out = self.reflect_pad(x)
        out = self.conv(out)
        return out

class ResidualBlock(torch.nn.Module):
    """
    architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """
    
    def __init__(self, channel):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(in_channel=channel, out_channel=channel, kernel_size=3, stride=1)
        self.ins_norm1 = torch.nn.InstanceNorm2d(channel, affine=True)
        self.conv2 = ConvLayer(in_channel=channel, out_channel=channel, kernel_size=3, stride=1)
        self.ins_norm2 = torch.nn.InstanceNorm2d(channel, affine=True)
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.ins_norm1(out)
        out = self.relu(out)
        
        out = self.ins_norm2(self.conv2(out))
        out = out + residual
        
        return out
    
class UpsampleConvLayer(torch.nn.Module):
    """
    https://distill.pub/2016/deconv-checkerboard/
    """
    
    def __init__(self, in_channel, out_channel, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        self.reflect_pad = torch.nn.ReflectionPad2d(kernel_size // 2)
        self.conv = torch.nn.Conv2d(in_channel, out_channel, kernel_size, stride)
        
    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflect_pad(x_in)
        out = self.conv(out)
        return out
    
    
class ImageTransformerNet(torch.nn.Module):
    def __init__(self):
        super(ImageTransformerNet, self).__init__()
        
        # conv layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=4, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=4, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        
        # Upsampling layers
        self.up1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.inup1 = torch.nn.InstanceNorm2d(64, affine=True)
        self.up2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.inup2 = torch.nn.InstanceNorm2d(32, affine=True)
        self.up3 = UpsampleConvLayer(32, 3, kernel_size=9, stride=1)
        
        self.relu = torch.nn.ReLU()
        
    def forward(self, X):
        out = self.relu(self.in1(self.conv1(X)))
        out = self.relu(self.in2(self.conv2(out)))
        out = self.relu(self.in3(self.conv3(out)))
        
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)
        
        out = self.relu(self.inup1(self.up1(out)))
        out = self.relu(self.inup2(self.up2(out)))
        out = self.up3(out)
        
        return out
        
