import torch
import torch.nn as nn

class MyConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation=1):
        super(MyConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty((out_channels, in_channels, *kernel_size)))
        self.bias = nn.Parameter(torch.empty((out_channels)))
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.dilation = dilation

    def forward(self, x):
        """
        input: 
            x:[B, C, H, W]
        output:
            y
        """
        padding = self.padding
        stride = self.stride
        dilation = self.dilation
        kernel_size = self.kernel_size
        
        B,C,H,W = x.shape
        h_out = (H + 2 * padding - ((kernel_size - 1) * dilation + 1)) // stride + 1
        w_out = (W + 2 * padding - ((kernel_size - 1) * dilation + 1)) // stride + 1
        out = torch.empty((B, self.out_channels, h_out, w_out))
        
        
        # w_out = 
        
if __name__=='__main__':
    conv2d = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=(3,3),
        stride=(1,1),
        padding=(1,1),
        bias=True
    )
    x = torch.randn(2, 3, 540, 960)
    y = conv2d(x)
    print(y.shape)
    print(conv2d.weight.shape)
    print(conv2d.bias.shape)
    
    my_conv2d = MyConv2d(3, 64, (3, 3))
    