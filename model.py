import torch
import torch.nn as nn
import torch.nn.functional as F

class DSConv(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c, bias=False)
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))

class ASPP(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(in_c, out_c, 1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(True))
        self.b2 = nn.Sequential(nn.Conv2d(in_c, out_c, 3, padding=6, dilation=6, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(True))
        self.b3 = nn.Sequential(nn.Conv2d(in_c, out_c, 3, padding=12, dilation=12, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(True))
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_c, out_c, 1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(True))
        self.out = nn.Sequential(nn.Conv2d(out_c * 4, out_c, 1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[2:]
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)
        gap = F.interpolate(self.gap(x), size=(h, w), mode="bilinear", align_corners=False)
        return self.out(torch.cat([b1, b2, b3, gap], dim=1))

class OffRoadSegNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Encoder (Lightweight)
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True))
        self.e1 = DSConv(32, 64, stride=2)   # 1/4
        self.e2 = DSConv(64, 128, stride=2)  # 1/8
        self.e3 = DSConv(128, 256, stride=2) # 1/16
        
        # Context
        self.aspp = ASPP(256, 256)
        
        # Decoder FPN Up
        self.up3 = nn.Conv2d(256, 128, 1)
        self.dec3 = DSConv(128 * 2, 128)
        
        self.up2 = nn.Conv2d(128, 64, 1)
        self.dec2 = DSConv(64 * 2, 64)
        
        self.up1 = nn.Conv2d(64, 32, 1)
        self.dec1 = DSConv(32 + 32, 32) # concat with stem
        
        # Heads
        self.head_main = nn.Conv2d(32, num_classes, 1)
        self.head_aux = nn.Conv2d(128, num_classes, 1) # Deep Supervision from dec3

    def forward(self, x):
        h, w = x.shape[2:]
        s0 = self.stem(x)
        e1 = self.e1(s0)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        
        ctx = self.aspp(e3)
        
        # P3
        p3 = self.up3(ctx)
        p3 = F.interpolate(p3, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([p3, e2], dim=1))
        
        # Aux output
        aux_out = self.head_aux(d3)
        
        # P2
        p2 = self.up2(d3)
        p2 = F.interpolate(p2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([p2, e1], dim=1))
        
        # P1
        p1 = self.up1(d2)
        p1 = F.interpolate(p1, size=s0.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([p1, s0], dim=1))
        
        main_out = self.head_main(d1)
        
        main_out = F.interpolate(main_out, size=(h, w), mode="bilinear", align_corners=False)
        
        if self.training:
            aux_out = F.interpolate(aux_out, size=(h, w), mode="bilinear", align_corners=False)
            return main_out, aux_out
        
        return main_out
