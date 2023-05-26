# import torch.nn as nn
import torch
from torch.autograd import Variable
# from core.modules import (
#     ResidualConv,
#     ASPP,
#     AttentionBlock,
#     Upsample_,
#     Squeeze_Excite_Block,
# )

# +
# class ResUnetPlusPlus(nn.Module):
#     def __init__(self, channel, filters=[32, 64, 128, 256, 512]):
#         super(ResUnetPlusPlus, self).__init__()

#         self.input_layer = nn.Sequential(
#             nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
#             nn.BatchNorm2d(filters[0]),
#             nn.ReLU(),
#             nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
#         )
#         self.input_skip = nn.Sequential(
#             nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
#         )

#         self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])

#         self.residual_conv1 = ResidualConv(filters[0], filters[1], 2, 1)

#         self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])

#         self.residual_conv2 = ResidualConv(filters[1], filters[2], 2, 1)

#         self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])

#         self.residual_conv3 = ResidualConv(filters[2], filters[3], 2, 1)

#         self.aspp_bridge = ASPP(filters[3], filters[4])

#         self.attn1 = AttentionBlock(filters[2], filters[4], filters[4])
#         self.upsample1 = Upsample_(2)
#         self.up_residual_conv1 = ResidualConv(filters[4] + filters[2], filters[3], 1, 1)

#         self.attn2 = AttentionBlock(filters[1], filters[3], filters[3])
#         self.upsample2 = Upsample_(2)
#         self.up_residual_conv2 = ResidualConv(filters[3] + filters[1], filters[2], 1, 1)

#         self.attn3 = AttentionBlock(filters[0], filters[2], filters[2])
#         self.upsample3 = Upsample_(2)
#         self.up_residual_conv3 = ResidualConv(filters[2] + filters[0], filters[1], 1, 1)

#         self.aspp_out = ASPP(filters[1], filters[0])

#         self.output_layer = nn.Sequential(nn.Conv2d(filters[0], 1, 1), nn.Sigmoid())

#     def forward(self, x):
#         x1 = self.input_layer(x) + self.input_skip(x)

#         x2 = self.squeeze_excite1(x1)
#         x2 = self.residual_conv1(x2)

#         x3 = self.squeeze_excite2(x2)
#         x3 = self.residual_conv2(x3)

#         x4 = self.squeeze_excite3(x3)
#         x4 = self.residual_conv3(x4)

#         x5 = self.aspp_bridge(x4)

#         x6 = self.attn1(x3, x5)
#         x6 = self.upsample1(x6)
#         x6 = torch.cat([x6, x3], dim=1)
#         x6 = self.up_residual_conv1(x6)

#         x7 = self.attn2(x2, x6)
#         x7 = self.upsample2(x7)
#         x7 = torch.cat([x7, x2], dim=1)
#         x7 = self.up_residual_conv2(x7)

#         x8 = self.attn3(x1, x7)
#         x8 = self.upsample3(x8)
#         x8 = torch.cat([x8, x1], dim=1)
#         x8 = self.up_residual_conv3(x8)

#         x9 = self.aspp_out(x8)
#         out = self.output_layer(x9)

#         return out

# +
import torch
import torch.nn as nn

class Squeeze_Excitation(nn.Module):
    def __init__(self, channel, r=8):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        b, c, _, _ = inputs.shape
        x = self.pool(inputs).view(b, c)
        x = self.net(x).view(b, c, 1, 1)
        x = inputs * x
        return x

class Stem_Block(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_c),
        )

        self.attn = Squeeze_Excitation(out_c)

    def forward(self, inputs):
        x = self.c1(inputs)
        s = self.c2(inputs)
        y = self.attn(x + s)
        return y

class ResNet_Block(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_c),
        )

        self.attn = Squeeze_Excitation(out_c)

    def forward(self, inputs):
        x = self.c1(inputs)
        s = self.c2(inputs)
        y = self.attn(x + s)
        return y

class ASPP(nn.Module):
    def __init__(self, in_c, out_c, rate=[1, 6, 12, 18]):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[0], padding=rate[0]),
            nn.BatchNorm2d(out_c)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[1], padding=rate[1]),
            nn.BatchNorm2d(out_c)
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[2], padding=rate[2]),
            nn.BatchNorm2d(out_c)
        )

        self.c4 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[3], padding=rate[3]),
            nn.BatchNorm2d(out_c)
        )

        self.c5 = nn.Conv2d(out_c, out_c, kernel_size=1, padding=0)


    def forward(self, inputs):
        x1 = self.c1(inputs)
        x2 = self.c2(inputs)
        x3 = self.c3(inputs)
        x4 = self.c4(inputs)
        x = x1 + x2 + x3 + x4
        y = self.c5(x)
        return y

class Attention_Block(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        out_c = in_c[1]

        self.g_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[0]),
            nn.ReLU(),
            nn.Conv2d(in_c[0], out_c, kernel_size=3, padding=1),
            nn.MaxPool2d((2, 2))
        )

        self.x_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[1]),
            nn.ReLU(),
            nn.Conv2d(in_c[1], out_c, kernel_size=3, padding=1),
        )

        self.gc_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[1]),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        )

    def forward(self, g, x):
        g_pool = self.g_conv(g)
        x_conv = self.x_conv(x)
        gc_sum = g_pool + x_conv
        gc_conv = self.gc_conv(gc_sum)
        y = gc_conv * x
        return y

class Decoder_Block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.a1 = Attention_Block(in_c)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.r1 = ResNet_Block(in_c[0]+in_c[1], out_c, stride=1)

    def forward(self, g, x):
        d = self.a1(g, x)
        d = self.up(d)
        d = torch.cat([d, g], axis=1)
        d = self.r1(d)
        return d

class ResUnetPlusPlus(nn.Module):
    def __init__(self):
        super().__init__()
#         n_filters = [32, 64, 128, 256, 512]

#         self.c1 = Stem_Block(3, 32, stride=1)
#         self.c1 = Stem_Block(1, 16, stride=1)
#         self.c2 = ResNet_Block(16, 32, stride=2)
#         self.c3 = ResNet_Block(32, 64, stride=2)
#         self.c4 = ResNet_Block(64, 128, stride=2)
        
#         self.b1 = nn.Conv2d(16,16, (3, 3), padding=1)
#         self.b2 = nn.Conv2d(32, 16, (3, 3), padding=1)
#         self.ab2 = nn.Conv2d(32, 16, (1, 1))
#         self.s2 = nn.Conv2d(16, 128, (1, 1), stride=2)
#         self.b3 = nn.Conv2d(64, 128, (3, 3), padding=1)
#         self.ab3 = nn.Conv2d(256, 128, (1, 1))
#         self.s3 = nn.Conv2d(128, 128, (1, 1), stride=2)
#         self.b4 = nn.Conv2d(128, 128, (3, 3), padding=1)
#         self.ab4 = nn.Conv2d(256, 128, (1, 1))
#         self.pred = nn.Conv2d(128, 128, (1, 1))
#         self.batch16 = nn.BatchNorm2d(16)
#         self.batch32 = nn.BatchNorm2d(32)
#         self.batch64 = nn.BatchNorm2d(64)
#         self.batch128 = nn.BatchNorm2d(128)
#         self.rel = nn.ReLU()
#         self.b11 = nn.Conv2d(16, 16, (3, 3), padding=1, stride=2)
#         self.b22 = nn.Conv2d(16, 16, (3, 3), padding=1)
#         self.b33 = nn.Conv2d(128, 128, (3, 3), padding=1)
#         self.b44 = nn.Conv2d(128, 128, (3, 3), padding=1)

#         self.c1 = Stem_Block(1, 16, stride=1)
#         self.c2 = ResNet_Block(16, 32, stride=2)
#         self.c3 = ResNet_Block(32, 64, stride=2)
#         self.c4 = ResNet_Block(64, 128, stride=2)
        
#         self.b1 = nn.Conv2d(16,32, (3, 3), padding=1)
#         self.b2 = nn.Conv2d(32, 32, (3, 3), padding=1)
#         self.ab2 = nn.Conv2d(64, 32, (1, 1))
#         self.s2 = nn.Conv2d(32, 128, (1, 1), stride=2)
#         self.b3 = nn.Conv2d(64, 128, (3, 3), padding=1)
#         self.ab3 = nn.Conv2d(256, 128, (1, 1))
#         self.s3 = nn.Conv2d(128, 128, (1, 1), stride=2)
#         self.b4 = nn.Conv2d(128, 128, (3, 3), padding=1)
#         self.ab4 = nn.Conv2d(256, 128, (1, 1))
#         self.pred = nn.Conv2d(128, 128, (1, 1))
#         self.batch16 = nn.BatchNorm2d(16)
#         self.batch32 = nn.BatchNorm2d(32)
#         self.batch64 = nn.BatchNorm2d(64)
#         self.batch128 = nn.BatchNorm2d(128)
#         self.rel = nn.ReLU()
#         self.b11 = nn.Conv2d(32, 32, (3, 3), padding=1, stride=2)
#         self.b22 = nn.Conv2d(32, 32, (3, 3), padding=1)
#         self.b33 = nn.Conv2d(128, 128, (3, 3), padding=1)
#         self.b44 = nn.Conv2d(128, 128, (3, 3), padding=1)

#         self.mean1 = ASPP(256, 256)
# #         self.mean1 = ASPP(128, 256)

#         self.d1 = Decoder_Block([64, 256], 128)
#         self.d2 = Decoder_Block([32, 128], 64)
#         self.d3 = Decoder_Block([16, 64], 32)

#         self.aspp = ASPP(32, 16)
#         self.output = nn.Conv2d(16, 1, kernel_size=1, padding=0)
        
        self.c1 = Stem_Block(1, 16, stride=1)
        self.c2 = ResNet_Block(16, 32, stride=2)
        self.c3 = ResNet_Block(32, 64, stride=2)
        self.c4 = ResNet_Block(64, 128, stride=2)
        self.bc1 = nn.Conv2d(16,16, (1, 1), padding=0)
        self.bc2 = nn.Conv2d(32,32, (1, 1), padding=0)
        self.bc3 = nn.Conv2d(64,64, (1, 1), padding=0)
        self.bc4 = nn.Conv2d(128,128, (1, 1), padding=0)
        self.sb2 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.sb3 = nn.Conv2d(128, 128, (3, 3), padding=1)
        
        self.b1 = nn.Conv2d(16,32, (3, 3), padding=1)
        self.b2 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.ab2 = nn.Conv2d(64, 32, (1, 1))
        self.s2 = nn.Conv2d(32, 64, (1, 1), stride=2)
        self.b3 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.ab3 = nn.Conv2d(128, 64, (1, 1))
        self.s3 = nn.Conv2d(64, 128, (1, 1), stride=2)
        self.b4 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.ab4 = nn.Conv2d(256, 128, (1, 1))
        self.pred = nn.Conv2d(128, 128, (1, 1))
        self.batch16 = nn.BatchNorm2d(16)
        self.batch32 = nn.BatchNorm2d(32)
        self.batch64 = nn.BatchNorm2d(64)
        self.batch128 = nn.BatchNorm2d(128)
        self.rel = nn.ReLU()
        self.b11 = nn.Conv2d(32, 32, (3, 3), padding=1, stride=2)
        self.b22 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.b33 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.b44 = nn.Conv2d(128, 128, (3, 3), padding=1)

        self.mean1 = ASPP(256, 256)
#         self.mean1 = ASPP(128, 256)

        self.d1 = Decoder_Block([64, 256], 128)
        self.d2 = Decoder_Block([32, 128], 64)
        self.d3 = Decoder_Block([16, 64], 32)

        self.aspp = ASPP(32, 16)
        self.output = nn.Conv2d(16, 1, kernel_size=1, padding=0)
        

    def multiply(self, image, mask):
#             print(image.shape, mask.shape, "shapes")
#             image,mask = x
#             mask = K.expand_dims(mask, axis=-1) #could be K.stack([mask]*3, axis=-1) too
#             mask = torch.unsqueeze(mask, 0)
#             mask = K.stack([mask]*2, axis=3)
            return mask*image

    def forward(self, inputs):
        
        
#         n_filters = [32, 64, 128, 256, 512]
        c1 = self.c1(inputs)
        ###########
        
        bc1 = self.bc1(c1)
        
        ###########
        b1 = self.batch16(c1)
        b1 = self.rel(b1)
        b1 = self.b1(b1)
        b1 = self.batch32(b1)
        b1 = self.rel(b1)
        b1 = self.b11(b1)
        ###########
        c2 = self.c2(c1)
        ###########
        
        bc2 = self.bc2(c2)
        
        #########
        b2 = self.batch32(bc2)
        b2 = self.rel(b2)
        b2 = self.b2(b2)
        b2 = self.batch32(b2)
        b2 = self.rel(b2)
        b2 = self.b22(b2)
        
        a2 = torch.cat([b1, b2], axis=1)
        ab2 = self.ab2(a2)
        
        at2 = nn.Sigmoid()(ab2)
        s2 = torch.mul(b1, at2)
        s2 = self.s2(s2)
#         s2[s2>0.7] = 1
#         s2[s2<0.7] = 0
        ###########
        # #??????????
        s2 = self.batch64(s2)
        s2 = self.rel(s2)
        s2 = self.sb2(s2)
        s2 = self.batch64(s2)
        s2 = self.rel(s2)
        s2 = self.sb2(s2)
        # #??????????
#       ###########
        c3 = self.c3(c2)
        ###########
        
        bc3 = self.bc3(c3)
        
        ###########
        b3 = self.batch64(bc3)
        b3 = self.rel(b3)
        b3 = self.b3(b3)
        b3 = self.batch64(b3)
        b3 = self.rel(b3)
        b3 = self.b33(b3)

        a3 = torch.cat([s2, b3], axis=1)
        ab3 = self.ab3(a3) #????
        at3 = nn.Sigmoid()(ab3)
        s3 = torch.mul(s2, at3)
        s3 = self.s3(s3)
        
        ###########
        # #??????????
        s3 = self.batch128(s3)
        s3 = self.rel(s3)
        s3 = self.sb3(s3)
        s3 = self.batch128(s3)
        s3 = self.rel(s3)
        s3 = self.sb3(s3)
        # #??????????
        ###########
        
        c4 = self.c4(c3)
        
        ###########
        
        bc4 = self.bc4(c4)
        
        ###########
        
        b4 = self.b4(bc4)
        b4 = self.batch128(c4)
        b4 = self.rel(b4)
        b4 = self.b4(b4)
        b4 = self.batch128(b4)
        b4 = self.rel(b4)
        b4 = self.b44(b4)
        
        a4 = torch.cat([s3, b4], axis=1)
        ab4 = self.ab4(a4)
        
        at4 = nn.Sigmoid()(ab4)
        
        s4 = torch.mul(s3, at4)

        s_pred = self.pred(s4)
        ###########

#         mean1 = self.mean1(c4) #the og one
#         
        mid = torch.cat([s4, c4], axis=1)
        mean1 = self.mean1(mid)

        d1 = self.d1(c3, mean1)
        d2 = self.d2(c2, d1)
        d3 = self.d3(c1, d2)

        output = self.aspp(d3)
        output = self.output(output)
#         output = (output > 0.8).float()
#         print(output, "1")
#         output = torch.round(output)
#         output = torch.where(output > 0.2, 1, 0)
        output = nn.Sigmoid()(output)
        
#         output = torch.as_tensor((output - 0.5) < 0, dtype=torch.int32)
#         output = (output>0.5).float()
#         t = Variable(torch.Tensor([0.5]), requires_grad=True)  # threshold
#         output = output > t
#         output = (output > t).float()
#         output = torch.round(output)
#         output = torch.where(output > 0.6, 1, 0)
        
#         print(output, "2")
#         output = torch.Sigmoid(output)
#         output = (output > 0.8).float()
        
        ####
        
#         s_pred = torch.reshape(s_pred, shape=inputs.shape)
        s_pred = torch.reshape(s_pred, shape=(inputs.shape[0], 2, inputs.shape[2], inputs.shape[3]))
        
        ####

        return output, s_pred, c1, c2, c3, c4, d1, d2, d3, output, s2, s3, s4
#         return output, output
