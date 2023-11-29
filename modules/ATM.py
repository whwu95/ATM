import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as tr
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

try:
    from spatial_correlation_sampler import SpatialCorrelationSampler
except ImportError:
    print('[ImportError] Cannot import SpatialCorrelationSampler')


class DiffBlock(nn.Module):
    def __init__(self,
                 d_in,
                 d_hid=64,
                 num_segments=8,
                 context=5,
                 res=True,
                 downsample=True
                ):
        super(DiffBlock, self).__init__()
        self.num_segments = num_segments
        self.context = context
   
        # Resize spatial resolution to 14x14
        self.downsample = nn.Sequential(
            nn.Conv2d(d_in, d_hid, kernel_size=3, stride=2 if downsample else 1, padding=1, bias=False),
            nn.BatchNorm2d(d_hid),
            nn.ReLU(inplace=True)
        )

        self.diff2vis = DomainTransfer(
            d_in=d_hid,
            d_out=d_in,
            num_segments=num_segments,
            context = context,
            downsample=downsample
        )         

        self.res = res

    def forward(self, x):
        # bt c h w
        identity = x
        x = self.downsample(x)

        if self.context > 1:
            x = rearrange(x, '(b t) c h w -> b t c h w', t=self.num_segments)
            x_pre = repeat(x, 'b t c h w -> (b t z) c h w', z=self.context)
            x_post = F.pad(x, (0,0,0,0,0,0,self.context//2,self.context//2), 'constant', 0).unfold(1,self.context,1)
            x_post = rearrange(x_post, 'b t c h w z -> (b t z) c h w')     
        else:
            x_post = x
            bt, c, h, w = x.size()
            x = rearrange(x, '(b t) c h w -> b t c h w', t=self.num_segments)
            x_pre = torch.cat((x[:, 0].unsqueeze(1), x[:, :-1]), 1).view(-1, c, h, w)
            
        diff = x_post - x_pre # btz c h w
        x = rearrange(diff, '(b t z) c h w -> (b z) c t h w', t=self.num_segments, z=self.context)
        

        # (b l) 64 t h w
        out = self.diff2vis(x)


        if self.res:
            out = out + identity
        out = F.relu(out)
        
        return out



class IntegrationBlock(nn.Module):
    def __init__(self,
                 d_in,
                 d_hid,
                 num_segments=8,
                 context=5,
                 res=True,
                 downsample=True
                ):
        super(IntegrationBlock, self).__init__()
        self.num_segments = num_segments
        self.context = context  


        # Resize spatial resolution to 14x14
        self.downsample = nn.Sequential(
            nn.Conv2d(d_in, d_hid, kernel_size=3, stride=2 if downsample else 1, padding=1, bias=False),
            nn.BatchNorm2d(d_hid),
            nn.ReLU(inplace=True)
        )

        self.inte2vis = DomainTransfer(
            d_in=d_hid,
            d_out=d_in,
            num_segments=num_segments,
            context = context,
            downsample=downsample
        )    

        self.res = res

    def forward(self, x):
        # bt c h w
        identity = x
        x = self.downsample(x)

        if self.context > 1:
            x = rearrange(x, '(b t) c h w -> b t c h w', t=self.num_segments)
            x_pre = repeat(x, 'b t c h w -> (b t z) c h w', z=self.context)
            x_post = F.pad(x, (0,0,0,0,0,0,self.context//2,self.context//2), 'constant', 0).unfold(1,self.context,1)
            x_post = rearrange(x_post, 'b t c h w z -> (b t z) c h w')     
        else:
            x_post = x
            bt, c, h, w = x.size()
            x = rearrange(x, '(b t) c h w -> b t c h w', t=self.num_segments)
            x_pre = torch.cat((x[:, 0].unsqueeze(1), x[:, :-1]), 1).view(-1, c, h, w)
            
        inter = x_post + x_pre # btz c h w
        x = rearrange(inter, '(b t z) c h w -> (b z) c t h w', t=self.num_segments, z=self.context)
        

        # (b l) 64 t h w
        out = self.inte2vis(x)


        if self.res:
            out = out + identity
        out = F.relu(out)
        
        return out


class DivBlock(nn.Module):
    def __init__(self,
                 d_in,
                 d_hid=64,
                 num_segments=8,
                 context=5,
                 res=True,
                 downsample=False
                ):
        super(DivBlock, self).__init__()
        self.num_segments = num_segments
        self.context = context
   
        # Resize spatial resolution to 14x14
        self.downsample = nn.Sequential(
            nn.Conv2d(d_in, d_hid, kernel_size=3, stride=2 if downsample else 1, padding=1, bias=False),
            nn.BatchNorm2d(d_hid),
            nn.ReLU(inplace=True)
        )

        self.div2vis = DomainTransfer(
            d_in=d_hid,
            d_out=d_in,
            num_segments=num_segments,
            context = context,
            downsample=downsample
        )         

        self.res = res

    def forward(self, x):
        # bt c h w
        identity = x
        x = self.downsample(x)

        if self.context > 1:
            x = rearrange(x, '(b t) c h w -> b t c h w', t=self.num_segments)
            x_pre = repeat(x, 'b t c h w -> (b t z) c h w', z=self.context)
            x_post = F.pad(x, (0,0,0,0,0,0,self.context//2,self.context//2), 'constant', 0).unfold(1,self.context,1)
            x_post = rearrange(x_post, 'b t c h w z -> (b t z) c h w')     
        else:
            x_post = x
            bt, c, h, w = x.size()
            x = rearrange(x, '(b t) c h w -> b t c h w', t=self.num_segments)
            x_pre = torch.cat((x[:, 0].unsqueeze(1), x[:, :-1]), 1).view(-1, c, h, w)
            
        # div = torch.div(x_post, x_pre)  # doesn't work
        div = torch.log1p(torch.abs(x_post)) - torch.log1p(torch.abs(x_pre))  #  btz c h w 
        x = rearrange(div, '(b t z) c h w -> (b z) c t h w', t=self.num_segments, z=self.context)
        

        # (b l) 64 t h w
        out = self.div2vis(x)


        if self.res:
            out = out + identity
        out = F.relu(out)
        
        return out



class SimTransform(nn.Module):
    def __init__(self, d_in, d_hid, num_segments, window=(5,9,9), use_corr_sampler=False, downsample=True):
        super(SimTransform, self).__init__()
        self.num_segments = num_segments
        self.window = window
        assert window[1] == window[2]
        self.use_corr_sampler = use_corr_sampler
        
        if use_corr_sampler:
            try:
                self.correlation_sampler = SpatialCorrelationSampler(1, window[1], 1, 0, 1)
            except:
                print("[Warning] SpatialCorrelationSampler cannot be used.")
                self.use_corr_sampler = False
            
        # Resize spatial resolution to 14x14
        self.downsample = nn.Sequential(
            nn.Conv2d(d_in, d_hid, kernel_size=3, stride=2 if downsample else 1, padding=1, bias=False),
            nn.BatchNorm2d(d_hid),
            nn.ReLU(inplace=True)
        )


    def _L2normalize(self, x, d=1):
        eps = 1e-6
        norm = x ** 2
        norm = norm.sum(dim=d, keepdim=True) + eps
        norm = norm ** (0.5)
        
        return (x / norm)

    def _corr_abs_to_rel(self, corr, h, w):
        # Naive implementation of spatial correlation sampler
        max_d = self.window[1] // 2

        b,c,s = corr.size()        
        corr = corr.view(b,h,w,h,w)



        w_diag = tr.zeros((b,h,h,self.window[1],w),device='cuda') ###!!! cuda
        for i in range(max_d+1):
            if (i==0):
                w_corr_offset = tr.diagonal(corr, offset=0, dim1=2, dim2=4)       
                w_diag[:,:,:,max_d] = w_corr_offset
            else:
                w_corr_offset_pos = tr.diagonal(corr, offset=i, dim1=2, dim2=4) 
                w_corr_offset_pos = F.pad(w_corr_offset_pos, (i,0))
                w_diag[:,:,:,max_d-i] = w_corr_offset_pos
                w_corr_offset_neg = tr.diagonal(corr, offset=-i, dim1=2, dim2=4) 
                w_corr_offset_neg = F.pad(w_corr_offset_neg, (0,i))
                w_diag[:,:,:,max_d+i] = w_corr_offset_neg

        hw_diag = tr.zeros((b,self.window[1],w,self.window[1],h), device='cuda')  ###!!! cuda
        for i in range(max_d+1):
            if (i==0):
                h_corr_offset = tr.diagonal(w_diag, offset=0, dim1=1, dim2=2)
                hw_diag[:,:,:,max_d] = h_corr_offset
            else:
                h_corr_offset_pos = tr.diagonal(w_diag, offset=i, dim1=1, dim2=2) 
                h_corr_offset_pos = F.pad(h_corr_offset_pos, (i,0))
                hw_diag[:,:,:,max_d-i] = h_corr_offset_pos
                h_corr_offset_neg = tr.diagonal(w_diag, offset=-i,dim1=1, dim2=2) 
                h_corr_offset_neg = F.pad(h_corr_offset_neg, (0,i))     
                hw_diag[:,:,:,max_d+i] = h_corr_offset_neg 

        hw_diag = hw_diag.permute(0,3,1,4,2).contiguous()
        hw_diag = hw_diag.view(-1, self.window[1], self.window[1], h, w)      

        return hw_diag         

    def _correlation(self, feature1, feature2):
        feature1 = self._L2normalize(feature1) # btl, c, h, w
        feature2 = self._L2normalize(feature2) # btl, c, h, w

        if self.use_corr_sampler:
            corr = self.correlation_sampler(feature1, feature2)
        else:
            b, c, h, w = feature1.size()
            feature1 = rearrange(feature1, 'b c h w -> b c (h w)')
            feature2 = rearrange(feature2, 'b c h w -> b c (h w)')
            corr = tr.einsum('bcn,bcm->bnm',feature2, feature1)

            corr = self._corr_abs_to_rel(corr, h, w)

        return corr
        
    def forward(self, x):
        # resize spatial resolution to 14x14
        x = self.downsample(x)
        
        if self.window[0] > 1:
            x = rearrange(x, '(b t) c h w -> b t c h w', t=self.num_segments)
            x_pre = repeat(x, 'b t c h w -> (b t l) c h w', l=self.window[0])

            x_post = F.pad(x, (0,0,0,0,0,0,self.window[0]//2,self.window[0]//2), 'constant', 0).unfold(1,self.window[0],1)
            x_post = rearrange(x_post, 'b t c h w l -> (b t l) c h w')     
        else:
            x_post = x
            bt, c, h, w = x.size()
            x = rearrange(x, '(b t) c h w -> b t c h w', t=self.num_segments)
            x_pre = torch.cat((x[:, 0].unsqueeze(1), x[:, :-1]), 1).view(-1, c, h, w)
            
        sim = self._correlation(x_pre, x_post)
        sim = rearrange(sim, '(b t l) u v h w -> b t h w 1 l u v', t=self.num_segments, l=self.window[0])
        return sim

    
class Conv2D_UV2C(nn.Module):
    def __init__(self, num_segments, window=(5,9,9), chnls=(4,16,64,64)):
        super(Conv2D_UV2C, self).__init__()
        self.num_segments = num_segments
        self.window = window
        self.chnls = chnls
        
        self.conv0 = nn.Sequential(
            nn.Conv3d(1, chnls[0], kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(chnls[0]),
            nn.ReLU(inplace=True))
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(chnls[0], chnls[1], kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(chnls[1]),
            nn.ReLU(inplace=True))
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(chnls[1], chnls[2], kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(chnls[2]),
            nn.ReLU(inplace=True))
        
        self.conv3 = nn.Sequential(
            nn.Conv3d(chnls[2], chnls[3], kernel_size=(1,3,3), stride=(1,1,1), padding=(0,0,0), bias=False),
            nn.BatchNorm3d(chnls[3]),
            nn.ReLU(inplace=True))    
        
    def forward(self, x):
        b,t,h,w,_,l,u,v = x.size()
        x = rearrange(x, 'b t h w 1 l u v -> (b t h w) 1 l u v')
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = rearrange(x, '(b t h w) c l 1 1 -> (b l) c t h w', t=t, h=h, w=w)
        
        return x
    


class Conv1D_UV2C(nn.Module):
    def __init__(self, num_segments, window=(5,9,9), chnls=(64,64)):
        super(Conv1D_UV2C, self).__init__()
        self.num_segments = num_segments
        self.window = window
        self.chnls = chnls
        
        self.conv0 = nn.Sequential(
            nn.Conv3d(window[1] * window[2], chnls[0], kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(chnls[0]),
            nn.ReLU(inplace=True))
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(chnls[0], chnls[1], kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(chnls[1]),
            nn.ReLU(inplace=True))
        
    def forward(self, x):
        b,t,h,w,_,l,u,v = x.size()
        x = rearrange(x, 'b t h w 1 l u v -> (b l) (u v) t h w')
        x = self.conv0(x)
        x = self.conv1(x)   # bl 64 t h w        
        return x


class DomainTransfer(nn.Module):
    def __init__(self, d_in, d_out, num_segments, context=5, chnls=(64,64,64), downsample=True):
        super(DomainTransfer, self).__init__()
        self.num_segments = num_segments
        self.chnls = chnls
                
        self.conv1 = nn.Sequential(
            nn.Conv3d(d_in, chnls[0], kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(chnls[0]),
            nn.ReLU(inplace=True))
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(chnls[0], chnls[1], kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(chnls[1]),
            nn.ReLU(inplace=True))
        
        self.conv3_fuse = nn.Sequential(
            Rearrange('(b z) c t h w -> b (z c) t h w', z=context),
            nn.Conv3d(chnls[1]*context, chnls[2], kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(chnls[2]),
            nn.ReLU(inplace=True)
        )
        
        if downsample:
            self.upsample = nn.Sequential(
                nn.ConvTranspose3d(chnls[2], d_out, kernel_size=1, stride=(1,2,2), padding=(0,0,0), output_padding=(0,1,1), bias=False),
                # nn.BatchNorm3d(d_out),
                Rearrange('b c t h w -> (b t) c h w')
            )
        else:
            self.upsample = nn.Sequential(
                nn.Conv3d(chnls[2], d_out, kernel_size=1, stride=(1,1,1), padding=(0,0,0), bias=False),
                # nn.BatchNorm3d(d_out),
                Rearrange('b c t h w -> (b t) c h w')
            )            
        
        # Zero-initialize the last conv in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        nn.init.constant_(self.upsample[0].weight, 0)
        
        
    def forward(self, x):
        # (b l) 64 t h w
        x = self.conv1(x)
        x = self.conv2(x)
        # (b l) 64 t h w
        x = self.conv3_fuse(x)
        x = self.upsample(x)
        
        return x
    
    

    
class SimBlock(nn.Module):
    def __init__(self,
                 d_in,
                 d_hid,
                 num_segments=8,
                 window=(5,9,9),
                 res=True,
                 downsample=True
                ):
        super(SimBlock, self).__init__()
        
        self.sim = SimTransform(
            d_in,
            d_hid,
            num_segments=num_segments,
            window=window,
            downsample=downsample
        )

        # self.uv2c = Conv1D_UV2C(
        #     num_segments=num_segments,
        #     window = window,
        # )        

        self.uv2c = Conv2D_UV2C(
            num_segments=num_segments,
            window = window,
        )


        self.sim2vis = DomainTransfer(
            64,
            d_in,
            num_segments=num_segments,
            context = window[0],
            downsample=downsample
        )
        self.res = res
        
    def forward(self, x):
        identity = x
        out = self.sim(x)
        #b t h w 1 l u v
        

        if self.uv2c.conv1[0].weight.dtype is torch.float32:
            out = self.uv2c(out) ###!!!
        else:
            out = self.uv2c(out.half())
        
        out = self.sim2vis(out)
        
        if self.res:
            out = out + identity
        out = F.relu(out)
        
        return out


class ATMBlock(nn.Module):
    def __init__(self,
                 d_in,
                 d_hid,
                 num_segments=8,
                 window=(5,9,9),
                 downsample=False,
                 eva_trans=False,
                ):
        super(ATMBlock, self).__init__()
        self.eva_trans = eva_trans ###!!!
        self.n_segment = num_segments

        self.corrlayer = SimBlock(
            d_in=d_in,
            d_hid=d_hid,
            num_segments=num_segments,
            window=window,
            res=True,
            downsample=downsample
        )        
        self.difflayer = DiffBlock(
            d_in=d_in,
            d_hid=d_hid,
            num_segments=num_segments,
            context=window[0],
            res=True,
            downsample=downsample
        )
        #self.intelayer = IntegrationBlock(
        #    d_in=d_in,
        #    d_hid=d_hid,
        #    num_segments=num_segments,
        #    context=window[0],
        #    res=True
        #)

        #self.divlayer = DivBlock(
        #    d_in=d_in,
        #    d_hid=d_hid,
        #    num_segments=num_segments,
        #    context=window[0],
        #    res=True,
        #    downsample=downsample
        #)
        
        
    def forward(self, x):
        if self.eva_trans:
            x = x.transpose(0,1)
            
        # n+1 bt c
        xt = x[1:, :, :]
        n, bt, c = xt.size()
        b = bt // self.n_segment
        H = W = int(n ** 0.5)
        xt = xt.view(H, W, b, self.n_segment, c).permute(2, 3, 4, 0, 1).contiguous().view(bt, c, H, W)  # bt c h w
        # bt c h w
        
        xt = self.corrlayer(xt)
        xt = self.difflayer(xt)
        #xt = self.intelayer(xt)
        #xt = self.divlayer(xt)
        
        # bt c h w -> n+1 bt c
        xt = xt.view(bt, c, n).permute(2, 0, 1).contiguous()  # n bt c
        x = torch.cat([x[:1, :, :], xt], dim=0)  # n+1 bt c

        if self.eva_trans:
            x = x.transpose(0,1)
            
        return x


def make_ATM(net, n_segment, locations_list=[]):
    if hasattr(net, 'transformer'):
        for idx, block in enumerate(net.transformer.resblocks):
            if idx in locations_list:
                block.control_point_c = ATMBlock(
                    d_in=block.control_point_c.inplanes,
                    d_hid=64,
                    num_segments=n_segment,
                    window=(5,9,9),
                    downsample=False,
                    )
    elif hasattr(net, 'blocks'):
        for idx, block in enumerate(net.blocks):
            if idx in locations_list:
                block.control_point_c = ATMBlock(
                    d_in=block.control_point_c.inplanes,
                    d_hid=64,
                    num_segments=n_segment,
                    window=(5,9,9),
                    downsample=False,
                    eva_trans=True,
                    )


if __name__ == '__main__':
    x = torch.randn([8,64,28,28]).cpu()
    window=(5,9,9)
    corrlayer = SimBlock(
            d_in=64,
            d_hid=64,
            num_segments=8,
            window=window,
            res=True,
            downsample=False
        )   
    corrlayer.cpu()
    x = corrlayer(x)
    print(x.shape)
