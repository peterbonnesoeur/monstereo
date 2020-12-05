import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


import math
from einops import rearrange, repeat




class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000, kind = "add"):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.kind = kind
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        print("position", position)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.kind == "add":
            x = x + self.pe[:x.size(0), :]
        elif self.kind == "cat":
            print(self.pe)
            print("SIZES", x.size(), self.pe.size(), self.pe.permute(1,0, 2).repeat(x.size(0),1, 1).size())
            pe = self.pe.permute(1,0, 2).repeat(x.size(0),1, 1)
            print("SIZES", x.size(), pe[:, :x.size(1)].size())
            print(pe)
            x = torch.cat((x, pe[:, :x.size(1)]), dim = 2)
            print(x)
            return x
        else:
            return x
        return self.dropout(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            #print(dots.size(), mask.size())
            #mask = F.pad(mask.flatten(1), (1, 0), value = True)
            #print(mask.size(), mask.shape[-1], dots.shape[-1])
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask.unsqueeze(dim = 1), mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.gelu(input)
    
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):

        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs)
        #return self.norm(self.fn(x, **kwargs))
        #return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    
class Transformer(nn.Module):
    def __init__(self, ninp, nhid, nhead, nlayers, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layers2 = nn.ModuleList([])

        self.feed_forward = Residual(PreNorm(ninp, FeedForward(ninp, nlayers, dropout = dropout)))

        for i in range(nhid):
            if i == -10:
                self.layers.append(nn.ModuleList([
                    PreNorm(ninp, Attention(ninp, heads = nhead, dropout = dropout)),
                    PreNorm(ninp, FeedForward(ninp, nlayers, dropout = dropout))
                ]))
                
                self.layers2.append(nn.ModuleList([
                    PreNorm(ninp,  Attention(ninp, heads = nhead, dropout = dropout)),
                    PreNorm(ninp, FeedForward(ninp, nlayers, dropout = dropout))
                ]))
            else:
                self.layers.append(nn.ModuleList([
                    Residual(PreNorm(ninp, Attention(ninp, heads = nhead, dropout = dropout))),
                    Residual(PreNorm(ninp, FeedForward(ninp, nlayers, dropout = dropout)))
                ]))

                self.layers2.append(nn.ModuleList([
                    PreNorm(ninp, Residual( Attention(ninp, heads = nhead, dropout = dropout))),
                    PreNorm(ninp, Residual( FeedForward(ninp, nlayers, dropout = dropout)))
                ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)

        #x = self.feed_forward(x)
        return x
    


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)#.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        #print("PE SIZE", self.pe.size())
        x = x + self.pe[:,:x.size(1), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
                #      num_classes,  dim, heads, depth, mlp_dim   
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.2, max_len = 24):
        super(TransformerModel, self).__init__()
        
        
        self.ntoken = ntoken

        self.model_type = 'Transformer'
        #self.pos_encoder = PositionalEncoding(ninp, dropout, max_len = max_len)
        
   
        #encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        #self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        #self.encoder = nn.Embedding(ntoken, ninp)
        
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.patch_to_embedding = nn.Linear(ntoken, ninp)
        
        self.transformer = Transformer(ninp, nhid, nhead, nlayers, dropout)
        
        self.init_weights()

        
        
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def generate_mask_keypoints(self,kps):
        if len(kps.size())==3:
            mask = torch.tensor( kps[:,:,2]>0)
        else:
            mask = torch.tensor( kps[:,2]>0)

        return mask 

    def conf_remover(self, src):
        return src[:,:,:2]   

    def init_weights(self):
        initrange = 0.1
        #self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask= None):
        
        #print("INPUT",src.size())
        
        assert src.size(1)%3==0, "Wrong input, we need the flattened keypoints with the confidence"
        
        #print(src.size())
        #print(src)
        #print("SIZE INPUT",src.size())
        src = rearrange(src, 'b (n t) -> b n t', t = 3)
        #print("KPS SIZE",kps.size())
        #print(kps)
        mask = self.generate_mask_keypoints(src).clone().detach()
        #print("Mask size", mask.size())
        
        src = self.conf_remover(src)

        #src = self.patch_to_embedding(src)
        
        #print("ENCODED INPUT", src.size())
        
        #src = self.pos_encoder(src)
        
        #print("POS ENCODED INPUT", src.size())
        #print("Mask size", src_mask.size())
        
        #output = self.transformer_encoder(src, src_mask)
        if src_mask is not None:
            output = self.transformer(src, src_mask)
        else:
            output = self.transformer(src, mask)
        #print("EXIT TRANSFORMER SIZE", output.size())
        output = self.decoder(output)
        #print(output.size())
        
        return rearrange(output, 'b n t -> b (n t)')#output