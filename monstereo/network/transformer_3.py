import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import math
import numpy as np
from einops import rearrange, repeat


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000, kind = "add"):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.kind = kind
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.kind == "add":
            pe = self.pe.permute(1,0, 2).repeat(x.size(0),1, 1)
            x = x + pe[:, :x.size(1)]
        elif self.kind == "cat":
            pe = self.pe.permute(1,0, 2).repeat(x.size(0),1, 1)
            x = torch.cat((x, pe[:, :x.size(1)]), dim = 2)
            #print(x)
            return x
        else:
            return x
        return self.dropout(x)



class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads#int(embed_dim/d_k)
        #self.W_K = nn.Parameter(torch.randn((embed_dim, d_k)) * .01)
        #self.W_Q = nn.Parameter(torch.randn((embed_dim, d_k)) * .01)
        #self.W_V = nn.Parameter(torch.randn((embed_dim, d_k)) * .01)
        self.W_K = nn.Linear(embed_dim, embed_dim)
        self.W_Q = nn.Linear(embed_dim, embed_dim)
        self.W_V = nn.Linear(embed_dim, embed_dim)
        self.scaling = torch.Tensor(np.array(1 / np.sqrt(embed_dim)))
    def _weight_value(self, Q_vec, K_vec, V_vec, mask):
        
        b, n, _, h = *Q_vec.shape, self.num_heads
        #print(K_vec.size())
        K_vec = rearrange(K_vec, 'b n (h d) -> b h n d', h = h)
        Q_vec = rearrange(Q_vec, 'b n (h d) -> b h n d', h = h)
        V_vec = rearrange(V_vec, 'b n (h d) -> b h n d', h = h)

        q = self.W_Q(Q_vec)
        v = self.W_V(V_vec)
        k = self.W_K(K_vec)
        
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scaling
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            #print("Size analysis",dots.size(), mask.size())
            #mask = F.pad(mask.flatten(1), (1, 0), value = True)
            #print(mask.size(), mask.shape[-1], dots.shape[-1])
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            #print("MASK", mask)
            #print("Dots", dots)
            dots.masked_fill_(~mask.unsqueeze(dim = 1), mask_value)
            
            del mask
        
        attn = dots.softmax(dim=-1)
        #weight = self.scaling * Q @ K.transpose(0, 1)
        #exp_weight = torch.exp(torch.clamp(weight, max=25)) * mask.float()
        #attn = exp_weight / (torch.sum(exp_weight, dim=1, keepdim=True) + 1e-5)
        return attn, v
    def forward(self, Q_vec, K_vec, V_vec, mask):
        weight, v = self._weight_value(Q_vec, K_vec, V_vec, mask)
        out = torch.einsum('bhij,bhjd->bhid', weight, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return out #weight @ V
        
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, d_k, d_v, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.attns = nn.ModuleList([Attention(embed_dim,1) for _ in range(num_heads)])
        self.linear = nn.Linear(num_heads * embed_dim, embed_dim)
    def forward(self, Q_vec, K_vec, V_vec, mask):
        results = [attn(Q_vec, K_vec, V_vec, mask) for attn in self.attns]
        return self.linear(torch.cat(results, dim=2))

class InputEmbedding(nn.Module):
    def __init__(self, n_words, n_embed, n_token, max_len=100, kind = "add", confidence = True, scene_disp = False):
        super().__init__()
        self.n_embed = n_embed
        
        self.confidence = confidence
        self.scene_disp = scene_disp
        if kind == "cat":
            self.position_emb =  PositionalEncoding(2, kind = kind, max_len = max_len)
        else:
            self.position_emb =  PositionalEncoding(n_token, kind = kind, max_len = max_len)
    def forward(self, x, surround = None):

        
        assert x.size(-1)%3==0, "Wrong input, we need the flattened keypoints with the confidence"
        
        if self.scene_disp:
            out = rearrange(x, 'b x (n d) -> b x n d', d = 3)

            if not self.confidence:
                out= self.conf_remover(out)

            out = rearrange(out, 'b x n d -> b x (n d)')


            mask = torch.sum(out, dim = 2)!=0

            #print(mask)

            out =self.position_emb(out)

            return out, mask
    
        else:
            if len (x.size()) == 3:
                print("second loop")
            
                return x, torch.ones(x.size()[:-1])* True
            
            kps = rearrange(x, 'b (n t) -> b n t', t = 3)
            mask = self.generate_mask_keypoints(kps)

            if not self.confidence:
                out = self.conf_remover(kps, mask)

            out =self.position_emb(out)

            out[mask == False] = 0


            return out, mask 

    
    def conf_remover(self, src):
        if self.scene_disp:
            return src[:,:,:,:2]
        else:
            return src[:,:,:2]
    
    def generate_mask_keypoints(self,kps):
        if len(kps.size())==3:
            mask = torch.tensor( kps[:,:,2]>0)
        else:
            mask = torch.tensor( kps[:,2]>0)
        return mask


class EncoderLayer(nn.Module):
    def __init__(self,
                 embed_dim=64 * 8,
                 d_k=64, d_v=64,
                 num_heads=8):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, d_k, d_v, num_heads)
        self.ffn = nn.Linear(embed_dim, embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
    def forward(self, x, mask):
        attn_out = self.attn(x, x, x, mask)
        ln1_out = self.ln1(attn_out + x)
        ffn_out = nn.functional.gelu(self.ffn(ln1_out)) #gelu
        return self.ln2(ffn_out + ln1_out)
    
class Encoder(nn.Module):
    def __init__(self, n_words,n_token = 2, embed_dim=512, n_layers=3, num_heads = 1,kind = "add",
                 confidence = True, scene_disp = False):
        super().__init__()
        self.input_enc = InputEmbedding(n_words, embed_dim, n_token, kind = kind, 
                                        confidence = confidence, scene_disp = scene_disp)
        self.layers = nn.ModuleList([EncoderLayer(embed_dim =embed_dim, d_k =int(embed_dim/num_heads),
                                                  d_v =int(embed_dim/num_heads) , num_heads = num_heads) for _ in range(n_layers)])
    def forward(self, x, surround = None,mask = None):
        
        out, mask = self.input_enc(x, surround = surround)

        for i, layer in enumerate(self.layers):
            if i> 0:
                pass

            if mask is not None:
                pass
            out = layer(out, mask )
        return out


class DecoderLayer(nn.Module):
    def __init__(self,
                 embed_dim=64 * 8,
                 d_k=64, d_v=64,
                 num_heads=8):
        super().__init__()
        self.attn1 = MultiHeadAttention(embed_dim, d_k, d_v, num_heads)
        self.attn2 = MultiHeadAttention(embed_dim, d_k, d_v, num_heads)
        self.ffn = nn.Linear(embed_dim, embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)
    def forward(self, x, encoder_output, mask=None):
        attn1_out = self.attn1(x, x, x, mask)
        ln1_out = self.ln1(attn1_out + x)
        attn2_out = self.attn2(ln1_out,
                               encoder_output,
                               encoder_output,
                               mask)
        ln2_out = self.ln2(attn2_out + ln1_out)
        ffn_out = nn.functional.gelu(self.ffn(ln2_out))
        return self.ln3(ffn_out + ln2_out)
    
class Decoder(nn.Module):
    def __init__(self, n_words,n_token = 2, embed_dim=512, n_layers=3, num_heads = 1, kind = "add", 
                 confidence = True, scene_disp = False):
        super().__init__()
        self.input_enc = InputEmbedding(n_words, embed_dim, n_token, kind = kind, 
                                        confidence = confidence, scene_disp = scene_disp)
        self.layers = nn.ModuleList([DecoderLayer(embed_dim =embed_dim, d_k =int(embed_dim/num_heads),
                                                  d_v =int(embed_dim/num_heads) , num_heads = num_heads)for _ in range(n_layers)])
    def forward(self, x,encoder_output, surround = None, mask = None):
        
        out, mask = self.input_enc(x, surround = surround)
        self.mask = mask
        for i, layer in enumerate(self.layers):
            #print("decoder number :", i)
            if i>0:
                #mask = None 
                pass
            #print("MASK is :", mask)
            #if mask is not None:
                #print(mask.size())

            out = layer(out, encoder_output,  mask)
        return out
    
    def get_mask(self):
        return self.mask
    
    def decoder_mask(self, mask):
        n = len(mask)
        #print("BEGIN MASK",mask)
        look_ahead = np.tril(np.ones(n))
        return mask * look_ahead
        
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class Transformer(nn.Module):
    def __init__(self, n_base_words, n_target_words, n_token,  n_layers = 3, kind = "add",embed_dim=512, 
                 num_heads = 1, confidence = True, scene_disp = False):
        super().__init__()
        self.encoder = Encoder(n_base_words, n_token, embed_dim=embed_dim, kind = kind, num_heads = num_heads, 
                               n_layers = n_layers, confidence = confidence, scene_disp = scene_disp)
        self.decoder = Decoder(n_target_words, n_token, embed_dim=embed_dim, kind = kind, num_heads= num_heads, 
                               n_layers = n_layers, confidence = confidence, scene_disp = scene_disp)
        self.linear = nn.Linear(embed_dim, n_target_words)
        
        self.scene_disp = scene_disp
    def forward(self,
                encoder_input,
                decoder_input,
                surround = None, 
                encoder_mask=None,
                decoder_mask = None):
        
        encoder_out = self.encoder(encoder_input, surround = surround, mask = encoder_mask)
        decoder_out = self.decoder(decoder_input, encoder_out, surround = surround, mask = decoder_mask)
        mask = self.decoder.get_mask()
        if self.scene_disp:            
            return rearrange(self.linear(decoder_out), 'b n t -> (b n) t')
        else:
            return rearrange(self.linear(decoder_out), 'b n t -> b (n t)')

    