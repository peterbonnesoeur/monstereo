import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import math
import numpy as np
from einops import rearrange, repeat


class InputEmbedding(nn.Module):
    def __init__(self, n_embed, n_token, max_len=100, kind = "add", confidence = True, 
                scene_disp = False, reordering = False):
        super().__init__()
        self.n_embed = n_embed
        self.n_token = n_token
        
        self.reordering = reordering
        self.confidence = confidence
        self.scene_disp = scene_disp
        if kind == "cat":
            self.position_emb =  PositionalEncoding(2, kind = kind, max_len = max_len)
        elif kind == "num":
            self.position_emb =  PositionalEncoding(1, kind = kind, max_len = max_len)
        elif kind == "add":
            self.position_emb =  PositionalEncoding(n_token, kind = kind, max_len = max_len)
        else:
            self.position_emb =  PositionalEncoding(2, kind = kind, max_len = max_len)

        #? The padding can be the mean over the whole dataset. 
        self.pad = torch.load('docs/tensor.pt')

    def forward(self, x, surround = None, mask_off = False):


        if self.scene_disp:
            #? The padded scenes are arrays of 0
            mask = torch.sum(x, dim = -1)!=0
            condition = (mask == False)

            pad = self.pad

            if x.size(-1)>10 and x.size(-1)<100:
                out = rearrange(x, 'b x (n d) -> b x n d', d = 3)
                pad = rearrange(pad, 'h (n d) -> h n d', d = 3)
                if not self.confidence:
                    #? remove the confidence term of the inputs (and the pads)
                    out = self.conf_remover(out)
                    pad = pad[:,:, :-1]
                out = rearrange(out, 'b x n d -> b x (n d)')
                pad = rearrange(pad, 'h n d -> h (n d)')
            else:
                out = x



            if not mask_off and x.size(-1)<100:                
                out[condition] = 0
                #? the padder function will change the kind of pads to improve the results (This is mostly due to the LayerNorm step that
                #?  is having a huge influence on the end result)
                out = self.padder(out,pad)
                
            #? Add the positional embedding at the end of the array (yes, the positional embedding is happening after the padder)
            #? this is a design choice, not an error
            out = self.position_emb(out)

            return out, mask
    
        else:
        
            out = rearrange(x, 'b (n d) -> b n d', d = 3)
            pad = rearrange(self.pad, 'h (n d) -> h n d', d = 3)[0]

            mask = self.generate_mask_keypoints(out)
            condition = (mask == False)


            if not self.confidence:
                out = self.conf_remover(out)
                pad = pad[:, :-1]

            
            #? Add the positionnal embedding at the end of the array
            out =self.position_emb(out)

            pad = torch.cat((torch.mean(pad, dim = 0), torch.zeros(out.size(-1)- pad.size(-1) )), dim = -1).to(out.device)
            if not mask_off:
                #out[condition] = repeat(pad, 'w -> h w', h = out[condition].size(0))
                out[condition] = 0

            return out, mask 

    
    def conf_remover(self, src):
        if self.scene_disp:
            return src[:,:,:,:-1]
        else:
            return src[:,:,:-1]


    def padder(self, inputs, pad = None):
        EOS = repeat(torch.tensor([5]),'h -> h w', w = inputs.size(-1) )

        for index in range(len(inputs)):
            masks = torch.sum(inputs[index], dim = -1) != 0
            for i, mask in enumerate(masks):
                if mask ==False:
                    #? Add the EOS pad at the end of each valid value for each scene
                    inputs[index, (i):,: ]=repeat(pad, "h w -> (h c) w ", c = (masks.size(0)-i))
                    #inputs[index, i, :] = EOS
                    break

        return inputs
    
    def generate_mask_keypoints(self,kps):
        if len(kps.size())==3:
            mask = (kps[:,:,-1]>0).clone().detach()
        else:
            mask = ( kps[:,-1]>0).clone().detach()
        return mask


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
            x = torch.cat((x, pe[:, :x.size(1)]), dim = -1)

        elif self.kind == "num":
            #? We enumerate the number of instances / scenes
            pe = torch.arange(0,x.size(1)).repeat(x.size(0),1,1).permute(0,2,1).to(x.device)
            x = torch.cat((x, pe), dim = -1)

        return x



class Attention(nn.Module):


    def __init__(self, embed_dim, d_attention,num_heads, embed_dim2 = None):

        super().__init__()
        self.num_heads = num_heads

        if embed_dim2 is None:
            embed_dim2 = embed_dim

        #? To represent the Matrix multiplications, we are using fully conected layers with no biasses
        self.W_K = nn.Linear(embed_dim2, d_attention, bias = False)
        self.W_Q = nn.Linear(embed_dim, d_attention, bias = False)
        self.W_V = nn.Linear(embed_dim2, d_attention, bias = False)      

        self.scaling = torch.Tensor(np.array(1 / np.sqrt(d_attention)))

    def weight_value(self, Q_vec, K_vec, V_vec, mask):
        
        b, n, _, h = *Q_vec.shape, self.num_heads
        K_vec = rearrange(K_vec, 'b n (h d) -> b h n d', h = h)
        V_vec = rearrange(V_vec, 'b n (h d) -> b h n d', h = h)
        #? In the case of the decoder, the query comes form the decoder's self attention layer
        Q_vec = rearrange(Q_vec, 'b n (h d) -> b h n d', h = h) 
  
        q = self.W_Q(Q_vec)
        v = self.W_V(V_vec)
        k = self.W_K(K_vec)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scaling
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            #? Remove the contribution of the occluded/padded elements during the training phase
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask.unsqueeze(dim = 1), mask_value)
            del mask
        
        
        attn = dots.softmax(dim=-1)
        return attn, v

    def forward(self, Q_vec, K_vec, V_vec, mask):
        weight, v = self.weight_value(Q_vec, K_vec, V_vec, mask)
        out = torch.einsum('bhij,bhjd->bhid', weight, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return out
        
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, d_attention, num_heads, embed_dim2 = None):
        super().__init__()
        self.num_heads = num_heads
        #? In our current implementation, the multi-headed attention mechanism is handeld by the usage 
        # of severall single headed attention mechanisms
        self.attns = nn.ModuleList([Attention(embed_dim, d_attention,1, embed_dim2) for _ in range(num_heads)])

        #? To represent the Matrix multiplications, we are using fully connected layers with no biasses
        self.linear = nn.Linear(num_heads * d_attention, embed_dim, bias = False)
    def forward(self, Q_vec, K_vec, V_vec, mask):
        results = [attn(Q_vec, K_vec, V_vec, mask) for attn in self.attns]
        return self.linear(torch.cat(results, dim=-1))

    
class Encoder(nn.Module):
    def __init__(self, n_words,n_token = 2, embed_dim=3, n_layers=3, d_attention = None, num_heads = 1,kind = "add",
                 confidence = True, scene_disp = False, reordering = False, refining = False):
        super().__init__()
        if d_attention is None:
            d_attention = embed_dim

        self.refining = refining
        self.input_enc = InputEmbedding(embed_dim, n_token, kind = kind, 
                                        confidence = confidence, scene_disp = scene_disp, 
                                        reordering = reordering)
        self.layers = nn.ModuleList([EncoderLayer(embed_dim =embed_dim, d_attention =d_attention, n_words = n_words,
                                                num_heads = num_heads) for _ in range(n_layers)])
        if refining:
            self.layers = nn.ModuleList([EncoderLayer_refine(embed_dim =embed_dim, d_attention =d_attention,n_words = n_words,
                                                num_heads = num_heads) for _ in range(n_layers)])
        
    def forward(self, x, surround = None,mask = None):
        
        mask_off = False
        out, mask = self.input_enc(x, surround = surround, mask_off = mask_off)
        self.mask = mask.detach()

        for i, layer in enumerate(self.layers):
            out = layer(out, self.mask)
        return out

    def get_mask(self):
        return self.mask

class EncoderLayer(nn.Module):
    def __init__(self,
                 embed_dim=3,
                 d_attention = 3,
                 n_words = 20,
                 num_heads=1):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, d_attention, num_heads)
        self.ffn = nn.Linear(embed_dim, embed_dim)

        self.ln1 = nn.LayerNorm([n_words,embed_dim])
        self.ln2 = nn.LayerNorm([n_words,embed_dim])

    def forward(self, x, mask):


        attn_out = self.attn(x, x, x, mask)

        ln1_out = self.ln1(attn_out + x)
        
        ffn_out = nn.functional.relu(self.ffn(ln1_out))

        return self.ln2(ffn_out+ln1_out)

class EncoderLayer_refine(nn.Module):
    #! Experimental, not used
    def __init__(self,
                 embed_dim=3,
                 d_attention = 3,
                 n_words = 20,
                 num_heads=1):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, d_attention, num_heads)
        self.ffn = nn.Linear(embed_dim, embed_dim)

        self.ln1 = nn.LayerNorm([n_words,embed_dim])
        self.ln2 = nn.LayerNorm([n_words,embed_dim])
    def forward(self, x, mask):

        attn_out = self.attn(x, x, x, mask)

        ln1_out = self.ln1(attn_out + x)
        
        ffn_out = nn.functional.relu(self.ffn(ln1_out))

        return self.ln2(ffn_out+ln1_out)


    
class Decoder(nn.Module):
    def __init__(self, n_words,n_token = 2, embed_dim=3, n_layers=3, d_attention = None, num_heads = 1, kind = "add", 
                 confidence = True, scene_disp = False, reordering = False, refining = False, embed_dim2 = None):

        super().__init__()

        if d_attention is None:
            d_attention = embed_dim
        self.input_enc = InputEmbedding(embed_dim, n_token, kind = kind, 
                                        confidence = confidence, scene_disp = scene_disp, reordering = reordering)
        self.layers = nn.ModuleList([DecoderLayer(embed_dim =embed_dim, d_attention =d_attention , n_words = n_words,
                                    num_heads = num_heads)for _ in range(n_layers)])
        if refining:
            self.layers = nn.ModuleList([DecoderLayer_refine(embed_dim =embed_dim, d_attention =d_attention, n_words = n_words,
                                        num_heads = num_heads, embed_dim2 = embed_dim2)for _ in range(n_layers)])

    def forward(self, x,encoder_output, surround = None, mask = None):

        
        if mask is not None:
           out, _ = self.input_enc(x, surround = surround, mask_off = False)
        else:
            out, mask = self.input_enc(x, surround = surround, mask_off = False)

        self.mask = mask.detach()
        for i, layer in enumerate(self.layers):
            out = layer(out, encoder_output,  mask)
        return out
    
    def get_mask(self):
        return self.mask

class DecoderLayer(nn.Module):
    def __init__(self,
                 embed_dim=3,
                 d_attention =3,
                 n_words = 20,
                 num_heads= 1,):
        super().__init__()

        self.attn1 = MultiHeadAttention(embed_dim, d_attention, num_heads)
        self.attn2 = MultiHeadAttention(embed_dim, d_attention, num_heads)
        self.ffn = nn.Linear(embed_dim, embed_dim)


        self.ln1 = nn.LayerNorm([n_words,embed_dim])
        self.ln2 = nn.LayerNorm([n_words,embed_dim])
        self.ln3 = nn.LayerNorm([n_words,embed_dim])

    def forward(self, x, encoder_output, mask=None):
        

        attn1_out = self.attn1(x, x, x, mask)
        ln1_out = self.ln1(attn1_out + x)

        attn2_out = self.attn2(ln1_out,ln1_out,ln1_out,mask)

        ln2_out = self.ln2(attn2_out + ln1_out)

        ffn_out = nn.functional.relu(self.ffn(ln2_out))

        return self.ln3(ffn_out + ln2_out)


class DecoderLayer_refine(nn.Module):

    #! Experimental, not used
    def __init__(self,
                 embed_dim=3,

                 d_attention = 3,
                 n_words = 20,

                 num_heads= 1,
                 embed_dim2 = None):
        super().__init__()
        self.attn1 = MultiHeadAttention(embed_dim, d_attention, num_heads)
        self.attn2 = MultiHeadAttention(embed_dim, d_attention, num_heads, embed_dim2)
        self.ffn = nn.Linear(embed_dim, embed_dim)
        self.ffn2 = nn.Linear(embed_dim, embed_dim)

        self.ln1 = nn.LayerNorm([n_words,embed_dim])
        self.ln2 = nn.LayerNorm([n_words,embed_dim])
        self.ln3 = nn.LayerNorm([n_words,embed_dim])
    def forward(self, x, encoder_output, mask=None):
        
        ln1_out = self.ln1(x)
        #return ln1_out
        attn1_out = self.attn1(ln1_out, ln1_out, ln1_out, mask)

        out = attn1_out + x
        
        ffn_out = out + self.ffn2(nn.functional.relu(self.ffn(self.ln2(out))))

        return ffn_out
       



class Transformer(nn.Module):
    #! Reformating of the transformer name is necessary
    #! Adapting the recall for the evaluation is also a necessity
    def __init__(self, n_base_dim, n_target_dim, n_words, n_layers = 3, kind = "add",embed_dim=512, d_attention = None, 
                 num_heads = 1, confidence = True, scene_disp = False, reordering = False, embed_dim2 = None, d_attention2 = None, n_layers2 = None, num_heads2 = None):

        super().__init__()

        if d_attention is not None:
            assert d_attention > 0, "d_attention is negative or equal to 0"

        #? Mixed case where the encoder's output is of a different size compared to the decoder's output -> this is quite experimental
        #? But it is one of the fields that can be explored: using different data for the encoder and decoder and extract the relationships presents in those 
        #? 2 sets individually
        if embed_dim2 == None:
            embed_dim2 = embed_dim
            d_attention2 = d_attention
            n_layers2 = n_layers
            num_heads2 = num_heads


        self.embed_dim2 = embed_dim2
        self.embed_dim = embed_dim

        self.n_target_dim = n_target_dim
        self.encoder = Encoder(n_words, n_base_dim, embed_dim=embed_dim2, kind = kind, num_heads = num_heads2, d_attention = d_attention2,
                               n_layers = n_layers2, confidence = confidence, scene_disp = scene_disp, reordering = reordering, refining = (embed_dim2!= embed_dim))


        self.decoder = Decoder(n_words, n_target_dim, embed_dim=embed_dim, kind = kind, num_heads= num_heads, d_attention = d_attention,
                               n_layers = n_layers, confidence = confidence, scene_disp = scene_disp, reordering = reordering, refining = (embed_dim2!= embed_dim), embed_dim2= embed_dim2)
    
        #? Flag in case, you do not want an higher number of outputs than the embeding, this will not use an additional layer
        if embed_dim == n_target_dim:
            self.end_layer = False
        else:
            self.end_layer = True

        self.linear = nn.Linear(embed_dim, n_target_dim)
               
        self.scene_disp = scene_disp
    def forward(self,
                encoder_input,
                decoder_input,
                surround = None, 
                encoder_mask=None,
                decoder_mask = None):
        
        #? get the output of the encoder mechanism
        #encoder_out = self.encoder(encoder_input, surround = surround, mask = encoder_mask)
        encoder_out, decoder_mask = self.decoder.input_enc(encoder_input, surround = surround, mask_off = False)
        if self.embed_dim != self.embed_dim2 and decoder_mask is None:
            decoder_mask = self.encoder.get_mask()
            
        decoder_out = self.decoder(decoder_input, encoder_out, surround = surround, mask = decoder_mask)
        mask = self.decoder.get_mask()


        if self.scene_disp:    
            if self.end_layer:
                return rearrange(self.linear(decoder_out), 'b n t -> (b n) t')
            else:
                return rearrange(decoder_out, 'b n t -> (b n) t')
        else:
            if self.end_layer:
                return rearrange(self.linear(decoder_out), 'b n t -> b (n t)')
            else:
                return rearrange(decoder_out, 'b n t -> b (n t)')


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_hid, d_inner_hid, dropout=0.1):

        super().__init__()
        self.w_1 = nn.Linear(d_hid, d_inner_hid)
        self.w_2 = nn.Linear(d_inner_hid ,d_hid)
        self.layer_norm = nn.LayerNorm(d_hid)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        output = self.w_1(x)
        output = nn.functional.relu(output)
        output = self.w_2(output)
        output = self.dropout(output)
        output = output + x
        return self.layer_norm(output)