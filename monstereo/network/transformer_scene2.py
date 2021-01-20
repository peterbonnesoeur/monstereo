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
            print("Additional embedding",self.pe.size())
            pe = self.pe.permute(1,0, 2).repeat(x.size(0),1, 1)

            print(x.size(), pe.size())
            x = x + pe[:, :x.size(1)]
        elif self.kind == "cat":
            pe = self.pe.permute(1,0, 2).repeat(x.size(0),1, 1)
            x = torch.cat((x, pe[:, :x.size(1)]), dim = 2)

        elif self.kind == "num":
            #We enumerate the number of instances / scenes
            pe = torch.arange(0,x.size(1)).repeat(x.size(0),1,1).permute(0,2,1).to(x.device)
            x = torch.cat((x, pe), dim = 2)

        return x#self.dropout(x)



class Attention(nn.Module):

    #TODO: For a cleaner code, remove the head contribution from the rearranging part -> no perf change but just easier to understand
    def __init__(self, embed_dim, d_attention,num_heads, embed_dim2 =None):
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

        #? To represent the Matrix multiplications, we are using fully conected layers with no biasses
        self.linear = nn.Linear(num_heads * d_attention, embed_dim, bias = False)
    def forward(self, Q_vec, K_vec, V_vec, mask):
        results = [attn(Q_vec, K_vec, V_vec, mask) for attn in self.attns]
        return self.linear(torch.cat(results, dim=2))

class InputEmbedding(nn.Module):
    def __init__(self, n_words, n_embed, n_token, max_len=100, kind = "add", confidence = True, 
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

    def forward(self, x, surround = None, mask_off = False):

        if  x.size(-1)>10: #? In case we use a refinement of the outputs
            if x.size(-1)<100:
                assert  x.size(-1)%3==0, "Wrong input, we need the flattened keypoints with the confidence"
        
        if self.scene_disp:

            #? The padded scenes are arrays of 0
            mask = torch.sum(x, dim = 2)!=0
            condition = (mask == False)

            if self.reordering:
                out = self.reorderer(out,condition)

            if x.size(-1)>10 and x.size(-1)<100:

                out = rearrange(x, 'b x (n d) -> b x n d', d = 3)
                if not self.confidence:
                    out= self.conf_remover(out)
                out = rearrange(out, 'b x n d -> b x (n d)')
            else:
                out = x

            #? Add the positionnal embedding at the end of the array
            out =self.position_emb(x)
            if not mask_off and x.size(-1)<100:
                out[condition] = 0

            return out, mask
    
        else:
            
            out = rearrange(x, 'b (n d) -> b n d', d = 3)
            mask = self.generate_mask_keypoints(out)
            condition = (mask == False)


            if self.reordering:
                out = self.reorderer(out,condition)

            if not self.confidence:
                out = self.conf_remover(out)

            #? Add the positionnal embedding at the end of the array
            out =self.position_emb(out)
            if not mask_off:
                out[condition] = 0

            return out, mask 

    
    def conf_remover(self, src):
        if self.scene_disp:
            return src[:,:,:,:-1]
        else:
            return src[:,:,:-1]

    #TODO: remove this option as well as env
    def reorderer(self,out, condition):
        
        if out.size(-1)<5:
            #? Reorder the array in the case of the transformer of the keypoints
            #? This functions put each keypoint instances one after the other (the one with a positive confidence) 
            #? and pad the rest of the array
            indexes = torch.arange(0, out.size(1)).repeat(out.size(0), 1,1).permute(0,2,1).to(out.device)
            test = torch.cat((out, indexes), dim = 2)
            test[condition] =10000
            _, indices = torch.sort(test[:,:,-1], dim = 1, descending = False)
                

            final = test
            for index in range(final.size(0)):
                final[index] = test[index,indices[index]]
                
            final = final[:,:,:-1]
            mask2 = (final[:,:] ==10000).to(out.device)
            final[mask2] = 0
            
            return final
        else:
            final = out
            for index in range(final.size(0)):
                indices = torch.arange(0, out.size(1)).to(final.device)
                final[index] = out[index, indices]
        
        return final
    
    def generate_mask_keypoints(self,kps):
        if len(kps.size())==3:
            mask = torch.tensor( kps[:,:,-1]>0)
        else:
            mask = torch.tensor( kps[:,-1]>0)
        return mask

class EncoderLayer_refine(nn.Module):
    def __init__(self,
                 embed_dim=3,
                 d_attention = 3,
                 num_heads=1):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, d_attention, num_heads)
        self.ffn = nn.Linear(embed_dim, embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        #self.ln1 = torch.nn.InstanceNorm1d(embed_dim)
        #self.ln2 = torch.nn.InstanceNorm1d(embed_dim)
    def forward(self, x, mask):

        
        """inp = self.ln1(x)
        attn_out = self.attn(inp, inp, inp, mask)

        ln1_out = x + attn_out     

        ffn_out = nn.functional.relu(self.ffn(self.ln2(ln1_out)))
        return ffn_out + ln1_out"""



        attn_out = self.attn(x, x, x, mask)

        ln1_out = self.ln1(attn_out + x)
        #attn_out = self.attn2(ln1_out, ln1_out, x, mask)
        
        ffn_out = nn.functional.relu(self.ffn(ln1_out))

        return self.ln2(ffn_out+ln1_out)

class EncoderLayer(nn.Module):
    def __init__(self,
                 embed_dim=3,
                 d_attention = 3,
                 num_heads=1):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, d_attention, num_heads)
        self.ffn = nn.Linear(embed_dim, embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
    def forward(self, x, mask):
        attn_out = self.attn(x, x, x, mask)

        ln1_out = self.ln1(attn_out + x)
        
        ffn_out = nn.functional.relu(self.ffn(ln1_out))
        return self.ln2(ffn_out+ln1_out)

    
class Encoder(nn.Module):
    def __init__(self, n_words,n_token = 2, embed_dim=3, n_layers=3, d_attention = None, num_heads = 1,kind = "add",
                 confidence = True, scene_disp = False, reordering = False, refining = False):
        super().__init__()
        if d_attention is None:
            d_attention = embed_dim
        self.input_enc = InputEmbedding(n_words, embed_dim, n_token, kind = kind, 
                                        confidence = confidence, scene_disp = scene_disp, 
                                        reordering = reordering)
        self.layers = nn.ModuleList([EncoderLayer(embed_dim =embed_dim, d_attention =d_attention,
                                                num_heads = num_heads) for _ in range(n_layers)])

        if refining:
            self.layers = nn.ModuleList([EncoderLayer_refine(embed_dim =embed_dim, d_attention =d_attention,
                                                num_heads = num_heads) for _ in range(n_layers)])
    def forward(self, x, surround = None,mask = None):
        
        out, mask = self.input_enc(x, surround = surround, mask_off = False)
        self.mask = mask.detach()

        for i, layer in enumerate(self.layers):
            out = layer(out, self.mask)
        return out

    def get_mask(self):
        return self.mask


class DecoderLayer(nn.Module):
    def __init__(self,
                 embed_dim=3,
                 d_attention =3,
                 num_heads= 1):
        super().__init__()
        self.attn1 = MultiHeadAttention(embed_dim, d_attention, num_heads)
        self.attn2 = MultiHeadAttention(embed_dim, d_attention, num_heads)
        self.ffn = nn.Linear(embed_dim, embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)


        #self.ln1 = torch.nn.InstanceNorm1d(embed_dim)
        #self.ln2 = torch.nn.InstanceNorm1d(embed_dim)
        #self.ln3 = torch.nn.InstanceNorm1d(embed_dim)
        #self.ln1 = torch.nn.BatchNorm1d(24)
        #self.ln2 = torch.nn.BatchNorm1d(24)
        #self.ln3 = torch.nn.BatchNorm1d(24)
    def forward(self, x, encoder_output, mask=None):
        attn1_out = self.attn1(x, x, x, mask)
        ln1_out = self.ln1(attn1_out + x)
        #attn2_out = self.attn2(ln1_out,encoder_output,encoder_output,mask)
        attn2_out = self.attn2(ln1_out,ln1_out,ln1_out,mask)
        ln2_out = self.ln2(attn2_out + ln1_out)
        ffn_out = nn.functional.relu(self.ffn(ln2_out))
        return self.ln3(ffn_out + ln2_out)

 

class DecoderLayer_refine(nn.Module):
    def __init__(self,
                 embed_dim=3,
                 d_attention =3,
                 num_heads= 1,
                 embed_dim2 = None):
        super().__init__()
        self.attn1 = MultiHeadAttention(embed_dim, d_attention, num_heads)
        #self.attn2 = MultiHeadAttention(embed_dim, d_attention, num_heads, embed_dim2)
        self.attn2 = MultiHeadAttention(embed_dim, d_attention, num_heads, embed_dim)
        self.pfn = PositionwiseFeedForward(embed_dim, embed_dim*2)
        self.ffn = nn.Linear(embed_dim, embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)

        #self.ln1 = torch.nn.InstanceNorm1d(embed_dim)
        #self.ln2 = torch.nn.InstanceNorm1d(embed_dim)
        #self.ln3 = torch.nn.InstanceNorm1d(embed_dim)



    def forward(self, x, encoder_output, mask=None):
        #attn1_out = self.attn1(x, x, x, mask)
        #ln1_out = attn1_out + x
        #ln1_out = self.ln1(attn1_out + x)

        """inp = self.ln1(x)
        attn1_out = self.attn1(inp, inp, inp, mask)
        ln1_out = attn1_out + x

        attn2_out = self.attn2(self.ln2(ln1_out),self.ln2(encoder_output),self.ln2(encoder_output),mask)


        #attn2_out = self.attn2(self.ln2(ln1_out),self.ln2(ln1_out),self.ln2(ln1_out),mask)
        #attn2_out = self.attn2(ln1_out,x,x,mask)
        ln2_out = attn2_out + ln1_out


        ffn_out = nn.functional.relu(self.ffn(self.ln3(ln2_out)))
        return ffn_out + ln2_out"""


        attn1_out = self.attn1(x, x, x, mask)

        ln1_out = self.ln1(attn1_out) + x

        #attn2_out = self.attn2(ln1_out,encoder_output,encoder_output,mask)

        attn2_out = self.attn2(ln1_out,ln1_out,ln1_out,mask)

        ln2_out = self.ln2(attn2_out) + ln1_out

        #return self.pfn(ln2_out)

        ffn_out = nn.functional.relu(self.ffn(ln2_out))

        return ffn_out + self.ln3(ln2_out)
    
class Decoder(nn.Module):
    def __init__(self, n_words,n_token = 2, embed_dim=3, n_layers=3, d_attention = None, num_heads = 1, kind = "add", 
                 confidence = True, scene_disp = False, reordering = False, refining = False, embed_dim2=None):
        super().__init__()

        if d_attention is None:
            d_attention = embed_dim
        self.input_enc = InputEmbedding(n_words, embed_dim, n_token, kind = kind, 
                                        confidence = confidence, scene_disp = scene_disp, reordering = reordering)
        self.layers = nn.ModuleList([DecoderLayer(embed_dim =embed_dim, d_attention =d_attention , 
                                    num_heads = num_heads)for _ in range(n_layers)])
        if refining:
            self.layers = nn.ModuleList([DecoderLayer_refine(embed_dim =embed_dim, d_attention =d_attention , 
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
    
    """def decoder_mask(self, mask):
        n = len(mask)
        look_ahead = np.tril(np.ones(n))
        return mask * look_ahead
        
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask"""


class Transformer(nn.Module):
    def __init__(self, n_base_words, n_target_words,  n_layers = 3, kind = "add",embed_dim=512, d_attention = None, 
                num_heads = 1, confidence = True, scene_disp = False, reordering = False, embed_dim2 = None,  d_attention2 = None,
                n_layers2 = None, num_heads2 = None):
        super().__init__()

        if d_attention is not None:
            assert d_attention > 0, "d_attention is negative or equal to 0"


        if embed_dim2 == None:
            embed_dim2 = embed_dim
            d_attention2 = d_attention
            n_layers2 = n_layers
            num_heads2 = num_heads


        self.embed_dim2 = embed_dim2
        self.embed_dim = embed_dim

        self.n_target_words = n_target_words
        self.encoder = Encoder(n_base_words, n_base_words, embed_dim=embed_dim2, kind = kind, num_heads = num_heads2, d_attention = d_attention2,
                               n_layers = n_layers2, confidence = confidence, scene_disp = scene_disp, reordering = reordering, refining = (embed_dim2!= embed_dim))

        self.decoder = Decoder(n_target_words, n_target_words, embed_dim=embed_dim, kind = kind, num_heads= num_heads, d_attention = d_attention,
                               n_layers = n_layers, confidence = confidence, scene_disp = scene_disp, reordering = reordering, refining = (embed_dim2!= embed_dim), embed_dim2= embed_dim2)

    
        #self.encoder_2 = Encoder(n_base_words, n_base_words, embed_dim=embed_dim, kind = kind, num_heads = num_heads, d_attention = d_attention,
        #                       n_layers = n_layers, confidence = confidence, scene_disp = False, reordering = reordering)

    
        #? Flag in case, you do not want an higher number of ouptus than the embeeding, this will not use an additionnal layer
        if embed_dim == n_target_words:
            self.end_layer = False
        else:
            self.end_layer = True

        self.linear = nn.Linear(embed_dim, n_target_words)
       
        self.linear_test = nn.Linear(2*embed_dim, n_target_words)
        
        self.scene_disp = scene_disp
    def forward(self,
                encoder_input,
                decoder_input,
                surround = None, 
                encoder_mask=None,
                decoder_mask = None):
        

        """if self.scene_disp and False:
            encoder_input = rearrange(encoder_input, 'b n t  -> (b n) t')
            encoder_kps = self.encoder_2(encoder_input, surround = surround, mask = encoder_mask)
            #encoder_kps = self.decoder_2(encoder_input, encoder_kps, surround = surround, mask = decoder_mask)

            encoder_kps = encoder_kps[:,:,:2]    
            encoder_kps = rearrange(encoder_kps, '(b n) t d-> b n (t d)', b = decoder_input.size(0))
            
            encoder_scene = self.encoder(decoder_input, surround = surround, mask = encoder_mask)

            encoder_kps = self.decoder.input_enc.position_emb(encoder_kps)

            test = torch.cat((encoder_kps, encoder_scene), dim = -1)


            #test = rearrange(self.linear_test(test), 'b n t d -> b n (t d)')
            #print(test.size())

            return rearrange(self.linear_test(test), 'b n t -> (b n) t')
        if self.scene_disp and False:
            encoder_input = rearrange(encoder_input, 'b n t  -> (b n) t')
            encoder_out = self.encoder_2(encoder_input, surround = surround, mask = encoder_mask)
            encoder_out = encoder_out[:,:,:2]    
            encoder_out = rearrange(encoder_out, '(b n) t d-> b n (t d)', b = decoder_input.size(0))
            encoder_out = self.decoder.input_enc.position_emb(encoder_out)
            
            
            #decoder_out = self.encoder(encoder_out, surround = surround, mask = encoder_mask)
            
            decoder_out = self.decoder(decoder_input, encoder_out, surround = surround, mask = decoder_mask)




            mask = self.decoder.get_mask()
            #mask = self.encoder.get_mask()
        else:"""
            
        #encoder_out = self.encoder(encoder_input, surround = surround, mask = encoder_mask)
        encoder_out, decoder_mask = self.decoder.input_enc(encoder_input, surround = surround, mask_off = True)

        if self.embed_dim != self.embed_dim2 and decoder_mask is None:
            #print("WAHOUUUUU")
            decoder_mask = self.encoder.get_mask()

        decoder_out = self.decoder(decoder_input, encoder_out, surround = surround, mask = decoder_mask)
        mask = self.decoder.get_mask()
        #mask = self.encoder.get_mask()

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