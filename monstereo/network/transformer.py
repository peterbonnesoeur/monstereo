import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import math
import numpy as np
from einops import rearrange, repeat


"""
Disclaimer: If you want to dig deeper into this code on self-attention mechanism,
I highly recommend looking at the following explanation on attention mechanism 
if the reader is not used to them : https://jalammar.github.io/illustrated-transformer/

Thank you for using my code, I am open to any feedback if an aberration is found.
"""


class InputEmbedding(nn.Module):
    """ Embed a positional embedding, reshape and pad the inputs of the network"""

    def __init__(self, n_embed, n_token, max_len=100, kind = "add", confidence = True, 
                scene_disp = False, reordering = False):
        super().__init__()
        self.n_embed = n_embed
        self.n_token = n_token
        
        self.reordering = reordering
        self.confidence = confidence
        self.scene_disp = scene_disp
        if kind == "cat":
            #? add an unique  mix of a sinus and a cosinus at the end of the array
            self.position_emb =  PositionalEncoding(2, kind = kind, max_len = max_len)
        elif kind == "num":
            #? add an unique index at the end of the array
            self.position_emb =  PositionalEncoding(1, kind = kind, max_len = max_len)
        else:
            self.position_emb =  PositionalEncoding(2, kind = kind, max_len = max_len)

        #? The padding can be the mean over the whole dataset. 
        self.pad = torch.load('docs/tensor.pt')

    def forward(self, x,  mask_off = False):

        if self.scene_disp:
            #? The padded scenes are arrays of 0
            mask = torch.sum(x, dim = -1)!=0
            condition = (mask == False)

            pad = self.pad

            out = rearrange(x, 'b x (n d) -> b x n d', d = 3)
            pad = rearrange(pad, 'h (n d) -> h n d', d = 3)
            if not self.confidence:
                #? remove the confidence term of the inputs (and the pads)
                out = self.conf_remover(out)
                pad = pad[:,:, :-1]
            out = rearrange(out, 'b x n d -> b x (n d)')
            pad = rearrange(pad, 'h n d -> h (n d)')

            if not mask_off :                
                out[condition] = 0
                #? the padder function will change the kind of pads to improve the results 
                #? (This is mostly due to the LayerNorm step that
                #?  is having a huge influence on the end result)
                out = self.padder(out,pad)
                
            #! Add the positional embedding at the end of the array (yes, the positional embedding is 
            #! happening after the padder)
            #! this is a design choice, not an error
            out = self.position_emb(out)

            return out, mask
    
        else:
        
            out = rearrange(x, 'b (n d) -> b n d', d = 3)

            mask = self.generate_mask_keypoints(out)
            condition = (mask == False)


            if not self.confidence:
                #? remove the confidence term of each key-points
                out = self.conf_remover(out)

            
            #? Add the positionnal embedding at the end of the array
            out =self.position_emb(out)

            if not mask_off:
                #? pad the occluded keypoints with a 0
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
                    #? pad the rest of the array with the mean value of the non occluded keypoints
                    inputs[index, (i):,: ]=repeat(pad, "h w -> (h c) w ", c = (masks.size(0)-i))
                    #? Add the EOS pad at the end of each valid value for each scene
                    #inputs[index, i, :] = EOS
                    break

        return inputs
    
    def generate_mask_keypoints(self,kps):
        # ? Use the confidence to identify occluded keypoints (a confidence of 0 is equal to an occlude keypoints)
        if len(kps.size())==3:
            mask = (kps[:,:,-1]>0).clone().detach()
        else:
            mask = ( kps[:,-1]>0).clone().detach()
        return mask


class PositionalEncoding(nn.Module):
    """Add a positional embedding to the set of instances.
    It is used by the attention mechanism to recognize the position of each
    "word" in a "sentence" while drawing relationships between the words"""

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

        if self.kind == "cat":
            pe = self.pe.permute(1,0, 2).repeat(x.size(0),1, 1)
            x = torch.cat((x, pe[:, :x.size(1)]), dim = -1)
        elif self.kind == "num":
            #? We enumerate the number of instances / scenes
            pe = torch.arange(0,x.size(1)).repeat(x.size(0),1,1).permute(0,2,1).to(x.device)
            x = torch.cat((x, pe), dim = -1)

        return x



class Attention(nn.Module):
    """Implementation of the attention mechanism
    - extract the key, query and value vector from the original inputs
    - perform a dot product between the key and query vector over the whole "sentence"
    - apply a mask on the dotted product to remove the contribution of the occluded/padded instances
    - multiply the values and the softmax of the dotted product"""

    def __init__(self, embed_dim, d_attention, embed_dim2 = None):
        """"""
        super().__init__()

        if embed_dim2 is None:
            embed_dim2 = embed_dim

        #? To represent the Matrix multiplications, we are using fully connected layers with no biasses
        self.W_K = nn.Linear(embed_dim2, d_attention, bias = False)
        self.W_Q = nn.Linear(embed_dim, d_attention, bias = False)
        self.W_V = nn.Linear(embed_dim2, d_attention, bias = False)      

        self.scaling = torch.Tensor(np.array(1 / np.sqrt(d_attention)))

    def weight_value(self, Q_vec, K_vec, V_vec, mask):
        
        b, n, _ = Q_vec.shape
  
        q = self.W_Q(Q_vec)
        v = self.W_V(V_vec)
        k = self.W_K(K_vec)

        dots = torch.einsum('bid,bjd->bij', q, k) * self.scaling
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            #? Remove the contribution of the occluded/padded elements during the training phase
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask
        
        attn = dots.softmax(dim=-1)
        return attn, v


    def forward(self, Q_vec, K_vec, V_vec, mask):
        #? extract the relationship between the instances
        weight, v = self.weight_value(Q_vec, K_vec, V_vec, mask)
        out = torch.einsum('bij,bjd->bid', weight, v)
        return out

        
class MultiHeadAttention(nn.Module):
    """Generate a stack of attention-based mechanism with different initialization.
    The objective is to extract different "relationships" between the instances.
    The results of those attention mechanisms are then stacked together and reduced 
    back to the desired dimension"""
    def __init__(self, embed_dim, d_attention, num_heads, embed_dim2 = None):
        super().__init__()
        self.num_heads = num_heads
        self.attns = nn.ModuleList([Attention(embed_dim, d_attention, embed_dim2) for _ in range(num_heads)])

        #? To represent the Matrix multiplications, we are using fully connected layers with no biasses
        self.linear = nn.Linear(num_heads * d_attention, embed_dim, bias = False)
    def forward(self, Q_vec, K_vec, V_vec, mask):
        results = [attn(Q_vec, K_vec, V_vec, mask) for attn in self.attns]
        return self.linear(torch.cat(results, dim=-1))

    
class Encoder(nn.Module):
    """Encoder style self-attention module.
    The design is similar to the encoder mechanism described in the paper "Attention is all you need" by A. Vaswani

    This class encodes the inputs and then call our encoder_layer class to use attention mechanisms"""

    def __init__(self, n_words,n_token = 2, embed_dim=3, n_layers=3, d_attention = None, num_heads = 1,kind = "add",
                 confidence = True, scene_disp = False, reordering = False, refining = False):
        super().__init__()
        if d_attention is None:
            d_attention = embed_dim

        self.refining = refining
        self.input_enc = InputEmbedding(embed_dim, n_token, kind = kind, 
                                        confidence = confidence, scene_disp = scene_disp, 
                                        reordering = reordering)
        #? define a stack of encoder. The layers a repeated n_layers time
        self.layers = nn.ModuleList([EncoderLayer(embed_dim =embed_dim, d_attention =d_attention, n_words = n_words,
                                                num_heads = num_heads) for _ in range(n_layers)])

        
    def forward(self, x, mask = None):
        
        mask_off = False
        
        #? encoding of the inputs
        out, mask = self.input_enc(x,  mask_off = mask_off)
        self.mask = mask.detach()

        for i, layer in enumerate(self.layers):
            out = layer(out, self.mask)
        return out

    def get_mask(self):
        return self.mask

class EncoderLayer(nn.Module):
    """instantiate a self-attention mechanism similar to the encoder in Vaswani's paper.
    This network contains an attention step followed by a feed-forward one. """

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

        #? First attention-based layer
        attn_out = self.attn(x, x, x, mask)

        ln1_out = self.ln1(attn_out + x)

        #? Feed forward layer
        ffn_out = nn.functional.relu(self.ffn(ln1_out))

        return self.ln2(ffn_out+ln1_out)


    
class Decoder(nn.Module):
    """Decoder style self-attention module.
    The design is similar to the decoder mechanism described in the paper "Attention is all you need" by A. Vaswani.
    
    This class encodes the inputs and then call our decoder_layer class to use attention mechanisms"""

    def __init__(self, n_words,n_token = 2, embed_dim=3, n_layers=3, d_attention = None, num_heads = 1, kind = "add", 
                 confidence = True, scene_disp = False, reordering = False, refining = False, embed_dim2 = None):

        super().__init__()

        if d_attention is None:
            d_attention = embed_dim
        self.input_enc = InputEmbedding(embed_dim, n_token, kind = kind, 
                                        confidence = confidence, scene_disp = scene_disp, reordering = reordering)
       #? define a stack of decoder. The layers a repeated n_layers time
        self.layers = nn.ModuleList([DecoderLayer(embed_dim =embed_dim, d_attention =d_attention, n_words = n_words,
                                    num_heads = num_heads, embed_dim2 = embed_dim2)for _ in range(n_layers)])

    def forward(self, x,encoder_output,  mask = None):

        
        if mask is not None:
           out, _ = self.input_enc(x,  mask_off = False)
        else:
            out, mask = self.input_enc(x,  mask_off = False)

        self.mask = mask.detach()
        for i, layer in enumerate(self.layers):
            out = layer(out, encoder_output,  mask)
        return out
    
    def get_mask(self):
        return self.mask

class DecoderLayer(nn.Module):
    """instantiate a self-attention mechanism similar to the decoder in Vaswani's paper.
    This network contains two attention steps followed by a feed-forward one. """
    def __init__(self,
                 embed_dim = 3,
                 d_attention = 3,
                 embed_dim2 = 3,
                 n_words = 20,
                 num_heads= 1,):
        super().__init__()

        self.attn1 = MultiHeadAttention(embed_dim, d_attention, num_heads)
        self.attn2 = MultiHeadAttention(embed_dim, d_attention, num_heads, embed_dim2)
        self.ffn = nn.Linear(embed_dim, embed_dim)

        self.ln1 = nn.LayerNorm([n_words,embed_dim])
        self.ln2 = nn.LayerNorm([n_words,embed_dim])
        self.ln3 = nn.LayerNorm([n_words,embed_dim])

    def forward(self, x, encoder_output, mask=None):
        
        #? First attention-based layer
        attn1_out = self.attn1(x, x, x, mask)

        ln1_out = self.ln1(attn1_out + x)

        #? Second attention-based layer
        attn2_out = self.attn2(ln1_out,ln1_out,ln1_out,mask)

        ln2_out = self.ln2(attn2_out + ln1_out)

        #? Feed forward layer
        ffn_out = nn.functional.relu(self.ffn(ln2_out))

        return self.ln3(ffn_out + ln2_out)



class Transformer(nn.Module):
    """"Instantiate the attention mechanism and reshape the data before going through a fully connected layer"""

    def __init__(self, n_base_dim, n_target_dim, n_words, n_layers = 3, kind = "add",embed_dim=512, d_attention = None, 
                 num_heads = 1, confidence = True, scene_disp = False, reordering = False, embed_dim2 = None, 
                 d_attention2 = None, n_layers2 = None, num_heads2 = None):

        super().__init__()

        if d_attention is not None:
            assert d_attention > 0, "d_attention is negative or equal to 0"

        #? Mixed case where the encoder's output is of a different size compared to 
        #? the decoder's output -> this is quite experimental
        #? But it is one of the fields that can be explored: using different data for 
        #? the encoder and decoder and extract the relationships that we can extract from those 2 sets
        
        if embed_dim2 == None:
            embed_dim2 = embed_dim 
            d_attention2 = d_attention
            n_layers2 = n_layers
            num_heads2 = num_heads


        self.embed_dim2 = embed_dim2
        self.embed_dim = embed_dim

        self.n_target_dim = n_target_dim
        self.encoder = Encoder(n_words, n_base_dim, embed_dim=embed_dim2, kind = kind, num_heads = num_heads2, 
                                d_attention = d_attention2, n_layers = n_layers2, confidence = confidence, 
                                scene_disp = scene_disp, reordering = reordering, refining = (embed_dim2!= embed_dim))


        self.decoder = Decoder(n_words, n_target_dim, embed_dim=embed_dim, kind = kind, num_heads= num_heads, 
                                d_attention = d_attention, n_layers = n_layers, confidence = confidence, 
                                scene_disp = scene_disp, reordering = reordering, 
                                refining = (embed_dim2!= embed_dim), embed_dim2= embed_dim2)
    
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
                encoder_mask=None,
                decoder_mask = None):
        
        #? get the output of the encoder mechanism
        #encoder_out = self.encoder(encoder_input,  mask = encoder_mask)

        #! in our case, we use only a tack of encoding layers. 
        #! the encoder output is more a formalism in case we want to use a more "transformer-like" kind of architecture
        encoder_out, decoder_mask = self.decoder.input_enc(encoder_input, mask_off = False)

        if self.embed_dim != self.embed_dim2 and decoder_mask is None:
            decoder_mask = self.encoder.get_mask()
            
        decoder_out = self.decoder(decoder_input, encoder_out,  mask = decoder_mask)
        mask = self.decoder.get_mask()


        if self.scene_disp: # in the scene disposition the current formatting is [scene; instance_in_scene; flattened_key-points] 
            if self.end_layer:
                return rearrange(self.linear(decoder_out), 'b n t -> (b n) t')
            else:
                return rearrange(decoder_out, 'b n t -> (b n) t')

        else:   # in the regular disposition, the formatting is [instance, set_of_key-points, key-point_coordinates]
            if self.end_layer:
                return rearrange(self.linear(decoder_out), 'b n t -> b (n t)')
            else:
                return rearrange(decoder_out, 'b n t -> b (n t)')