
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .transformer import TransformerModel
from .transformer_scene import Transformer as TransformerModel_scene
from .transformer_scene2 import Transformer as TransformerModel_scene2
from einops import rearrange, repeat



class MyLinearSimple(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(MyLinearSimple, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):

        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out

class Refiner(nn.Module):
    def __init__(self, input_size, length_sentence =12, p_dropout=0.2, num_stage=4, num_heads = 3,device='cuda'):
        super(Refiner, self).__init__()

        self.input_size = input_size
        self.length_sentence = length_sentence
        self.p_dropout = p_dropout

    
        n_output_token = self.input_size
        kind ='cat'
        if kind == 'cat':
            n_output = n_output_token + 2
        else:
            n_output = n_output_token
        mul_output = 1
        
        self.transformer_scene=  TransformerModel_scene2(n_base_words = length_sentence, n_target_words = n_output_token*mul_output, n_token = n_output_token, kind = kind, embed_dim = n_output
                                                        , num_heads = num_heads, n_layers = num_stage, confidence = True, scene_disp = True)

        #self.w_out = nn.Linear(int(n_output_token*mul_output), self.input_size)

    def forward(self, y,batch_size):
        
        #print("HERE",batch_size)
        env = None
        output = rearrange(y, '(b n) d -> b n d', b = batch_size)
        output = self.transformer_scene(output,output,env)
        #output = self.w_out(output)

        return output
        
class SimpleModel(nn.Module):

    def __init__(self, input_size, output_size=2, linear_size=512, p_dropout=0.2, num_stage=4, device='cuda', transformer = False, 
                surround = False, lstm = False, scene_disp = False):
        super(SimpleModel, self).__init__()

        self.stereo_size = input_size
        self.mono_size = int(input_size / 2)
        self.output_size = output_size - 1
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.linear_stages = []
        self.device = device
        self.transformer = transformer
        self.surround = surround
        self.lstm = lstm
        self.scene_disp = scene_disp
        self.scene_refine = False
        self.refiner_flag = True

        assert (not (self.transformer and self.lstm)) , "The network cannot implement a transformer and a LSTM at the same time"
        # Initialize weights

        n_head = 3
        n_token = 3
        n_hidden = self.num_stage
        mul_output = 3
        # Preprocessing
        if self.transformer:
            assert self.stereo_size%3 == 0, "The confidence needs to be in the keypoints [x, y, conf]"
            # The max 
            #ntoken = 3
            if not self.scene_disp:
                if self.surround:
                    n_token+=10
                kind = "cat"
                if kind == 'cat':
                    n_inp = n_token + 2
                else:
                    n_inp = n_token

                print(n_token, n_inp, kind)
                self.transformer_model = TransformerModel(ntoken = n_token, ninp = n_inp, nhead = 1,  nhid = 2, nlayers = 2,  dropout = p_dropout, kind = kind)
                self.transformer_kps=  TransformerModel_scene2(n_base_words = n_token, n_target_words = n_token*mul_output, n_token = n_token, kind = kind, embed_dim = n_inp
                                                            , num_heads = n_head, n_layers = n_hidden,confidence = True, scene_disp = False)
                
                #self.transformer_model= TransformerModel_2(n_base_words = n_token, n_target_words = n_token*mul_output, n_token = n_token, kind = kind, embed_dim = n_inp, num_heads = n_head, n_layers = n_hidden) 

                self.w1 = nn.Linear(int(self.stereo_size/3*n_token*mul_output), self.linear_size)

            elif self.scene_disp and self.scene_refine:

               
                kind = "cat"
                if kind == 'cat':
                    n_inp = n_token + 2
                else:
                    n_inp = n_token

                self.transformer_kps=  TransformerModel_scene2(n_base_words = n_token, n_target_words = n_token*mul_output, n_token = n_token, kind = kind, embed_dim = n_inp
                                        , num_heads = n_head, n_layers = n_hidden,confidence = True, scene_disp = False)


                self.w1 = nn.Linear(int(self.stereo_size/3*n_token*mul_output), self.linear_size) 
            else:
                assert self.transformer, "Currently, the scene disposition method is only compatible with the transformer"
                n_token = int(self.stereo_size/3*2)
                n_token = self.stereo_size

                kind = "cat"
                if kind == 'cat':
                    n_inp = n_token + 2
                else:
                    n_inp = n_token
                assert self.stereo_size%3 == 0, "The confidence needs to be in the keypoints [x, y, conf]"

                self.transformer_scene = TransformerModel_scene2(n_base_words = 12, n_target_words = n_token*mul_output, n_token = n_token, kind = kind, embed_dim = n_inp
                                                            , num_heads = 3, n_layers = n_hidden,confidence = True, scene_disp = True)

                self.w1 = nn.Linear(n_token*mul_output, self.linear_size) 

        elif self.lstm:
            
            ntoken = 3                
            kind = "cat"
            if kind == 'cat':
                n_inp = n_token + 2
            else:
                n_inp = n_token
            bidirectional = True
            # This transformer is only instanciated to use the same exact input encoding for both the LSTM and the transformer
            self.transformer_kps=  TransformerModel_scene(n_base_words = n_token, n_target_words = n_token, n_token = n_token, kind = kind, embed_dim = n_inp
                                                            ,confidence = True, scene_disp = False)            
            self.inputEmbedding = InputLSTM(dim_embed = 3,conf = True, mask = True )
            #self.inputEmbedding = self.transformer_kps.encoder.input_enc()
            self.LSTM = torch.nn.LSTM(input_size = n_inp, hidden_size = int(n_token*mul_output), num_layers = n_hidden, 
                                bidirectional = bidirectional, dropout = p_dropout)
            if bidirectional:
                mul_output*=2
            self.w1 = nn.Linear(int(self.stereo_size/3*n_token*mul_output), self.linear_size)

        else:
            self.w1 = nn.Linear(self.stereo_size, self.linear_size)

        self.n_token = n_token
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)
        self.group_norm1 = nn.GroupNorm(self.linear_size, int(self.linear_size/100))

        # Internal loop
        for _ in range(num_stage):
            self.linear_stages.append(MyLinearSimple(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # Post processing
        self.w2 = nn.Linear(self.linear_size, self.linear_size)
        self.w3 = nn.Linear(self.linear_size, self.linear_size)
        self.batch_norm3 = nn.BatchNorm1d(self.linear_size)
        self.group_norm3 =  nn.GroupNorm(self.linear_size, int(self.linear_size/100))

        # ------------------------Other----------------------------------------------
        # Auxiliary
        self.w_aux = nn.Linear(self.linear_size, 1)

        # Final
        self.w_fin = nn.Linear(self.linear_size, self.output_size)

        self.refiner = Refiner(input_size=self.output_size+1, length_sentence=12, p_dropout=self.p_dropout, num_stage=n_hidden, device=self.device)
        # NO-weight operations
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def generate_square_subsequent_mask(self, sz):
        mask = self.transformer.generate_square_subsequent_mask(sz)
        return mask

    def get_output(self, input, output):
        mask = torch.sum(input, dim = 2)==0
        #output = rearrange(output, 'b n k -> (b n) k')
        mask = rearrange(mask, 'b n -> (b n)')
        output[mask] = 0
        return output

        
    def forward(self, x, env= None):

        if self.transformer or self.lstm:
            if self.transformer:
                #y = self.transformer(x, env)

                if self.scene_disp:
                    
                    if self.scene_refine:
                        inp = rearrange(x, 'b n d -> (b n) d')
                        y = self.transformer_kps(inp,inp, env)
                    else:
                        y = self.transformer_scene(x,x, env)
                else:
                    y = self.transformer_kps(x,x, env)
                    #y = self.transformer_model(x,x,env)
            else:
                y,_ = self.transformer_kps.encoder.input_enc(x)
                y = rearrange(y, 'b n t -> n b t')
                y, _ = self.LSTM(y)
                y = rearrange(y, 'n b d -> b (n d)')

            y = self.w1(y)
            
            aux = self.w_aux(y)

            y = self.batch_norm1(y)
            y = self.relu(y)
            y = self.dropout(y)

            y = self.w_fin(y)
            
            y = torch.cat((y, aux), dim=1)
            
            if self.scene_refine and self.refiner_flag:
                y = self.refiner(y, x.size(0))

            return y
        else:
            y = self.w1(x)

        y = self.batch_norm1(y)
        y = self.relu(y)
        
        if not self.transformer:
            y = self.dropout(y)

        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        # Auxiliary task
        y = self.w2(y)
        aux = self.w_aux(y)

        # Final layers
        y = self.w3(y)
        y = self.batch_norm3(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.w_fin(y)

        # Cat with auxiliary task
        y = torch.cat((y, aux), dim=1)
        return y

class DecisionModel(nn.Module):

    def __init__(self, input_size, output_size=2, linear_size=512, p_dropout=0.2, num_stage=3, device='cuda:1'):
        super(DecisionModel, self).__init__()

        self.num_stage = num_stage
        self.stereo_size = input_size
        self.mono_size = int(input_size / 2)
        self.output_size = output_size - 1
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.linear_stages_mono, self.linear_stages_stereo, self.linear_stages_dec = [], [], []
        self.device = device

        # Initialize weights

        # ------------------------Stereo----------------------------------------------
        # Preprocessing
        self.w1_stereo = nn.Linear(self.stereo_size, self.linear_size)
        self.batch_norm_stereo = nn.BatchNorm1d(self.linear_size)

        # Internal loop
        for _ in range(num_stage):
            self.linear_stages_stereo.append(MyLinear_stereo(self.linear_size, self.p_dropout))
        self.linear_stages_stereo = nn.ModuleList(self.linear_stages_stereo)

        # Post processing
        self.w2_stereo = nn.Linear(self.linear_size, self.output_size)

        # ------------------------Mono----------------------------------------------
        # Preprocessing
        self.w1_mono = nn.Linear(self.mono_size, self.linear_size)
        self.batch_norm_mono = nn.BatchNorm1d(self.linear_size)

        # Internal loop
        for _ in range(num_stage):
            self.linear_stages_mono.append(MyLinear_stereo(self.linear_size, self.p_dropout))
        self.linear_stages_mono = nn.ModuleList(self.linear_stages_mono)

        # Post processing
        self.w2_mono = nn.Linear(self.linear_size, self.output_size)

        # ------------------------Decision----------------------------------------------
        # Preprocessing
        self.w1_dec = nn.Linear(self.stereo_size, self.linear_size)
        self.batch_norm_dec = nn.BatchNorm1d(self.linear_size)
        #
        # Internal loop
        for _ in range(num_stage):
            self.linear_stages_dec.append(MyLinear(self.linear_size, self.p_dropout))
        self.linear_stages_dec = nn.ModuleList(self.linear_stages_dec)

        # Post processing
        self.w2_dec = nn.Linear(self.linear_size, 1)

        # ------------------------Other----------------------------------------------

        # NO-weight operations
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x, label=None):

        # Mono
        y_m = self.w1_mono(x[:, 0:34])
        y_m = self.batch_norm_mono(y_m)
        y_m = self.relu(y_m)
        y_m = self.dropout(y_m)

        for i in range(self.num_stage):
            y_m = self.linear_stages_mono[i](y_m)
        y_m = self.w2_mono(y_m)

        # Stereo
        y_s = self.w1_stereo(x)
        y_s = self.batch_norm_stereo(y_s)
        y_s = self.relu(y_s)
        y_s = self.dropout(y_s)

        for i in range(self.num_stage):
            y_s = self.linear_stages_stereo[i](y_s)
        y_s = self.w2_stereo(y_s)

        # Decision
        y_d = self.w1_dec(x)
        y_d = self.batch_norm_dec(y_d)
        y_d = self.relu(y_d)
        y_d = self.dropout(y_d)

        for i in range(self.num_stage):
            y_d = self.linear_stages_dec[i](y_d)
        aux = self.w2_dec(y_d)

        # Combine
        if label is not None:
            gate = label
        else:
            gate = torch.where(torch.sigmoid(aux) > 0.3,
                               torch.tensor([1.]).to(self.device), torch.tensor([0.]).to(self.device))
        y = gate * y_s + (1-gate) * y_m

        # Cat with auxiliary task
        y = torch.cat((y, aux), dim=1)
        return y


class AttentionModel(nn.Module):

    def __init__(self, input_size, output_size=2, linear_size=512, p_dropout=0.2, num_stage=3, device='cuda'):
        super(AttentionModel, self).__init__()

        self.num_stage = num_stage
        self.stereo_size = input_size
        self.mono_size = int(input_size / 2)
        self.output_size = output_size - 1
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.linear_stages_mono, self.linear_stages_stereo, self.linear_stages_comb = [], [], []
        self.device = device

        # Initialize weights
        # ------------------------Stereo----------------------------------------------
        # Preprocessing
        self.w1_stereo = nn.Linear(self.stereo_size, self.linear_size)
        self.batch_norm_stereo = nn.BatchNorm1d(self.linear_size)

        # Internal loop
        for _ in range(num_stage):
            self.linear_stages_stereo.append(MyLinear_stereo(self.linear_size, self.p_dropout))
        self.linear_stages_stereo = nn.ModuleList(self.linear_stages_stereo)

        # Post processing
        self.w2_stereo = nn.Linear(self.linear_size, self.linear_size)

        # ------------------------Mono----------------------------------------------
        # Preprocessing
        self.w1_mono = nn.Linear(self.mono_size, self.linear_size)
        self.batch_norm_mono = nn.BatchNorm1d(self.linear_size)

        # Internal loop
        for _ in range(num_stage):
            self.linear_stages_mono.append(MyLinear_stereo(self.linear_size, self.p_dropout))
        self.linear_stages_mono = nn.ModuleList(self.linear_stages_mono)

        # Post processing
        self.w2_mono = nn.Linear(self.linear_size, self.linear_size)

        # ------------------------Combined----------------------------------------------
        # Preprocessing
        self.w1_comb = nn.Linear(self.linear_size, self.linear_size)
        self.batch_norm_comb = nn.BatchNorm1d(self.linear_size)
        #
        # Internal loop
        for _ in range(num_stage):
            self.linear_stages_comb.append(MyLinear(self.linear_size, self.p_dropout))
        self.linear_stages_comb = nn.ModuleList(self.linear_stages_comb)

        # Post processing
        self.w2_comb = nn.Linear(self.linear_size, self.linear_size)

        # ------------------------Other----------------------------------------------
        # Auxiliary
        self.w_aux = nn.Linear(self.linear_size, 1)

        # Final
        self.w_fin = nn.Linear(self.linear_size, self.output_size)

        # NO-weight operations
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x, label=None):


        # Mono
        y_m = self.w1_mono(x[:, 0:34])
        y_m = self.batch_norm_mono(y_m)
        y_m = self.relu(y_m)
        y_m = self.dropout(y_m)

        for i in range(self.num_stage):
            y_m = self.linear_stages_mono[i](y_m)
        y_m = self.w2_mono(y_m)

        # Stereo
        y_s = self.w1_stereo(x)
        y_s = self.batch_norm_stereo(y_s)
        y_s = self.relu(y_s)
        y_s = self.dropout(y_s)

        for i in range(self.num_stage):
            y_s = self.linear_stages_stereo[i](y_s)
        y_s = self.w2_stereo(y_s)

        # Auxiliary task
        aux = self.w_aux(y_s)

        # Combined
        if label is not None:
            gate = label
        else:
            gate = torch.where(torch.sigmoid(aux) > 0.3,
                               torch.tensor([1.]).to(self.device), torch.tensor([0.]).to(self.device))
        y_c = gate * y_s + (1-gate) * y_m
        y_c = self.w1_comb(y_c)
        y_c = self.batch_norm_comb(y_c)
        y_c = self.relu(y_c)
        y_c = self.dropout(y_c)
        y_c = self.w_fin(y_c)

        # Cat with auxiliary task
        y = torch.cat((y_c, aux), dim=1)
        return y


class MyLinear_stereo(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(MyLinear_stereo, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        # self.w0_a = nn.Linear(self.l_size, self.l_size)
        # self.batch_norm0_a = nn.BatchNorm1d(self.l_size)
        # self.w0_b = nn.Linear(self.l_size, self.l_size)
        # self.batch_norm0_b = nn.BatchNorm1d(self.l_size)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        #
        # x = self.w0_a(x)
        # x = self.batch_norm0_a(x)
        # x = self.w0_b(x)
        # x = self.batch_norm0_b(x)

        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out


class MonolocoModel(nn.Module):
    """
    Architecture inspired by https://github.com/una-dinosauria/3d-pose-baseline
    Pytorch implementation from: https://github.com/weigq/3d_pose_baseline_pytorch
    """

    def __init__(self, input_size, output_size=2, linear_size=256, p_dropout=0.2, num_stage=3):
        super(MonolocoModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for _ in range(num_stage):
            self.linear_stages.append(MyLinear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):
        # pre-processing
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)
        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)
        y = self.w2(y)
        return y


class MyLinear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(MyLinear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):

        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out

class InputLSTM(nn.Module):



    def __init__(self, dim_embed = 3, conf = False, mask = False):
        super().__init__()
        self.dim_embed = dim_embed
        self.conf = conf
        self.mask = mask
        
        
    def forward(self, x, surround = None):

        assert x.size(1)%self.dim_embed==0, "Wrong input, we need the flattened keypoints with the confidence"
        
        out = rearrange(x, 'b (n t) -> b n t', t = self.dim_embed)
        mask = self.generate_mask_keypoints(out)
        
        
        if self.mask:
            mask = self.generate_mask_keypoints(out)
            out[mask == False] = 0
        
        if not self.conf : 
            out = self.conf_remover(out)
        
        out = rearrange(out, 'b n t -> n b t')
        return out

    
    def conf_remover(self, kps):
        return kps[:,:,:2]
    
    def generate_mask_keypoints(self,kps):
        #print(kps)
        if len(kps.size())==3:
            mask = torch.tensor( kps[:,:,2]>0)
        else:
            mask = torch.tensor( kps[:,2]>0)
        return mask