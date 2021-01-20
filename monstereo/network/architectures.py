
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#from .transformer import TransformerModel
#from .transformer_scene import Transformer as TransformerModel_scene
from .transformer_scene2 import Transformer as TransformerModel_scene2
from einops import rearrange, repeat

#from ..train import SCENE_INSTANCE_SIZE

SCENE_INSTANCE_SIZE = 20

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
    def __init__(self, input_size, length_sentence =12, p_dropout=0.2, num_stage=3, num_heads = 3,device='cuda'):
        super(Refiner, self).__init__()

        self.input_size = input_size
        self.length_sentence = length_sentence
        self.p_dropout = p_dropout

    
        n_output_token = self.input_size
        kind ='cat'
        if kind == 'cat':
            n_output = n_output_token + 2
        elif kind == 'num':
            embed_dim = n_output_token +1
        else:
            n_output = n_output_token
        mul_output = 1
        
        self.transformer_scene=  TransformerModel_scene2(n_base_words = length_sentence, n_target_words = n_output_token*mul_output, kind = kind, embed_dim = n_output
                                                        , num_heads = num_heads, n_layers = 1, confidence = True, scene_disp = True)

        
        self.w_out = nn.Linear(int(n_output_token*mul_output), self.input_size)

    def forward(self, y,batch_size):
        
        env = None
        #output = rearrange(y, '(b n) d -> b n d', b = batch_size)
        #output = self.transformer_scene(output,output,env)
        #output = self.w_out(output)
        output = self.w_out(y)
        return output
        
class SimpleModel(nn.Module):

    def __init__(self, input_size, output_size=2, linear_size=512, p_dropout=0.2, num_stage=3, device='cuda', transformer = False, 
                confidence = True, surround = False, lstm = False, scene_disp = False, scene_refine = False):
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
        self.scene_refine = scene_refine
        self.confidence = confidence
        self.refiner_flag = True

        assert (not (self.transformer and self.lstm)) , "The network cannot implement a transformer and a LSTM at the same time"
        # Initialize weights

        n_hidden = self.num_stage #? Number of stages for the transformer (a stage for the transfromer is the number of layer for the encoder/decoder). Of course, more layers means more calculations
        mul_output = 1  #? determine if at the end of the transformer we add a fully connected layer to go from N to mul_output*N outputs. If mul_output = 1, there will be no such fully connected layer
        n_head = 4 #? the number of heads of of multi_headed attention model
        n_inp = 3  #? the original input size for the keypoints (X,Y,C)
        if not self.confidence:
            n_inp = 2
        kind = "cat" #? The kind of position embedding between [cat, add, num, none]
        #? Cat adds a complex of a sin and cos after the data of the keypoints (hence, the inputs for the transformer grows from n_inp to n_inp + 2)
        #? add adds a complex of a sin and a cos on top of the X and Y coordinate. We are doing X = X+cos(wt) and Y = Ysin(wt)
        #? num adds a counter at the end of the data of the keypoints. It is a simple index going from [0, N-1] where N is the size of the sequence (hence, the inputs for the transformer grows from n_inp to n_inp + 1).  
        
        

        reordering = False

        length_scene_sequence = SCENE_INSTANCE_SIZE #? in the case of the scene disposition (where we do not look at the keypoints but at the sequence of keypoints in our transformer),
                                                    #? we needed to create a padded array of fixed size to put our instances. In this case, the instances in the sequence are the set of 
                                                    #? 2d keypoints with their confidence and flattened. This constant is defined in the train/dataset part.

        
        # Preprocessing
        if self.transformer:
            #assert self.stereo_size%3 == 0, "The confidence needs to be in the keypoints [x, y, conf]"

            if not self.scene_disp:

                #! In the future, the surround parameter will be removed
                if self.surround:
                    n_inp+=10

                #? Embed_dim is the embeding dimension that the transformer will efectively see. Hence, it is the dimension obtained after the embedding and position embedding step
                if kind == 'cat':
                    embed_dim = n_inp + 2 #? For an explanation, see the comment on top for the variable "kind"
                elif kind == 'num':
                    embed_dim = n_inp +1 #? For an explanation, see the comment on top for the variable "kind"
                else:
                    embed_dim = n_inp

                d_attention = int(embed_dim/2) #? The dimesion of the key, query and value vector in the attention mechanism. Being an embedding, its dimension should be inferior to the embed_dim
                                               #? In the original paper of the transformer, d_attention = int(embed_dim/n_head)

                print( n_inp, embed_dim, kind)
                #self.transformer_model = TransformerModel(ntoken = n_token, ninp = n_inp, nhead = 1,  nhid = 2, nlayers = 2,  dropout = p_dropout, kind = kind)
                self.transformer_kps=  TransformerModel_scene2(n_base_words = n_inp, n_target_words = embed_dim*mul_output, kind = kind, embed_dim = embed_dim, 
                                                                d_attention =d_attention, num_heads = n_head, n_layers = n_hidden,confidence = self.confidence, 
                                                                scene_disp = False, reordering = reordering)
                                                                #? The confidence flag tells us if we should take into account the confidence for each keypoints, by design, yes
                                                                #? The scene_disp flag tells us if we are reasonning with scenes or keypoints 
                                                                #? the reordering flag is there to order the inputs in a peculiar way (in the scene case, order
                                                                #  the instances depending on their height for example)

                #self.transformer_model= TransformerModel_2(n_base_words = n_token, n_target_words = n_token*mul_output, n_token = n_token, kind = kind, embed_dim = n_inp, num_heads = n_head, n_layers = n_hidden)
                if self.confidence:
                    self.w1 = nn.Linear(int(self.stereo_size/3*embed_dim*mul_output), self.linear_size)
                else:
                    self.w1 = nn.Linear(int(self.stereo_size/2*embed_dim*mul_output), self.linear_size)

            elif self.scene_refine and False: 
                #? This method is a bit peculiar since we are at first using our regular keypoint-based transformer to obtain some results. 
                #? But right after that step, we are using a scene transformer reasonning on the outputs of the transformer.
                #! The system obtained acts as a refining step for our model

                #? Embed_dim is the embeding dimension that the transformer will efectively see. Hence, it is the dimension obtained after the embedding and position embedding step
                if kind == 'cat':
                    embed_dim = n_inp + 2 #? For an explanation, see the comment on top for the variable "kind"
                elif kind == 'num':
                    embed_dim = n_inp +1 #? For an explanation, see the comment on top for the variable "kind"

                d_attention = int(embed_dim/n_head) #? The dimesion of the key, query and value vector in the attention mechanism. Being an embedding, its dimension should be inferior to the embed_dim
                                               #? In the original paper of the transformer, d_attention = int(embed_dim/n_head)




                self.transformer_kps=  TransformerModel_scene2(n_base_words = n_inp, n_target_words = embed_dim*mul_output, kind = kind, embed_dim = embed_dim, 
                                                                d_attention =d_attention, num_heads = n_head, n_layers = n_hidden, confidence = self.confidence, 
                                                                scene_disp = False, reordering = reordering)
                                                                #? The confidence flag tells us if we should take into account the confidence for each keypoints, by design, yes
                                                                #? The scene_disp flag tells us if we are reasonning with scenes or keypoints 
                                                                #? the reordering flag is there to order the inputs in a peculiar way (in the scene case, order
                                                                #  the instances depending on their height for example)

                if self.confidence:
                    self.w1 = nn.Linear(int(self.stereo_size/3*embed_dim*mul_output), self.linear_size)
                else:
                    self.w1 = nn.Linear(int(self.stereo_size/2*embed_dim*mul_output), self.linear_size)

                #TODO for the refiner, remove the auxiliary term
                self.refiner = Refiner(input_size=self.output_size+1, length_sentence=length_scene_sequence, p_dropout=self.p_dropout, num_stage=n_hidden, device=self.device)

            elif scene_refine:
                assert self.transformer, "Currently, the scene disposition method is only compatible with the transformer"
                n_inp = self.linear_size
                n_hidden = 1
                n_target_words = self.output_size#embed_dim*mul_output

                if kind == 'cat':
                    embed_dim = n_inp + 2
                elif kind == 'num':
                    embed_dim = n_inp +1
                else:
                    embed_dim = n_inp

                d_attention = int(embed_dim/n_head)

                #! TEST
                if kind == 'cat':
                    embed_dim2 = self.stereo_size + 2 #? For an explanation, see the comment on top for the variable "kind"
                elif kind == 'num':
                    embed_dim2 = self.stereo_size + 1 #? For an explanation, see the comment on top for the variable "kind"
                else:
                    embed_dim2 = self.stereo_size

                d_attention2 = int(embed_dim2/n_head)

                n_head2 = 3
                d_attention2 = int(embed_dim2/n_head2) #embed_dim2 -3
                n_hidden2 = self.num_stage

                assert self.stereo_size%3 == 0, "The confidence needs to be in the keypoints [x, y, conf]"

                self.transformer_scene = TransformerModel_scene2(n_base_words = n_inp, n_target_words = n_target_words, kind = kind, embed_dim = embed_dim,
                                                                d_attention =d_attention, num_heads = n_head, n_layers = n_hidden,confidence = self.confidence, 
                                                                scene_disp = True, reordering = reordering, embed_dim2 = embed_dim2, d_attention2 = d_attention2, n_layers2 = n_hidden2, num_heads2 = n_head2)
                                                                #? The confidence flag tells us if we should take into account the confidence for each keypoints, by design, yes
                                                                #? The scene_disp flag tells us if we are reasonning with scenes or keypoints 
                                                                #? the reordering flag is there to order the inputs in a peculiar way (in the scene case, order
                                                                #? the instances depending on their height for example)

                mul_output = 1
                self.LSTM = torch.nn.LSTM(input_size = embed_dim, hidden_size = int(embed_dim*mul_output), num_layers = n_hidden, 
                                    bidirectional = True, dropout = 0.1)
                mul_output*=2

                self.w1 = nn.Linear(self.stereo_size, self.linear_size)
                self.w_scene_refine = nn.Linear(int(embed_dim*mul_output), self.output_size)

                self.refiner = Refiner(input_size=self.output_size+1, length_sentence=length_scene_sequence, p_dropout=self.p_dropout, num_stage=n_hidden, device=self.device)

            else:
                assert self.transformer, "Currently, the scene disposition method is only compatible with the transformer"
                n_inp = self.stereo_size


                if kind == 'cat':
                    embed_dim = n_inp + 2
                elif kind == 'num':
                    embed_dim = n_inp +1
                else:
                    embed_dim = n_inp

                d_attention = int(embed_dim/2)

                assert self.stereo_size%3 == 0, "The confidence needs to be in the keypoints [x, y, conf]"

                self.transformer_scene = TransformerModel_scene2(n_base_words = n_inp, n_target_words = embed_dim*mul_output, kind = kind, embed_dim = embed_dim,
                                                                d_attention =d_attention, num_heads = n_head, n_layers = n_hidden,confidence = self.confidence, 
                                                                scene_disp = True, reordering = False)
                                                                #? The confidence flag tells us if we should take into account the confidence for each keypoints, by design, yes
                                                                #? The scene_disp flag tells us if we are reasonning with scenes or keypoints 
                                                                #? the reordering flag is there to order the inputs in a peculiar way (in the scene case, order
                                                                #?   the instances depending on their height for example)

                self.w1 = nn.Linear(embed_dim*mul_output, self.linear_size) 
        elif self.lstm:

            #? To benchmark our algotrtihm, the LSTM is also implemented. This is a simple bi-directioonal LSTM working in both the scene and keypoint situation
            
            if self.scene_disp:
                n_inp = self.stereo_size 
                     
            if kind == 'cat':
                embed_dim = n_inp + 2

            elif kind == 'num':
                embed_dim = n_inp +1
            else:
                embed_dim = n_inp

            d_attention = int(embed_dim/2)

            bidirectional = True

            #! This transformer is only instanciated to use the same exact input encoding for both the LSTM and the transformer
            self.transformer_kps=  TransformerModel_scene2(n_base_words = n_inp, n_target_words = embed_dim*mul_output,  kind = kind, embed_dim = embed_dim, 
                                                            d_attention =embed_dim, num_heads = n_head, n_layers = n_hidden,confidence = self.confidence, 
                                                            scene_disp = self.scene_disp, reordering = reordering)       

            self.LSTM = torch.nn.LSTM(input_size = embed_dim, hidden_size = int(embed_dim*mul_output), num_layers = n_hidden, 
                                    bidirectional = bidirectional, dropout = p_dropout)
            if bidirectional:
                mul_output*=2
            if self.confidence:
                self.w1 = nn.Linear(int(self.stereo_size/3*embed_dim*mul_output), self.linear_size)
            else:
                self.w1 = nn.Linear(int(self.stereo_size/2*embed_dim*mul_output), self.linear_size)

        else:
            self.w1 = nn.Linear(self.stereo_size, self.linear_size)

        #self.n_token = n_token
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)
        # self.group_norm1 = nn.GroupNorm(self.linear_size, int(self.linear_size/100))

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
        # NO-weight operations
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def generate_square_subsequent_mask(self, sz):
        mask = self.transformer.generate_square_subsequent_mask(sz)
        return mask

        
    def forward(self, x, env= None):

        if (self.transformer or self.lstm) and not self.scene_refine:
            if self.transformer:
                if self.scene_disp:

                    if self.scene_refine:
                        inp = rearrange(x, 'b n d -> (b n) d')
                        y = self.transformer_kps(inp,inp, env)
                    else:
                        y = self.transformer_scene(x,x, env)
                else:
                    y = self.transformer_kps(x,x, env)
            else:
                #? LSTM 
                y,_ = self.transformer_kps.encoder.input_enc(x)
                y = rearrange(y, 'b n t -> n b t')
                y, _ = self.LSTM(y)
                y = rearrange(y, 'n b d -> b (n d)')

            y = self.w1(y)
            if True:
                aux = self.w_aux(y)

                y = self.batch_norm1(y)
                y = self.relu(y)
                y = self.dropout(y)

                y = self.w_fin(y)
                
                y = torch.cat((y, aux), dim=1)
                
                if self.scene_refine and self.refiner_flag:
                    y = self.refiner(y, x.size(0))

                return y
        elif self.scene_refine:
            b_size = x.size(0)
            x = rearrange(x, 'b n d -> (b n) d')
            y = self.w1(x)
        else:
            y = self.w1(x)

        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        # Auxiliary task

        y = self.w2(y)
        aux = self.w_aux(y)

        # Final layers
        if self.scene_refine and False:
            y = rearrange(y, ' (b n) d -> b n d', b = b_size)

            #y = self.transformer_scene(y, y, env)

            #? TEST

            x = rearrange(x,' (b n) d -> b n d', b = b_size)

            y = self.transformer_scene(x, y, env)

            #y = rearrange(y,'b n d -> (b n) d')
            y = self.w_scene_refine(y)
        else:
            y = self.w3(y)
            y = self.batch_norm3(y)
            y = self.relu(y)
            y = self.dropout(y)

            if self.scene_refine: 
                y = rearrange(y, ' (b n) d -> b n d', b = b_size)
                x = rearrange(x,' (b n) d -> b n d', b = b_size)
                y = self.transformer_scene(x, y, env)
            else:
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
