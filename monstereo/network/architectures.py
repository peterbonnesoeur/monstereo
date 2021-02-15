
import torch
import torch.nn as nn
from einops import rearrange
from .transformer import Transformer as TransformerModel


#? Define the size of a scene (For Kitti : 20)
SCENE_INSTANCE_SIZE = 20

#? divide the scenes into lines (instances that are superposing themselves)
SCENE_LINE = True

#? define if we need to have an unique instance per scene
SCENE_UNIQUE = False

if SCENE_UNIQUE:
    SCENE_LINE = True

#? factor to have more or less overlap between the instances during the division of the scenes into lines
BOX_INCREASE = 0.2

class MyLinearSimple(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super().__init__()
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


class SimpleModel(nn.Module):

    def __init__(self, input_size, output_size=2, linear_size=512, p_dropout=0.2, num_stage=3,
                num_heads = 4, device='cuda', transformer = False,
                confidence = True,  lstm = False, scene_disp = False, scene_refine = False):
        super().__init__()

        self.stereo_size = input_size
        self.mono_size = int(input_size / 2)
        self.output_size = output_size - 1
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.num_heads = num_heads
        self.linear_stages = []
        self.device = device
        self.transformer = transformer
        self.lstm = lstm
        self.scene_disp = scene_disp
        self.scene_refine = scene_refine
        self.confidence = confidence

        assert (not (self.transformer and self.lstm)) , "The network cannot implement a transformer and"\
                                                        "a LSTM at the same time (are you a psycho ??)"
        # Initialize weights

        n_hidden = self.num_stage #? Number of stages for the transformer (a stage for the transfromer
        #?is the number of layer for the encoder/decoder). Of course, more layers means more calculations
        mul_output = 1 #? determine if at the end of the transformer we add a fully connected layer to go
        #?from N to mul_output*N outputs. If mul_output = 1, there will be no such fully connected layer
        n_head = self.num_heads #? the number of heads of of multi_headed attention model
        if self.confidence:
            n_inp = 3  #? the original input size for the keypoints (X,Y,C)
        else:
            n_inp = 2 #? keypoints must only contains the X and Y coordinate
        kind = "cat" #? The kind of position embedding between [cat, num, none] -> best one is cat
        #? Cat adds a complex of a sin and cos after the data of the keypoints (hence, the inputs
        #? for the transformer grows from n_inp to n_inp + 2)
        #? num adds a counter at the end of the data of the keypoints. It is a simple index going
        #? from [0, N-1] where N is the size of the sequence (hence, the inputs for the transformer grows
        #?  from n_inp to n_inp + 1).  
        
        


        length_scene_sequence = SCENE_INSTANCE_SIZE 
        #? in the case of the scene disposition (where we do not look at the keypoints but at 
        #? the sequence of keypoints in our transformer), we needed to create a padded array 
        #? of fixed size to put our instances. In this case, the instances in the sequence are the set of
        #? 2d keypoints with their confidence and flattened. This constant is defined in the train/dataset part.

        
        # Preprocessing
        if self.transformer:            
            if self.scene_disp:
                assert self.transformer, "Currently, the scene disposition method is only"\
                                        " compatible with the transformer"
                n_inp = self.stereo_size

                if kind == 'cat':
                    embed_dim = n_inp + 2
                elif kind == 'num':
                    embed_dim = n_inp +1
                else:
                    embed_dim = n_inp

                d_attention = int(embed_dim/n_head)#? The dimension of the key, query and value vector in the attention mechanism.
                                                    #? Being an embedding, its dimension should be inferior to the embed_dim
                                                    #? In the original paper of the transformer, d_attention = int(embed_dim/n_head)

                #? the input is in a format ([B, N, :]) in this case, N is the variable N_words
                n_words = SCENE_INSTANCE_SIZE

                self.transformer_scene = TransformerModel(n_base_dim = n_inp, n_target_dim = embed_dim*mul_output, n_words =  n_words, kind = kind,
                                                        embed_dim = embed_dim, d_attention =d_attention, num_heads = n_head, n_layers = n_hidden,
                                                        confidence = self.confidence, scene_disp = True)
                                                                #? The confidence flag tells us if we should take into account the confidence 
                                                                # ?for each keypoints, by design, yes
                                                                #? The scene_disp flag tells us if we are reasoning with scenes or keypoints 
                                                                #? the reordering flag is there to order the inputs in a peculiar way 
                                                                #? (in the scene case, order the instances depending on their height for example)

                self.w1 = nn.Linear(embed_dim*mul_output, self.linear_size) 

            else:
                #? Embed_dim is the embeding dimension that the transformer will effectively see. Hence, it is the dimension obtained after the embedding and position embedding step
                if kind == 'cat':
                    embed_dim = n_inp + 2 #? For an explanation, see the comment on top for the variable "kind"
                elif kind == 'num':
                    embed_dim = n_inp + 1 #? For an explanation, see the comment on top for the variable "kind"
                else:
                    embed_dim = n_inp

                d_attention = int(embed_dim/2) #? The dimesion of the key, query and value vector in the attention mechanism. Being an embedding, its dimension should be inferior to the embed_dim
                #? In the original paper of the transformer, d_attention = int(embed_dim/n_head)

                n_words = int(self.stereo_size/3) if self.confidence else int(self.stereo_size/2)
                self.transformer_kps=  TransformerModel(n_base_dim = n_inp, n_target_dim = embed_dim*mul_output, n_words = n_words ,kind = kind, embed_dim = embed_dim, 
                                                        d_attention =d_attention, num_heads = n_head, n_layers = n_hidden,confidence = self.confidence, 
                                                        scene_disp = False)
                                                                #? The confidence flag tells us if we should take into account the confidence for each keypoints, by design, yes
                                                                #? The scene_disp flag tells us if we are reasoning with scenes or keypoints 
                                                                #? the reordering flag is there to order the inputs in a peculiar way (in the scene case, order
                                                                #?  the instances depending on their height for example)

                if self.confidence:
                    self.w1 = nn.Linear(int(self.stereo_size/3*embed_dim*mul_output), self.linear_size)
                else:
                    self.w1 = nn.Linear(int(self.stereo_size/2*embed_dim*mul_output), self.linear_size)
        elif self.lstm:

            #? To benchmark our algorithm, the LSTM is also implemented (for both the scene and instance level). This is a simple bi-directional LSTM working in both the scene and keypoint situation
            
            if self.scene_disp:
                n_inp = self.stereo_size 
                #? The dimesion of the key, query and value vector in the attention mechanism. Being an embedding, its dimension should be inferior to the embed_dim
                #? In the original paper of the transformer, d_attention = int(embed_dim/n_head)
                n_words = SCENE_INSTANCE_SIZE
            else:
                n_words = int(self.stereo_size/3) if self.confidence else int(self.stereo_size/2)

                     
            if kind == 'cat':
                embed_dim = n_inp + 2

            elif kind == 'num':
                embed_dim = n_inp +1
            else:
                embed_dim = n_inp

            d_attention = int(embed_dim/2)

            bidirectional = True

            #! This transformer is only instanciated to use the same exact input encoding for both the LSTM and the transformer
            self.transformer_kps=  TransformerModel(n_base_dim = n_inp, n_target_dim = embed_dim*mul_output,  n_words = n_words,kind = kind, embed_dim = embed_dim, 
                                                            d_attention =embed_dim, num_heads = n_head, n_layers = n_hidden,confidence = self.confidence, 
                                                            scene_disp = self.scene_disp)       

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

        #!Regular Monstereo/Monoloco_pp implementation
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        # Internal loop
        for _ in range(self.num_stage):
            self.linear_stages.append(MyLinearSimple(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # Post processing
        self.w2 = nn.Linear(self.linear_size, self.linear_size)
        self.w3 = nn.Linear(self.linear_size, self.linear_size)
        self.batch_norm3 = nn.BatchNorm1d(self.linear_size)

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

        if (self.transformer or self.lstm):

            #? Call the attnetion mechanism layers
            if self.transformer:
                if self.scene_disp:
                    #? self-attention mechanism happening at the scene level
                    y = self.transformer_scene(x,x, env)
                else:
                    #? self-attention mechanism happening at the keypoint level
                    y = self.transformer_kps(x,x, env)
            else:
                #? LSTM 
                #? the keypoint encoding of our self-attention mechanism is used
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

            return y
        else:

            #! This is the regular monoloco_pp/monstereo network
            y = self.w1(x)

            y = self.batch_norm1(y)
            y = self.relu(y)
            y = self.dropout(y)

            for i in range(self.num_stage):
                y = self.linear_stages[i](y)

            # Auxiliary task

            y = self.w2(y)
            aux = self.w_aux(y)

            y = self.w3(y)
            y = self.batch_norm3(y)
            y = self.relu(y)
            y = self.dropout(y)
            y = self.w_fin(y)

            # Cat with auxiliary task
            y = torch.cat((y, aux), dim=1)
            return y

class DecisionModel(nn.Module):

    def __init__(self, input_size, output_size=2, linear_size=512, 
                p_dropout=0.2, num_stage=3, device='cuda:1'):
        super().__init__()

        self.num_stage = num_stage
        self.stereo_size = input_size
        self.mono_size = int(input_size / 2)
        self.output_size = output_size - 1
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.linear_stages_mono, self.linear_stages_stereo, self.linear_stages_dec = [], [], []
        self.device = device

        # Initialize weights

        # ------------------------Stereo----------------------------------------------
        # Preprocessing
        self.w1_stereo = nn.Linear(self.stereo_size, self.linear_size)
        self.batch_norm_stereo = nn.BatchNorm1d(self.linear_size)

        # Internal loop
        for _ in range(self.num_stage):
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
        super().__init__()

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
        super().__init__()
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
        super().__init__()

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
        super().__init__()
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
