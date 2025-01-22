import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class ContextAwareModelOG(nn.Module):
    def __init__(self, weights=None, input_size=512, num_classes=3, chunk_size=240, dim_capsule=16, receptive_field=80, num_detections=5, framerate=2):
        """
        INPUT: a Tensor of the form (batch_size,1,chunk_size,input_size)
        OUTPUTS:    1. The segmentation of the form (batch_size,chunk_size,num_classes)
                    2. The action spotting of the form (batch_size,num_detections,2+num_classes)
        """

        super(ContextAwareModelOG, self).__init__()

        self.load_weights(weights=weights)

        self.input_size = input_size
        self.num_classes = num_classes
        self.dim_capsule = dim_capsule
        self.receptive_field = receptive_field
        self.num_detections = num_detections
        self.chunk_size = chunk_size
        self.framerate = framerate

        self.pyramid_size_1 = int(np.ceil(receptive_field/7))
        self.pyramid_size_2 = int(np.ceil(receptive_field/3))
        self.pyramid_size_3 = int(np.ceil(receptive_field/2))
        self.pyramid_size_4 = int(np.ceil(receptive_field))

        # Base Convolutional Layers
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1,input_size))
        self.conv_2 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1,1))

        # Temporal Pyramidal Module
        self.pad_p_1 = nn.ZeroPad2d((0,0,(self.pyramid_size_1-1)//2, self.pyramid_size_1-1-(self.pyramid_size_1-1)//2))
        self.pad_p_2 = nn.ZeroPad2d((0,0,(self.pyramid_size_2-1)//2, self.pyramid_size_2-1-(self.pyramid_size_2-1)//2))
        self.pad_p_3 = nn.ZeroPad2d((0,0,(self.pyramid_size_3-1)//2, self.pyramid_size_3-1-(self.pyramid_size_3-1)//2))
        self.pad_p_4 = nn.ZeroPad2d((0,0,(self.pyramid_size_4-1)//2, self.pyramid_size_4-1-(self.pyramid_size_4-1)//2))
        self.conv_p_1 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(self.pyramid_size_1,1))
        self.conv_p_2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(self.pyramid_size_2,1))
        self.conv_p_3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(self.pyramid_size_3,1))
        self.conv_p_4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(self.pyramid_size_4,1))

        # -------------------
        # Segmentation module
        # -------------------

        self.kernel_seg_size = 3
        self.pad_seg = nn.ZeroPad2d((0,0,(self.kernel_seg_size-1)//2, self.kernel_seg_size-1-(self.kernel_seg_size-1)//2))
        self.conv_seg = nn.Conv2d(in_channels=152, out_channels=dim_capsule*num_classes, kernel_size=(self.kernel_seg_size,1))
        self.batch_seg = nn.BatchNorm2d(num_features=self.chunk_size, momentum=0.01,eps=0.001) 


        # -------------------
        # detection module
        # -------------------       
        self.max_pool_spot = nn.MaxPool2d(kernel_size=(3,1),stride=(2,1))
        self.kernel_spot_size = 3
        self.pad_spot_1 = nn.ZeroPad2d((0,0,(self.kernel_spot_size-1)//2, self.kernel_spot_size-1-(self.kernel_spot_size-1)//2))
        self.conv_spot_1 = nn.Conv2d(in_channels=num_classes*(dim_capsule+1), out_channels=32, kernel_size=(self.kernel_spot_size,1))
        self.max_pool_spot_1 = nn.MaxPool2d(kernel_size=(3,1),stride=(2,1))
        self.pad_spot_2 = nn.ZeroPad2d((0,0,(self.kernel_spot_size-1)//2, self.kernel_spot_size-1-(self.kernel_spot_size-1)//2))
        self.conv_spot_2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(self.kernel_spot_size,1))
        self.max_pool_spot_2 = nn.MaxPool2d(kernel_size=(3,1),stride=(2,1))

        # Confidence branch
        self.conv_conf = nn.Conv2d(in_channels=16*(chunk_size//8-1), out_channels=self.num_detections*2, kernel_size=(1,1))

        # Class branch
        self.conv_class = nn.Conv2d(in_channels=16*(chunk_size//8-1), out_channels=self.num_detections*self.num_classes, kernel_size=(1,1))
        self.softmax = nn.Softmax(dim=-1)


    def load_weights(self, weights=None):
        if(weights is not None):
            print("=> loading checkpoint '{}'".format(weights))
            checkpoint = torch.load(weights)
            self.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(weights, checkpoint['epoch']))

    def forward(self, inputs):

        # -----------------------------------
        # Feature input (chunks of the video)
        # -----------------------------------
        # input_shape: (batch,channel,frames,dim_features)
        #print("Input size: ", inputs.size())

        # -------------------------------------
        # Temporal Convolutional neural network
        # -------------------------------------


        # Base Convolutional Layers
        conv_1 = F.relu(self.conv_1(inputs))
        #print("Conv_1 size: ", conv_1.size())
        
        conv_2 = F.relu(self.conv_2(conv_1))
        #print("Conv_2 size: ", conv_2.size())


        # Temporal Pyramidal Module
        conv_p_1 = F.relu(self.conv_p_1(self.pad_p_1(conv_2)))
        #print("Conv_p_1 size: ", conv_p_1.size())
        conv_p_2 = F.relu(self.conv_p_2(self.pad_p_2(conv_2)))
        #print("Conv_p_2 size: ", conv_p_2.size())
        conv_p_3 = F.relu(self.conv_p_3(self.pad_p_3(conv_2)))
        #print("Conv_p_3 size: ", conv_p_3.size())
        conv_p_4 = F.relu(self.conv_p_4(self.pad_p_4(conv_2)))
        #print("Conv_p_4 size: ", conv_p_4.size())

        concatenation = torch.cat((conv_2,conv_p_1,conv_p_2,conv_p_3,conv_p_4),1)
        #print("Concatenation size: ", concatenation.size())


        # -------------------
        # Segmentation module
        # -------------------

        conv_seg = self.conv_seg(self.pad_seg(concatenation))
        #print("Conv_seg size: ", conv_seg.size())

        conv_seg_permuted = conv_seg.permute(0,2,3,1)
        #print("Conv_seg_permuted size: ", conv_seg_permuted.size())

        conv_seg_reshaped = conv_seg_permuted.view(conv_seg_permuted.size()[0],conv_seg_permuted.size()[1],self.dim_capsule,self.num_classes)
        #print("Conv_seg_reshaped size: ", conv_seg_reshaped.size())


        #conv_seg_reshaped_permuted = conv_seg_reshaped.permute(0,3,1,2)
        #print("Conv_seg_reshaped_permuted size: ", conv_seg_reshaped_permuted.size())

        conv_seg_norm = torch.sigmoid(self.batch_seg(conv_seg_reshaped))
        #print("Conv_seg_norm: ", conv_seg_norm.size())


        #conv_seg_norm_permuted = conv_seg_norm.permute(0,2,3,1)
        #print("Conv_seg_norm_permuted size: ", conv_seg_norm_permuted.size())

        output_segmentation = torch.sqrt(torch.sum(torch.square(conv_seg_norm-0.5), dim=2)*4/self.dim_capsule)
        #print("Output_segmentation size: ", output_segmentation.size())


        # ---------------
        # Spotting module
        # ---------------

        # Concatenation of the segmentation score to the capsules
        output_segmentation_reverse = 1-output_segmentation
        #print("Output_segmentation_reverse size: ", output_segmentation_reverse.size())

        output_segmentation_reverse_reshaped = output_segmentation_reverse.unsqueeze(2)
        #print("Output_segmentation_reverse_reshaped size: ", output_segmentation_reverse_reshaped.size())


        output_segmentation_reverse_reshaped_permutted = output_segmentation_reverse_reshaped.permute(0,3,1,2)
        #print("Output_segmentation_reverse_reshaped_permutted size: ", output_segmentation_reverse_reshaped_permutted.size())

        concatenation_2 = torch.cat((conv_seg, output_segmentation_reverse_reshaped_permutted), dim=1)
        #print("Concatenation_2 size: ", concatenation_2.size())

        conv_spot = self.max_pool_spot(F.relu(concatenation_2))
        #print("Conv_spot size: ", conv_spot.size())

        conv_spot_1 = F.relu(self.conv_spot_1(self.pad_spot_1(conv_spot)))
        #print("Conv_spot_1 size: ", conv_spot_1.size())

        conv_spot_1_pooled = self.max_pool_spot_1(conv_spot_1)
        #print("Conv_spot_1_pooled size: ", conv_spot_1_pooled.size())

        conv_spot_2 = F.relu(self.conv_spot_2(self.pad_spot_2(conv_spot_1_pooled)))
        #print("Conv_spot_2 size: ", conv_spot_2.size())

        conv_spot_2_pooled = self.max_pool_spot_2(conv_spot_2)
        #print("Conv_spot_2_pooled size: ", conv_spot_2_pooled.size())

        spotting_reshaped = conv_spot_2_pooled.view(conv_spot_2_pooled.size()[0],-1,1,1)
        #print("Spotting_reshape size: ", spotting_reshaped.size())

        # Confindence branch
        conf_pred = torch.sigmoid(self.conv_conf(spotting_reshaped).view(spotting_reshaped.shape[0],self.num_detections,2))
        #print("Conf_pred size: ", conf_pred.size())

        # Class branch
        conf_class = self.softmax(self.conv_class(spotting_reshaped).view(spotting_reshaped.shape[0],self.num_detections,self.num_classes))
        #print("Conf_class size: ", conf_class.size())

        output_spotting = torch.cat((conf_pred,conf_class),dim=-1)
        #print("Output_spotting size: ", output_spotting.size())


        return output_segmentation, output_spotting

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Global pooling over spatial/temporal dimensions
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch, channels, _, _ = x.size()
        squeeze = self.global_avg_pool(x).view(batch, channels)
        excitation = self.fc2(F.relu(self.fc1(squeeze)))
        excitation = self.sigmoid(excitation).view(batch, channels, 1, 1)
        return x + (x * excitation)

class ContextAwareModelSeb(nn.Module):
    def __init__(self, weights=None, input_size=512, num_classes=3, chunk_size=240, dim_capsule=16,
                 receptive_field=80, num_detections=5, framerate=2):
        super(ContextAwareModelSeb, self).__init__()

        self.load_weights(weights=weights)

        self.input_size = input_size
        self.num_classes = num_classes
        self.dim_capsule = dim_capsule
        self.receptive_field = receptive_field
        self.num_detections = num_detections
        self.chunk_size = chunk_size
        self.framerate = framerate

        self.pyramid_size_1 = int(np.ceil(receptive_field / 7))
        self.pyramid_size_2 = int(np.ceil(receptive_field / 3))
        self.pyramid_size_3 = int(np.ceil(receptive_field / 2))
        self.pyramid_size_4 = int(np.ceil(receptive_field))

        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1, input_size))
        self.conv_2 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1))

        self.pad_p_1 = nn.ZeroPad2d((0, 0, (self.pyramid_size_1 - 1) // 2, self.pyramid_size_1 - 1 - (self.pyramid_size_1 - 1) // 2))
        self.pad_p_2 = nn.ZeroPad2d((0, 0, (self.pyramid_size_2 - 1) // 2, self.pyramid_size_2 - 1 - (self.pyramid_size_2 - 1) // 2))
        self.pad_p_3 = nn.ZeroPad2d((0, 0, (self.pyramid_size_3 - 1) // 2, self.pyramid_size_3 - 1 - (self.pyramid_size_3 - 1) // 2))
        self.pad_p_4 = nn.ZeroPad2d((0, 0, (self.pyramid_size_4 - 1) // 2, self.pyramid_size_4 - 1 - (self.pyramid_size_4 - 1) // 2))
        
        self.conv_p_1 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(self.pyramid_size_1, 1), groups=8)
        self.conv_p_2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(self.pyramid_size_2, 1), groups=16)
        self.conv_p_3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(self.pyramid_size_3, 1), groups=32)
        self.conv_p_4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(self.pyramid_size_4, 1), groups=32)
        
        # self.conv_p_1 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(self.pyramid_size_1, 1))
        # self.conv_p_2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(self.pyramid_size_2, 1))
        # self.conv_p_3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(self.pyramid_size_3, 1))
        # self.conv_p_4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(self.pyramid_size_4, 1))

        # SE Blocks for Attention
        self.se_p_1 = SEBlock(in_channels=8)
        self.se_p_2 = SEBlock(in_channels=16)
        self.se_p_3 = SEBlock(in_channels=32)
        self.se_p_4 = SEBlock(in_channels=64)
        
        # self.ta_p_1 = TemporalAttention(in_channels=8, chunk_size=self.chunk_size)
        # self.ta_p_2 = TemporalAttention(in_channels=16, chunk_size=self.chunk_size)
        # self.ta_p_3 = TemporalAttention(in_channels=32, chunk_size=self.chunk_size)
        # self.ta_p_4 = TemporalAttention(in_channels=64, chunk_size=self.chunk_size)
        
        self.dropout_seg = nn.Dropout(p=0.3)
        self.dropout_spot = nn.Dropout(p=0.3)

        self.conv_seg_bn = nn.BatchNorm2d(num_features=dim_capsule * num_classes)

        # Segmentation Module
        self.kernel_seg_size = 3
        self.pad_seg = nn.ZeroPad2d((0, 0, (self.kernel_seg_size - 1) // 2, self.kernel_seg_size - 1 - (self.kernel_seg_size - 1) // 2))
        self.conv_seg = nn.Conv2d(in_channels=152, out_channels=dim_capsule * num_classes, kernel_size=(self.kernel_seg_size, 1))
        self.batch_seg = nn.BatchNorm2d(num_features=self.chunk_size, momentum=0.01, eps=0.001)
        self.temporal_refinement_module = TemporalRefinementModule(num_classes=self.num_classes, kernel_size=5)

        # Detection Module
        self.max_pool_spot = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1))
        self.kernel_spot_size = 3
        self.pad_spot_1 = nn.ZeroPad2d((0, 0, (self.kernel_spot_size - 1) // 2, self.kernel_spot_size - 1 - (self.kernel_spot_size - 1) // 2))
        self.conv_spot_1 = nn.Conv2d(in_channels=num_classes * (dim_capsule + 1), out_channels=32, kernel_size=(self.kernel_spot_size, 1))
        self.max_pool_spot_1 = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1))
        self.pad_spot_2 = nn.ZeroPad2d((0, 0, (self.kernel_spot_size - 1) // 2, self.kernel_spot_size - 1 - (self.kernel_spot_size - 1) // 2))
        self.conv_spot_2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(self.kernel_spot_size, 1))
        self.max_pool_spot_2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1))

        # Confidence Branch
        self.conv_conf = nn.Conv2d(in_channels=16 * (chunk_size // 8 - 1), out_channels=self.num_detections * 2, kernel_size=(1, 1))

        # Class Branch
        self.conv_class = nn.Conv2d(in_channels=16 * (chunk_size // 8 - 1), out_channels=self.num_detections * self.num_classes, kernel_size=(1, 1))
        self.softmax = nn.Softmax(dim=-1)

    def load_weights(self, weights=None):
        if weights is not None:
            print(f"=> loading checkpoint '{weights}'")
            checkpoint = torch.load(weights)
            self.load_state_dict(checkpoint['state_dict'])
            print(f"=> loaded checkpoint '{weights}' (epoch {checkpoint['epoch']})")
    
    def forward(self, inputs):
        conv_1 = F.relu(self.conv_1(inputs))
        conv_2 = F.relu(self.conv_2(conv_1))

        conv_p_1 = self.se_p_1(F.relu(self.conv_p_1(self.pad_p_1(conv_2))))
        conv_p_2 = self.se_p_2(F.relu(self.conv_p_2(self.pad_p_2(conv_2))))
        conv_p_3 = self.se_p_3(F.relu(self.conv_p_3(self.pad_p_3(conv_2))))
        conv_p_4 = self.se_p_4(F.relu(self.conv_p_4(self.pad_p_4(conv_2))))
        concatenation = torch.cat((conv_2, conv_p_1, conv_p_2, conv_p_3, conv_p_4), 1)

        # Segmentation Module
        conv_seg = self.conv_seg(self.pad_seg(concatenation))
        # conv_seg = self.conv_seg_bn(conv_seg)
        # conv_seg = self.dropout_seg(conv_seg)
        conv_seg_permuted = conv_seg.permute(0, 2, 3, 1)
        conv_seg_reshaped = conv_seg_permuted.view(conv_seg_permuted.size()[0], conv_seg_permuted.size()[1], self.dim_capsule, self.num_classes)
        conv_seg_norm = torch.sigmoid(self.batch_seg(conv_seg_reshaped))
        output_segmentation = torch.sqrt(torch.sum(torch.square(conv_seg_norm - 0.5), dim=2) * 4 / self.dim_capsule)
        output_segmentation = self.temporal_refinement_module(output_segmentation)

        # Spotting Module
        output_segmentation_reverse = 1 - output_segmentation
        output_segmentation_reverse_reshaped = output_segmentation_reverse.unsqueeze(2).permute(0, 3, 1, 2)
        concatenation_2 = torch.cat((conv_seg, output_segmentation_reverse_reshaped), dim=1)
        conv_spot = self.max_pool_spot(F.relu(concatenation_2))
        # conv_spot = self.dropout_spot(conv_spot)  # Add Dropout
        conv_spot_1 = F.relu(self.conv_spot_1(self.pad_spot_1(conv_spot)))
        conv_spot_1_pooled = self.max_pool_spot_1(conv_spot_1)
        conv_spot_2 = F.relu(self.conv_spot_2(self.pad_spot_2(conv_spot_1_pooled)))
        conv_spot_2_pooled = self.max_pool_spot_2(conv_spot_2)
        spotting_reshaped = conv_spot_2_pooled.view(conv_spot_2_pooled.size()[0], -1, 1, 1)

        conf_pred = torch.sigmoid(self.conv_conf(spotting_reshaped).view(spotting_reshaped.shape[0], self.num_detections, 2))

        conf_class = self.softmax(self.conv_class(spotting_reshaped).view(spotting_reshaped.shape[0], self.num_detections, self.num_classes))
        output_spotting = torch.cat((conf_pred, conf_class), dim=-1)

        return output_segmentation, output_spotting

class FixedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(FixedPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe) 

    def forward(self, x):
        return x + self.pe[:, :x.size(1)] 
        
class ContextAwareModelTran(nn.Module):
    def __init__(self, weights=None, input_size=512, num_classes=3, chunk_size=240, dim_capsule=16, receptive_field=80, num_detections=5, framerate=2):
        super(ContextAwareModelTran, self).__init__()

        self.load_weights(weights=weights)

        self.input_size = input_size
        self.num_classes = num_classes
        self.dim_capsule = dim_capsule
        self.receptive_field = receptive_field
        self.num_detections = num_detections
        self.chunk_size = chunk_size
        self.framerate = framerate

        self.transformer_layer = TransformerEncoderLayer(
            d_model=input_size,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1, 
            batch_first = True
        )
        self.transformer_encoder = TransformerEncoder(self.transformer_layer, num_layers=2)

        # segmentation Module
        self.conv_seg = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size,  
                out_channels=dim_capsule * num_classes,
                kernel_size=3, 
                padding=1       
            ),
            nn.ReLU(),
            nn.BatchNorm1d(dim_capsule * num_classes)  # Normalize output
        )

        # detection module
        self.max_pool_spot = nn.MaxPool2d(kernel_size=(3,1),stride=(2,1))
        self.kernel_spot_size = 3
        self.pad_spot_1 = nn.ZeroPad2d((0,0,(self.kernel_spot_size-1)//2, self.kernel_spot_size-1-(self.kernel_spot_size-1)//2))
        self.conv_spot_1 = nn.Conv2d(in_channels=num_classes*(dim_capsule+1), out_channels=32, kernel_size=(self.kernel_spot_size,1))
        self.max_pool_spot_1 = nn.MaxPool2d(kernel_size=(3,1),stride=(2,1))
        self.pad_spot_2 = nn.ZeroPad2d((0,0,(self.kernel_spot_size-1)//2, self.kernel_spot_size-1-(self.kernel_spot_size-1)//2))
        self.conv_spot_2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(self.kernel_spot_size,1))
        self.max_pool_spot_2 = nn.MaxPool2d(kernel_size=(3,1),stride=(2,1))

        self.conv_conf = nn.Conv2d(in_channels=16*(chunk_size//8-1), out_channels=self.num_detections*2, kernel_size=(1,1))

        self.conv_class = nn.Conv2d(in_channels=16*(chunk_size//8-1), out_channels=self.num_detections*self.num_classes, kernel_size=(1,1))
        self.softmax = nn.Softmax(dim=-1)


    def load_weights(self, weights=None):
        if(weights is not None):
            print("=> loading checkpoint '{}'".format(weights))
            checkpoint = torch.load(weights)
            self.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(weights, checkpoint['epoch']))

    def forward(self, inputs):
        inputs_flattened = inputs.squeeze(1)  # (batch, chunk_size, feature_dim)

        transformed = self.transformer_encoder(inputs_flattened)  # (batch, chunk_size, feature_dim)

        transformed = transformed.permute(0, 2, 1)  # (batch, feature_dim, chunk_size)

        # Segmentation module
        conv_seg = self.conv_seg(transformed)  # (batch, dim_capsule * num_classes, chunk_size)

        conv_seg_permuted = conv_seg.permute(0, 2, 1)  # (batch, chunk_size, dim_capsule * num_classes)
        conv_seg_reshaped = conv_seg_permuted.view(
            conv_seg_permuted.size(0),
            conv_seg_permuted.size(1),
            self.dim_capsule,
            self.num_classes
        )  # (batch, chunk_size, dim_capsule, num_classes)

        conv_seg_norm = torch.sigmoid(conv_seg_reshaped)  # Normalize the capsules
        output_segmentation = torch.sqrt(torch.sum(torch.square(conv_seg_norm - 0.5), dim=2) * 4 / self.dim_capsule)

        # Spotting module
        output_segmentation_reverse = 1 - output_segmentation
        output_segmentation_reverse_reshaped = output_segmentation_reverse.unsqueeze(2)
        output_segmentation_reverse_reshaped_permutted = output_segmentation_reverse_reshaped.permute(0, 3, 1, 2)

        concatenation_2 = torch.cat((conv_seg.unsqueeze(-1), output_segmentation_reverse_reshaped_permutted), dim=1)
        conv_spot = self.max_pool_spot(F.relu(concatenation_2))

        conv_spot_1 = F.relu(self.conv_spot_1(self.pad_spot_1(conv_spot)))
        conv_spot_1_pooled = self.max_pool_spot_1(conv_spot_1)
        conv_spot_2 = F.relu(self.conv_spot_2(self.pad_spot_2(conv_spot_1_pooled)))
        conv_spot_2_pooled = self.max_pool_spot_2(conv_spot_2)

        spotting_reshaped = conv_spot_2_pooled.view(conv_spot_2_pooled.size(0), -1, 1, 1)
        conf_pred = torch.sigmoid(self.conv_conf(spotting_reshaped).view(spotting_reshaped.shape[0], self.num_detections, 2))
        conf_class = self.softmax(self.conv_class(spotting_reshaped).view(spotting_reshaped.shape[0], self.num_detections, self.num_classes))

        output_spotting = torch.cat((conf_pred, conf_class), dim=-1)

        return output_segmentation, output_spotting
        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe[:, :x.size(1), :].to(x.device)
        return x + pe

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, input_dim)
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, chunk_size, channels)

        attn_output, _ = self.self_attn(x, x, x)  # (batch_size, chunk_size, channels)
        x = self.norm1(x + self.dropout(attn_output))

        ffn_output = self.ffn(x)  # (batch_size, chunk_size, channels)
        x = self.norm2(x + self.dropout(ffn_output))

        return x

class TemporalRefinementModule(nn.Module):
    def __init__(self, num_classes, kernel_size=5):
        super(TemporalRefinementModule, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=num_classes, 
                                out_channels=num_classes, 
                                kernel_size=kernel_size,
                                padding=kernel_size // 2)

    def forward(self, x):
        # x - (batch, chunk_size, num_classes)
        x = x.permute(0, 2, 1) #(batch, num_classes, chunk_size)
        x = self.conv1d(x)  # (batch, num_classes, chunk_size)
        x = x.permute(0, 2, 1) #(batch, chunk_size, num_classes)
        
        x = torch.sigmoid(x)
        return x

class ContextAwareModelMHA(nn.Module):
    def __init__(self, weights=None, input_size=512, num_classes=3, chunk_size=240, dim_capsule=16, 
                 receptive_field=80, num_detections=5, framerate=2, transformer_heads=4, transformer_ff_dim=512):
        super(ContextAwareModelMHA, self).__init__()

        self.load_weights(weights=weights)
        self.input_size = input_size
        self.num_classes = num_classes
        self.dim_capsule = dim_capsule
        self.receptive_field = receptive_field
        self.num_detections = num_detections
        self.chunk_size = chunk_size
        self.framerate = framerate
        print("NUM HEADS:", transformer_heads)

        self.pyramid_size_1 = int(np.ceil(receptive_field/7))
        self.pyramid_size_2 = int(np.ceil(receptive_field/3))
        self.pyramid_size_3 = int(np.ceil(receptive_field/2))
        self.pyramid_size_4 = int(np.ceil(receptive_field))

        # Base Convolutional Layers
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1, input_size))
        self.conv_2 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1))

        # Temporal Pyramidal Module
        self.pad_p_1 = nn.ZeroPad2d((0, 0, (self.pyramid_size_1-1)//2, self.pyramid_size_1-1-(self.pyramid_size_1-1)//2))
        self.pad_p_2 = nn.ZeroPad2d((0, 0, (self.pyramid_size_2-1)//2, self.pyramid_size_2-1-(self.pyramid_size_2-1)//2))
        self.pad_p_3 = nn.ZeroPad2d((0, 0, (self.pyramid_size_3-1)//2, self.pyramid_size_3-1-(self.pyramid_size_3-1)//2))
        self.pad_p_4 = nn.ZeroPad2d((0, 0, (self.pyramid_size_4-1)//2, self.pyramid_size_4-1-(self.pyramid_size_4-1)//2))
        self.conv_p_1 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(self.pyramid_size_1, 1))
        self.conv_p_2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(self.pyramid_size_2, 1))
        self.conv_p_3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(self.pyramid_size_3, 1))
        self.conv_p_4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(self.pyramid_size_4, 1))

        self.positional_encoding = PositionalEncoding(d_model=152)
        self.transformer = TransformerEncoder(input_dim=152, num_heads=transformer_heads, ff_dim=transformer_ff_dim)
        
        # Segmentation Module
        self.kernel_seg_size = 3
        self.pad_seg = nn.ZeroPad2d((0, 0, (self.kernel_seg_size-1)//2, self.kernel_seg_size-1-(self.kernel_seg_size-1)//2))
        self.conv_seg = nn.Conv2d(in_channels=152, out_channels=dim_capsule * num_classes, kernel_size=(self.kernel_seg_size, 1))
        self.batch_seg = nn.BatchNorm2d(num_features=self.chunk_size, momentum=0.01, eps=0.001)
        self.temporal_refinement_module = TemporalRefinementModule(num_classes=self.num_classes, kernel_size=5)

        # Detection Module
        self.max_pool_spot = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1))
        self.kernel_spot_size = 3
        self.pad_spot_1 = nn.ZeroPad2d((0, 0, (self.kernel_spot_size-1)//2, self.kernel_spot_size-1-(self.kernel_spot_size-1)//2))
        self.conv_spot_1 = nn.Conv2d(in_channels=num_classes * (dim_capsule + 1), out_channels=32, kernel_size=(self.kernel_spot_size, 1))
        self.max_pool_spot_1 = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1))
        self.pad_spot_2 = nn.ZeroPad2d((0, 0, (self.kernel_spot_size-1)//2, self.kernel_spot_size-1-(self.kernel_spot_size-1)//2))
        self.conv_spot_2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(self.kernel_spot_size, 1))
        self.max_pool_spot_2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1))
        self.conv_conf = nn.Conv2d(in_channels=16 * (chunk_size // 8 - 1), out_channels=self.num_detections * 2, kernel_size=(1, 1))
        self.conv_class = nn.Conv2d(in_channels=16 * (chunk_size // 8 - 1), out_channels=self.num_detections * self.num_classes, kernel_size=(1, 1))
        self.softmax = nn.Softmax(dim=-1)

    def load_weights(self, weights=None):
        if weights is not None:
            checkpoint = torch.load(weights)
            self.load_state_dict(checkpoint['state_dict'])

    def forward(self, inputs):
        conv_1 = F.relu(self.conv_1(inputs))
        conv_2 = F.relu(self.conv_2(conv_1))

        conv_p_1 = F.relu(self.conv_p_1(self.pad_p_1(conv_2)))
        conv_p_2 = F.relu(self.conv_p_2(self.pad_p_2(conv_2)))
        conv_p_3 = F.relu(self.conv_p_3(self.pad_p_3(conv_2)))
        conv_p_4 = F.relu(self.conv_p_4(self.pad_p_4(conv_2)))

        concatenation = torch.cat((conv_2, conv_p_1, conv_p_2, conv_p_3, conv_p_4), 1)  # (batch, channels, chunk_size, 1)

        transformer_input = concatenation.squeeze(-1).permute(0, 2, 1)  # (batch, chunk_size, channels)
        transformer_input = self.positional_encoding(transformer_input) 
        transformer_output = self.transformer(transformer_input)  # (batch, chunk_size, channels)
        
        transformer_output = transformer_output.permute(0, 2, 1).unsqueeze(-1)  # (batch, channels, chunk_size, 1)

        # Segmentation Module
        conv_seg = self.conv_seg(self.pad_seg(transformer_output))
        conv_seg_permuted = conv_seg.permute(0, 2, 3, 1)
        conv_seg_reshaped = conv_seg_permuted.view(conv_seg_permuted.size(0), conv_seg_permuted.size(1), self.dim_capsule, self.num_classes)
        conv_seg_norm = torch.sigmoid(self.batch_seg(conv_seg_reshaped))
        output_segmentation = torch.sqrt(torch.sum(torch.square(conv_seg_norm - 0.5), dim=2) * 4 / self.dim_capsule)
        output_segmentation = self.temporal_refinement_module(output_segmentation)

        # Detection Module
        output_segmentation_reverse = 1 - output_segmentation
        output_segmentation_reverse_reshaped = output_segmentation_reverse.unsqueeze(2).permute(0, 3, 1, 2)
        concatenation_2 = torch.cat((conv_seg, output_segmentation_reverse_reshaped), dim=1)
        conv_spot = self.max_pool_spot(F.relu(concatenation_2))
        conv_spot_1 = F.relu(self.conv_spot_1(self.pad_spot_1(conv_spot)))
        conv_spot_1_pooled = self.max_pool_spot_1(conv_spot_1)
        conv_spot_2 = F.relu(self.conv_spot_2(self.pad_spot_2(conv_spot_1_pooled)))
        conv_spot_2_pooled = self.max_pool_spot_2(conv_spot_2)
        spotting_reshaped = conv_spot_2_pooled.view(conv_spot_2_pooled.size(0), -1, 1, 1)
        conf_pred = torch.sigmoid(self.conv_conf(spotting_reshaped).view(spotting_reshaped.shape[0], self.num_detections, 2))
        conf_class = self.softmax(self.conv_class(spotting_reshaped).view(spotting_reshaped.shape[0], self.num_detections, self.num_classes))
        output_spotting = torch.cat((conf_pred, conf_class), dim=-1)

        return output_segmentation, output_spotting