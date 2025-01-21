
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math

from layers.StandardNorm import Normalize
from layers.Autoformer_EncDec import series_decomp


def compute_lagged_difference(x, lag=1, dim=1):
    lagged_x = torch.roll(x, shifts=lag, dims=dim)
    diff_x = x - lagged_x
    diff_x[:, :lag, :] = x[:, :lag, :]
    return diff_x
    

class PatchTimeStepAttention(nn.Module):
    def __init__(self, input_dim, patch_size=16):
        super().__init__()
        self.input_dim = input_dim
        self.patch_size = patch_size
        
        # Projections for patch-level attention
        self.patch_query = nn.Linear(input_dim, input_dim)
        self.patch_key = nn.Linear(input_dim, input_dim)
        self.patch_value = nn.Linear(input_dim, input_dim)
        
        # Projections for within-patch attention
        self.local_query = nn.Linear(input_dim, input_dim)
        self.local_key = nn.Linear(input_dim, input_dim)
        self.local_value = nn.Linear(input_dim, input_dim)
        
        # Scale factors
        self.scale = math.sqrt(input_dim)
        
    def _make_patches(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        
        # Pad sequence if needed
        if seq_len % self.patch_size != 0:
            pad_len = self.patch_size - (seq_len % self.patch_size)
            x = F.pad(x, (0, 0, 0, pad_len))
            seq_len = x.size(1)
            
        # Reshape into patches
        num_patches = seq_len // self.patch_size
        patches = x.view(batch_size, num_patches, self.patch_size, self.input_dim)
        return patches, num_patches
        
    def _patch_attention(self, patches_q, patches_k, patches_v):
        # patches shape: (batch_size, num_patches, patch_size, input_dim)
        batch_size, num_patches, patch_size, _ = patches_q.size()
        
        # Compute patch representations using mean pooling
        patch_repr_q = patches_q.mean(dim=2)  # (batch_size, num_patches, input_dim)
        patch_repr_k = patches_k.mean(dim=2)
        patch_repr_v = patches_v.mean(dim=2)

        # Compute patch-level attention
        q_patch = self.patch_query(patch_repr_q)
        k_patch = self.patch_key(patch_repr_k)
        v_patch = self.patch_value(patch_repr_v)
        
        scores = torch.matmul(q_patch, k_patch.transpose(-2, -1)) / self.scale
            
        attention_weights = torch.softmax(scores, dim=-1)
        patch_context = torch.matmul(attention_weights, v_patch)
        
        return patch_context, attention_weights
        
    def _local_attention(self, patches_q, patches_k, patches_v):
        # patches shape: (batch_size, num_patches, patch_size, input_dim)
        batch_size, num_patches, patch_size, _ = patches_q.size()
        
        # Reshape for local attention
        local_x_q = patches_q.reshape(batch_size * num_patches, patch_size, self.input_dim)
        local_x_k = patches_k.reshape(batch_size * num_patches, patch_size, self.input_dim)
        local_x_v = patches_v.reshape(batch_size * num_patches, patch_size, self.input_dim)
        
        # Compute local attention within each patch
        q_local = self.local_query(local_x_q)
        k_local = self.local_key(local_x_k)
        v_local = self.local_value(local_x_v)
        
        scores = torch.matmul(q_local, k_local.transpose(-2, -1)) / self.scale
        
        attention_weights = torch.softmax(scores, dim=-1)
        local_context = torch.matmul(attention_weights, v_local)
        
        # Reshape back
        local_context = local_context.view(batch_size, num_patches, patch_size, self.input_dim)
        return local_context, attention_weights
        
    def forward(self, x_query, x_key1, x_key2, x_val):
        # Make patches
        patches_q, num_patches = self._make_patches(x_query)
        patches_k_1, num_patches = self._make_patches(x_key1)
        patches_k_2, num_patches = self._make_patches(x_key2)
        patches_v, num_patches = self._make_patches(x_val)
        
        # Compute patch-level attention
        patch_context, patch_weights = self._patch_attention(patches_q, patches_k_1, patches_v)
        
        # Compute local attention within patches
        local_context, local_weights = self._local_attention(patches_q, patches_k_2, patches_v)
        
        # Combine patch and local attention
        # Broadcast patch context to all timesteps within each patch
        patch_context = patch_context.unsqueeze(2).expand(-1, -1, self.patch_size, -1)
        
        # Combine the two contexts (you could also use other combination strategies)
        combined_context = patch_context + local_context
        
        # Reshape back to original sequence shape
        batch_size, seq_len, _ = x_query.size()
        output = combined_context.view(batch_size, -1, self.input_dim)[:, :seq_len, :]
        
        return output, patch_weights.mean(1)
    

class Encoder(nn.Module):
    def __init__(self, configs, seq_len, pred_len):
        super(Encoder, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_dim = configs.enc_in
        self.feature_dim_output = configs.c_out
        self.channel_independence = configs.channel_independence

        self.linear_final = nn.Linear(self.seq_len, self.pred_len)

        self.temporal = nn.Sequential(
            nn.Linear(self.seq_len, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, self.seq_len),
            nn.Dropout(configs.dropout)
        )

        self.channel = nn.Sequential(
                nn.Linear(self.feature_dim, configs.d_model),
                nn.ReLU(),
                nn.Linear(configs.d_model, self.feature_dim),
                nn.Dropout(configs.dropout)
            )

    def forward(self, x_enc):

        x_temp = self.temporal(x_enc.permute(0, 2, 1)).permute(0, 2, 1)
        x_temp = torch.multiply(x_temp, compute_lagged_difference(x_enc))
        x = x_enc + x_temp

        if not self.channel_independence:
            x = x + self.channel(x_temp)
        
        return self.linear_final(x.permute(0, 2, 1)).permute(0, 2, 1)


class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.d_model = d_model
        self.pe = torch.zeros(max_len, d_model)
        
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        # Sine for even indices
        self.pe[:, 0::2] = torch.sin(position * div_term)
        
        # Cosine for odd indices
        if d_model % 2 == 0:
            # If even dimension, use full div_term
            self.pe[:, 1::2] = torch.cos(position * div_term)
        else:
            # If odd dimension, create a separate slicing for cos
            cos_div_term = div_term[:d_model // 2]
            self.pe[:, 1::2] = torch.cos(position * cos_div_term)
        
        self.pe = self.pe.unsqueeze(0)  # Add batch dimension
    
    def forward(self, x):
        # x: B x seq x F
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(x.device)
    


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.feature_dim = configs.enc_in
        self.feature_dim_output = configs.c_out
        self.d_model = configs.d_model
        self.down_sampling_layers = 3
        self.down_sampling_window = 2

        self.use_attention = True

        if self.task_name == 'anomaly_detection':
            self.pred_len = self.seq_len

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sequence_list = [1]
        current = 2
        for _ in range(1, self.down_sampling_layers+1):
            sequence_list.append(current)
            current *= 2
                
        num_scales = len(sequence_list)        
        

        patch_size = 16
        self.decomp = series_decomp(configs.moving_avg)
        self.pos_encoding = AbsolutePositionalEncoding(self.feature_dim, self.seq_len)

        self.attention = torch.nn.ModuleList([PatchTimeStepAttention(self.feature_dim, patch_size=patch_size//i)for i in sequence_list[1:]])

        self.scale_weights = nn.Parameter(torch.ones(len(sequence_list)))  # One weight per scale

        self.encoder_Seasonal = torch.nn.ModuleList([Encoder(configs, self.seq_len//i, self.pred_len) for i in sequence_list])
        self.encoder_Trend = torch.nn.ModuleList([Encoder(configs, self.seq_len//i, self.pred_len) for i in sequence_list])

        self.normalize_layer = Normalize(configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
        self.projection = nn.Linear(self.pred_len * num_scales, self.pred_len)   

    
    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        
        down_pool = torch.nn.AvgPool1d(self.down_sampling_window)
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        x_mark_enc = x_mark_sampling_list if x_mark_enc is not None else None

        return x_enc, x_mark_enc

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
                
        x_enc = self.normalize_layer(x_enc, 'norm')
        x_enc = self.pos_encoding(x_enc)

        output_list = []
        trend_list, season_list = [], []
        attn_list = []
        # ******************* SCALED INPUTS *******************************
        x_enc_list, x_mark_enc_list = self.__multi_scale_process_inputs(x_enc, x_mark_enc)
        for i, x in zip(range(len(x_enc_list)), x_enc_list):
            

            if self.use_attention and i > 0:
                tmp = F.interpolate(output_list[0].permute(0, 2, 1), size=x.shape[1], mode='linear').permute(0, 2, 1)
                # tmp_trend = F.interpolate(trend_list[0].permute(0, 2, 1), size=x.shape[1], mode='linear').permute(0, 2, 1)
                tmp_season = F.interpolate(season_list[0].permute(0, 2, 1), size=x.shape[1], mode='linear').permute(0, 2, 1)
                x, patch_weight = self.attention[i-1](x, tmp, tmp_season, x)
                attn_list.append(patch_weight)


            x_season, x_trend = self.decomp(x)
            
            seasonal_output = self.encoder_Seasonal[i](x_season)
            trend_output = self.encoder_Trend[i](x_trend)

            output = seasonal_output + trend_output
            output = output * torch.sigmoid(self.scale_weights[i])

            output_list.append(output)
            trend_list.append(trend_output)
            season_list.append(seasonal_output)

        
        output = torch.cat(output_list, dim=1)
        output = self.projection(output.permute(0,2,1)).permute(0,2,1)        

        attention_maps = []
        for i, attn in enumerate(attn_list):
            attn = attn.unsqueeze(-1) * torch.sigmoid(self.scale_weights[i+1])
            attn = F.interpolate(attn.permute(0, 2, 1), size=self.seq_len, mode='linear').permute(0, 2, 1)
            attention_maps.append(attn)


        if self.use_attention:
            self.time_step_importance = sum(attention_map / attention_map.sum(axis=1, keepdims=True) for attention_map in attention_maps).squeeze()
        else:
            self.time_step_importance = torch.ones(x_enc.shape[0], self.seq_len)

        output = self.normalize_layer(output, 'denorm')

        if self.feature_dim_output == 1:
            return output[:,:,-1].unsqueeze(-1)
        return output
    

    def anomaly_detection(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
                
        x_enc = self.normalize_layer(x_enc, 'norm')
        x_enc = self.pos_encoding(x_enc)

        output_list = []
        trend_list, season_list = [], []
        attn_list = []
        # ******************* SCALED INPUTS *******************************
        x_enc_list, x_mark_enc_list = self.__multi_scale_process_inputs(x_enc, x_mark_enc)
        for i, x in zip(range(len(x_enc_list)), x_enc_list):
            
            x, patch_weight = self.attention[i-1](x, x, x, x)
            attn_list.append(patch_weight)

            x_season, x_trend = self.decomp(x)
            
            seasonal_output = self.encoder_Seasonal[i](x_season)
            trend_output = self.encoder_Trend[i](x_trend)

            output = seasonal_output + trend_output
            output = output * torch.sigmoid(self.scale_weights[i])

            output_list.append(output)
            trend_list.append(trend_output)
            season_list.append(seasonal_output)

        
        output = torch.cat(output_list, dim=1)
        output = self.projection(output.permute(0,2,1)).permute(0,2,1)        

        attention_maps = []
        for i, attn in enumerate(attn_list):
            attn = attn.unsqueeze(-1) * torch.sigmoid(self.scale_weights[i])
            attn = F.interpolate(attn.permute(0, 2, 1), size=self.seq_len, mode='linear').permute(0, 2, 1)
            attention_maps.append(attn)


        if self.use_attention:
            self.time_step_importance = sum(attention_map / attention_map.sum(axis=1, keepdims=True) for attention_map in attention_maps).squeeze()
        else:
            self.time_step_importance = torch.ones(x_enc.shape[0], self.seq_len)

        output = self.normalize_layer(output, 'denorm')

        if self.feature_dim_output == 1:
            return output[:,:,-1].unsqueeze(-1)
        return output


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]