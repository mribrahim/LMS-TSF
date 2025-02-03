
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from layers.StandardNorm import Normalize

# https://arxiv.org/pdf/2412.06866

def compute_lagged_difference(x, lag=1):
    lagged_x = torch.roll(x, shifts=lag, dims=1)
    diff_x = x - lagged_x
    diff_x[:, :lag, :] = x[:, :lag, :]
    return diff_x


class Encoder(nn.Module):
    def __init__(self, configs, seq_len, pred_len):
        super(Encoder, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_dim = configs.enc_in
        self.channel_independence = configs.channel_independence

        self.linear_final = nn.Linear(self.seq_len, self.pred_len)

        self.temporal = nn.Sequential(
            nn.Linear(self.seq_len, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, self.seq_len),
            nn.Dropout(configs.dropout)
        )

        if not self.channel_independence:
            self.channel = nn.Sequential(
                nn.Linear(self.feature_dim, configs.d_model),
                nn.ReLU(),
                nn.Linear(configs.d_model, self.feature_dim),
                nn.Dropout(configs.dropout)
            )

    def forward(self, x_enc):

        # Temporal and channel processing
        x_temp = self.temporal(x_enc.permute(0, 2, 1)).permute(0, 2, 1)
        x_temp = torch.multiply(x_temp, compute_lagged_difference(x_enc))
        x = x_enc + x_temp

        if not self.channel_independence:
            x = x + self.channel(x_temp)
        
        return self.linear_final(x.permute(0, 2, 1)).permute(0, 2, 1)


class MSEFeedbackRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(MSEFeedbackRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)

        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # Forward pass through RNN
        out, _ = self.rnn(x, h0)
        out = self.fc(out)

        return out


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.feature_dim = configs.enc_in
        self.d_model = configs.d_model
        self.down_sampling_layers = 3
        self.down_sampling_window = 2

        if self.task_name == 'anomaly_detection':
            self.pred_len = self.seq_len
            rnn_hidden_size = self.feature_dim // 2
            rnn_output_size = self.feature_dim
            self.rnn = MSEFeedbackRNN(input_size= configs.enc_in, hidden_size=rnn_hidden_size, output_size=rnn_output_size, num_layers=8)  
            

        sequence_list = [1]
        current = 2
        for _ in range(1, self.down_sampling_layers+1):
            sequence_list.append(current)
            current *= 2
                
        num_scales = len(sequence_list)        
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cutoff_frequencies = nn.Parameter(torch.ones(num_scales, self.feature_dim, device=self.device) * torch.tensor(0.2))
        self.stepness = nn.Parameter(torch.ones(num_scales, self.feature_dim, device=self.device) * torch.tensor(10))

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


    def low_pass_filter(self, x_freq, seq_len, cutoff_frequency, stepness):
        freqs = torch.fft.fftfreq(seq_len, d=1.0).to(x_freq.device)
        mask = torch.sigmoid(-(freqs.unsqueeze(-1) - cutoff_frequency) * stepness)  # Apply different cutoff for each feature
        mask = mask.to(x_freq.device)
        x_freq_filtered = x_freq * mask
        return x_freq_filtered

    def high_pass_filter(self, x_freq, seq_len, cutoff_frequency, stepness):
        freqs = torch.fft.fftfreq(seq_len, d=1.0).to(x_freq.device)
        mask = torch.sigmoid((freqs.unsqueeze(-1) - cutoff_frequency) * stepness)  # Apply different cutoff for each feature
        mask = mask.to(x_freq.device)
        x_freq_filtered = x_freq * mask
        return x_freq_filtered
    
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
                
        x_enc = self.normalize_layer(x_enc, 'norm')

        output_list = []
        # ******************* SCALED INPUTS *******************************
        x_enc_list, x_mark_enc_list = self.__multi_scale_process_inputs(x_enc, x_mark_enc)
        for i, x in zip(range(len(x_enc_list)), x_enc_list):
            seq_len = x.shape[1]
            # Frequency domain processing
            x_freq = torch.fft.fft(x, dim=1)

            x_freq_low = self.low_pass_filter(x_freq, seq_len, self.cutoff_frequencies[i], self.stepness[i])
            x_freq_high = self.high_pass_filter(x_freq, seq_len, self.cutoff_frequencies[i], self.stepness[i])
        
            x_low = torch.fft.ifft(x_freq_low, dim=1).real
            x_high = torch.fft.ifft(x_freq_high, dim=1).real
            
            seasonal_output = self.encoder_Seasonal[i](x_high)
            trend_output = self.encoder_Trend[i](x_low)
            output = seasonal_output + trend_output

            output_list.append(output)

        
        output = torch.cat(output_list, dim=1)
        output = self.projection(output.permute(0,2,1)).permute(0,2,1)

        output = self.normalize_layer(output, 'denorm')
        return output
    

    def anomaly_detection(self, x_enc):
                
        x_enc = self.normalize_layer(x_enc, 'norm')

        output_list = []
        # ******************* SCALED INPUTS *******************************
        x_enc_list, x_mark_enc_list = self.__multi_scale_process_inputs(x_enc, x_mark_enc=None)
        for i, x in zip(range(len(x_enc_list)), x_enc_list):
            seq_len = x.shape[1]
            # Frequency domain processing
            x_freq = torch.fft.fft(x, dim=1)

            x_freq_low = self.low_pass_filter(x_freq, seq_len, self.cutoff_frequencies[i], self.stepness[i])
            x_freq_high = self.high_pass_filter(x_freq, seq_len, self.cutoff_frequencies[i], self.stepness[i])
        
            x_low = torch.fft.ifft(x_freq_low, dim=1).real
            x_high = torch.fft.ifft(x_freq_high, dim=1).real
            
            seasonal_output = self.encoder_Seasonal[i](x_high)
            trend_output = self.encoder_Trend[i](x_low)
            output = seasonal_output + trend_output

            output_list.append(output)

        
        output = torch.cat(output_list, dim=1)
        output = self.projection(output.permute(0,2,1)).permute(0,2,1)

        mse_error = ((x_enc - output) ** 2)
        combined_input = output + mse_error
        
        rnn_output = self.rnn(combined_input)
        output = output +  rnn_output.squeeze(1)

        output = self.normalize_layer(output, 'denorm')
        return output
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]