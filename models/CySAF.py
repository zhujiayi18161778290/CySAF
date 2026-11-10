import torch
import torch.nn as nn

class RecurrentCycle(nn.Module):
    def __init__(self, cycle_len, channel_size):
        super().__init__()
        self.cycle_len = cycle_len
        self.data = nn.Parameter(torch.zeros(cycle_len, channel_size))

    def forward(self, index, length):
        gather_index = (
            index.view(-1, 1)
            + torch.arange(length, device=index.device).view(1, -1)
        ) % self.cycle_len
        return self.data[gather_index]

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.use_revin = configs.use_revin
        self.use_x_period_filter = configs.use_x_period_filter
        self.use_y_period_filter = configs.use_y_period_filter

        self.cycleQueue = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)
        self.ssf = SSF(
            self.seq_len,
            self.pred_len,
            configs.seg_window,
            configs.seg_stride,
            configs.d_model,
            configs.window_type,
        )
        self.Cycfilter =  AdaptiveThresholdFilter(
            self.pred_len,
            mask_temperature=configs.mask_temperature,
            keep_bins=configs.keep_bins,
        )

    def forward(self, x, cycle_index):
        # Instance normalization
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)
        # Decomposition and Forecasting
        x_period = self.cycleQueue(cycle_index, self.seq_len)
        if self.use_x_period_filter:
            x_period = self.Cycfilter(x_period)
        x = x - x_period
        y = self.ssf(x.permute(0, 2, 1)).permute(0, 2, 1)
        y_period = self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len,)
        if self.use_y_period_filter:
            y_period = self.Cycfilter(y_period)
        y = y + y_period
        # Inverse Instance normalization
        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean

        return y


class SSF(nn.Module):
    def __init__(self, input_len, pred_len, window_size, stride, d_model, window_type):
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.seg_num = (input_len - window_size) // stride + 1
        self.freq_len = input_len // 2 + 1

        windows = torch.zeros(self.seg_num, self.input_len)
        for i in range(self.seg_num):
            start = i * stride
            end = start + window_size

            if window_type == 'hamming':
                window = torch.hamming_window(window_size, periodic=False)
            elif window_type == 'hann':
                window = torch.hann_window(window_size, periodic=False)
            else:
                window = torch.ones(window_size)
            windows[i, start:end] = window
            
        self.register_buffer("windows", windows)
        self.time_linear = nn.Sequential(
            nn.Linear(self.input_len, d_model),
            nn.ReLU(),
            nn.Linear(d_model, self.pred_len),
        )

        scale = 0.02
        self.fr1 = nn.Parameter(scale * torch.randn(self.seg_num, self.freq_len))
        self.fi1 = nn.Parameter(scale * torch.randn(self.seg_num, self.freq_len))
        self.fbr1 = nn.Parameter(scale * torch.randn(self.seg_num, self.freq_len))
        self.fbi1 = nn.Parameter(scale * torch.randn(self.seg_num, self.freq_len))

        # Zero initialization makes the first forward pass equivalent to mean(dim=2).
        self.fusion_logits = nn.Parameter(torch.zeros(self.seg_num, self.freq_len))
    def freq_adaptive_filter(self, x):
        # x: (B, D, S, F), complex
        xr, xi = x.real, x.imag

        wr1 = self.fr1.unsqueeze(0).unsqueeze(0)
        wi1 = self.fi1.unsqueeze(0).unsqueeze(0)
        br1 = self.fbr1.unsqueeze(0).unsqueeze(0)
        bi1 = self.fbi1.unsqueeze(0).unsqueeze(0)

        h_real = xr * wr1 - xi * wi1 + br1
        h_imag = xi * wr1 + xr * wi1 + bi1
        return torch.view_as_complex(torch.stack([h_real, h_imag], dim=-1))

    def forward(self, x):
        segx = x.unsqueeze(2) * self.windows
        segw_fft = torch.fft.rfft(segx, dim=-1, norm="ortho")
        segw = segw_fft * self.freq_adaptive_filter(segw_fft)

        fusion_weight = torch.softmax(self.fusion_logits, dim=0)
        segw_all = (segw * fusion_weight.unsqueeze(0).unsqueeze(0)).sum(dim=2)

        y = torch.fft.irfft(segw_all, n=self.input_len, dim=-1, norm="ortho")
        y = self.time_linear(y)
        return y

class AdaptiveThresholdFilter(nn.Module):
    def __init__(self, pred_len, mask_temperature=16.0, keep_bins=2):
        super().__init__()
        self.pred_len = pred_len
        self.threshold_param = nn.Parameter(torch.tensor(0.5))
        self.mask_temperature = mask_temperature
        self.keep_bins = keep_bins

    def create_adaptive_high_freq_mask(self, x_fft):
        energy = x_fft.abs().pow(2).sum(dim=-1)
        median_energy = energy.median(dim=1, keepdim=True).values
        normalized_energy = energy / (median_energy + 1e-6)

        # threshold = self.threshold_param.clamp(0.0, 4.0)
        threshold = self.threshold_param
        soft_mask = torch.sigmoid((normalized_energy - threshold) * self.mask_temperature)
        adaptive_mask = (normalized_energy > threshold).float().detach() - soft_mask.detach() + soft_mask

        keep_bins = min(self.keep_bins, adaptive_mask.size(1))
        if keep_bins > 0:
            adaptive_mask = adaptive_mask.clone()
            adaptive_mask[:, :keep_bins] = 1.0

        return adaptive_mask.unsqueeze(-1)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        input_len = x.size(-1)
        x_fft = torch.fft.rfft(x).permute(0, 2, 1)
        adaptive_mask = self.create_adaptive_high_freq_mask(x_fft)
        fx = x_fft * adaptive_mask
        fx = fx.permute(0, 2, 1)
        y = torch.fft.irfft(fx, n=input_len)
        y = y.permute(0, 2, 1)
        return y
