import torch
import torch.nn as nn
import math

device = torch.device('cuda')

def get_activation(activation_name='swish', init_alpha=1.0):
    if activation_name == 'swish':
        return Swish()
    if activation_name == 'tanh':
        return nn.Tanh()
    if activation_name =='relu':
        return nn.relu()
    if activation_name =='GELU':
        return nn.GELU(approximate='tanh')
    if activation_name =='sigmoid':
        return nn.sigmoid()
    if activation_name == 'adap_swish':
        return AdaptiveSwish()
    if activation_name=='adap_tanh':
        return AdaptiveTanh(init_alpha)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class AdaptiveSwish(nn.Module):
    def __init__(self):
        super(AdaptiveSwish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))  # Learnable parameter
    
    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

class AdaptiveTanh(nn.Module):
    def __init__(self, init_alpha=1.0):
        super(AdaptiveTanh, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(init_alpha))

    def forward(self, x):
        return torch.tanh(self.alpha * x)

class SIRENLayer(nn.Module):
    def __init__(self, in_features, out_features, w0=1.0, is_first=False, h_siren=False, learning_w0=False):
        super().__init__()
        self.in_features = in_features
        self.is_first = is_first
        self.h_siren = h_siren
        if learning_w0:
            self.w0 = nn.Parameter(torch.tensor(float(w0)))
            print("adaptive SIREN is TRUE")
        else:
            self.w0 = w0

        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                bound = math.sqrt(6 / self.in_features) / self.w0
                self.linear.weight.uniform_(-bound, bound)    
    def forward(self, x):
        if self.h_siren:
            return torch.sin(torch.sinh(self.w0 * self.linear(x)))
        else:    
            return torch.sin(self.w0 * self.linear(x))

class layer(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super(layer,self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            # nn.BatchNorm1d(out_channels),
            activation
        )
    def forward(self, x):
        return self.layer(x)

class FourierFeatureLayer(nn.Module):
    def __init__(self, in_channels, num_frequencies=4):
        super().__init__()
        self.in_channels = in_channels
        self.num_frequencies = num_frequencies
        self.register_buffer(
            "freq_bands",
            torch.arange(1, num_frequencies + 1, dtype=torch.float32) * math.pi
        )

    def forward(self, x):
        x_proj = x.unsqueeze(-1) * self.freq_bands  # (B, C, F)
        sin_features = torch.sin(x_proj)
        cos_features = torch.cos(x_proj)
        sin_flat = sin_features.reshape(x.shape[0], -1)
        cos_flat = cos_features.reshape(x.shape[0], -1)
        return torch.cat([x, sin_flat, cos_flat], dim=-1)

class RandomFourierFeatureLayer(nn.Module):
    def __init__(self, in_channels, num_features=16, scale=1.0):
        """
        in_channels: number of input dimensions
        num_features: number of random Fourier features per input dimension
        scale: standard deviation of the Gaussian used for frequency sampling
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_features = num_features

        # Random Gaussian frequencies
        # Shape: (in_channels, num_features)
        B = torch.randn(in_channels, num_features) * scale * math.pi
        self.register_buffer("B", B)

    def forward(self, x):
        """
        x: (B, in_channels)
        returns: (B, in_channels * 2 * num_features)
        """
        # x: (B, C) -> (B, C, 1)
        x_proj = x.unsqueeze(-1) * self.B  # (B, C, F)
        sin_features = torch.sin(x_proj)
        cos_features = torch.cos(x_proj)

        # Flatten last two dims: (B, C*F)
        sin_flat = sin_features.reshape(x.shape[0], -1)
        cos_flat = cos_features.reshape(x.shape[0], -1)

        # Concatenate sin and cos (original x can be added optionally)
        return torch.cat([x, sin_flat, cos_flat], dim=-1)

class NN(nn.Module):
    def __init__(self, in_c, out_c, features = [100,100,100,100], activation_name='tanh', num_frequencies=5, fourier_type='random',
    fourier_scale=1.0,use_siren=False, siren_w0=30.0, h_siren=False, learning_w0=False, init_alpha=1.0):
        super(NN,self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.features = features
        self.activation_name = activation_name
        # self.activation = get_activation(activation_name=self.activation_name)
        self.use_siren = use_siren
        self.h_siren = h_siren
        self.learning_w0 = learning_w0
        self.init_alpha = init_alpha
        
        fourier_in_c = 2 + (in_c - 1)
        # Fourier feature layer
        if fourier_type == 'linear':
            self.fourier_layer = FourierFeatureLayer(fourier_in_c, num_frequencies)
            fourier_out_dim = fourier_in_c + 2*num_frequencies*fourier_in_c
        elif fourier_type == 'random':
            self.fourier_layer = RandomFourierFeatureLayer(fourier_in_c, num_frequencies, scale=fourier_scale)
            fourier_out_dim = fourier_in_c + 2*num_frequencies*fourier_in_c
        else:
            raise ValueError(f"Unknown fourier_type: {fourier_type}")

        self.layers = nn.ModuleList()

        if use_siren:
            self.layers.append(SIRENLayer(fourier_out_dim, features[0], w0=siren_w0, is_first=True, h_siren=self.h_siren, learning_w0=self.learning_w0))
            for i in range(1, len(features)):
                self.layers.append(SIRENLayer(features[i-1], features[i], w0=1.0, is_first=False, h_siren=self.h_siren, learning_w0=self.learning_w0))

        else:
            self.layers.append(layer(fourier_out_dim, features[0], get_activation(self.activation_name,self.init_alpha)))
            for i in range(1,len(features)-1):
                self.layers.append(layer(features[i],features[i+1], get_activation(self.activation_name,self.init_alpha)))
        
        self.final_layer = nn.Linear(features[-1],self.out_c)
    
        # self.log_vars = nn.Parameter(torch.zeros(5), requires_grad=False)
    def forward(self, x):
        x = self.fourier_layer(x)
        for i, layer in enumerate(self.layers):
            skip_connection = x
            x = layer(x)
            if i>0:
                x = x + skip_connection

        x = self.final_layer(x)
        return x

    def get_alpha_loss(self):
        alpha_params = []

        # 1. Collect alpha parameters from SIREN layers (w0)
        for layer in self.layers:
            if isinstance(layer, SIRENLayer):
                if isinstance(layer.w0, nn.Parameter):
                    alpha_params.append(layer.w0)

        # 2. Collect alpha parameters from AdaptiveTanh activations
        for layer in self.layers:
            # Your layer wrapper: layer(..., activation)
            if hasattr(layer, "layer"):  
                for mod in layer.layer:
                    if isinstance(mod, AdaptiveTanh):
                        alpha_params.append(mod.alpha)

        # If zero or one alpha, regularization returns 0
        if len(alpha_params) <= 1:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        # Compute your custom alpha loss
        D = len(alpha_params)
        alpha_loss = 0.0
        for k, a in enumerate(alpha_params):
            alpha_loss = alpha_loss + torch.exp(a ** k)

        alpha_loss = (D - 1) / alpha_loss

        return alpha_loss

class NN_Periodic(NN):
    def __init__(self, in_c, out_c, features = [100,100,100,100], activation_name='tanh', num_frequencies=4,\
        fourier_type='random',fourier_scale=1.0, use_siren=False, siren_w0=30.0, x_left = -1.0, x_right=1.0, skip_con = True, h_siren=False, learning_w0=False, init_alpha=1.0):

        super().__init__(in_c, out_c, features, activation_name,\
            num_frequencies, fourier_type, fourier_scale, use_siren, siren_w0, h_siren, learning_w0, init_alpha)

        self.x_left = x_left
        self.x_right = x_right
        self.period = x_right - x_left
        self.skip_con = skip_con
        self.h_siren = h_siren
        self.learning_w0 = learning_w0

    def forward(self,x):
        x_input = x.clone()

        x_spatial = x_input[:,0:1]
        x_periodic = torch.cat([
                torch.sin(2*math.pi*(x_spatial-self.x_left)/self.period),
                torch.cos(2 * math.pi * (x_spatial - self.x_left)/self.period)
                ], dim=-1)
        
        if x.shape[1] > 1:
            x_rest = x_input[:, 1:]
            x_input = torch.cat([x_periodic, x_rest], dim=-1)
        else:
            x_input = x_periodic

        # Pass through Fourier layers if exists
        x_input = self.fourier_layer(x_input)
        for i, layer in enumerate(self.layers):
            skip_connection = x_input
            x_input = layer(x_input)
            if i > 0:
                if self.skip_con:
                    x_input = x_input + skip_connection
        
        x_input = self.final_layer(x_input)
        return x_input
        

def xavier_init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# x = torch.rand(100,2)
# model = NN_Periodic(in_c = 2, out_c = 1, features=[20,20,20,20],activation_name='tanh', num_frequencies=0)
# out = model(x)
# print(out.shape)
# print("number of model parameters:",sum(p.numel() for p in model.parameters() if p.requires_grad))

# x = torch.rand(2,4)
# fourier = FourierFeatureLayer(in_channels=4,num_frequencies=1)
# output = fourier(x)
# print(output.shape)
# print(output[:,0])
# print(torch.sin(output[:,0]))
# print(output[:,3])