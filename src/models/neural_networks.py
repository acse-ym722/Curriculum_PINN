"""
Neural Network Architectures for PINNs
"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with flexible architecture
    
    Args:
        layers: List of layer sizes [input_dim, hidden1, hidden2, ..., output_dim]
        activation: Activation function ('tanh', 'relu', 'sigmoid', 'gelu')
        use_batch_norm: Whether to use batch normalization
        dropout_rate: Dropout rate (0 for no dropout)
    """
    
    def __init__(self, layers, activation='tanh', use_batch_norm=False, dropout_rate=0.0):
        super(MLP, self).__init__()
        
        self.layers = layers
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        
        # Select activation function
        activation_dict = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'gelu': nn.GELU(),
            'elu': nn.ELU(),
        }
        self.activation = activation_dict.get(activation.lower(), nn.Tanh())
        
        # Build network
        self.network = self._build_network()
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_network(self):
        """Build the network architecture"""
        modules = []
        
        for i in range(len(self.layers) - 1):
            # Linear layer
            modules.append(nn.Linear(self.layers[i], self.layers[i+1]))
            
            # Add batch norm, activation, dropout (except for output layer)
            if i < len(self.layers) - 2:
                if self.use_batch_norm:
                    modules.append(nn.BatchNorm1d(self.layers[i+1]))
                
                modules.append(self.activation)
                
                if self.dropout_rate > 0:
                    modules.append(nn.Dropout(self.dropout_rate))
        
        return nn.Sequential(*modules)
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Forward pass"""
        if x.dim() == 1:
            x = x.view(-1, x.shape[0])
        return self.network(x)


class ResidualBlock(nn.Module):
    """Residual block for deep networks"""
    
    def __init__(self, dim, activation='tanh'):
        super(ResidualBlock, self).__init__()
        
        activation_dict = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
        }
        self.activation = activation_dict.get(activation.lower(), nn.Tanh())
        
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
    
    def forward(self, x):
        residual = x
        out = self.activation(self.fc1(x))
        out = self.fc2(out)
        out += residual
        out = self.activation(out)
        return out


class ResNet(nn.Module):
    """
    Residual Network for PINNs
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_blocks: Number of residual blocks
        activation: Activation function
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks=4, activation='tanh'):
        super(ResNet, self).__init__()
        
        activation_dict = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
        }
        self.activation = activation_dict.get(activation.lower(), nn.Tanh())
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, activation) for _ in range(num_blocks)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Forward pass"""
        if x.dim() == 1:
            x = x.view(-1, x.shape[0])
        
        x = self.activation(self.input_layer(x))
        
        for block in self.blocks:
            x = block(x)
        
        x = self.output_layer(x)
        return x


class FourierFeatureNet(nn.Module):
    """
    Network with Fourier feature embedding
    
    Args:
        layers: List of layer sizes (after embedding)
        input_dim: Input dimension
        fourier_dim: Dimension of Fourier features
        sigma: Standard deviation for Fourier feature sampling
        activation: Activation function
    """
    
    def __init__(self, layers, input_dim=2, fourier_dim=256, sigma=1.0, activation='tanh'):
        super(FourierFeatureNet, self).__init__()
        
        self.input_dim = input_dim
        self.fourier_dim = fourier_dim
        
        # Fourier feature mapping
        self.B = nn.Parameter(torch.randn(input_dim, fourier_dim // 2) * sigma, requires_grad=False)
        
        # MLP network (input is now 2*fourier_dim from cos and sin)
        mlp_layers = [fourier_dim] + layers[1:]
        self.mlp = MLP(mlp_layers, activation=activation)
    
    def forward(self, x):
        """Forward pass with Fourier features"""
        if x.dim() == 1:
            x = x.view(-1, x.shape[0])
        
        # Fourier feature mapping
        x_proj = 2 * torch.pi * x @ self.B
        x_fourier = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)
        
        return self.mlp(x_fourier)


def create_model(model_type='mlp', **kwargs):
    """
    Factory function to create different model types
    
    Args:
        model_type: Type of model ('mlp', 'resnet', 'fourier')
        **kwargs: Model-specific arguments
    
    Returns:
        Neural network model
    """
    if model_type.lower() == 'mlp':
        return MLP(**kwargs)
    elif model_type.lower() == 'resnet':
        return ResNet(**kwargs)
    elif model_type.lower() == 'fourier':
        return FourierFeatureNet(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")