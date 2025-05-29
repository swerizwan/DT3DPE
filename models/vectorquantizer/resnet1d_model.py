import torch.nn as nn
import torch

# Custom nonlinearity class implementing the SiLU (Sigmoid Linear Unit) activation function
class nonlinearity(nn.Module):
    def __init__(self):
        """
        Initialize the nonlinearity module.
        """
        super().__init__()

    def forward(self, x):
        """
        Apply the SiLU activation function.

        :param x: Input tensor
        :return: Output tensor after applying SiLU
        """
        return x * torch.sigmoid(x)

# Residual Convolutional 1D Block
class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, activation='silu', norm=None, dropout=0.2):
        """
        Initialize a Residual Convolutional 1D Block.

        :param n_in: Number of input channels
        :param n_state: Number of hidden channels
        :param dilation: Dilation rate for the convolution
        :param activation: Activation function ('relu', 'silu', 'gelu')
        :param norm: Normalization type ('LN', 'GN', 'BN', or None)
        :param dropout: Dropout rate
        """
        super(ResConv1DBlock, self).__init__()

        padding = dilation
        self.norm = norm

        # Normalization layers
        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        # Activation functions
        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()
        elif activation == "silu":
            self.activation1 = nonlinearity()
            self.activation2 = nonlinearity()
        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()

        # Convolutional layers
        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass of the ResConv1DBlock.

        :param x: Input tensor
        :return: Output tensor after applying the residual block
        """
        x_orig = x
        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1))
            x = self.activation1(x.transpose(-2, -1))
        else:
            x = self.norm1(x)
            x = self.activation1(x)

        x = self.conv1(x)

        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1))
            x = self.activation2(x.transpose(-2, -1))
        else:
            x = self.norm2(x)
            x = self.activation2(x)

        x = self.conv2(x)
        x = self.dropout(x)
        x = x + x_orig  # Residual connection
        return x

# ResNet 1D model
class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, dilation_growth_rate=1, reverse_dilation=True, activation='relu', norm=None):
        """
        Initialize the ResNet 1D model.

        :param n_in: Number of input channels
        :param n_depth: Number of residual blocks
        :param dilation_growth_rate: Dilation growth rate
        :param reverse_dilation: Whether to reverse the dilation order
        :param activation: Activation function ('relu', 'silu', 'gelu')
        :param norm: Normalization type ('LN', 'GN', 'BN', or None)
        """
        super().__init__()

        # Create a list of ResConv1DBlock layers
        blocks = [ResConv1DBlock(n_in, n_in, dilation=dilation_growth_rate ** depth, activation=activation, norm=norm)
                  for depth in range(n_depth)]
        if reverse_dilation:
            blocks = blocks[::-1]  # Reverse the order of blocks if specified

        self.model = nn.Sequential(*blocks)  # Stack the blocks into a sequential model

    def forward(self, x):
        """
        Forward pass of the ResNet 1D model.

        :param x: Input tensor
        :return: Output tensor after passing through the ResNet model
        """
        return self.model(x)