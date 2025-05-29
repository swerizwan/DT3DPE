import torch.nn as nn
from models.vectorquantizer.resnet1d_model import Resnet1D

# Define the Encoder class, which inherits from nn.Module
class Encoder(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        """
        Initialize the Encoder model.
        
        Args:
            input_emb_width: Width of the input embeddings.
            output_emb_width: Width of the output embeddings.
            down_t: Number of downsampling steps.
            stride_t: Stride for downsampling.
            width: Width of the model.
            depth: Depth of the Resnet1D blocks.
            dilation_growth_rate: Growth rate for dilation in Resnet1D blocks.
            activation: Activation function to use.
            norm: Normalization method to apply.
        """
        super().__init__()

        blocks = []
        # Calculate filter and padding sizes for downsampling
        filter_t, pad_t = stride_t * 2, stride_t // 2
        # Initial convolutional layer to transform input embeddings to the desired width
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        # Create downsampling blocks
        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        # Final convolutional layer to transform to the output embedding width
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        """
        Forward pass of the Encoder model.
        
        Args:
            x: Input data.
        
        Returns:
            Encoded output.
        """
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        """
        Initialize the Decoder model.
        
        Args:
            input_emb_width: Width of the input embeddings.
            output_emb_width: Width of the output embeddings.
            down_t: Number of upsampling steps.
            stride_t: Stride for upsampling.
            width: Width of the model.
            depth: Depth of the Resnet1D blocks.
            dilation_growth_rate: Growth rate for dilation in Resnet1D blocks.
            activation: Activation function to use.
            norm: Normalization method to apply.
        """
        super().__init__()
        blocks = []

        # Initial convolutional layer to transform from output embedding width to the model width
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        # Create upsampling blocks
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        # Final convolutional layers to transform back to the input embedding width
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        """
        Forward pass of the Decoder model.
        
        Args:
            x: Input data.
        
        Returns:
            Decoded output with dimensions permuted to (batch_size, sequence_length, input_emb_width).
        """
        x = self.model(x)
        # Permute dimensions to match the expected output format
        return x.permute(0, 2, 1)