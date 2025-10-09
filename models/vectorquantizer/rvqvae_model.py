# Import necessary libraries
import random

import torch.nn as nn
from models.vectorquantizer.seq_encoder_decoder import Encoder, Decoder
from models.vectorquantizer.rvq_module import ResidualVQ

# Define the RVQVAE class, which inherits from nn.Module
class RVQVAE(nn.Module):
    def __init__(self,
                 args,
                 input_width=263,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        """
        Initialize the RVQVAE model.
        
        Args:
            args: Arguments containing model configuration.
            input_width: Width of the input data.
            nb_code: Number of codes in the codebook.
            code_dim: Dimension of each code.
            output_emb_width: Width of the output embedding.
            down_t: Number of downsampling steps.
            stride_t: Stride for downsampling.
            width: Width of the model.
            depth: Depth of the model.
            dilation_growth_rate: Growth rate for dilation.
            activation: Activation function to use.
            norm: Normalization method to apply.
        """
        super().__init__()
        # Ensure the output embedding width matches the code dimension
        assert output_emb_width == code_dim
        self.code_dim = code_dim
        self.num_code = nb_code
        # Initialize the encoder
        self.encoder = Encoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        # Initialize the decoder
        self.decoder = Decoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        # Configure the ResidualVQ module
        rvqvae_config = {
            'num_quantizers': args.num_quantizers,
            'shared_codebook': args.shared_codebook,
            'quantize_dropout_prob': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': nb_code,
            'code_dim': code_dim, 
            'args': args,
        }
        self.quantizer = ResidualVQ(**rvqvae_config)

    def preprocess(self, x):
        """
        Preprocess the input data by permuting dimensions and converting to float.
        
        Args:
            x: Input data.
        
        Returns:
            Preprocessed data.
        """
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        """
        Postprocess the output data by permuting dimensions.
        
        Args:
            x: Output data.
        
        Returns:
            Postprocessed data.
        """
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x):
        """
        Encode the input data using the encoder and quantizer.
        
        Args:
            x: Input data.
        
        Returns:
            code_idx: Indices of the quantized codes.
            all_codes: All quantized codes.
        """
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        code_idx, all_codes = self.quantizer.quantize(x_encoder, return_latent=True)

        return code_idx, all_codes

    def forward(self, x):
        """
        Forward pass of the RVQVAE model.
        
        Args:
            x: Input data.
        
        Returns:
            x_out: Reconstructed output.
            commit_loss: Commitment loss.
            perplexity: Perplexity of the quantizer.
        """
        x_in = self.preprocess(x)

        x_encoder = self.encoder(x_in)
        # Quantize the encoder output
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5)

        x_out = self.decoder(x_quantized)
        return x_out, commit_loss, perplexity

    def forward_decoder(self, x):
        """
        Forward pass through the decoder using quantized codes.
        
        Args:
            x: Indices of the quantized codes.
        
        Returns:
            x_out: Reconstructed output.
        """
        x_d = self.quantizer.get_codes_from_indices(x)
        x = x_d.sum(dim=0).permute(0, 2, 1)

        x_out = self.decoder(x)
        return x_out

class LengthEstimator(nn.Module):
    def __init__(self, input_size, output_size):
        """
        Initialize the LengthEstimator model.
        
        Args:
            input_size: Size of the input data.
            output_size: Size of the output data.
        """
        super(LengthEstimator, self).__init__()
        nd = 512
        # Define a sequential neural network for estimating length
        self.output = nn.Sequential(
            nn.Linear(input_size, nd),
            nn.LayerNorm(nd),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(0.2),
            nn.Linear(nd, nd // 2),
            nn.LayerNorm(nd // 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(0.2),
            nn.Linear(nd // 2, nd // 4),
            nn.LayerNorm(nd // 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(nd // 4, output_size)
        )

        # Initialize the weights of the network
        self.output.apply(self.__init_weights)

    def __init_weights(self, module):
        """
        Initialize the weights of the neural network modules.
        
        Args:
            module: Neural network module.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, text_emb):
        """
        Forward pass of the LengthEstimator model.
        
        Args:
            text_emb: Text embeddings.
        
        Returns:
            Estimated length.
        """
        return self.output(text_emb)