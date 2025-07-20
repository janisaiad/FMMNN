import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import List, Optional
from flax.linen import initializers



class FMMNNJax(nn.Module):
    ranks: List[int] # i list where the i-th element represents output dimension of i-th layer
    widths: List[int] # list specifying width of each layer
    resnet: bool # whether to use resnet architecture with identity connections
    fix_wb: bool # if true, weights and biases not updated during training
    learning_rate: float = 0.01 # learning rate for training
    init_scaling: bool = True # if true, initialize weights and biases with scaling factor
    
    
    @nn.compact
    def setup(self):
        """initialize the model layers."""
        self.depth = len(self.widths)
        
        fc_sizes = [self.ranks[0]]
        for j in range(self.depth):
            fc_sizes += [self.widths[j], self.ranks[j+1]]
        
        self.fc_sizes = fc_sizes
        #list of dense layers
        fcs = []

        #we just define a helper function for scaled initialization
        def scaled_initializer(initializer, scale):
            def init(key, shape, dtype=jnp.float32):
                return scale * initializer(key, shape, dtype)
            return init

        if self.init_scaling:
            s = self.widths[0] / 2
            s = s**(1 / self.ranks[0]) * self.ranks[0]**0.5
            # custom initializers for the 1st layer
            custom_kernel_init = scaled_initializer(initializers.lecun_normal(), s)
            custom_bias_init = scaled_initializer(initializers.zeros, s)

        for j in range(len(self.fc_sizes)):
            if j == 0 and self.init_scaling: # iknow it's bad practice but to show how to do it
                # we use custom initializers for the first layer
                fc = nn.Dense(self.fc_sizes[j], use_bias=True, kernel_init=custom_kernel_init, bias_init=custom_bias_init)
            else:
                # we use default initializers for other layers
                fc = nn.Dense(self.fc_sizes[j], use_bias=True)
            fcs.append(fc)
        self.fcs = fcs
                
                
# Handle fix_wb parameter by marking parameters as trainable/non-trainable
# Note: In Flax/JAX, parameter updates are controlled during training
# We'll need to handle the fix_wb logic in the training loop
# by filtering the parameters

    @nn.compact
    def __call__(self, x):
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        for j in range(self.depth):
            if self.resnet:
                if 0 < j < self.depth-1:
                    x_id = x + 0.0  # make a copy to avoid inplace operations
                    
            x = self.fcs[2*j](x)
            x = jax.nn.relu(x)
            x = self.fcs[2*j+1](x)
            
            if self.resnet:
                if 0 < j < self.depth-1:
                    n = min(x.shape[1], x_id.shape[1])
                    x = x.at[:,:n].add(x_id[:,:n])
    
        x = jax.nn.sin(x)
        x = self.fcs[-1](x)
        return x
