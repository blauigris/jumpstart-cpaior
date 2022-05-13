from torch import max

# Inherit from Function
from torch.autograd import Function
import torch
import pandas as pd

class RandomBackpropMax(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input_, dim):
        output = input_.amax(dim=dim)
        max, argmax = input_.max(dim=dim)
        argmax_mask = input_ == output

        output = torch.gather(input_, 0, torch.unsqueeze(argmax, 0))
        ctx.save_for_backward(input_, argmax, torch.tensor(dim))
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input_, argmax, dim = ctx.saved_tensors
        grad_input = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = torch.gather(input_, 0, torch.unsqueeze(argmax, 0))

        return grad_input