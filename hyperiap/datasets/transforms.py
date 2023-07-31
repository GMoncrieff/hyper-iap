import torch


class VectorNorm:
    """unit vector normalization of tensor"""

    def __call__(self, input_tensor):
        # Calculate the magnitude for each batch and level along the wavelength dimension
        norm = input_tensor.norm(p=2, dim=2, keepdim=True)
        input_tensor = input_tensor.div(norm)
        return input_tensor


class NormalizeZ:
    """z normalization of tensor"""

    def __call__(self, tensor):
        # Calculate the mean for each batch and level along the wavelength dimension
        mean = tensor.mean(dim=-1, keepdim=True)
        # Calculate the mean for each batch and level along the wavelength dimension
        stds = tensor.std(dim=-1, keepdim=True)
        # Normalize the tensor along the wavelength dimension
        normalized_tensor = (tensor - mean) / stds

        return normalized_tensor


class Normalize01:
    """01 normalization of tensor"""

    def __call__(self, input_tensor):
        min_vals, _ = torch.min(input_tensor, dim=2, keepdim=True)
        max_vals, _ = torch.max(input_tensor, dim=2, keepdim=True)
        input_tensor = (input_tensor - min_vals) / (max_vals - min_vals)
        return input_tensor


class NoNorm:
    """01 normalization of tensor"""

    def __call__(self, input_tensor):
        return input_tensor / 10000
