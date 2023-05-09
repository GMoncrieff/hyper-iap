import torch


class UnitVectorNorm:
    """unit vector normalization of tensor"""

    def __call__(self, tensor):
        # Calculate the magnitude for each batch and level along the wavelength dimension
        magnitude = torch.norm(tensor, dim=-1, keepdim=True)
        # Normalize the tensor along the wavelength dimension
        normalized_tensor = tensor / magnitude

        return normalized_tensor
