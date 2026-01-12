import numpy as np
import torch

def rotate_points(coordinates, angle_deg, rot_coord=None):
    """
    Rotates a set of 2D coordinates by a given angle in degrees.

    Parameters:
    - coordinates: numpy array or torch tensor of shape (n, 2), where each row is a point [x, y].
    - angle_deg: float, the angle by which to rotate the points, in degrees.

    Returns:
    - rotated_coords: numpy array or torch tensor of shape (n, 2), the rotated coordinates.
    """
    if rot_coord is None:
        t = [coordinates[:, 0].mean(), coordinates[:, 1].mean()]
    else:
        t = rot_coord

    # Check if coordinates is a numpy array
    if isinstance(coordinates, np.ndarray):
        # Convert angle from degrees to radians
        angle_rad = np.deg2rad(angle_deg)

        # Calculate cosine and sine of the angle
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        # Create the rotation matrix using NumPy
        rotation_matrix = np.array([
            [cos_angle, -sin_angle],
            [sin_angle,  cos_angle]
        ])

    # Check if coordinates is a torch tensor
    elif isinstance(coordinates, torch.Tensor):
        # Convert angle from degrees to radians (ensure angle_rad is a torch scalar)
        angle_rad = torch.tensor(angle_deg, dtype=coordinates.dtype, device=coordinates.device) * (torch.pi / 180.0)

        # Calculate cosine and sine of the angle using PyTorch
        cos_angle = torch.cos(angle_rad)
        sin_angle = torch.sin(angle_rad)

        # Create the rotation matrix using PyTorch
        rotation_matrix = torch.tensor([
            [cos_angle, -sin_angle],
            [sin_angle,  cos_angle]
        ], dtype=coordinates.dtype, device=coordinates.device)

        t = torch.tensor(t, device=coordinates.device)

    else:
        raise TypeError("coordinates must be a numpy array or a torch tensor")
    
    # Rotate the coordinates
    coords_zero_translated = coordinates - t
    rotated_coords = coords_zero_translated @ rotation_matrix.T

    return rotated_coords + t