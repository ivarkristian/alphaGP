import warnings
from xarray.coding.times import SerializationWarning

import numpy as np
import pandas as pd
import xarray as xr
import torch

def load_chemical_dataset(chem_data_path):
    """
    Loads a chemical dataset from a NetCDF file, fixes the time issue, and returns the dataset.

    This function reads a NetCDF file containing chemical data, fixes the time issue by converting
    ITIME to actual timestamps, and returns the corrected dataset.

    Parameters
    ----------
    chem_data_path : str
        The file path to the NetCDF file containing the chemical dataset.

    Returns
    -------
    xr.Dataset
        The loaded and corrected chemical dataset.
    """
    # Ignore the time warning
    warnings.filterwarnings("ignore", category=SerializationWarning)

    # Load the NetCDF file
    chem_dataset = xr.open_dataset(chem_data_path, decode_times=False)

    # Fix time issue by converting ITIME to actual timestamps
    time_reference = pd.Timestamp("1970-01-01")
    times_in_seconds = chem_dataset["Itime2"].values / 1000
    converted_times = time_reference + pd.to_timedelta(times_in_seconds, unit="s")
    chem_dataset["time"] = ("time", converted_times)

    return chem_dataset

def extract_synoptic_chemical_data_from_depth(x_coords, y_coords, values, sample_coords, radius=1.0, method='mean'):
    """
    Extracts chemical data within a specified spherical volume and computes the average data value.

    This function calculates the average of a data value within a spherical volume specified by the
    target coordinates (x, y, z), time, and radius. It crops the data to exclude the outer boundary
    condition mesh, calculates 3D distances, identifies points within the radius, and computes the
    average value for those points.

    Parameters
    ----------
    x_coords : array-like
    y_coords : array-like
    values : array-like
        The chemical values for a compound, siglay and time.
    sample_coords : array-like
        An array-like or tuple containing the target coordinates (x_target, y_target)
        of the spherical volume.

    Returns
    -------
    float
        The average data value within the specified spherical volume.
    """
    x_target, y_target = sample_coords

    # Calculate 3D distances from the target point
    x_diff = x_coords - x_target
    y_diff = y_coords - y_target

    # Calculation of distances
    distances = np.sqrt(
        x_diff[:, np.newaxis]**2 + 
        y_diff[:, np.newaxis]**2
    )

    # Find indices within the specified radius
    within_radius_indices = np.where(distances <= radius)
    #print(f'within_radius_indices: {(within_radius_indices)}')

    # Extract unique indices within the radius
    unique_indices_within_radius = np.unique(within_radius_indices[0])
    #print(f'unique_indices_within_radius: {(unique_indices_within_radius)}')

    # Extract data within the radius for the specified time and depth
    data_within_radius = values[unique_indices_within_radius]
    #print(f'data_within_radius: {(data_within_radius)}')

    if method == 'mean':
        # Compute the average data value
        average_data_value = data_within_radius.mean()
    elif method == 'max':
        average_data_value = data_within_radius.max()
    else:
        print(f'Method {method} not recognized.')
        average_data_value = 0

    return average_data_value


def torch_extract_synoptic_chemical_data_from_depth(x_coords, y_coords, values, sample_coords, radius=1.0, method='mean'):
    """
    Torch implementation of extract_synoptic_chemical_data_from_depth.

    Supports single sample coords (shape (2,)) or a batch of coords (shape (N, 2)).
    Returns a torch scalar for a single sample or a 1-D tensor for a batch.
    """
    x_coords_t = x_coords if isinstance(x_coords, torch.Tensor) else torch.as_tensor(x_coords)
    y_coords_t = y_coords if isinstance(y_coords, torch.Tensor) else torch.as_tensor(y_coords)
    values_t = values if isinstance(values, torch.Tensor) else torch.as_tensor(values)

    device = x_coords_t.device
    dtype = x_coords_t.dtype
    y_coords_t = y_coords_t.to(device=device, dtype=dtype)
    values_t = values_t.to(device=device)

    sample_coords_t = sample_coords if isinstance(sample_coords, torch.Tensor) else torch.as_tensor(sample_coords)
    sample_coords_t = sample_coords_t.to(device=device, dtype=dtype)

    if sample_coords_t.ndim == 1:
        if sample_coords_t.numel() != 2:
            raise ValueError("sample_coords must have shape (2,) for single-sample mode.")
        x_target, y_target = sample_coords_t
        x_diff = x_coords_t - x_target
        y_diff = y_coords_t - y_target
        distances = torch.sqrt(x_diff * x_diff + y_diff * y_diff)
        within_radius = distances <= radius
        data_within_radius = values_t[within_radius]
        if method == 'mean':
            return data_within_radius.mean()
        if method == 'max':
            return data_within_radius.max()
        raise ValueError(f"Method {method} not recognized.")

    if sample_coords_t.ndim == 2:
        if sample_coords_t.shape[1] != 2:
            raise ValueError("sample_coords must have shape (N, 2) for batch mode.")
        dx = x_coords_t[:, None] - sample_coords_t[None, :, 0]
        dy = y_coords_t[:, None] - sample_coords_t[None, :, 1]
        distances = torch.sqrt(dx * dx + dy * dy)
        within_radius = distances <= radius
        if method == 'mean':
            within_radius_f = within_radius.to(values_t.dtype)
            sums = (values_t[:, None] * within_radius_f).sum(dim=0)
            counts = within_radius_f.sum(dim=0)
            return sums / counts
        if method == 'max':
            neg_inf = torch.tensor(float("-inf"), device=values_t.device, dtype=values_t.dtype)
            masked = torch.where(within_radius, values_t[:, None], neg_inf)
            return masked.max(dim=0).values
        raise ValueError(f"Method {method} not recognized.")

    raise ValueError("sample_coords must be a 1-D or 2-D tensor-like object.")


def extract_chemical_data_from_dataset(dataset, metadata, data_variable):
    """
    Extracts chemical data within a specified spherical volume and computes the average data value.

    This function calculates the average of a data value within a spherical volume specified by the
    target coordinates (x, y, z), time, and radius. It crops the data to exclude the outer boundary
    condition mesh, calculates 3D distances, identifies points within the radius, and computes the
    average value for those points.

    Parameters
    ----------
    dataset : xr.Dataset
        The chemical dataset loaded from a NetCDF file.
    metadata : tuple
        A tuple containing the target coordinates (x_frac, y_frac, z_target, time_target, radius)
        of the spherical volume.
    data_variable : str
        The name of the data variable to extract and average.

    Returns
    -------
    float
        The average data value within the specified spherical volume.
    np.ndarray
        The data values within the specified spherical volume.
    """
    x_target, y_target, z_target, time_target, radius = metadata

    # Crop data to exclude the outer boundary condition mesh
    x_coords = dataset["x"].values[:72710]
    y_coords = dataset["y"].values[:72710]

    # Calculate target coordinates
    x_coord_target = np.min(x_coords) + x_target
    y_coord_target = np.min(y_coords) + y_target
    z_coord_target = np.float64(z_target)

    # Get the depth levels (siglay)
    siglay_depths = np.arange(len(dataset["siglay"]))

    # Calculate 3D distances from the target point
    x_diff = x_coords - x_coord_target
    y_diff = y_coords - y_coord_target
    z_diff = siglay_depths - z_coord_target

    # Calculation of distances
    distances = np.sqrt(
        x_diff[:, np.newaxis]**2 + 
        y_diff[:, np.newaxis]**2 + 
        z_diff[np.newaxis, :]**2
    )

    # Find indices within the specified radius
    within_radius_indices = np.where(distances <= radius)

    # Extract unique indices within the radius
    unique_indices_within_radius = np.unique(within_radius_indices[0])

    # Extract data within the radius for the specified time and depth
    data_within_radius = dataset[data_variable].isel(time=time_target, siglay=int(z_coord_target)).values[unique_indices_within_radius]

    # Compute the average data value
    average_data_value = np.mean(data_within_radius)

    return average_data_value, data_within_radius


def torch_extract_chemical_data_from_dataset(dataset, metadata, data_variable, device=None):
    """
    Torch implementation of extract_chemical_data_from_dataset.

    The dataset is still read via xarray (CPU); tensors can optionally be moved
    to a device for downstream GPU processing.
    """
    x_target, y_target, z_target, time_target, radius = metadata

    device = torch.device(device) if device is not None else torch.device("cpu")

    x_coords = torch.as_tensor(dataset["x"].values[:72710], device=device)
    y_coords = torch.as_tensor(dataset["y"].values[:72710], device=device)

    x_coord_target = x_coords.min() + torch.as_tensor(x_target, device=device, dtype=x_coords.dtype)
    y_coord_target = y_coords.min() + torch.as_tensor(y_target, device=device, dtype=y_coords.dtype)
    z_coord_target = torch.as_tensor(z_target, device=device, dtype=torch.float32)

    siglay_depths = torch.arange(len(dataset["siglay"]), device=device, dtype=torch.float32)

    x_diff = x_coords - x_coord_target
    y_diff = y_coords - y_coord_target
    z_diff = siglay_depths - z_coord_target

    distances = torch.sqrt(
        x_diff[:, None] * x_diff[:, None] +
        y_diff[:, None] * y_diff[:, None] +
        z_diff[None, :] * z_diff[None, :]
    )

    within_radius_indices = torch.where(distances <= radius)
    unique_indices_within_radius = torch.unique(within_radius_indices[0])

    data_array = dataset[data_variable].isel(time=time_target, siglay=int(z_coord_target.item())).values
    data_values = torch.as_tensor(data_array, device=device)[unique_indices_within_radius]

    average_data_value = data_values.mean()

    return average_data_value, data_values


if __name__ == "__main__":
    # File path to the chemical data NetCDF file
    chemical_file_path = "utils_and_data/plume_data/SMART-AUVs_OF_0001-ph.nc"
    chemical_dataset = load_chemical_dataset(chemical_file_path)

    # Define target coordinates and parameters for volume extraction
    x_target = 100
    y_target = 100
    z_target = 30
    time_target = 2
    radius = 4
    metadata = x_target, y_target, z_target, time_target, radius
    data_variable = 'pCO2'

    # Extract chemical data and compute average value within the specified volume
    chemical_volume_data_mean, data_within_radius = extract_chemical_data_from_dataset(
        chemical_dataset, metadata, data_variable
    )

    # Print the results
    print(
        f"Average {data_variable} value within spherical radius of {radius} "
        f"around the point ({x_target}, {y_target}, {z_target}) is: {chemical_volume_data_mean:.4f}."
    )
    print(f"Number of points used in the average: {len(data_within_radius)}.")
