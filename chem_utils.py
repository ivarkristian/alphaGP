import warnings
from xarray.coding.times import SerializationWarning

import numpy as np
import pandas as pd
import xarray as xr

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
