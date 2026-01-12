"""
This script simulates the path of an Autonomous Underwater Vehicle (AUV) as it collects and visualizes chemical data (e.g., pCO2) in a 3D space.

Key functionalities include:
1. **Interpolation**: Precomputes interpolation weights to efficiently map chemical data from source points to a target grid for multiple time steps.
2. **Path Simulation**: Generates waypoints for the AUV's path (e.g., lawnmower or spiral pattern) and simulates its movement while collecting chemical data.
3. **Data Visualization**: Animates and plots the AUV's path along with the interpolated chemical concentration data, with options to save the result as a GIF.

The script utilizes Delaunay triangulation for efficient interpolation and includes various utility functions for path generation, data extraction, and plotting.
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from tqdm import tqdm
from scipy.spatial import Delaunay
from matplotlib.ticker import ScalarFormatter
import pandas as pd


from lawnmower_path import generate_lawnmower_waypoints
from bubble_utils import get_bubbles_from_beam, load_bubble_dataset
from chem_utils import load_chemical_dataset, extract_chemical_data_from_dataset
from patterns import *


# plt.rcParams.update({
#     'font.family': 'serif',
#     'font.serif': ['Computer Modern Roman'],  # Use Computer Modern
#     'text.usetex': True,  # Use LaTeX to render the text with Computer Modern
#     'font.size': 12  # Set the default font size
# })


def compute_interpolation_weights(source_points, target_points):
    """
    Compute the interpolation weights for mapping source points to target points.

    Parameters
    ----------
    source_points : ndarray
        The coordinates of the source points (n, 3).
    target_points : ndarray
        The coordinates of the target points (m, 3).

    Returns
    -------
    vertices : ndarray
        The indices of the vertices of the simplex that contains each target point.
    bary : ndarray
        The barycentric coordinates of the target points relative to the simplices.
    """
    tri = Delaunay(source_points)
    simplex = tri.find_simplex(target_points)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = target_points - temp[:, 2]
    bary = np.einsum('njk,nk->nj', temp[:, :2, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

def interpolate_values(values, vertices, weights, fill_value=np.nan):
    """
    Interpolate values based on precomputed weights.

    Parameters
    ----------
    values : ndarray
        The values at the source points.
    vertices : ndarray
        The indices of the vertices of the simplex that contains each target point.
    weights : ndarray
        The interpolation weights for each target point.
    fill_value : float, optional
        The value to use for points outside the convex hull (default is np.nan).

    Returns
    -------
    ndarray
        The interpolated values at the target points.
    """
    interpolated = np.einsum('nj,nj->n', np.take(values, vertices), weights)
    interpolated[np.any(weights < 0, axis=1)] = fill_value
    return interpolated


def animate(chemical_dataset, bubble_dataset, x_coords, y_coords, z_coords, sample_coords, data_var, title, siglay=69, save_as_gif=True, gif_filename='auv_path.gif'):
    """
    Plots the 3D path of an AUV along with interpolated chemical data and sample points.

    Parameters
    ----------
    chemical_dataset : xarray.Dataset
        The dataset containing chemical data.
    bubble_dataset : DataFrame
        The dataset containing bubble data.
    x_coords : np.ndarray
        X coordinates of the AUV path.
    y_coords : np.ndarray
        Y coordinates of the AUV path.
    z_coords : np.ndarray
        Z coordinates (depth) of the AUV path.
    sample_coords : np.ndarray
        Coordinates and times of the sample points.
    data_var : str
        The chemical data variable to be plotted (e.g., 'pCO2').
    title : str
        Title of the plot.
    siglay : The depth to run at(?)
    save_as_gif : bool, optional
        Whether to save the animation as a GIF (default is True).
    gif_filename : str, optional
        Filename for the GIF output (default is 'auv_path.gif').
    """

    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    x_min, x_max = np.min(x_coords) - 10, np.max(x_coords) + 10
    y_min, y_max = np.min(y_coords) - 10, np.max(y_coords) + 10

    x_target = np.linspace(int(np.floor(x_min)), np.ceil(x_max), int(2 * (np.ceil(x_max) - np.floor(x_min))) + 1)
    y_target = np.linspace(int(np.floor(y_min)), np.ceil(y_max), int(2 * (np.ceil(y_max) - np.floor(y_min))) + 1)
    X_target, Y_target = np.meshgrid(x_target, y_target)

    # Use x_data and y_data from the chemical dataset
    x_data = chemical_dataset['x'].values[:72710]
    y_data = chemical_dataset['y'].values[:72710]

    source_points = np.column_stack((x_data, y_data))
    target_points = np.column_stack((X_target.ravel(), Y_target.ravel()))

    # Precompute interpolation weights and vertices
    vertices, weights = compute_interpolation_weights(source_points, target_points)

    # Precompute interpolations for all timesteps
    precomputed_interpolations = []
    for time_index in tqdm(range(len(chemical_dataset['time'].values)), desc=f"Interpolation of siglay number {siglay} for Visualisation"):
        data_sample = chemical_dataset[data_var].isel(time=time_index, siglay=siglay).values[:72710]
        interpolated_layer = interpolate_values(data_sample, vertices, weights).reshape(X_target.shape)
        precomputed_interpolations.append(interpolated_layer)

    # Initial plot setup
    contour = ax.contourf(X_target, Y_target, precomputed_interpolations[0], zdir='z', offset=69, cmap='viridis', alpha=0.9, zorder=1)

    # Add colorbar for the chemical concentration
    cbar_c = fig.colorbar(contour, ax=ax, shrink=0.5, aspect=10, location='left', pad=0.05)
    bar_label = f'Concentration of {data_var}'
    cbar_c.set_label(bar_label)

    # Adjust the colorbar position
    cbar_c.ax.set_position([0.1, 0.3, 0.03, 0.4])

    # Plot the AUV path
    ax.plot(x_coords, y_coords, z_coords, label='AUV Path', color='crimson', zorder=10)
    sample_coords = np.array(sample_coords)

    # Plot sample points
    line, = ax.plot([], [], [], color='r', label='Sample Points', linestyle='None', marker='o', markersize=5, zorder=10)

    # Set plot limits and invert Z-axis (depth)
    ax.set_xlim([min(x_target), max(x_target)])
    ax.set_ylim([min(y_target), max(y_target)])
    ax.set_zlim(58, 69)
    ax.invert_zaxis()

    # Set the default view angle
    ax.view_init(elev=38, azim=-58)

    # Labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Meters Below Sea Level')
    ax.set_title(title, fontsize=18)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.001), ncol=3)

    # Adjust the y-axis format
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=False))
    ax.yaxis.get_major_formatter().set_scientific(False)

    # Adjust layout
    plt.subplots_adjust(left=0.1, right=0.9)

    # Convert the dataset time coordinates to numpy datetime64 for comparison
    dataset_times = chemical_dataset['time'].values

    # Animation update function
    def update(num, sample_coords, line):
        """
        Updates the animation by setting the data for the line and updating the interpolated layer.

        Parameters
        ----------
        num : int
            Frame number.
        sample_coords : numpy.ndarray
            Array of sample coordinates and times.
        line : matplotlib.lines.Line2D
            Line object to be updated.

        Returns
        -------
        matplotlib.lines.Line2D
            Updated line object.
        """
        # Start from the first timestep instead of the 0th
        current_time = sample_coords[num + 1, 3]

        # Find the nearest available timestep in the dataset
        timestep_idx = np.argmin(np.abs(dataset_times - current_time))

        # Retrieve the precomputed interpolation
        interpolated_layer = precomputed_interpolations[timestep_idx]

        # Clear the axis and redraw the contour
        ax.clear()

        # Redraw the path and the updated contour
        contour = ax.contourf(X_target, Y_target, interpolated_layer, zdir='z', offset=siglay, cmap='viridis', alpha=0.9, zorder=1)
        ax.plot(x_coords, y_coords, z_coords, label='AUV Path', color='crimson', zorder=10)

        # Update the line for sample points
        line.set_data(sample_coords[:num + 1, 0], sample_coords[:num + 1, 1])
        line.set_3d_properties(sample_coords[:num + 1, 2])
        ax.add_line(line)

        # Redraw static elements
        ax.set_xlim([min(x_target), max(x_target)])
        ax.set_ylim([min(y_target), max(y_target)])
        ax.set_zlim(58, 69)
        ax.invert_zaxis()
        ax.view_init(elev=38, azim=-58)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Meters Below Sea Level')
        ax.set_title(title, fontsize=18)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.001), ncol=3)

        return line, contour

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(sample_coords) - 1, fargs=(sample_coords, line), interval=100, blit=False)

    if save_as_gif:
        writer = PillowWriter(fps=10)
        ani.save(gif_filename, writer=writer, dpi=300)
    else:
        plt.show()

def plot(waypoints, sample_coords, sample_values, data_var='Default [m]', vminmax=None):
    # Extract X and Y coordinates
    x_coords = np.array([coord[0] for coord in sample_coords], dtype=float)
    y_coords = np.array([coord[1] for coord in sample_coords], dtype=float)

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the sample locations with values indicated by color
    if not vminmax:
        vmin = min(sample_values)
        vmax = max(sample_values)
    else:
        vmin = vminmax[0]
        vmax = vminmax[1]

    scatter = ax.scatter(
        x_coords,
        y_coords,
        c=sample_values,
        cmap='coolwarm',
        s=2,
        vmin=vmin,
        vmax=vmax
    )

    # Plot the path between waypoints
    #ax.plot(
    #    waypoints[:, 0],
    #    waypoints[:, 1],
    #    linestyle='-',
    #    color='red',
    #    linewidth=0,
    #    marker=None
    #)

    # Create an axis on the left for the colorbar
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("left", size="5%", pad=0.1)

    # Add the colorbar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(data_var)

    # Adjust the main plot
    #ax.yaxis.set_label_position("right")
    #ax.yaxis.tick_right()
    #fig.subplots_adjust(left=0.2)

    # Set labels and title
    ax.set_xlabel('Easting [m]')
    ax.set_ylabel('Northing [m]')
    ax.set_title('Sample Locations')

    return fig

def beam_angles(points):
    """
    Calculates the angles and opposite angles between adjacent points.

    Parameters
    ----------
    points : np.ndarray
        Array of points (x, y).

    Returns
    -------
    list of tuples
        List of tuples containing (angle, opposite angle) in degrees.
    """
    # Calculate the differences between adjacent points in the x and y directions
    x_diff = points[1:, 0] - points[:-1, 0]
    y_diff = points[1:, 1] - points[:-1, 1]

    # Calculate the angles using arctan2
    angles = np.arctan2(y_diff, x_diff)

    # Convert angles from radians to degrees
    angles_degrees = np.degrees(angles)

    # Calculate the opposite angles (180 degrees from the original)
    opposite_angles = (angles_degrees + 180) % 360

    # Combine the angles into tuples (angle, opposite angle)
    angle_tuples = list(zip(angles_degrees, opposite_angles))

    return angle_tuples


def path_with_samples(dataset, way_points, start_time, speed, sample_frequency, threshold=np.inf, pattern_func=None, data_variable='pCO2', sphere_radius=1, synoptic=False):
    """
    Simulates the path of an Autonomous Underwater Vehicle (AUV) through waypoints and collects chemical data.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset containing chemical data.
    bubble_dataset : DataFrame
        The dataset containing bubble data.
    start_time : str
        The start time of the simulation.
    speed : float
        The speed of the AUV in m/s.
    way_points : list
        List of waypoints (x, y, z) for the AUV to follow.
    sample_frequency : float
        The frequency of sampling in Hz.
    threshold : float
        The threshold for the chemical data variable to trigger pattern sampling.
    pattern_func : function
        The function to generate sampling patterns.
    data_variable : str, optional
        The data variable to be explored (default is 'pCO2').
    sphere_radius : float, optional
        The radius of the sphere for volume sampling (default is 4).

    Returns
    -------
    tuple
        Measurements and sample coordinates (including time) collected along the path.
    """
    measurements = []
    sample_coords = []

    way_points = np.array(way_points)
    distances = np.sqrt(np.sum(np.diff(way_points, axis=0) ** 2, axis=1))

    times = distances / speed
    cumulative_times = np.cumsum(times)

    total_time = np.sum(times)
    num_samples = int(total_time * sample_frequency)
    sample_interval = 1 / sample_frequency
    # sample_times = np.array([np.datetime64(start_time) + np.timedelta64(round(t * 1000), 'ms') for t in np.arange(0, total_time, sample_interval)])
    sample_times = np.array([np.datetime64(start_time) + np.timedelta64(round(t * 1000), 'ms') for t in np.arange(0, total_time, sample_interval)])

    for i in tqdm(range(len(way_points) - 1), desc="Path Waypoints and Sensor Sample Calculations"):
        start_point = way_points[i]
        end_point = way_points[i + 1]

        # Determine the time points within the current segment
        segment_times = sample_times[
            (sample_times >= (np.datetime64(start_time) + np.timedelta64(round((cumulative_times[i] - times[i]) * 1000), 'ms'))) &
            (sample_times <= (np.datetime64(start_time) + np.timedelta64(round(cumulative_times[i] * 1000), 'ms')))
        ]

        for t in segment_times:
            # Calculate the ratio of the current time to the segment duration
            ratio = (t - (np.datetime64(start_time) + np.timedelta64(round((cumulative_times[i] - times[i]) * 1000), 'ms'))) / np.timedelta64(round(times[i] * 1000), 'ms')
            ratio = np.clip(ratio, 0, 1)

            # Compute the sample coordinates based on the ratio
            x_sample = round(start_point[0] + ratio * (end_point[0] - start_point[0]), 2)
            y_sample = round(start_point[1] + ratio * (end_point[1] - start_point[1]), 2)
            z_sample = round(start_point[2] + ratio * (end_point[2] - start_point[2]), 2)

            # Append time information to the sample coordinates
            if synoptic:
                use_t = np.datetime64(start_time)
            else:
                use_t = t
            
            sample_coords.append((x_sample, y_sample, z_sample, use_t))

            # Collect chemical data
            nearest_t = abs((dataset['time'].values - use_t)).argmin()
            metadata = (x_sample, y_sample, z_sample, nearest_t, sphere_radius)
            chemical_volume_data_mean, data = extract_chemical_data_from_dataset(dataset, metadata, data_variable)
            measurements.append(chemical_volume_data_mean)

            # Check if the chemical data exceeds the threshold and sample in a pattern if true
            if chemical_volume_data_mean > threshold:
                pattern_coords = pattern_func(np.array([x_sample, y_sample, z_sample]))
                for coord in pattern_coords:
                    metadata_pattern = (coord[0], coord[1], coord[2], str(t), sphere_radius)
                    chemical_volume_data_mean, data = extract_chemical_data_from_dataset(dataset, metadata_pattern, data_variable)
                    measurements.append(chemical_volume_data_mean)
                    sample_coords.append((coord[0], coord[1], coord[2], t))

    return measurements, sample_coords

def path(waypoints, start_time, speed, sample_frequency, synoptic):
    """
    Simulates the path of an Autonomous Underwater Vehicle (AUV) through waypoints.

    Parameters
    ----------
    way_points : list
        List of waypoints
    start_time : str
        The start time of the simulation.
    speed : float
        The speed of the AUV in m/s.
    way_points : list
        List of waypoints (x, y, z) for the AUV to follow.
    sample_frequency : float
        The frequency of sampling in Hz.
    synoptic : bool
        Use synoptic sampling

    Returns
    -------
    tuple
        Sample coordinates (including time) along the path.
    """
    
    sample_coords = []

    waypoints = np.array(waypoints)
    distances = np.sqrt(np.sum(np.diff(waypoints, axis=0) ** 2, axis=1))

    times = distances / speed
    cumulative_times = np.cumsum(times)

    total_time = np.sum(times)
    num_samples = int(total_time * sample_frequency)
    sample_interval = 1 / sample_frequency
    # sample_times = np.array([np.datetime64(start_time) + np.timedelta64(round(t * 1000), 'ms') for t in np.arange(0, total_time, sample_interval)])
    sample_times = np.array([np.datetime64(start_time) + np.timedelta64(round(t * 1000), 'ms') for t in np.arange(0, total_time, sample_interval)])

    for i in range(len(waypoints) - 1):
        start_point = waypoints[i]
        end_point = waypoints[i + 1]

        # Determine the time points within the current segment
        segment_times = sample_times[
            (sample_times >= (np.datetime64(start_time) + np.timedelta64(round((cumulative_times[i] - times[i]) * 1000), 'ms'))) &
            (sample_times <= (np.datetime64(start_time) + np.timedelta64(round(cumulative_times[i] * 1000), 'ms')))
        ]

        for t in segment_times:
            # Calculate the ratio of the current time to the segment duration
            ratio = (t - (np.datetime64(start_time) + np.timedelta64(round((cumulative_times[i] - times[i]) * 1000), 'ms'))) / np.timedelta64(round(times[i] * 1000), 'ms')
            ratio = np.clip(ratio, 0, 1)

            # Compute the sample coordinates based on the ratio
            x_sample = round(start_point[0] + ratio * (end_point[0] - start_point[0]), 2)
            y_sample = round(start_point[1] + ratio * (end_point[1] - start_point[1]), 2)
            z_sample = round(start_point[2] + ratio * (end_point[2] - start_point[2]), 2)

            # Append time information to the sample coordinates
            if synoptic:
                use_t = np.datetime64(start_time)
            else:
                use_t = t
            
            sample_coords.append((x_sample, y_sample, z_sample, use_t))

    return sample_coords


def make_spiral_path(x_data, y_data, radius=30, num_t_points=50, min_siglay=69, max_siglay=55):
    """
    Generates a spiral path for the AUV based on the given parameters.

    Parameters
    ----------
    x_data : np.ndarray
        X coordinate data.
    y_data : np.ndarray
        Y coordinate data.
    radius : float, optional
        Radius of the spiral (default is 30).
    num_t_points : int, optional
        Number of time points (default is 50).
    min_siglay : int, optional
        Minimum depth layer (default is 69).
    max_siglay : int, optional
        Maximum depth layer (default is 55).

    Returns
    -------
    tuple
        Waypoints and coordinate arrays (x_coords, y_coords, z_coords).
    """
    t_way_points = np.linspace(0, 4 * np.pi, num_t_points)

    # Calculate waypoints for the spiral
    x_coords = np.min(x_data) + 105 + radius * np.cos(t_way_points)
    y_coords = np.min(y_data) + 105 + radius * np.sin(t_way_points)
    z_coords = np.linspace(min_siglay, max_siglay, len(t_way_points))

    way_points = [(x, y, z) for x, y, z in zip(x_coords, y_coords, z_coords)]

    return way_points, x_coords, y_coords, z_coords


def plot_waypoints_and_data(waypoints, x_coords, y_coords, title):
    """
    Plots waypoints and corresponding data points on a 2D grid.

    Parameters
    ----------
    waypoints : list
        List of waypoints.
    x_coords : np.ndarray
        X coordinates of the waypoints.
    y_coords : np.ndarray
        Y coordinates of the waypoints.
    title : str
        Title of the plot.
    """
    num_points = len(x_coords)
    color_values = np.linspace(0, 1, num_points)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(x_coords, y_coords, c=color_values, cmap='viridis', label='Waypoints')
    plt.colorbar(scatter, label='Waypoint Index')

    wp_x = [wp[0] for wp in waypoints]
    wp_y = [wp[1] for wp in waypoints]

    plt.plot(wp_x, wp_y, c='red', label='Lawnmower Path')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(title)

    # Adjust the legend to be below the plot
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f'AUV_waypoints_{title}.png', dpi=300)
    # plt.show()


if __name__ == '__main__':
    # Load bubble dataset and adjust coordinates
    bubble_file_path = "1-plume.dat"
    bubble_dataset = load_bubble_dataset(bubble_file_path)

    closest_x = 923323.625
    closest_y = 6614356.5

    bubble_dataset['shifted_x'] = bubble_dataset['x_coordinate'] + closest_x
    bubble_dataset['shifted_y'] = bubble_dataset['y_coordinate'] + closest_y

    # Load chemical dataset
    chemical_file_path = "SMART-AUVs_OF-June-1a-0001.nc"
    chemical_dataset = load_chemical_dataset(chemical_file_path)

    x_data = chemical_dataset['x'].values[:72710]
    y_data = chemical_dataset['y'].values[:72710]

    # Example simulation parameters
    sphere_radius = 4
    dist_between_bars = 20
    speed = 1.5  # m/s
    data_var = 'pCO2'
    siglay = 69
    start_time = '2020-01-01T02:10:00.000000000'
    sample_frequency = 0.1  # Sample frequency in Hz (samples per second)
    threshold = 545  # Threshold for the chemical data variable

    """ Run the path function with the bowtie pattern as an example """
    
    # Generate waypoints
    # way_points, x_coords, y_coords, z_coords = make_spiral_path(x_data, y_data, radius=30, num_t_points=50, min_siglay=69, max_siglay=55)
    way_points, x_coords, y_coords, z_coords = generate_lawnmower_waypoints(x_data, y_data, width=dist_between_bars, min_turn_radius=dist_between_bars, siglay=siglay, direction='x')

    # Plot waypoints
    # plot_waypoints_and_data(way_points, x_coords, y_coords, title='AUV Lawnmower Path (Clover Pattern)')

    # Simulate AUV path and collect data
    measurements, sample_coords = path_with_samples(chemical_dataset, bubble_dataset, start_time, speed, way_points, sample_frequency, threshold, bowtie, data_var, sphere_radius=sphere_radius)

    # Plot the results
    plot(chemical_dataset, bubble_dataset, x_coords, y_coords, z_coords, sample_coords, data_var, title='AUV Lawnmower Path (Clover Pattern)', save_as_gif=False, gif_filename='auv_path_LM_clover.gif')
