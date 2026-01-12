import numpy as np
import path_utils
import dubins

def bowtie(center, a=25, steps=30):
    """
    Generates a bowtie pattern centered at a specified point.

    Parameters
    ----------
    center : array-like
        The center of the bowtie.
    a : float, optional
        The amplitude of the bowtie pattern. Defaults to 10.
    steps : int, optional
        Number of points in the pattern. Defaults to 30.

    Returns
    -------
    numpy.ndarray
        Coordinates of the bowtie pattern.
    """
    t = np.linspace(0, 2 * np.pi, steps)
    x = a * np.sin(t)
    y = a * np.sin(t) * np.cos(t)
    z = np.zeros(steps)
    return np.column_stack((x, y, z)) + center

def bowtie_tilted(center, a=10, steps=30, tilt_angle=0):
    waypoints = bowtie(center, a, steps)
    xy_coords = [(item[0], item[1]) for item in waypoints]
    z_coords = [item[2] for item in waypoints]
    
    xy_coords_translated = xy_coords - center[:2]
    rotated_coords = path_utils.rotate_points(np.array(xy_coords_translated), tilt_angle)
    
    rotated_coords_xyz = [(rot_coord[0], rot_coord[1], z_coords[i]) for i, rot_coord in enumerate(rotated_coords)]
    return rotated_coords_xyz + center

def bowtie_double(center, a=10, steps=30):
    bowtie_coords = bowtie(center, a, steps)
    bowtie_tilted_coords = bowtie_tilted(center, a, steps, tilt_angle=90)
    return np.vstack((bowtie_coords, bowtie_tilted_coords))

def cross(center, length=10):
    """
    Generates a cross pattern centered at a specified point.

    Parameters
    ----------
    center : array-like
        The center of the cross.
    length : float, optional
        The length of each line in the cross. Defaults to 10.

    Returns
    -------
    numpy.ndarray
        Coordinates of the cross pattern.
    """
    lines = [
        [-length/2, 0, 0], [length/2, 0, 0],
        [0, -length/2, 0], [0, length/2, 0]
    ]
    return np.array([line + center for line in lines])

def crisscross(center, length=10):
    """
    Generates a crisscross pattern centered at a specified point.

    Parameters
    ----------
    center : array-like
        The center of the crisscross.
    length : float, optional
        The length of each line in the crisscross. Defaults to 10.

    Returns
    -------
    numpy.ndarray
        Coordinates of the crisscross pattern.
    """
    lines = [
        [-length/2, 0, 0], [length/2, 0, 0],
        [0, -length/2, 0], [0, length/2, 0],
        [-length/2, -length/2, 0], [length/2, length/2, 0],
        [-length/2, length/2, 0], [length/2, -length/2, 0]
    ]
    return np.array([line + center for line in lines])

def drifting_circle(center, radius=10, drift=1, steps=30):
    """
    Generates a drifting circle pattern centered at a specified point.

    Parameters
    ----------
    center : array-like
        The center of the drifting circle.
    radius : float, optional
        The radius of the circle. Defaults to 10.
    drift : float, optional
        The drift along the z-axis. Defaults to 1.
    steps : int, optional
        Number of points in the pattern. Defaults to 30.

    Returns
    -------
    numpy.ndarray
        Coordinates of the drifting circle pattern.
    """
    t = np.linspace(0, 2 * np.pi, steps)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = drift * t / (2 * np.pi)
    return np.column_stack((x, y, z)) + center

def square(center, length=10):
    """
    Generates a square pattern centered at a specified point.

    Parameters
    ----------
    center : array-like
        The center of the square.
    length : float, optional
        The side length of the square. Defaults to 10.

    Returns
    -------
    numpy.ndarray
        Coordinates of the square pattern.
    """
    half = length / 2
    lines = [
        [half, half, 0], [-half, half, 0],
        [-half, half, 0], [-half, -half, 0],
        [-half, -half, 0], [half, -half, 0],
        [half, -half, 0], [half, half, 0]
    ]
    return np.array([line + center for line in lines])

def square_dubins(center, length=20, turn_radius=10, point_sep=1.0):
    """
    Generates a square pattern centered at a specified point. The lines
    in the square are connected with dubins paths with the specified turn radius.
    """
    waypoints = square(center, length)
    orientations = np.deg2rad([180, 180, -90, -90, 0, 0, 90, 90])

    # verify that orientations and waypoints have same length
    planner = dubins.Dubins(radius=turn_radius, point_separation=point_sep)
    start = (center[0], center[1], 0)
    end = (waypoints[0][0], waypoints[0][1], orientations[0])
    print(f'start:{start} end: {end}')
    waypoints_dubins = planner.dubins_path(start, end)

    for i in range(len(orientations)-1):
        start = (waypoints[i][0], waypoints[i][1], orientations[i])
        end = (waypoints[i+1][0], waypoints[i+1][1], orientations[i+1])
        print(f'start:{start} end: {end}')
        new_points = planner.dubins_path(start, end)
        waypoints_dubins = np.vstack((waypoints_dubins, new_points))

    start = (waypoints[-1][0], waypoints[-1][1], orientations[-1])
    end = (center[0], center[1], 0)
    print(f'start:{start} end: {end}')
    new_points = planner.dubins_path(start, end)
    waypoints_dubins = np.vstack((waypoints_dubins, new_points))

    return [(wp[0], wp[1], center[2]) for wp in waypoints_dubins]

def leaf_clover(center, radius=10, steps=30):
    """
    Generates a leaf clover pattern centered at a specified point.

    Parameters
    ----------
    center : array-like
        The center of the leaf clover.
    radius : float, optional
        The radius of each leaf. Defaults to 10.
    steps : int, optional
        Number of points in each leaf. Defaults to 30.

    Returns
    -------
    numpy.ndarray
        Coordinates of the leaf clover pattern.
    """
    patterns = []
    for i in range(4):
        angle = np.pi/2 * i
        x = radius * np.cos(np.linspace(angle, angle + np.pi, steps))
        y = radius * np.sin(np.linspace(angle, angle + np.pi, steps))
        z = np.zeros(steps)
        patterns.append(np.column_stack((x, y, z)) + center)
    return np.vstack(patterns)

def spiral(center, b=1, steps=30):
    """
    Generates a spiral pattern centered at a specified point.

    Parameters
    ----------
    center : array-like
        The center of the spiral.
    b : float, optional
        The coefficient controlling the spiral tightness. Defaults to 1.
    steps : int, optional
        Number of points in the spiral. Defaults to 30.

    Returns
    -------
    numpy.ndarray
        Coordinates of the spiral pattern.
    """
    t = np.linspace(0, 4 * np.pi, steps)
    r = b * t
    x = r * np.cos(t)
    y = r * np.sin(t)
    z = np.zeros(steps)
    return np.column_stack((x, y, z)) + center
