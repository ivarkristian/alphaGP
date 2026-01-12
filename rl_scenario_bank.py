# %%
import matplotlib.pyplot as plt
import torch
import os
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import chem_utils

Record = Dict[str, Any]

# %%
# Disable LaTeX rendering to avoid the need for an external LaTeX installation
# Use MathText for LaTeX-like font rendering
plt.rcParams.update({
    "text.usetex": False,  # Disable external LaTeX usage
    "font.family": "Dejavu Serif",  # Use a serif font that resembles LaTeX's default
    "mathtext.fontset": "dejavuserif"  # Use DejaVu Serif font for mathtext, similar to LaTeX fonts
})

# Define the Scenario Bank class
class ScenarioBank:
    def __init__(self, data_dir = '../scenario_1c_medium/'):
        self.data_dir = data_dir
        self.data_file = None
        self.dataset = None
        self.environments = []

        # Read and clean list of .nc files
        files = os.listdir(data_dir)
        # Create a new list with strings that end with '.nc'
        self.nc_files = [s for s in files if s.endswith('.nc')]
        self.nc_files.sort()
        self.print_data_files()
    
    def print_data_files(self):
        print(f'Directory {self.data_dir} contains these files of type .nc:\n{self.nc_files}')
        print(f'Run .load_dataset(nc_file) to load a file as a dataset')
        #print(f'Use .convert_files_to_tensors() to save datasets as tensors for RL')

    def load_dataset(self, nc_file=None):
        if isinstance(nc_file, int):
            data_file = self.data_dir + self.nc_files[nc_file]
        elif isinstance(nc_file, str):
            if nc_file in self.nc_files:
                data_file = self.data_dir + nc_file
            else:
                print(f'File {nc_file} not found in data_dir {self.data_dir}')
                return
        else:
            print(f'EnvironmentWrapper.load_dataset: Parameter {nc_file} not recognized as int or str')
            return

        self.data_file = data_file
        self.dataset = chem_utils.load_chemical_dataset(self.data_file)
    
    def get_env(self, parameter='pH', depth=67, time=1):
        self.parameter = parameter
        self.depth = depth
        self.time = time
        val_dataset = self.dataset[self.parameter].isel(time=self.time, siglay=self.depth)
        values = val_dataset.values[:72710]
        x_dataset = val_dataset['x'].values[:72710]
        y_dataset = val_dataset['y'].values[:72710]
        x = x_dataset - x_dataset.min()
        y = y_dataset - y_dataset.min()
        env_xy = torch.tensor(np.column_stack((x, y)), dtype=torch.float32)

        # add mean current strength and direction
        u = self.dataset['u'].isel(time=time, siglay=depth)
        v = self.dataset['v'].isel(time=time, siglay=depth)
        u = u.values[:72710]
        v = v.values[:72710]
        cur_dir = np.atan2(v.mean(), u.mean())/np.pi*180.0
        cur_str = np.sqrt(u.mean()**2 + v.mean()**2)

        metadata = {'parameter': parameter, 'depth': depth, 'time': str(self.dataset['time'].values[time]).split('T')[1].split('.')[0], 'cur_dir': cur_dir, 'cur_str': cur_str, 'data_file': self.data_file}
        return env_xy, torch.tensor(values), metadata
    
    def add_env(self, parameter='pH', depth=67, time=1):
        env_xy, values, metadata = self.get_env(parameter, depth, time)
        self.environments.append({'coords': env_xy, 'values': values, 'parameter': metadata['parameter'], 'depth': metadata['depth'], 'time': metadata['time'], 'cur_dir': metadata['cur_dir'], 'cur_str': metadata['cur_str']})
        print(f"Loaded environment: {parameter}, depth={metadata['depth']}, time={metadata['time']} ({self.data_file})")
        
    def print_info(self):
        print(f'Current directory: {self.data_dir}')
        print(f'Data file: {self.data_file}')
        print(f'Loaded dataset: {self.dataset}')
        self.print_envs_info()
    
    def print_envs_info(self):
        print(f"Loaded environments:")
        for env in self.environments:
            print(f"{env['parameter']} (depth={env['depth']}, time={env['time']})")
    
    def add_all_envs_in_data_dir(self, parameter, depth_range, time_range):

        for file in self.nc_files:
            self.load_dataset(nc_file=file)
            for time in range(time_range[0], time_range[1]):
                for depth in range(depth_range[0], depth_range[1]):
                    if time > 0 or file != self.nc_files[0]:
                        self.add_env(parameter, depth, time)

        return

    def downsample_all_envs(self, radius=1.0, method='mean'):

        for i in range(len(self.environments)):
            self.downsample_env(i, radius, method)

    def downsample_env(self, env_num, radius=1.0, method='mean'):

        # Radius of sample averaging
        downsampled = torch.zeros(len(self.coords_flat), dtype=torch.float32)
        env = self.environments[env_num]
        
        for c, coord in enumerate(self.coords_flat):
            downsampled[c] = chem_utils.extract_synoptic_chemical_data_from_depth(env['coords'][:, 0], env['coords'][:, 1], env['values'], coord.numpy(), radius, method)
        
        self.environments[env_num]['coords'] = self.coords_flat
        self.environments[env_num]['values'] = downsampled

        print(f"Downsampled env {env_num} ({env['parameter']} time: {env['time']} depth: {env['depth']})")
        
        return

    def save_envs(self, environments: List[Record], file_path: str | Path) -> None:
        """
        Save a list of dicts (tensors, strings, etc.) to disk with torch.save.

        Parameters
        ----------
        records   : list of dictionaries, each having keys
                    ["coords", "values", "parameter", "depth", "time"].
        file_path : destination file (.pt or .pkl extension recommended).
        """

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Move tensors to CPU so the file is device-agnostic
        safe_records: List[Record] = []
        for rec in environments:
            new_rec: Record = {}
            for k, v in rec.items():
                if torch.is_tensor(v):
                    new_rec[k] = v.detach().cpu()
                else:
                    new_rec[k] = v
            safe_records.append(new_rec)

        torch.save(safe_records, file_path)
        print(f'Saved {len(self.environments)} environments to {file_path}')
    
        return
    
    def load_envs(self, file_path: str | Path, device: str | torch.device = "cpu") -> List[Record]:
        """
        Load the list back into memory.

        Parameters
        ----------
        file_path : path produced by `save_records`.
        device    : "cpu", "cuda", or torch.device; tensors will be mapped here.
        """
        self.environments: List[Record] = torch.load(file_path, map_location=device, weights_only=False)
        print(f'Loaded {len(self.environments)} environments from {file_path}')

        return

    def plot_env(self, env_num=0, title_postfix=None, path=None, x_range=[0, 250], y_range=[0, 250]):
        if len(self.environments) <= env_num:
            print(f'Bank contains {len(self.environments)} environments (0-indexed). Tried to plot #{env_num}')
            return
        
        env = self.environments[env_num]
        if env['parameter']:
            x = env['coords'][:, 0]
            y = env['coords'][:, 1]
            c = env['values']
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(x, y, c=c, cmap='coolwarm', s=1, vmin=c.min(), vmax=c.max())
            if path is not None:
                ax.scatter(path[:, 0], path[:, 1], c='black', s=1)
            
            ax.set_xlim(x_range[0], x_range[1])
            ax.set_ylim(y_range[0], y_range[1])
            
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label('(Value)')

            # Add labels and title
            ax.set_xlabel('Easting [m]')
            ax.set_ylabel('Northing [m]')
            ax.set_title(f"Time {env['time']}, {env['parameter']} at {env['depth']}m depth ({title_postfix})")

            return fig, ax
        
        print(f"Could not plot dataset = {self.dataset}, parameter = {env['parameter']}")
    
    def plot_above_threshold(self, threshold=None, title_str='Environments'):

        if threshold is None:
            threshold = self.get_minmax()[0]

        # compute percentage above threshold for each environment
        percentages = []
        for env in self.environments:
            vals = np.asarray(env['values'])
            pct = (vals > threshold).sum() / len(vals) * 100
            percentages.append(pct)

        # sort values and create labels
        sorted_idx = np.argsort(percentages)[::-1]
        sorted_pcts = np.array(percentages)[sorted_idx]

        # plot
        fig = plt.figure(figsize=(8, 5))
        bars = plt.bar(range(len(sorted_pcts)), sorted_pcts, color='black', edgecolor=None, width=1.0)

        ax = plt.gca()
        ax.tick_params(labelsize=16)
        plt.ylabel(f'% of locations > {threshold}', fontsize=16)
        plt.xlabel(title_str, fontsize=16)
        #plt.title('Share of high-concentration locations', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        fig.savefig(f"figures/percentage_above_{threshold}_{title_str}.eps", format="eps", dpi=300, bbox_inches="tight")
        return fig

    def sample(self):
        return random.choice(self.environments)

    def get_minmax(self):
        maxes = torch.zeros(len(self.environments))
        mins = torch.zeros(len(self.environments))
        for c, env in enumerate(self.environments):
            maxes[c] = env['values'].max()
            mins[c] = env['values'].min()
        
        return mins.min(), maxes.max()
    
    def get_mu_sigma2(self, biased=True):
        ns = torch.zeros(len(self.environments))
        mus = torch.zeros(len(self.environments))
        sigma2s = torch.zeros(len(self.environments))
        for c, env in enumerate(self.environments):
            ns[c] = len(env['values'])
            mus[c] = env['values'].mean()
            sigma2s[c] = env['values'].var()
        
        N = ns.sum()
        mu_all = (ns * mus).sum() / N
        
        if biased:
            ss = ns * (sigma2s + mus**2)
            sigma2_all = ss.sum() / N - mu_all**2
        else:
            within  = ((ns - 1) * sigma2s).sum()
            between = (ns * (mus - mu_all) ** 2).sum()
            sigma2_all = (within + between) / (N - 1)

        return mu_all, sigma2_all
    
    def clip_sensor_range(self, parameter=None, min=0, max=2000):
        if not parameter:
            print(f'No sensor parameter given')
            return
        
        for i in range(len(self.environments)):
            torch.clamp_(self.environments[i]['values'], min, max)

    def gas_coverage_cutoff(
        self,
        cutoff_concentration=0,
        cutoff_percentage_min=0.0,
        cutoff_percentage_max=np.inf,
    ):
        """
        Filter environments based on percentage of grid points exceeding
        a concentration threshold.

        Keeps environments where:
            cutoff_percentage_min <= plume_coverage <= cutoff_percentage_max
        """
        percentages = np.zeros(len(self.environments))

        for i, env in enumerate(self.environments):
            vals = np.asarray(env["values"])
            pct = (vals > cutoff_concentration).sum() / len(vals) * 100.0
            percentages[i] = pct

        # Apply interval filter
        kept = [
            env
            for env, pct in zip(self.environments, percentages)
            if cutoff_percentage_min <= pct <= cutoff_percentage_max
        ]

        n_removed = len(self.environments) - len(kept)
        self.environments = kept

        print(
            f"Removed environments with gas plume coverage outside "
            f"[{cutoff_percentage_min}%, {cutoff_percentage_max}%]"
        )
        print(f"(Gas plume defined as concentration > {cutoff_concentration})")
        print(f"{len(self.environments)} environments left ({n_removed} removed)")

    def gas_coverage_cutoff_old(self, cutoff_concentration=0, cutoff_percentage=0):
        # compute percentage above threshold for each environment
        percentages = np.zeros(len(self.environments))
        for i, env in enumerate(self.environments):
            vals = np.asarray(env['values'])
            pct = (vals > cutoff_concentration).sum() / len(vals) * 100
            percentages[i] = pct

        self.environments = [env for env, pct in zip(self.environments, percentages) if pct >= cutoff_percentage]
        print(f'Removed environments where coverage of gas plume < {cutoff_percentage}%')
        print(f'(Gas plume defined as concentration >= {cutoff_concentration})')
        print(f'{len(self.environments)} environments left')

    def create_obs_coords(self, resolution):
        
        # Downsampling based on given pred_resolution
        obs_x, obs_y = resolution
        env_x_max = self.environments[-1]['coords'][:, 0].max()
        env_y_max = self.environments[-1]['coords'][:, 1].max()

        # -- 1. grid of query points -----------------
        #   (H*W, 2) tensor that GPyTorch will accept.
        xs = np.linspace(0, env_x_max, obs_x, dtype=np.float32)
        ys = np.linspace(0, env_y_max, obs_y, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)                        # shape (H, W)

        # Save as 2‑D field for coord‑channels and as flat list for GP queries
        coords = np.stack([gx, gy], axis=-1)      # (H, W, 2)
        self.coords_flat = torch.from_numpy(coords.reshape(-1, 2))
    
    def rotate_xy(self, env_xy: torch.Tensor, d: float | int) -> torch.Tensor:
        """
        Rotate 2-D coordinates `env_xy` by `d` degrees **clockwise**.

        Returns
        -------
        rotated : (N, 2) torch.Tensor
            Rotated coordinates, same dtype and device as `env_xy`.
        """
        t = torch.tensor([env_xy[:, 0].mean(), env_xy[:, 1].mean()], device=env_xy.device)
        env_xy_zero_translated = env_xy - t
        # ensure float dtype on the same device as the input
        theta = torch.deg2rad(torch.as_tensor(d, dtype=env_xy.dtype,
                                            device=env_xy.device))

        c, s = torch.cos(theta), torch.sin(theta)
        rot_mat = torch.stack((torch.stack(( c,  -s)),
                            torch.stack((s,  c))))
        
        env_xy_rot = env_xy_zero_translated @ rot_mat.T

        return env_xy_rot + t
    
    def offset_xy(self, env_xy, values, max_offset_factors):
        # Offset so that source is not always in the middle
        max_x_off = int(env_xy[:, 0].max()/2 * max_offset_factors[0])
        max_y_off = int(env_xy[:, 1].max()/2 * max_offset_factors[1])
        self.x_off = random.randint(-max_x_off, max_x_off)
        self.y_off = random.randint(-max_y_off, max_y_off)
        
        x_max = env_xy[:, 0].max()
        y_max = env_xy[:, 1].max()
        x_min = env_xy[:, 0].min()
        y_min = env_xy[:, 1].min()
        
        env_xy[:, 0] += self.x_off
        env_xy[:, 1] += self.y_off
        
        values[env_xy[:, 0] > x_max] = values.min()
        values[env_xy[:, 0] < x_min] = values.min()
        values[env_xy[:, 1] > y_max] = values.min()
        values[env_xy[:, 1] < y_min] = values.min()
        env_xy[:, 0][env_xy[:, 0] > x_max] -= x_max
        env_xy[:, 0][env_xy[:, 0] < x_min] += x_max
        env_xy[:, 1][env_xy[:, 1] > y_max] -= y_max
        env_xy[:, 1][env_xy[:, 1] < y_min] += y_max

        return env_xy, values
    
def plot_currents_in_dataset(bank, depths=[], times=[], save_prefix="currents"):

    if not hasattr(bank, 'dataset'):
        print('You have to load a dataset first')
        return
    
    if not len(depths) or not len(times):
        print(f'You have to give both depths and times')
        return
    
    cur_dirs = []
    cur_strs = []
    timestamps = []
    
    for d in depths:
        bank.environments = []
        for t in times:
            bank.add_env('pCO2', depth=d, time=t)
        #bank.plot_above_threshold(405)
        cur_dir_tmp = []
        cur_str_tmp = []
        for env in bank.environments:
            cur_dir_tmp.append(env['cur_dir'])
            cur_str_tmp.append(env['cur_str'])
            if d is depths[0]:
                timestamps.append(env['time'])
        cur_dirs.append(np.array(cur_dir_tmp))
        cur_strs.append(np.array(cur_str_tmp))
    
    plot_currents_by_depth(cur_dirs, cur_strs, depths, times=timestamps, save_prefix=save_prefix)
    #plt.show()


def plot_currents_by_depth(cur_dirs, cur_strs, depths, *, times=None,
                        dir_units="deg", unwrap=True,
                        figsize=(6.8, 3.2), dpi=300,
                        save_prefix=None):
    """
    Create two paper-ready line plots:
    1) Current direction vs time (one line per depth)
    2) Current strength vs time (one line per depth)

    Parameters
    ----------
    cur_dirs : list[np.ndarray]
        List of 1D arrays with current direction for each depth.
    cur_strs : list[np.ndarray]
        List of 1D arrays with current strength for each depth.
    depths : list[int|float]
        Depth labels (same length as cur_dirs/cur_strs).
    times : array-like or None
        X-axis values. If None, uses np.arange(len(cur_dirs[0])).
        If you want actual snapshot times, pass the same t values used when building the envs.
    dir_units : {"deg","rad"}
        Units of cur_dirs input.
    unwrap : bool
        If True, unwrap direction to avoid 0/360 jumps (recommended for line plots).
    figsize : tuple
        Figure size in inches.
    dpi : int
        Resolution for saving.
    save_prefix : str or None
        If provided, saves figures as f"{save_prefix}_cur_dir.png" and "_cur_str.png".

    Returns
    -------
    fig_dir, ax_dir, fig_str, ax_str
    """
    if len(cur_dirs) != len(cur_strs) or len(cur_dirs) != len(depths):
        raise ValueError("cur_dirs, cur_strs, and depths must have the same length")

    # Ensure all series have same length
    nT = len(cur_dirs[0])
    for arr in cur_dirs + cur_strs:
        if len(arr) != nT:
            raise ValueError("All cur_dirs/cur_strs arrays must have the same length")

    if times is None:
        times = np.arange(nT)
        time_label = "Snapshot index"
    else:
        times = np.asarray(times)
        if len(times) != nT:
            raise ValueError("times must have the same length as the direction/strength arrays")
        time_label = "Time index"

    # --------- Plot 1: Current direction ---------
    fig_dir, ax_dir = plt.subplots(figsize=figsize, constrained_layout=True)

    for d, cd in zip(depths, cur_dirs):
        cd = np.asarray(cd).astype(float)

        # Convert to radians for unwrap if needed
        if dir_units == "deg":
            cd_rad = np.deg2rad(cd)
        elif dir_units == "rad":
            cd_rad = cd
        else:
            raise ValueError("dir_units must be 'deg' or 'rad'")

        if unwrap:
            cd_rad = np.unwrap(cd_rad)

        # Back to degrees for plotting (most readable)
        cd_plot = np.rad2deg(cd_rad)

        ax_dir.plot(times, cd_plot, linewidth=1.6, label=f"{d} m")

    ax_dir.set_xticks(ax_dir.get_xticks()[::6])
    ax_dir.set_xticklabels(ax_dir.get_xticklabels(), rotation=45)
    ax_dir.set_xlabel(time_label)
    ax_dir.set_ylabel("Current direction [deg]")
    ax_dir.grid(True, which="both", linewidth=0.6, alpha=0.4)
    ax_dir.legend(title="Depth", frameon=True, fontsize=8, title_fontsize=9, ncols=2)

    # Optional: keep y-range sensible if you didn't unwrap
    if not unwrap:
        ax_dir.set_ylim(0, 360)

    # --------- Plot 2: Current strength ---------
    fig_str, ax_str = plt.subplots(figsize=figsize, constrained_layout=True)

    for d, cs in zip(depths, cur_strs):
        cs = np.asarray(cs).astype(float)
        ax_str.plot(times, cs, linewidth=1.6, label=f"{d} m")

    ax_str.set_xticks(ax_str.get_xticks()[::6])
    ax_str.set_xticklabels(ax_str.get_xticklabels(), rotation=45)
    ax_str.set_xlabel(time_label)
    ax_str.set_ylabel("Current strength")
    ax_str.grid(True, which="both", linewidth=0.6, alpha=0.4)
    ax_str.legend(title="Depth", frameon=True, fontsize=8, title_fontsize=9, ncols=2)

    # --------- Save (optional) ---------
    if save_prefix is not None:
        fig_dir.savefig(f"figures/{save_prefix}_cur_dir.png", dpi=dpi, bbox_inches="tight")
        fig_str.savefig(f"figures/{save_prefix}_cur_str.png", dpi=dpi, bbox_inches="tight")

    return fig_dir, ax_dir, fig_str, ax_str

# %%
if __name__ == '__main__':
    # Initialize bank object
    bank = ScenarioBank(data_dir='../my_data_dir/')

    # Load scenarios from netCDF files
    depth_range = [67, 70]
    time_range = [0, 12]
    bank.add_all_envs_in_data_dir('pCO2', depth_range, time_range)

    # Define downsampling resolution
    bank.create_obs_coords([250, 250])
    bank.downsample_all_envs()

    # Save to file
    bank.save_envs(bank.environments, 'tensor_envs/my_file.pt')

    # Load from file
    bank.load_envs('tensor_envs/my_file.pt')


# %%
