# %%
#from memory_profiler import profile
import gpytorch.constraints
import torch
import gpytorch
import gc
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import matplotlib.pyplot as plt
import time
from typing import Tuple, Union, Sequence
from scipy.ndimage import shift      # comes with SciPy
from scipy.spatial import cKDTree

import dubins
from gpt_class_exactgpmodel import ExactGPModel
import chem_utils
#import agents

# %%
# Definitions

# %%
class GasSurveyDubinsEnv(gym.Env):
    def __init__(self, scenario_bank=None, gp_ls_constraint=gpytorch.constraints.Interval(9, 11), gp_kernel_type='scale_rbf', gp_pred_resolution=[100, 100], r_weights=[1.0, 1.0, 1.0], turn_radius=250, channels=np.array([0, 1, 0, 0, 0]), reward_func='None', timer=False, debug=False, device=torch.device("cpu"), return_torch=False):
        super(GasSurveyDubinsEnv, self).__init__()
        self.debug = debug
        self.timer = timer
        self.return_torch = return_torch
        self.print_info_rate = 10
        self.turn_radius = turn_radius
        self.path_planner = dubins.Dubins(self.turn_radius-3, 1.0) #1.0 - sample every meter

        self.reward_func = reward_func
        self.a_gas, self.a_var, self.a_dist = map(float, r_weights)

        # Normalize device early so all tensors land on the intended backend.
        self.device = torch.device(device)
        # Load scenario bank
        if scenario_bank is None:
            raise ValueError("You have to provide a scenario bank.")
        
        self.scenario_bank = scenario_bank
        self.max_offset_factors = (0.7, 0.7)
        self.min_concentration, self.max_concentration = map(
            float, self.scenario_bank.get_minmax()
        )
        self.mu_all, self.sigma2_all = map(float, self.scenario_bank.get_mu_sigma2())
        
        # GP model parameters
        self.ls_const = gp_ls_constraint
        self.kernel_type = gp_kernel_type
        self.obs_x, self.obs_y = map(int, gp_pred_resolution)

        # Steps until truncated=True (done)
        self.n_steps = 0
        self.n_episodes = 0
        self.n_steps_max = 100
        self.total_steps = 0
        self.acc_reward = 0.0
        
        # μ, σ, location, coord‑Y, coord‑X  → 5 possible channels
        # Could instead of location include 'visited' channel
        # Including coord_x/y channels is a bit dangerous, should
        # randomize direction of scenarios, e.g. rotate by 90/180 deg
        # to avoid 'learning the coordinate system'
        # Make a private copy to avoid accidental in-place mutation by the caller.
        self.channels = np.asarray(channels, dtype=np.uint8).copy()
        self._coords_signature = None
        self._coord_tree = None

        self.max_samples = 0
        self.location_noise = 0.05
        self.location_radius = self.ls_const.upper_bound.item()
        # reset draws a random scenario, initializes GP model and sample memory
        self.reset()
    
        # observation space
        self.observation_layers = spaces.Box(
            low=0, high=255, shape=(self.channels.sum(), self.obs_x, self.obs_y), dtype=np.uint8
        )
        
        self.observation_space = spaces.Dict({
            "map": self.observation_layers,
            "loc": spaces.Box(-1.0, 1.0, (2,), np.float32),
            "hdg": spaces.MultiBinary(8)
        })

        #Discrete action space, left - 0, straight - 1, right - 2:
        self.action_space = spaces.Discrete(3)

        print(f'Init dubins env, \ndevice: {self.device}\nturn_radius: {self.turn_radius}\nchannels: {self.channels}')

    #@profile
    def reset(self, seed=None, options=None, random_scenario=None, env_xy=None, values=None):
        
        if self.n_episodes % self.print_info_rate == 0 and self.n_episodes:
            print(f'Ep {self.n_episodes}, mean reward = {(self.acc_reward/self.print_info_rate):.3}')
            self.acc_reward = 0

        if self.timer:
            t = time.process_time()
        self.n_episodes += 1
        self.total_steps += self.n_steps
        self.n_steps = 0
        self.terminated = False

        if random_scenario is None:
            # Draw a random scenario/snapshot
            random_scenario = self.scenario_bank.sample()
            self.rotation = random.choice([-90, 0, 90, 180])

            env_xy = self.scenario_bank.rotate_xy(random_scenario['coords'].to(self.device), self.rotation)
            values = random_scenario['values'].to(self.device)
            self.cur_dir = random_scenario['cur_dir'] + self.rotation

            self.env_xy, self.values = self.scenario_bank.offset_xy(env_xy, values, self.max_offset_factors)
        else:
            # Ensure provided data is on the target device for GP inference.
            if env_xy is None:
                raise ValueError("env_xy must be provided when random_scenario is not None")
            self.env_xy = env_xy.to(self.device) if isinstance(env_xy, torch.Tensor) else torch.as_tensor(env_xy, device=self.device)
            if values is None:
                self.values = None
            else:
                self.values = values.to(self.device) if isinstance(values, torch.Tensor) else torch.as_tensor(values, device=self.device)
            self.cur_dir = random_scenario['cur_dir']

        if self.env_xy is None:
            raise ValueError("env_xy is None after offset_xy")
        
        # Cache torch views for sampling to keep data on the target device.
        self.env_x = self.env_xy[:, 0]
        self.env_y = self.env_xy[:, 1]

        self.parameter = random_scenario['parameter']
        self.depth = random_scenario['depth']
        self.time = random_scenario['time']
        
        self.cur_str = random_scenario['cur_str']

        if self.debug:
            print(f"Sampled env '{self.parameter}', depth {self.depth}', time {self.time}")

        self.env_x_max = float(self.env_xy[:, 0].max())
        self.env_y_max = float(self.env_xy[:, 1].max())

        self.maxdist=2*math.pi*self.turn_radius/4.0

        # Rebuild coordinate grid only if the scenario bounds changed.
        coords_signature = (self.env_x_max, self.env_y_max, self.obs_x, self.obs_y)
        if coords_signature != self._coords_signature:
            self._create_obs_coords()
            self._coords_signature = coords_signature
            self._coord_tree = None  # cached KD-tree invalidated by new grid

        # Reuse observation buffers to avoid per-reset allocations.
        obs_shape = (self.obs_y, self.obs_x)
        if (not hasattr(self, "pred_mu_norm")) or (self.pred_mu_norm.shape != obs_shape):
            self.pred_mu_norm = np.zeros(obs_shape, dtype=np.uint8)
            self.pred_mu_norm_clipped = np.zeros(obs_shape, dtype=np.uint8)
            self.pred_var_norm = np.full(obs_shape, 255, dtype=np.uint8)
            self.pred_var_norm_clipped = np.full(obs_shape, 255, dtype=np.uint8)
            self.location = np.zeros(obs_shape, dtype=np.uint8)
        else:
            self.pred_mu_norm.fill(0)
            self.pred_mu_norm_clipped.fill(0)
            self.pred_var_norm.fill(255)
            self.pred_var_norm_clipped.fill(255)
            self.location.fill(0)
        if (not hasattr(self, "location_t")) or (self.location_t.shape != obs_shape):
            self.location_t = torch.zeros(obs_shape, device=self.device, dtype=torch.uint8)
        else:
            self.location_t.zero_()

        # Init GP model
        had_model = hasattr(self, 'mdl') or hasattr(self, 'llh')
        if hasattr(self, 'mdl'):
            self.mdl.cpu()
            del self.mdl
            
        if hasattr(self, 'llh'):
            self.llh.cpu()
            del self.llh
        
        if had_model:
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        self.llh = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        empty_x = torch.empty((0, 2), device=self.device)
        empty_y = torch.empty((0,), device=self.device)
        self.mdl = ExactGPModel(
            empty_x,
            empty_y,
            self.llh,
            type=self.kernel_type,
            lengthscale_constraint=self.ls_const,
        ).to(self.device)
        self.mdl.covar_module.outputscale = torch.tensor(self.sigma2_all, device=self.device)
        self.mdl.eval()
        self.llh.eval()
       
        #self.values_submuall = self.values-self.mu_all
        
        # Init sample memory. Could include lawnmower path samples.
        self.max_samples_old = self.max_samples
        self.max_samples = int((round(self.maxdist+0.5) + 1) * (1 + self.location_noise) *(self.n_steps_max + 1))
        self.sample_idx = 0
        self.sample_idx_mdl = 0

        if (hasattr(self, 'sampled_coords') is False) or (self.max_samples != self.max_samples_old):
            # Preallocate on device for GP updates without reallocating each step.
            self.sampled_coords = torch.empty((self.max_samples, 2), device=self.device, dtype=torch.float32)
        
        if (hasattr(self, 'sampled_vals') is False) or (self.max_samples != self.max_samples_old):
            self.sampled_vals = torch.empty(self.max_samples, device=self.device, dtype=torch.float32)
        
        # Init location and heading. Should be random, but for Dubins paths
        # we ensure that location is not too close to area boundaries
        rng_x = self.env_x_max - self.turn_radius*3
        rng_y = self.env_y_max - self.turn_radius*3
        loc_x = rng_x * random.random() + self.turn_radius*1.5
        loc_y = rng_y * random.random() + self.turn_radius*1.5
        #loc_x = (self.env_x_max-1) * random.random()
        #loc_y = (self.env_y_max-1) * random.random()
        #self.heading = np.zeros(4)
        #self.heading[random.choice([0, 1, 2, 3])] += 1
        self.heading = np.zeros(8, dtype=np.int8)
        self.heading[random.choice([0, 1, 2, 3, 4, 5, 6, 7])] += 1

        # Keep a CPU copy of XY to avoid GPU->CPU sync in geometry/path code.
        self.loc_xy_np = np.array([loc_x, loc_y], dtype=np.float32)
        self.loc = torch.tensor([loc_x, loc_y, self.depth], device=self.device, dtype=torch.float32)
        if self.debug:
            print(f'reset loc: {loc_x:.2f}, {loc_y:.2f} hdg: {self.heading}')

        self.make_circle(self.loc_xy_np[0], self.loc_xy_np[1], self.location_radius)

        # Init prediction tensors
        if (not hasattr(self, "pred_mu")) or (self.pred_mu.shape != obs_shape):
            self.pred_mu = np.full(obs_shape, self.mu_all, dtype=np.float32)
            self.pred_var = np.full(obs_shape, self.sigma2_all, dtype=np.float32)
        else:
            self.pred_mu.fill(self.mu_all)
            self.pred_var.fill(self.sigma2_all)

        # Torch copies keep reward computations on-device.
        self.pred_mu_t = torch.full(obs_shape, self.mu_all, device=self.device, dtype=torch.float32)
        self.pred_var_t = torch.full(obs_shape, self.sigma2_all, device=self.device, dtype=torch.float32)
        self.pred_mu_norm_t = (self.pred_mu_t - self.min_concentration) / (self.max_concentration - self.min_concentration) * 255
        self.pred_var_norm_t = self.pred_var_t / self.sigma2_all * 255
        
        if self.debug:
            self._assert_gpu_consistency()

        obs, _, info = self._get_obs_truncated_info()
        
        if self.timer:
            print(f'reset took: {time.process_time() - t}')

        return obs, info

    #@profile
    def step(self, action, speed=1.0, sample_freq=1.0):
        if self.timer:
            tt = time.process_time()
            t = tt
        self.n_steps += 1
        reward = 0.0
        # action = absolute (x,y) or Δx,Δy; clip, update GP, rewards...
        #old_ind_y, old_ind_x = np.argwhere(self.location)[0]
        # Copy only when needed to avoid large per-step allocations.
        use_var_reward = self.channels[0] == 0 and self.channels[1] == 1
        use_ch11000 = self.channels[0] == 1 and self.channels[1] == 1 and self.reward_func != 'e2e'
        old_var = self.pred_var_norm.copy() if use_var_reward else None
        old_var_t = None
        if use_ch11000:
            # Torch copy keeps reward computation on-device without extra host transfers.
            old_var_t = self.pred_var_norm_t.detach().clone()
        old_pred_mu = None
        if self.channels[0] == 1 and self.channels[1] == 1 and self.reward_func == 'e2e':
            old_pred_mu = self.pred_mu.copy()

        if self.debug:
            print(f'step: {self.n_steps} Q-action: {action}')
        # expects action to be left, forward, right

        n_headings = len(self.heading)
        delta_xy, new_heading = move_with_heading(
            heading_1hot=self.heading, action=action, turn_radius=self.turn_radius,
            turn_degrees=int(360/n_headings), n_headings=n_headings, straight_matches_arc=True
        )
        #delta_xy, new_heading = self._dubins_delta_90(action, self.heading, self.turn_radius)
        noise = self._delta_add_noise(delta_xy, self.turn_radius)
        # Use cached CPU position to avoid device sync on each step.
        new_xy = self.loc_xy_np + delta_xy + noise

        if self.debug:
            print(f'delta_xy: {delta_xy} ({noise}) new_xy: {new_xy} new_hdg: {new_heading}', end=' ')
        out_of_bounds = not ((0 <= new_xy[0] <= self.env_x_max) and (0 <= new_xy[1] <= self.env_y_max))
        facing_the_boundary = self._facing_the_boundary(new_xy, new_heading)
        if out_of_bounds or facing_the_boundary:
            obs, truncated, info = self._get_obs_truncated_info()
            reward += -5.0
            self.acc_reward += reward
            if self.debug:
                print(f'out_of_bounds or facing_the_boundary = True')
            if self.return_torch:
                reward = torch.as_tensor(reward, device=self.device, dtype=torch.float32)
            return obs, float(reward) if not self.return_torch else reward, self.terminated, truncated, info
        else:
            self.new_loc = torch.as_tensor([*new_xy, self.depth], dtype=self.loc.dtype, device=self.device)
        
        if self.timer:
            print(f't0 step: {time.process_time()-t}')

        if torch.allclose(self.loc, self.new_loc):
            obs, truncated, info = self._get_obs_truncated_info()
            reward += -5.0
            self.acc_reward += reward
            if self.debug:
                print(f'torch.allclose = True')
            if self.return_torch:
                reward = torch.as_tensor(reward, device=self.device, dtype=torch.float32)
            return obs, float(reward) if not self.return_torch else reward, self.terminated, truncated, info

        if self.timer:
            t = time.process_time()
        #sample_coords = path.path([self.loc.cpu(), self.new_loc.cpu()], start_time, speed, sample_freq, synoptic)
        start = (self.loc_xy_np[0], self.loc_xy_np[1], onehot_to_rad(self.heading))
        end = (new_xy[0], new_xy[1], onehot_to_rad(new_heading))
        sample_coords_xy = np.asarray(self.path_planner.dubins_path(start, end), dtype=np.float32)
        
        if self.timer:
            print(f't1 step: {time.process_time()-t}')
        
        radius = 1.0 # Radius of sample averaging
        
        if self.timer:
            t = time.process_time()
        # Sample directly on the torch device to avoid CPU/GPU round-trips.
        measurements_t = chem_utils.torch_extract_synoptic_chemical_data_from_depth(
            self.env_x,
            self.env_y,
            self.values,
            sample_coords_xy,
            radius,
        )

        if self.debug:
            if torch.isnan(measurements_t).any():
                print(f'Measurements contains nans')
                #agents.plot_n(x=sample_coords_xy[:, 0], y=sample_coords_xy[:, 1], data_list=[np.ones_like(sample_coords_xy[:, 0])], path=sample_coords_xy)

        if self.timer:
            print(f't2 step: {time.process_time()-t}')
        
        if self.debug:
            print(f'#Smp: {len(measurements_t)}', end=' ')

        end_idx = self.sample_idx + len(sample_coords_xy)
        if end_idx > self.max_samples:
            raise RuntimeError(f"Exceeded maximum number of samples ({end_idx} > {self.max_samples})")

        # Store new samples into the preallocated tensors
        self.sampled_coords[self.sample_idx:end_idx] = torch.as_tensor(sample_coords_xy, device=self.device, dtype=self.sampled_coords.dtype)
        self.sampled_vals[self.sample_idx:end_idx] = measurements_t.to(device=self.device, dtype=self.sampled_vals.dtype)
        self.sample_idx = end_idx

        if self.timer:
            t = time.process_time()
        self._estimate_local() # fill self.pred_mu, self.pred_var and norms
        if self.timer:
            print(f't3 step: {time.process_time()-t}')
        
        # Update location
        # Update both device and CPU locations in sync.
        self.loc = self.new_loc.detach()
        self.loc_xy_np = new_xy
        self.heading = new_heading

        #self.make_circle(self.loc[0].cpu().numpy(), self.loc[1].cpu().numpy(), self.location_radius)
        
        obs, truncated, info = self._get_obs_truncated_info()
        
        if self.channels[0] == 0 and self.channels[1] == 1:
            reward = self._reward_ch_01000(old_var)
        elif self.channels[0] == 1 and self.channels[1] == 1:
            if self.reward_func == 'e2e':
                reward = self._reward_e2e(old_pred_mu)
            else:
                reward = self._reward_ch_11000(old_var_t, measurements_t)

        self.acc_reward += reward
        
        if self.timer:
            print(f'step took: {time.process_time()-tt}')
        
            #self._assert_gpu_consistency()

        if self.return_torch:
            reward = torch.as_tensor(reward, device=self.device, dtype=torch.float32)
        return obs, float(reward) if not self.return_torch else reward, self.terminated, truncated, info
    
    def render():
        pass

    def close():
        pass
    
    def _reward_e2e(self, old_pred_mu):
        old_rms = np.sqrt((old_pred_mu - self.obs_truth).mean()**2)
        rms = np.sqrt((self.pred_mu - self.obs_truth).mean()**2)
        
        if old_pred_mu.mean() == 0:
            old_rms = rms

        if self.debug:
            print(f'old_rms.mean: {old_rms:.4} rms: {rms:.4}')

        r_rms = old_rms - rms
        r_dist = -1.0

        reward = self.a_var*r_rms + self.a_dist*r_dist

        if self.debug:
            print(f'r_rms: {r_rms:.4}, r_dist: {r_dist:.4}, r_tot: {reward:.4}')

        return reward
    
    def _reward_ch_01000(self, old_var):
        # compute reward (based on decrease in overall variance)
        if self.debug:
            print(f'old_var.mean: {old_var.mean():.4f} pred_var_norm.mean: {self.pred_var_norm.mean():.4f}')

        var_red = min(2.0, (old_var.mean() - self.pred_var_norm.mean()))#2.0 is max possible reward for step length 20
        r_var = var_red # reward for reducing variance
        #r_var = var_red/float(len(sample_coords_xy)*0.0694)
        r_dist = -1.0 # step penalty (for changing course)
        r_term = 0.0

        if self.pred_var_norm.mean() <= 100:
            #r_term = self.n_steps_max - self.n_steps
            r_term = 5.0
            self.terminated = True
        
        reward = self.a_var*r_var + self.a_dist*r_dist + r_term
        
        if self.debug:
            print(f'r_var: {r_var:.4f}, r_dist: {r_dist:.4f}, r_tot: {reward:.4f}')

        return reward
    
    def _reward_ch_11000(self, old_var, measurements):
        # compute reward (based on decrease in overall variance)
        # old_var is pred_var_norm from previous step
        if isinstance(measurements, torch.Tensor):
            old_var_t = old_var if isinstance(old_var, torch.Tensor) else torch.as_tensor(old_var, device=measurements.device)
            pred_var_norm_t = getattr(self, "pred_var_norm_t", None)
            if pred_var_norm_t is None:
                pred_var_norm_t = torch.as_tensor(self.pred_var_norm, device=measurements.device)

            if self.debug:
                print(
                    f'old_var_norm.mean: {old_var_t.mean().item():.4f} '
                    f'pred_var_norm.mean: {pred_var_norm_t.mean().item():.4f}'
                )

            diff = old_var_t.mean() - pred_var_norm_t.mean()
            cap = torch.tensor(2.0, device=measurements.device, dtype=diff.dtype)
            var_red = torch.minimum(cap, diff)
            # Max number of samples seems to be 22, so we scale with 22/len(measurements)
            r_var = var_red * 22 / measurements.numel()

            measurements_norm = (measurements - self.min_concentration) / (self.max_concentration - self.min_concentration) * 255
            r_gas = (measurements_norm >= 5).to(measurements_norm.dtype).mean()

            r_dist = torch.tensor(-1.0, device=measurements.device, dtype=measurements_norm.dtype)
            r_term = 0.0
            if pred_var_norm_t.mean().item() <= 125:
                r_term = 5.0
                self.terminated = True

            reward = self.a_gas * r_gas + self.a_var * r_var + self.a_dist * r_dist + r_term

            if self.debug:
                print(
                    f'r_gas: {r_gas.item():.4f}, r_var: {r_var.item():.4f}, '
                    f'r_dist: {r_dist.item():.4f}, r_tot: {reward.item():.4f}'
                )

            return float(reward.item())

        # NumPy fallback path.
        if self.debug:
            print(f'old_var_norm.mean: {old_var.mean():.4f} pred_var_norm.mean: {self.pred_var_norm.mean():.4f}')

        var_red = min(2.0, (old_var.mean() - self.pred_var_norm.mean()))
        # 2.0 is max possible reward
        # Max number of samples seems to be 22, so we scale with 22/len(measurements)
        r_var = var_red * 22/len(measurements) # reward for reducing variance

        # r_gas is based on the newly acquired samples.
        # measurements have to be normalized in the same manner as the GP estimate:
        measurements_norm = (measurements - self.min_concentration) / (self.max_concentration - self.min_concentration) * 255
        r_gas = (measurements_norm >= 5).sum()/len(measurements_norm)
        # Everything above n contributes to reward, max is 1.0

        r_dist = -1.0 # step penalty (for changing course)
        r_term = 0.0

        if self.pred_var_norm.mean() <= 125:
            #r_term = self.n_steps_max - self.n_steps
            r_term = 5.0
            self.terminated = True
        
        reward = self.a_gas*r_gas + self.a_var*r_var + self.a_dist*r_dist + r_term
        
        if self.debug:
            print(f'r_gas: {r_gas:.4f}, r_var: {r_var:.4f}, r_dist: {r_dist:.4f}, r_tot: {reward:.4f}')

        return reward

    def _dubins_delta_90(self, action, heading_1hot, turn_radius: float | int):
        """
        Parameters
        ----------
        action : {"left", "straight", "right"}
        heading_1hot : iterable of length 4 [north, south, west, east]  e.g. [1,0,0,0] for north.
        turn_radius : positive float (or int)

        Returns
        -------
        (Δ : shape (2,) torch.Tensor, heading)
            ([dx, dy], heading) after executing the action.
        """
        # decode the action
        actions = ["left", "straight", "right"]
        act = actions[action]

        # validate & decode the heading ------------------------------------------
        headings = ('north', 'south', 'west', 'east')
        try:
            h_idx = list(heading_1hot).index(1)
        except ValueError:
            raise ValueError("heading_1hot must have exactly one 1") from None

        heading = headings[h_idx]

        # canonical mapping -------------------------------------------------------
        r = float(turn_radius)
        mapping = {
            ('north', 'straight'): (( 0,  r*math.pi/2.0), [1, 0, 0, 0]),
            ('north', 'left')    : ((-r,  r), [0, 0, 1, 0]),
            ('north', 'right')   : (( r,  r), [0, 0, 0, 1]),

            ('south', 'straight'): (( 0, -r*math.pi/2.0), [0, 1, 0, 0]),
            ('south', 'left')    : (( r, -r), [0, 0, 0, 1]),
            ('south', 'right')   : ((-r, -r), [0, 0, 1, 0]),

            ('west',  'straight'): ((-r*math.pi/2.0,  0), [0, 0, 1, 0]),
            ('west',  'left')    : ((-r, -r), [0, 1, 0, 0]),
            ('west',  'right')   : ((-r,  r), [1, 0, 0, 0]),

            ('east',  'straight'): (( r*math.pi/2.0,  0), [0, 0, 0, 1]),
            ('east',  'left')    : (( r,  r), [1, 0, 0, 0]),
            ('east',  'right')   : (( r, -r), [0, 1, 0, 0]),
        }

        try:
            (dx, dy), new_heading = mapping[(heading, act)]
        except KeyError:
            raise ValueError(f"invalid action '{act}'") from None

        return np.array([dx, dy]), np.array(new_heading)

    def _dubins_delta_45(self, action, heading_hot, turn_radius: float | int):
        """
        Parameters
        ----------
        action : {"left", "straight", "right"}
        heading : iterable of length 4 [north, south, west, east]  e.g. [1,0,1,0] for northwest.
        turn_radius : positive float (or int)

        Returns
        -------
        (Δ : shape (2,) torch.Tensor, heading)
            ([dx, dy], heading) after executing the action.
        """
        # decode the action
        actions = ["left", "straight", "right"]
        act = actions[action]

        # validate & decode the heading ------------------------------------------
        headings = ('north', 'south', 'west', 'east')
        
        on_idx = [i for i, v in enumerate(heading_hot) if v == 1]
        # Single direction (1-hot)
        if len(on_idx) == 1:
            heading = headings[on_idx[0]]

        # Two directions (2-hot) → combine or reject
        else:
            d1, d2 = headings[on_idx[0]], headings[on_idx[1]]
            heading = d1+d2

        # canonical mapping -------------------------------------------------------
        r = float(turn_radius)
        mapping = {
            ('north', 'straight'): (( 0,  r*math.pi/2.0), [1, 0, 0, 0]),
            ('north', 'left')    : ((-r,  r), [0, 0, 1, 0]),
            ('north', 'right')   : (( r,  r), [0, 0, 0, 1]),

            ('south', 'straight'): (( 0, -r*math.pi/2.0), [0, 1, 0, 0]),
            ('south', 'left')    : (( r, -r), [0, 0, 0, 1]),
            ('south', 'right')   : ((-r, -r), [0, 0, 1, 0]),

            ('west',  'straight'): ((-r*math.pi/2.0,  0), [0, 0, 1, 0]),
            ('west',  'left')    : ((-r, -r), [0, 1, 0, 0]),
            ('west',  'right')   : ((-r,  r), [1, 0, 0, 0]),

            ('east',  'straight'): (( r*math.pi/2.0,  0), [0, 0, 0, 1]),
            ('east',  'left')    : (( r,  r), [1, 0, 0, 0]),
            ('east',  'right')   : (( r, -r), [0, 1, 0, 0]),
        }

        try:
            (dx, dy), new_heading = mapping[(heading, act)]
        except KeyError:
            raise ValueError(f"invalid action '{act}'") from None

        return np.array([dx, dy]), np.array(new_heading)

    def _delta_add_noise(self, delta_xy, step, max_percentage=0.05):
        max_noise_x = max_percentage * abs(delta_xy[0])
        max_noise_y = max_percentage * abs(delta_xy[1])
        x_noise = (random.random() - 0.5)*2 * max_noise_x
        y_noise = (random.random() - 0.5)*2 * max_noise_y
        
        return np.array([x_noise, y_noise])

    def _facing_the_boundary(self, new_loc, new_heading):
        if len(new_heading) == 4:
            headings = ('east', 'north', 'west', 'south')
        elif len(new_heading) == 8:
            headings = ('east', 'ne', 'north', 'nw', 'west', 'sw', 'south', 'se')

        # headings = ('north', 'south', 'west', 'east')
        h_idx = list(new_heading).index(1)
        
        match headings[h_idx]:
            case 'east':
                return new_loc[0] > self.env_x_max - self.turn_radius
            case 'ne':
                cx = self.env_x_max - self.turn_radius
                cy = self.env_y_max - self.turn_radius
                return (new_loc[0] > self.env_x_max - self.turn_radius/3 or
                    new_loc[1] > self.env_y_max - self.turn_radius/3 or
                    (new_loc[0] - cx)**2 + (new_loc[1] - cy)**2 < self.turn_radius**2)
            case 'north':
                return new_loc[1] > self.env_y_max - self.turn_radius
            case 'nw':
                cx = self.turn_radius
                cy = self.env_y_max - self.turn_radius
                return (new_loc[0] < self.turn_radius/3 or
                    new_loc[1] > self.env_y_max - self.turn_radius/3 or
                    (new_loc[0] - cx)**2 + (new_loc[1] - cy)**2 < self.turn_radius**2)
            case 'west':
                return new_loc[0] < self.turn_radius
            case 'sw':
                cx = self.turn_radius
                cy = self.turn_radius
                return (new_loc[0] < self.turn_radius/3 or
                    new_loc[1] < self.turn_radius/3 or
                    (new_loc[0] - cx)**2 + (new_loc[1] - cy)**2 < self.turn_radius**2)
            case 'south':
                return new_loc[1] < self.turn_radius
            case 'se':
                cx = self.env_x_max - self.turn_radius
                cy = self.turn_radius
                return (new_loc[0] > self.env_x_max - self.turn_radius/3 or
                    new_loc[1] < self.turn_radius/3 or
                    (new_loc[0] - cx)**2 + (new_loc[1] - cy)**2 < self.turn_radius**2)

        return False
 
    def loc_to_ind(self, loc: Tuple[float, float]) -> Tuple[int, int]:
        x_idx = min(int(round(loc[0] / (self.env_x_max / self.obs_x))), self.obs_x - 1)
        y_idx = min(int(round(loc[1] / (self.env_y_max / self.obs_y))), self.obs_y - 1)
        return x_idx, y_idx
    
    def _get_obs_truncated_info(self):
        if self.return_torch:
            layers = self._render_layers_torch()
            loc_x = (self.loc[0] / self.env_x_max) * 2.0 - 1.0
            loc_y = (self.loc[1] / self.env_y_max) * 2.0 - 1.0
            obs_dict = {
                "map": layers,  # (C,H,W) torch
                "loc": torch.stack((loc_x, loc_y)).to(dtype=torch.float32),
                "hdg": torch.as_tensor(self.heading, device=self.device, dtype=torch.float32),
            }
        else:
            layers_uint8  = self._render_layers()
            loc_x = ((self.loc[0]/self.env_x_max)*2.0 - 1.0).cpu()
            loc_y = ((self.loc[1]/self.env_y_max)*2.0 - 1.0).cpu()
            obs_dict = {
                "map": layers_uint8,           # (C,H,W)
                "loc": np.array([loc_x, loc_y], np.float32),
                "hdg": self.heading
            }

        truncated = (self.n_steps >= self.n_steps_max)
        info = {}
        return obs_dict, truncated, info

    def _assert_gpu_consistency(self):

        for p in self.mdl.parameters():
            if str(p.device.type) != str(self.device):
                print(f'p.device.type: {p.device.type}, self.device: {self.device}')
                print(f'{p} - {p.device}')
        
        for t in self.mdl.train_inputs + (self.mdl.train_targets,):
            if str(t.device.type) != str(self.device):
                print(f'{t} - {t.device}')
        
        for p in self.llh.parameters():
            if str(p.device.type) != str(self.device):
                print(f'{p} - {p.device}')

        # 1. parameters
        assert all(str(p.device.type) == str(self.device) for p in self.mdl.parameters()), \
            "Some model parameters are not on the target device"

        # 2. training data
        for t in self.mdl.train_inputs + (self.mdl.train_targets,):
            assert str(t.device.type) == str(self.device), "GP training tensor on wrong device"

        # 3. likelihood parameters
        assert all(str(p.device.type) == str(self.device) for p in self.llh.parameters()), \
            "Likelihood parameters not on target device"

    def debug_local_gp_update(self, idx_local, show=True, save_path=None):
        """
        Plot a diagnostic figure showing:
        - All samples so far (gray)
        - The most recent samples used for the GP update (red)
        - The area (points) recomputed by the local GP update (light blue)

        Parameters
        ----------
        idx_local : torch.Tensor or np.ndarray
            Indices of self._coords_flat that were recomputed.
        show : bool
            Whether to display the plot interactively.
        save_path : str or None
            If given, save the figure to this path.
        """

        # --- Prepare coordinate arrays ---
        coords = self._coords_flat.detach().cpu().numpy()
        if isinstance(idx_local, torch.Tensor):
            idx_local = idx_local.detach().cpu().numpy().astype(np.int64)

        # Local update region
        coords_local = coords[idx_local]

        # All sample coordinates (past and current)
        all_samples = self.sampled_coords[:self.sample_idx]
        if isinstance(all_samples, torch.Tensor):
            all_samples = all_samples.detach().cpu().numpy()

        # Most recent samples (used in GP update)
        recent_samples = self.sampled_coords[self.sample_idx_mdl:self.sample_idx]
        if isinstance(recent_samples, torch.Tensor):
            recent_samples = recent_samples.detach().cpu().numpy()

        # --- Create figure ---
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect('equal', adjustable='box')

        # Show local-update region (light blue)
        ax.scatter(coords_local[:, 0], coords_local[:, 1], s=8, color='lightskyblue', alpha=0.4, label='Recomputed area')

        # Show all previous samples (gray)
        if len(all_samples) > 0:
            ax.scatter(all_samples[:, 0], all_samples[:, 1], s=10, color='gray', alpha=0.5, label='All samples')

        # Show most recent samples (red, on top)
        if len(recent_samples) > 0:
            ax.scatter(recent_samples[:, 0], recent_samples[:, 1], s=30, color='red', edgecolor='k', label='Recent samples')

        # --- Styling ---
        ax.set_xlabel("X position")
        ax.set_ylabel("Y position")
        ax.set_title("Local GP Update Debug View")
        ax.legend(loc='best')
        ax.grid(True, linestyle=':', alpha=0.4)

        # --- Save or show ---
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        if show:
            plt.show()
        plt.close(fig)
    
    def debug_gp_update_values(self, new_pred, show=True, save_path=None):
        """
        Visualize how self.pred_mu changes after a local update.
        
        Shows:
        - self.pred_mu (current, after update)
        - self.pred_mu values before update (if provided)
        - difference (delta) map
        - scatter markers for updated cells (rows, cols)
        """

        # --- 2. Difference map ---
        delta = new_pred - self.pred_mu

        # --- 3. Plot setup ---
        fig, axs = plt.subplots(1, 3, figsize=(14, 4))
        extent = [0, self.env_x_max, 0, self.env_y_max]

        im0 = axs[0].imshow(self.pred_mu, origin='lower', cmap='viridis', extent=extent)
        axs[0].set_title("pred_mu BEFORE update")
        fig.colorbar(im0, ax=axs[0], fraction=0.046)

        im1 = axs[1].imshow(new_pred, origin='lower', cmap='viridis', extent=extent)
        axs[1].set_title("pred_mu AFTER update")
        axs[1].legend(loc='lower right', fontsize=8)
        fig.colorbar(im1, ax=axs[1], fraction=0.046)

        im2 = axs[2].imshow(delta, origin='lower', cmap='coolwarm', extent=extent)
        axs[2].set_title("Difference (After - Before)")
        fig.colorbar(im2, ax=axs[2], fraction=0.046)

        for ax in axs:
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.grid(True, linestyle=":", alpha=0.3)

        plt.tight_layout()

        # --- 4. Save or show ---
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        if show:
            plt.show()
        plt.close(fig)

    #@profile
    def _estimate_local(self):
        if self.mdl is None:
            if self.debug:
                print(f'Created model in ._estimate()')
            self.mdl = ExactGPModel(self.sampled_coords, self.sampled_vals-self.mu_all, self.llh, self.kernel_type, lengthscale_constraint=self.ls_const).to(self.device)
        
        t = time.process_time()
        #self.mdl.set_train_data(
        #    inputs=self.sampled_coords[:self.sample_idx], targets=self.sampled_vals[:self.sample_idx]-self.mu_all, strict=False)
        if len(self.mdl.train_targets) > 0:
            # not first prediction, use fantasy mdl
            self.mdl = self.mdl.get_fantasy_model(self.sampled_coords[self.sample_idx_mdl:self.sample_idx], self.sampled_vals[self.sample_idx_mdl:self.sample_idx]-self.mu_all).to(self.device)
        else:
            # first prediction must have train data
            self.mdl.set_train_data(
                inputs=self.sampled_coords[:self.sample_idx], targets=self.sampled_vals[:self.sample_idx]-self.mu_all, strict=False)
            
            if self.debug:
                self.mdl.print_named_parameters()
            
        if self.timer:
            print(f't3.1 step: {time.process_time()-t}')

        if not hasattr(self, "current_pred"):
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                current_pred = self.mdl(self._coords_flat)
                self.current_pred_mean = current_pred.mean + self.mu_all
                self.current_pred_variance = current_pred.variance

        # Then predict local coords around acquired samples
        t = time.process_time()
        lengthscale = float(self.mdl.covar_module.base_kernel.lengthscale.squeeze().cpu())
        self.idx_local = self._get_local_update_indices(corr_length=lengthscale)
        coords_local = self._coords_flat[self.idx_local]
        
        if len(coords_local) > 0:
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                local_pred = self.mdl(coords_local)
        
            self.current_pred_mean[self.idx_local] = local_pred.mean + self.mu_all
            self.current_pred_variance[self.idx_local] = local_pred.variance
            
            #self.debug_local_gp_update(self.idx_local)
            #self.debug_gp_update_values(self._tensor_to_obs_channel(self.current_pred_mean))
            
            self.pred_mu = self._tensor_to_obs_channel(self.current_pred_mean)
            self.pred_var = self._tensor_to_obs_channel(self.current_pred_variance)
            
        self.sample_idx_mdl = self.sample_idx

        if self.timer:
            print(f't3.2 step: {time.process_time()-t} - len(coords_local: {len(coords_local)})')
        
        # Scale to 0-255 ([min_conc, max_conc] from scenario bank)
        t = time.process_time()
        self.pred_mu_norm = (self.pred_mu - self.min_concentration) / (self.max_concentration - self.min_concentration) * 255
        self.pred_var_norm = self.pred_var/self.sigma2_all * 255
        self._normalize_pred_layers()

        # Maintain torch versions for on-device rewards.
        self.pred_mu_t = self.current_pred_mean.view(self.obs_y, self.obs_x)
        self.pred_var_t = self.current_pred_variance.view(self.obs_y, self.obs_x)
        self.pred_mu_norm_t = (self.pred_mu_t - self.min_concentration) / (self.max_concentration - self.min_concentration) * 255
        self.pred_var_norm_t = self.pred_var_t / self.sigma2_all * 255

        return
    
    def _estimate(self):
        if self.mdl is None:
            if self.debug:
                print(f'Created model in ._estimate()')
            self.mdl = ExactGPModel(self.sampled_coords, self.sampled_vals-self.mu_all, self.llh, self.kernel_type, lengthscale_constraint=self.ls_const).to(self.device)
        
        t = time.process_time()
        #self.mdl.set_train_data(
        #    inputs=self.sampled_coords[:self.sample_idx], targets=self.sampled_vals[:self.sample_idx]-self.mu_all, strict=False)
        if len(self.mdl.train_targets) > 0:
            # not first prediction, use fantasy mdl
            self.mdl = self.mdl.get_fantasy_model(self.sampled_coords[self.sample_idx_mdl:self.sample_idx], self.sampled_vals[self.sample_idx_mdl:self.sample_idx]-self.mu_all).to(self.device)
            self.sample_idx_mdl = self.sample_idx
        else:
            # first prediction must have train data
            self.mdl.set_train_data(
                inputs=self.sampled_coords[:self.sample_idx], targets=self.sampled_vals[:self.sample_idx]-self.mu_all, strict=False)
            self.sample_idx_mdl = self.sample_idx
            if self.debug:
                self.mdl.print_named_parameters()
            
        if self.timer:
            print(f't3.1 step: {time.process_time()-t}')

        # Then predict
        t = time.process_time()
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-2):
            current_pred = self.mdl(self._coords_flat)

        if self.timer:
            print(f't3.2 step: {time.process_time()-t}')

        t = time.process_time()
        self.pred_mu = self._tensor_to_obs_channel(current_pred.mean + self.mu_all)
        self.pred_var = self._tensor_to_obs_channel(current_pred.variance)
        if self.timer:
            print(f't3.3 step: {time.process_time()-t}')
        
        # Scale to 0-255 ([min_conc, max_conc] from scenario bank)
        t = time.process_time()
        self.pred_mu_norm = (self.pred_mu - self.min_concentration) / (self.max_concentration - self.min_concentration) * 255
        self.pred_var_norm = self.pred_var/self.sigma2_all * 255
        self._normalize_pred_layers()

        # Maintain torch versions for on-device rewards.
        self.pred_mu_t = current_pred.mean.view(self.obs_y, self.obs_x) + self.mu_all
        self.pred_var_t = current_pred.variance.view(self.obs_y, self.obs_x)
        self.pred_mu_norm_t = (self.pred_mu_t - self.min_concentration) / (self.max_concentration - self.min_concentration) * 255
        self.pred_var_norm_t = self.pred_var_t / self.sigma2_all * 255

        return

    def _get_local_update_indices(self, corr_length, scale=3.0):
        """
        Return indices of coords in self._coords_flat that lie within
        (scale * corr_length) of any recently sampled coordinate.
        
        Parameters
        ----------
        corr_length : float
            The GP kernel correlation length (in same units as self._coords_flat).
        scale : float
            Multiplier defining the radius of influence (~3 is typical).
        """
        # --- 1. Extract recent sample coordinates (convert to CPU numpy) ---
        new_samples = self.sampled_coords[self.sample_idx_mdl:self.sample_idx]
        
        if isinstance(new_samples, torch.Tensor):
            new_samples = new_samples.detach().cpu().numpy()
        elif isinstance(new_samples, list):
            new_samples = np.array(new_samples)

        # --- 2. Build KD-tree on all grid coordinates (cached between calls if possible) ---
        if not hasattr(self, "_coord_tree") or self._coord_tree is None:
            coords_np = self._coords_flat.detach().cpu().numpy()
            self._coord_tree = cKDTree(coords_np)

        # --- 3. Query nearby coordinates ---
        update_radius = scale * corr_length
        neighbor_indices = []
        for pt in new_samples:
            neighbor_indices.extend(self._coord_tree.query_ball_point(pt, r=update_radius))

        # --- 4. Deduplicate and return as tensor indices ---
        idx_unique = np.unique(neighbor_indices)
        
        idx_tensor = torch.as_tensor(idx_unique, dtype=torch.long, device=self.device)
        return idx_tensor

    def _norm_minmax(self):
        return (self.values - self.min_concentration)/(self.max_concentration - self.min_concentration)
    
    def _norm_zscale(self):
        return (self.values - self.mu_all)/(self.sigma2_all**(0.5))

    #@profile
    def _render_layers(self) -> np.ndarray:
        """
        Assemble the observation tensor.

        Channels (fixed order):
            0: μ‑field  (self.mu_norm)
            1: σ‑field  (self.sigma_norm)
            2: location  mask (self.location) (could be all visited locations)
            3: Coord‑Y  channel (self.coord_y)
            4: Coord‑X  channel (self.coord_x)

        Only the layers whose corresponding entry in `self.channels`
        is truthy (1 / True) are stacked.
        """

        # List all *possible* layers in a canonical order
        candidate_layers = [
            self.pred_mu_norm_clipped,     # idx 0
            self.pred_var_norm_clipped,  # idx 1
            self.location,     # idx 2
            self.coord_y_norm,     # idx 3
            self.coord_x_norm      # idx 4
        ]

        # Select the ones flagged by `self.channels`
        chosen_layers = [
            layer for layer, flag in zip(candidate_layers, self.channels) if flag
        ]

        # Sanity‑check: number of layers matches observation_space
        assert len(chosen_layers) == self.channels.sum(), \
            "Mismatch between channel mask and selected layers"

        # Stack into (C, H, W) NumPy array expected by Gym
        stacked = np.stack(chosen_layers, axis=0).astype(np.uint8)
        return stacked

    def _render_layers_torch(self) -> torch.Tensor:
        """
        Assemble the observation tensor on-device.
        """
        pred_mu = torch.clamp(self.pred_mu_norm_t, 0, 255)
        pred_var = torch.clamp(self.pred_var_norm_t, 0, 255)
        candidate_layers = [
            pred_mu,              # idx 0
            pred_var,             # idx 1
            self.location_t,      # idx 2
            self.coord_y_norm_t,  # idx 3
            self.coord_x_norm_t,  # idx 4
        ]
        chosen_layers = [
            layer for layer, flag in zip(candidate_layers, self.channels) if flag
        ]
        return torch.stack(chosen_layers, dim=0)
    
    def _normalize_pred_layers(self):
        self.pred_mu_norm_clipped = np.clip(self.pred_mu_norm, 0, 255).astype(np.uint8)
        self.pred_var_norm_clipped = np.clip(self.pred_var_norm, 0, 255).astype(np.uint8)

    #@profile
    def _create_obs_coords(self):

        # -- 1. grid of query points -----------------
        #   (H*W, 2) tensor that GPyTorch will accept.
        xs = np.linspace(0, self.env_x_max, self.obs_x, dtype=np.float32)
        ys = np.linspace(0, self.env_y_max, self.obs_y, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)                        # shape (H, W)

        # Save as 2‑D field for coord‑channels and as flat list for GP queries
        self._coord_x = gx           # (H, W)
        self._coord_y = gy           # (H, W)
        self._coords = np.stack([gx, gy], axis=-1).reshape(-1, 2)      # (H, W, 2)
        self._coords_flat = torch.as_tensor(self._coords, device=self.device)
        self._coord_x_t = torch.as_tensor(self._coord_x, device=self.device)
        self._coord_y_t = torch.as_tensor(self._coord_y, device=self.device)
        
        # -- 2. static coordinate channels, normalised [0, 255] --
        self.coord_x_norm = (gx / self.env_x_max) * 255  # (H, W)
        self.coord_y_norm = (gy / self.env_y_max) * 255  # (H, W)
        self.coord_x_norm_t = (self._coord_x_t / self.env_x_max) * 255
        self.coord_y_norm_t = (self._coord_y_t / self.env_y_max) * 255

        return
    
    #@profile
    def _tensor_to_obs_channel(self, t: torch.Tensor) -> np.ndarray:
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"Expected a torch.Tensor, got {type(t)}")

        # Ensure tensor is flat with expected size
        expected_size = self.obs_x * self.obs_y
        if t.numel() != expected_size:
            raise ValueError(f"Tensor has {t.numel()} elements, expected {expected_size}")

        return t.view(self.obs_y, self.obs_x).detach().cpu().numpy()
    
    def _print_info(self):
        print(f'Currently loaded env: {self.parameter} ({self.depth} {self.time})')
    
    def _append_z_to_xy(self, xy):
        if len(xy) == 2:
            return torch.cat((xy, torch.tensor([self.depth], dtype=torch.int)))
        else:
            return xy
    
    def make_circle(self, x: int, y: int, r: int):
        """
        Set all pixels inside radius `r` of (x, y) to 0 and the rest to 255.

        Parameters
        ----------
        self.location : np.ndarray              # 2-D array (H, W) you want to modify
        x, y  : int                     # centre coordinate (column, row)
        r     : int                     # radius in pixels (inclusive)

        Returns
        -------
        """
        # Boolean mask for the circle (≤ r²)
        mask = (self._coord_x - x)**2 + (self._coord_y - y)**2 <= r*r

        # Write values in place
        self.location.fill(255)     # everything white
        self.location[mask] = 0     # black disk

        if hasattr(self, "location_t"):
            mask_t = (self._coord_x_t - x)**2 + (self._coord_y_t - y)**2 <= r*r
            self.location_t.fill_(255)
            self.location_t[mask_t] = 0

        return

    def plot_env(self, x=None, y=None, c=None, path=None, x_range=[0, 250], y_range=[0, 250], value_title=''):

        if x is None:
            x = self.env_xy[:, 0]
        if y is None:
            y = self.env_xy[:, 1]
        if c is None:
            c = self.values

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(x, y, c=c, cmap='coolwarm', s=1, vmin=c.min(), vmax=c.max())
        if path is not None:
            ax.scatter(path[:, 0], path[:, 1], c='black', s=1)
        
        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(y_range[0], y_range[1])
        
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(f'Value ({value_title})')

        # Add labels and title
        ax.set_xlabel('Easting [m]')
        ax.set_ylabel('Northing [m]')
        ax.set_title(f"Time {self.time}, {self.parameter} at -{self.depth}m. ({self.cur_str:.2}m/s @ {round(self.cur_dir)} deg)")

        return fig, ax
    
    def translate_field(self, x, y, v, dx: int = 0, dy: int = 0):
        """
        Shift a 2-D scalar field and replicate edge pixels to keep it in [0, 250].

        Parameters
        ----------
        x, y : 1-D arrays (length N)
            Grid coordinates (assumed to form a full tensor grid).
        v    : 1-D array  (length N)
            Values at each (x, y) point.
        dx   : int
            Horizontal translation in **grid steps**.
            +dx → shift right, -dx → shift left.
        dy   : int
            Vertical   translation in **grid steps**.
            +dy → shift up,   -dy → shift down.

        Returns
        -------
        v_shift : 1-D array (length N)
            Translated values aligned with the *original* x, y.
        """

        # 1. Infer grid shape -----------------------------------------------------
        xs = np.unique(x)
        ys = np.unique(y)

        w, h = len(xs), len(ys)          # width (x-axis), height (y-axis)

        # Safety check: x and y really form a full grid
        if w * h != len(v):
            raise ValueError("x and y must form a complete tensor mesh")

        # 2. Put `v` into a 2-D image (row  = y, column = x) ----------------------
        # Sort indices so that increasing row index means increasing y
        order = np.lexsort((x, y))       # sort by y first, then x
        img   = v[order].reshape(h, w)

        # 3. Shift with edge replication -----------------------------------------
        # SciPy’s `shift` does exactly what we need with mode='nearest'
        img_shift = shift(img,
                        shift=( -dy,   # rows   (note: +dy = up  ⇒ negative row shift)
                                dx),   # columns
                        mode='nearest',
                        order=0)       # order=0 = nearest-neighbour, keeps uint8 exact

        # 4. Flatten back to 1-D in the *same order* as the input ---------------
        v_shift = img_shift.ravel()[np.argsort(order)]

        return v_shift

def rotate_xy(env_xy: torch.Tensor, d: float | int) -> torch.Tensor:
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

def onehot_to_rad(heading_1hot):
    '''
    Parameters
    ----------
    onehot - [north, south, west, east]

    Returns
    ----------
    one of [math.pi/2, -math.pi/2, -math.pi, math.pi]
    '''
    try:
        idx = list(heading_1hot).index(1)
    except ValueError:
        raise ValueError("heading_1hot must have exactly one 1") from None

    if len(heading_1hot) == 4:
        return idx*math.pi/2
    elif len(heading_1hot) == 8:
        return idx*math.pi/4
    
    return

def move_with_heading(
    heading_1hot: Sequence[int],
    action: Union[int, str],
    turn_radius: float,
    turn_degrees: float = 90.0,               # e.g. 45.0 for finer turning
    n_headings: int = 8,                      # 4 (NESW), 8 (N,NE,E,SE,...), etc.
    straight_matches_arc: bool = True,        # straight distance = r*theta
    forward_step: float = None,               # if provided, overrides above
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute (dx, dy) and the new heading after taking a discrete high-level action
    ('left', 'straight', 'right') from a quantized heading with N bins.

    The turn is a circular arc of radius r through angle θ = turn_degrees (in radians).
    The straight move is either length r*θ (to match arc length) or a fixed forward_step.

    Parameters
    ----------
    heading_1hot : one-hot of length n_headings
        Current heading bin as a one-hot vector (exactly one '1').
        Heading index 0 corresponds to angle ψ=0 (pointing along +x),
        indices increase CCW in steps of 2π / n_headings.
    action : int or str
        Either integer in {0,1,2} or string in {"left","straight","right"}.
    turn_radius : float
        Turning radius r (same units as your map coordinates).
    turn_degrees : float
        Turn angle in degrees for left/right (e.g. 45, 90). Internally converted to radians.
    n_headings : int
        Number of discrete heading bins (e.g., 4 or 8).
    straight_matches_arc : bool
        If True, straight move distance = r * theta (arc length), so all three actions
        traverse equal path length. Ignored if `forward_step` is provided.
    forward_step : float or None
        If not None, use this distance for the straight action.

    Returns
    -------
    dxdy : np.ndarray, shape (2,)
        World-frame displacement.
    new_onehot : np.ndarray, shape (n_headings,)
        One-hot vector for the new heading.
    new_idx : int
        New heading index (0..n_headings-1).
    """
    # -------- decode current heading index ψ ---------------------------
    try:
        h_idx = list(heading_1hot).index(1)
    except ValueError as e:
        raise ValueError("heading_1hot must have exactly one 1") from e
    if not (0 <= h_idx < n_headings):
        raise ValueError(f"heading index {h_idx} outside 0..{n_headings-1}")

    psi = 2.0 * math.pi * (h_idx / n_headings)   # radians; 0 = +x, CCW positive

    # -------- decode action -------------------------------------------
    if isinstance(action, str):
        action = action.lower()
        if action not in ("left", "straight", "right"):
            raise ValueError("action must be 'left', 'straight', or 'right'")
    elif isinstance(action, int) or isinstance(action, np.int64):
        if action not in (0, 1, 2):
            raise ValueError("int action must be 0:'left', 1:'straight', 2:'right'")
        action = ("left", "straight", "right")[action]
    else:
        print(f'action.dtype: {type(action)}')
        raise TypeError("action must be int or str")

    theta = math.radians(turn_degrees)          # arc angle for turns
    r = float(turn_radius)

    # -------- local-frame displacements --------------------------------
    # Define a local frame: +y forward (along current heading), +x to the right.
    # For a left turn by theta on a circle of radius r:
    #   dx_local = - r * sin(theta)
    #   dy_local =   r * (1 - cos(theta))
    # For a right turn: dx_local = + r * sin(theta), dy_local same.
    if action == "left":
        dx_local = r * math.sin(theta)
        dy_local = r * (1 - math.cos(theta))
        heading_delta_bins = +1                 # rotate CCW by one bin if bins match turn angle
    elif action == "right":
        dx_local =  r * math.sin(theta)
        dy_local =  -r * (1 - math.cos(theta))
        heading_delta_bins = -1                 # rotate CW by one bin
    else:  # "straight"
        # distance for straight move
        if forward_step is not None:
            dist = float(forward_step)
        else:
            dist = r * theta if straight_matches_arc else r
        dx_local = dist
        dy_local = 0.0
        heading_delta_bins = 0

    # -------- rotate local displacement into world frame ---------------
    # Local-to-world rotation by current heading angle ψ
    # local basis: [right, forward]; world x = cosψ*right - sinψ*forward
    # Using matrix for vector [dx_local, dy_local] where dy_local is along forward:
    cos_psi, sin_psi = math.cos(psi), math.sin(psi)
    dx_world =  cos_psi * dx_local - sin_psi * dy_local
    dy_world =  sin_psi * dx_local + cos_psi * dy_local

    # -------- update heading index -------------------------------------
    # If the heading lattice step equals the turn angle (e.g., 8 bins + 45°)
    # then moving left/right advances by exactly one bin. More generally,
    # we advance by round(theta / (2π / n_headings)) bins.
    bins_per_turn = int(round(theta / (2 * math.pi / n_headings)))
    if action == "straight":
        delta_bins = 0
    else:
        delta_bins = int(math.copysign(bins_per_turn, heading_delta_bins))
    new_idx = (h_idx + delta_bins) % n_headings

    new_onehot = np.zeros(n_headings, dtype=int)
    new_onehot[new_idx] = 1

    return np.array([dx_world, dy_world], dtype=float), new_onehot

def get_q_values(model, obs):
    """
    Return the Q-value vector (one value per discrete action) for a single observation.
    """
    # 1. Convert raw obs (np array, dict, …) to a batched torch.Tensor on the
    #    same device as the policy
    obs_tensor, _ = model.policy.obs_to_tensor(obs)

    # 2. Extract features (CNN/MLP) exactly as the policy does
    with torch.no_grad():
        q_values = model.policy.q_net(obs_tensor)          # shape (1, n_actions)

    return q_values.cpu().numpy().squeeze(0)             # -> (n_actions,)

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN

class MapPlusLocExtractor(BaseFeaturesExtractor):
    def __init__(self, obs_space: spaces.Dict, features_dim=512):
        super().__init__(obs_space, features_dim)
        self.cnn = NatureCNN(obs_space["map"], features_dim=256)
        self.linear = torch.nn.Linear(256 + 10, features_dim)

    def forward(self, obs):
        device = self.linear.weight.device          # extractor is on same device as policy
        map_t = obs["map"].to(device).float().div(255.0)  # scale 0-1
        loc_t = obs["loc"].to(device)
        hdg_t = obs["hdg"].to(device)
        map_feats = self.cnn(map_t)
        return torch.relu(self.linear(torch.cat([map_feats, loc_t, hdg_t], dim=1)))


def show_conv3_maps(model, obs):
    conv3 = model.policy.q_net.features_extractor.cnn.cnn[4]  # 3rd Conv2d
    feature_bank = {}
    def _save_features(_, __, output):
        feature_bank["conv3"] = output.detach().cpu()
    h = conv3.register_forward_hook(_save_features)

    # ---- 3. forward pass through the extractor -----------------------
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        _ = model.policy.q_net.features_extractor(obs_tensor)

    h.remove()
    # fmap from the forward hook: shape (1, 64, 9, 9)
    fmap = feature_bank["conv3"].squeeze(0)          # (64, 9, 9)  remove batch dim

    # per-channel activation energy
    energy = fmap.abs().mean(dim=(1, 2)).cpu().numpy()   # (64,)

    # bar plot
    channels = np.arange(len(energy))        # x-positions: 0 … 63
    fig, axes = plt.subplots(1, 1, figsize=(4.5, 2.0), dpi=300)
    axes.bar(channels, energy, width=0.8)
    plt.xlabel("Channel", fontsize=8)
    plt.ylabel("mean |activation|", fontsize=8)
    axes.tick_params(axis="both", labelsize=7)
    plt.tight_layout()
    plt.show()

    # Top activation channels
    k = 9
    top_idx = energy.argsort()[-k:]
    rows = int(np.ceil(np.sqrt(k)))
    fig, axes = plt.subplots(rows, rows, figsize=(rows*2, rows*2))

    for ax, idx in zip(axes.flat, top_idx):
        ax.imshow(fmap[idx], cmap="inferno")
        ax.set_title(f"ch {idx}", fontsize=6)
        ax.axis("off")
    plt.tight_layout(); plt.show()
