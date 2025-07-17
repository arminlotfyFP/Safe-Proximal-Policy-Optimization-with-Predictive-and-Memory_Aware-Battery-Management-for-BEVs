
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt
# import mat73
import os
import scipy.io as sio   #---> import matfile

import gymnasium as gym
import gymnasium.spaces as spaces
from   gymnasium import Env
from gymnasium.wrappers import NormalizeObservation, NormalizeReward
from collections import deque  # progress is tqdm

os.getcwd()
os.chdir("/home/armin/Desktop/paper3_python/paper3_py")


# Data preparation:
data = sio.loadmat('i_dc_estimate.mat')
data1 = sio.loadmat('i_dc.mat')
# with resolution of 0.1
# i_dc_estimate = data['i_dc_estimate'].transpose().flatten() # with this I must change SC and batter params
# with resolution of 1
temp = data1['i_dc'].transpose().flatten()
i_dc_estimate = []
i_dc_estimate.extend(temp[::10])
i_dc_estimate = np.array(i_dc_estimate, dtype=np.float32)



i_dc_future = np.roll(i_dc_estimate, -30)
i_dc_future_mean = np.zeros_like(i_dc_future)

for i in range(len(i_dc_future) - 30):
    i_dc_future_mean[i] = i_dc_future[i:i+30].mean()


plt.figure(figsize=(10, 6))
plt.plot(i_dc_estimate, label='Original Signal', color='blue')
plt.title('Low-pass Filtered Signal')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show()

# Low pass filter design
# Define a low-pass Butterworth filter function

def butter_lowpas_filter(i_input, cutoff,fs,order=1):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, i_input)
    return y

filtered_i = butter_lowpas_filter(i_dc_estimate, 0.00637, 1, order=1)

action = np.zeros((len(i_dc_estimate), 1), dtype=np.float32)
action[:, 0] = filtered_i #output is a numpy array


# Define Battery & SC

class Supercapacitor:
    def __init__(self, C=58.0, R_esr=0.01, V_init=12.0, dt=1.0):
        self.C = C                # Farads
        self.R_esr = R_esr        # Ohms
        self.V_sc = V_init        # Initial voltage (V)
        self.dt = dt              # Time step (seconds)

    def step(self, I_sc):
        # Update the supercapacitor voltage (ideal)
        self.V_sc -= (I_sc * self.dt) / self.C
        # Terminal voltage with ESR drop
        V_out = self.V_sc - I_sc * self.R_esr
        return V_out, self.V_sc
    



class ECM_RC_Battery:
    def __init__(self, Q_init=3.3, alpha=0.0002, R0=30e-3, R1=0.02, C1=2500, dt=1, Vmin=2.5, Vmax=4.2, SOC_init=0.8):
        self.Q_init = Q_init          # Initial capacity (Ah)
        self.alpha = alpha            # Capacity fade rate per cycle
        self.R0 = R0                  # Ohm, series resistance
        self.R1 = R1                  # Ohm, RC branch resistance
        self.C1 = C1                  # Farad, RC branch capacitance
        self.dt = dt                  # Simulation timestep (seconds)
        self.Vmin = Vmin              # Minimum voltage
        self.Vmax = Vmax              # Maximum voltage
        self.ai = 0.0
        self.i0 = 0.0

        # State variables
        self.Q = Q_init               # Current capacity (Ah)
        self.SOC = SOC_init                # Initial SOC (1.0 = 100%)
        self.V_RC = 0                 # Initial RC voltage
        self.cycle_count = 0          # Number of full cycles (for capacity fade)

    def voc(self, soc):
        """Simple linear OCV model, replace with lookup for real battery."""
        return self.Vmin + soc * (self.Vmax - self.Vmin)

    def step(self, I_bat):
        # Update RC branch (discretized)
        dV_RC = (-self.V_RC / (self.R1 * self.C1) + I_bat / self.C1) * self.dt
        self.V_RC += dV_RC

        # SOC update
        self.SOC -= (I_bat * self.dt) / (self.Q * 3600)  # Convert Ah to As

        # Enforce SOC limits
        self.SOC = np.clip(self.SOC, 0.0, 1.0)

        #ai
        self.ai = abs(self.i0-I_bat) #

        # Terminal voltage
        V_oc = self.voc(self.SOC)
        V_bat = V_oc - I_bat * self.R0 - self.V_RC

        # Capacity fade update (increment by one cycle if fully charged/discharged)
        aging_loss = 0.0
        if 0 < self.SOC < 1.0:
            aging_loss = self.ai * 0.00005

        # Coulombic loss
        if abs(I_bat) < 3 * self.Q:
            coulombic_loss = abs(I_bat) * self.dt / 3600
        else:
            coulombic_loss = 3 * abs(I_bat) * self.dt / 3600

        # Total capacity loss
        total_loss = aging_loss + coulombic_loss

        # Safe update
        Q_new = self.Q - total_loss
        self.Q = max(Q_new, 0.0)

        return V_bat, self.SOC, self.Q, self.V_RC

min_current =  0 #np.min(i_dc_estimate)    
max_current =  100 #np.max(i_dc_estimate)    
# Define my GymEnvironment:
# input_current, future_current, battery_voltage, battery_capacity,battery_SOC, SC_pack_voltage  6 inputs
class MyGymEnv(Env):
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        self.i_dc_estimate      = config.get("i_dc_estimate", np.zeros(1000))
        self.i_dc_future        = config.get("i_dc_future", np.zeros(1000))
        self.max_steps          = config.get("max_steps", 8000)
        # shuffle_indices         = np.random.permutation(len(self.i_dc_estimate_org))  # get shuffled order
        # self.i_dc_estimate      = self.i_dc_estimate_org[shuffle_indices]
        # self.i_dc_future        = self.i_dc_future_org[shuffle_indices] 
        
        self.observation_space = spaces.Box(low=np.array([-2, -2, -2, 0, 0, 0]), 
                                            high=np.array([2, 2, 2, 1, 1, 1]), 
                                            shape=(6,), dtype=np.float32
)
        
        self.action_space       = spaces.Box(low=-np.array([min_current]), 
                                            high=np.array([max_current]), shape=(1,), dtype=np.float32)
        
        self.input_current      = self.i_dc_estimate[0] #+ 10* np.random.uniform(-1, 1)
        self.output_current     = []
        self.future_current     = self.i_dc_future[0]   #+ 10* np.random.uniform(-1, 1)
        self.helper_current     = butter_lowpas_filter(self.i_dc_estimate, 0.00637, 1, order=1)

        # Deining Buffers
        self.buffer = deque(maxlen=3)
        self.current_buffer = deque(maxlen=3)
        self.std_buffer = deque(maxlen=3)

        # History
        self.dt                 = 1
        self.V_hist             = []
        self.SOC_hist           = []
        self.Q_hist             = []
        self.V_SC_hist          = []
        self.battery_I_hist     = []
        self.SC_I_hist          = []
        self.requested_I_hist   = []
        self.provided_I_hist    = []
        self.reward             = []
        self.R_std             = []
        # Battery
        self.battery_current    = 0
        self.battery_capacity   = 3.3 + 0.1 * np.random.uniform(-1, 1)
        self.battery_SOC        = 1.0 #0.60  + 0.20  * np.random.uniform(-1, 1)
        self.n_batt_par         = 22
        self.n_batt_seri        = 60
        self.battery_cell_voltage    = 2.5 + self.battery_SOC  * (4 - 2.5)
        self.battery_voltage    = self.battery_cell_voltage * self.n_batt_seri
        self.fading_coefficient = 2e-6  # alpha 2 coeff for derivative
        self.R0                 = 0.03
        self.R1                 = 0.04
        self.C1                 = 750
        # SC
        self.C_n_seri           = 7
        self.SC_current         = 0
        self.SC_voltage         = 16
        self.SC_pack_voltage    = self.SC_voltage  * self.C_n_seri 
        self.end_counter        = self.max_steps
        self.SC_C               = 58
        self.R_esr              = 22e-3
        


        self.terminated         = False
        self.truncated          = False
        self.info               = {}
        self.step_count         = 0
        # Defining Models
        self.battery    = ECM_RC_Battery(Q_init=self.battery_capacity, alpha=self.fading_coefficient, R0=self.R0, R1=self.R1, C1=self.C1, dt=self.dt, Vmin=2.5, Vmax=4.0, SOC_init=0.8)
        self.sc         = Supercapacitor(C=self.SC_C, R_esr=self.R_esr, V_init=self.SC_voltage , dt=self.dt)
        

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count         = 0
        #self.start_idx          = np.random.randint(0, len(self.i_dc_estimate) - 500)                   
        self.input_current      = self.i_dc_estimate[self.step_count ] #self.i_dc_estimate[self.start_idx] + 10* np.random.uniform(-1, 1) 
        self.future_current     = self.i_dc_future[self.step_count ] #self.i_dc_future[self.start_idx]   + 10* np.random.uniform(-1, 1) 
        self.output_current     = []

        self.battery_current    = 0
        self.battery_capacity   = 3.3 + 0.1 * np.random.uniform(-1, 1)
        

        self.buffer.clear()
        self.current_buffer.clear()

        self.SC_C               = 58
        self.R_esr              = 22e-3
        self.C_n_seri           = 7
        self.SC_current         = 0
        
        self.terminated         = False
        self.truncated          = False
        self.info               = {}

        #Random section
        self.SOC_init           = np.random.uniform(0.6, 0.8)
        self.SC_V_init          = 12 + 2 * np.random.uniform(-1, 1) # 16V is the nominal voltage of SCs
        

        self.battery    = ECM_RC_Battery(Q_init=self.battery_capacity, alpha=self.fading_coefficient, R0=self.R0, R1=self.R1, C1=self.C1, dt=self.dt, Vmin=2.5, Vmax=4.0, SOC_init=self.SOC_init)
        self.sc         = Supercapacitor(C=self.SC_C, R_esr=self.R_esr, V_init=self.SC_V_init , dt=self.dt)

        V_bat, self.SOC, self.Q, self.V_RC      = self.battery.step(0)
        V_out, self.V_sc                        = self.sc.step(0)

        self.battery_SOC                        = self.SOC
        self.battery_cell_voltage               = V_bat
        self.battery_voltage                    = self.battery_cell_voltage * self.n_batt_seri
        self.SC_voltage                         = V_out
        self.SC_pack_voltage                    = self.SC_voltage  * self.C_n_seri 




        self.state = np.array([self.input_current/np.max(self.i_dc_estimate),
                                self.future_current/np.max(self.i_dc_estimate),
                                self.battery_voltage/252,
                                1,
                                self.battery_SOC,
                                np.clip(self.SC_pack_voltage /112,0.0,1.0)], dtype=np.float32)
        
        self.state = np.clip(self.state, -1e6, 1e6)
        return self.state, {}

    def step(self, in_current):
        self.buffer.append(float(in_current))
        self.battery_current    = np.mean(self.buffer) if len(self.buffer) > 2 else 0.0
        self.SC_current         = self.input_current - self.battery_current
        battery_current_cell    = self.battery_current/self.n_batt_par
        SC_current_cell         = self.SC_current/1 # We do not have parallel SCs in this case  
        
        V_bat, SOC, Q, V_RC     = self.battery.step(battery_current_cell)
        V_SC, V_sc              = self.sc.step(SC_current_cell)
        self.battery_SOC        = SOC  
        self.SC_voltage         = V_SC
        self.SC_pack_voltage    = self.SC_voltage*self.C_n_seri 
        self.battery_voltage    = V_bat*self.n_batt_seri

        # Updating counter
        self.step_count += 1

        # Stop conditions section
        self.truncated = self.battery_SOC  <= 0.05 or self.SC_pack_voltage <= 60 or self.SC_pack_voltage >= 120 or Q <= 0 or self.battery_voltage <= 150 or self.battery_voltage >= 254 #some extra freedom
        self.terminated = self.step_count >= self.end_counter - 1  
                
        # Updating current values
        self.input_current      = self.i_dc_estimate[self.step_count ] 
        self.future_current     = self.i_dc_future[self.step_count ] if self.step_count < self.end_counter else self.i_dc_estimate[-1]  

        # Append to history
        self.V_hist.append(self.battery_voltage)                 
        self.SOC_hist.append(SOC*100)                                       
        self.Q_hist.append(Q)                                               
        self.V_SC_hist.append(V_SC*self.C_n_seri)                           
        self.battery_I_hist.append(self.battery_current)                    
        self.SC_I_hist.append(self.SC_current )                               
        self.requested_I_hist.append(self.input_current)                    
        self.provided_I_hist.append(self.battery_current + self.SC_current)

        # State signal section
        self.state = np.array([self.input_current/np.max(self.i_dc_estimate),
                                self.future_current/np.max(self.i_dc_estimate),
                                self.battery_voltage/252,
                                Q/self.battery_capacity,
                                self.battery_SOC,
                                np.clip(self.SC_pack_voltage/112,0.0,1.0)], dtype=np.float32)
        
        
        
        
        # Define reward sections:
        
        #Current
        sum_current = self.battery_current + self.SC_current
        self.output_current.append(sum_current)
        r_current = -10*abs(sum_current-self.input_current) 

        #Capacity
        r_capacity = -1000 * abs(self.battery_capacity - Q)

        #Derivative current
        self.current_buffer.append(self.battery_current)
        r_current_a_temp = -50*abs(self.current_buffer[-1] - self.current_buffer[0]) if len(self.current_buffer) > 2 else 0
        self.std_buffer.append(r_current_a_temp)
        r_current_a = np.clip(np.mean(self.std_buffer) if len(self.std_buffer) > 2 else 0,-1000,0)





        reward = 0.0
        if self.truncated :
            reward -= 1000
        reward += 10*(self.step_count / self.end_counter) # Reward for progress
        if self.terminated:
            reward += 200
        reward_cc = np.clip(10 * -((abs(self.battery_current) - abs(self.SC_current))/ (sum_current +0.1)),-1000,0) # Reward for balancing the current
        self.R_std.append(reward_cc)
        reward +=reward_cc

        self.reward.append(reward)
        assert not np.isnan(self.state).any(), "NaN in observation"
        assert not np.isnan(reward), "NaN in reward"
        assert np.isfinite(self.state).all(), "Inf in observation"
        # info section                                  
        if self.truncated or self.terminated:
            self.info = {
                'battery_voltage'       : self.V_hist,
                'battery_SOC'           : self.SOC_hist,
                'battery_capacity'      : self.Q_hist,
                'SC_voltage'            : self.V_SC_hist,
                'battery_I_history'     : self.battery_I_hist,
                'SC_I_history'          : self.SC_I_hist,
                'requested_I_history'   : self.requested_I_hist,
                'provided_I_history'    : self.provided_I_hist,
                'reward_history'        : self.reward,
            }
        else:
            self.info = {}
        
        return self.state, reward, self.terminated,self.truncated, self.info

    def render(self, mode='human'):
        pass

    def close(self):
        pass




# hybridESS = MyGymEnv(i_dc_estimate=i_dc_estimate, i_dc_future=i_dc_future)


from ray import tune
from ray.tune.logger import TBXLogger
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.env.env_context import EnvContext
from ray.tune.registry import register_env
import tensorflow as tf
import torch
print(torch.__version__)

register_env("MyCustomEnv", lambda config: MyGymEnv(config))

# --------- RLlib SAC-LSTM CONFIGURATION -----------

config_dict1 = {
    "log_level"         : "ERROR",
    "framework"         : "torch",   # <--- change to torch
    "num_workers"       : 10,
    "num_gpus"          : 0,          # Set to 0 if you don't have a GPU
    "num_envs_per_worker": 5,
    "gamma"             : tune.uniform(0.90, 0.99),
    "actor_lr"          : tune.loguniform(1e-5, 1e-3),
    "critic_lr"         : tune.loguniform(1e-5, 1e-3),
    "alpha_lr"          : tune.loguniform(1e-5, 1e-3),
    "normalize_actions" : True,
    "normalize_observations": True,
    "clip_rewards"      : False,
    "target_entropy"    : "auto",
    "entropy_coeff"     : "auto",  # Automatically adjust entropy coefficient
    "explore"           : True,  # Enable exploration
    "rollout_fragment_length": 75,   # Must be >= max_seq_len
    "train_batch_size"  : 1024,         # To hold multiple sequences
    "logger_config": {
        "type": TBXLogger,  # Ensures TensorBoard logging
    },
    "env": "MyCustomEnv",
    "env_config": {
        "i_dc_estimate": i_dc_estimate,   # these should be defined in your script!
        "i_dc_future": i_dc_future,
        "max_steps": len(i_dc_estimate),  # or any other per-episode limit you want
    },
    "rl_module": {
        "model_config": {
            "use_lstm": True,
            "lstm_cell_size": [128,64],
            "max_seq_len": 75,
            "lstm_use_prev_action": True,
            "lstm_use_prev_reward": True,
            "fcnet_hiddens": [128,64, 32],
            "fcnet_activation": "relu",
            "burn_in": 5, # Number of initial steps to ignore before LSTM starts processing
        }
    }
}

config_dict2 = {
    "log_level": "ERROR",
    "framework": "torch",   # <--- change to torch
    "num_workers": 10,
    "num_gpus": 0,          # Set to 0 if you don't have a GPU
    "num_envs_per_worker": 5,
    "actor_lr": 1e-3,
    "critic_lr": 1e-3,
    "alpha_lr": 1e-3,
    "normalize_actions": True,
    "normalize_observations": True,
    "clip_rewards": False,
    "target_entropy": "auto",
    "entropy_coeff": "auto",  # Automatically adjust entropy coefficient
    "explore": True,  # Enable exploration
    "rollout_fragment_length": 75,   # Must be >= max_seq_len
    "train_batch_size": 1024,         # To hold multiple sequences
    "logger_config": {
        "type": TBXLogger,  # Ensures TensorBoard logging
    },
    "env": "MyCustomEnv",
    "env_config": {
        "i_dc_estimate": i_dc_estimate,   # these should be defined in your script!
        "i_dc_future": i_dc_future,
        "max_steps": len(i_dc_estimate),  # or any other per-episode limit you want
    },
    "rl_module": {
        "model_config": {
            "use_lstm": False,
            "lstm_cell_size": [128,64],
            "max_seq_len": 75,
            "lstm_use_prev_action": True,
            "lstm_use_prev_reward": True,
            "fcnet_hiddens": [128, 64, 32],
            "fcnet_activation": "relu",
            "burn_in": 5, # Number of initial steps to ignore before LSTM starts processing
        }
    }
}


config_dict3 = {
    "log_level": "ERROR",
    "framework": "torch",   # <--- change to torch
    "num_workers": 4,
    "num_gpus": 0,          # Set to 0 if you don't have a GPU
    "num_envs_per_worker": 2,
    "actor_lr": 1e-4,
    "critic_lr": 1e-4,
    "alpha_lr": 1e-4,
    "normalize_actions": True,
    "normalize_observations": True,
    "clip_rewards": True,
    "logger_config": {
        "type": TBXLogger,  # Ensures TensorBoard logging
    },
    "env": "MyCustomEnv",
    "env_config": {
        "i_dc_estimate": i_dc_estimate,   # these should be defined in your script!
        "i_dc_future": i_dc_future,
        "max_steps": len(i_dc_estimate),  # or any other per-episode limit you want
    },
    "rl_module": {
        "model_config": {
            "use_lstm": False,
            "lstm_cell_size": 64,
            "max_seq_len": 20,
            "lstm_use_prev_action": True,
            "lstm_use_prev_reward": True,
            "fcnet_hiddens": [128, 64],
            "fcnet_activation": "relu",
        }
    }
}

config_dict4 = {
    "log_level": "ERROR",
    "framework": "torch",   # <--- change to torch
    "num_workers": 4,
    "num_gpus":0,          # Set to 0 if you don't have a GPU
    "num_envs_per_worker": 2,
    "actor_lr": 1e-4,
    "critic_lr": 1e-4,
    "alpha_lr": 1e-4,
    "normalize_actions": True,
    "normalize_observations": True,
    "clip_rewards": True,
    "logger_config": {
        "type": TBXLogger,  # Ensures TensorBoard logging
    },
    "env": "MyCustomEnv",
    "env_config": {
        "i_dc_estimate": i_dc_estimate,   # these should be defined in your script!
        "i_dc_future": i_dc_future,
        "max_steps": len(i_dc_estimate),  # or any other per-episode limit you want
    },
    "rl_module": {
        "model_config": {
            "use_lstm": False,
            "lstm_cell_size": 64,
            "max_seq_len": 20,
            "lstm_use_prev_action": True,
            "lstm_use_prev_reward": True,
            "fcnet_hiddens": [128, 64],
            "fcnet_activation": "relu",
        }
    }
}


config_dict5 = {
    "log_level": "ERROR",
    "framework": "torch",   # <--- change to torch
    "num_workers": 4,
    "num_gpus": 0,          # Set to 0 if you don't have a GPU
    "num_envs_per_worker": 2,
    "actor_lr": 1e-3,
    "critic_lr": 1e-3,
    "alpha_lr": 1e-3,
    "lr": 5e-5,
    "entropy_coeff": 0.001,
    "vf_loss_coeff": 0.2,
    "clip_param": 0.2,
    "grad_clip": 0.5,
    "lambda": 0.9,
    "explore": True,  # Enable exploration
    "train_batch_size": 4000,
    "sgd_minibatch_size": 256,
    "num_sgd_iter": 10,
    "normalize_actions": True,
    "normalize_observations": False,
    "clip_rewards": False,
    "logger_config": {
        "type": TBXLogger,  # Ensures TensorBoard logging
    },
    "clip_param": 1.0,  # PPO-specific
    "env": "MyCustomEnv",
    "env_config": {
        "i_dc_estimate": i_dc_estimate,   # these should be defined in your script!
        "i_dc_future": i_dc_future,
        "max_steps": len(i_dc_estimate),  # or any other per-episode limit you want
    },
    "rl_module": {
        "model_config": {
            "use_lstm": True,
            "lstm_cell_size": 64,
            "max_seq_len": 20,
            "lstm_use_prev_action": True,
            "lstm_use_prev_reward": True,
            "fcnet_hiddens": [128, 64],
            "fcnet_activation": "relu",
        }
    }
}

config_dict6 = {
    "log_level": "ERROR",
    "framework": "torch",   # <--- change to torch
    "num_workers": 4,
    "num_gpus":0,          # Set to 0 if you don't have a GPU
    "num_envs_per_worker": 2,
    "actor_lr": 1e-3,
    "critic_lr": 1e-3,
    "alpha_lr": 1e-3,
    "lr": 5e-5,
    "entropy_coeff": 0.001,
    "vf_loss_coeff": 0.2,
    "clip_param": 0.2,
    "grad_clip": 0.5,
    "explore": True,  # Enable exploration
    "lambda": 0.9,
    "train_batch_size": 4000,
    "sgd_minibatch_size": 256,
    "num_sgd_iter": 10,
    "normalize_actions": True,
    "normalize_observations": False,
    "clip_rewards": False,
    "logger_config": {
        "type": TBXLogger,  # Ensures TensorBoard logging
    },
    "clip_param": 1.0,  # PPO-specific
    "env": "MyCustomEnv",
    "env_config": {
        "i_dc_estimate": i_dc_estimate,   # these should be defined in your script!
        "i_dc_future": i_dc_future,
        "max_steps": len(i_dc_estimate),  # or any other per-episode limit you want
    },
    "rl_module": {
        "model_config": {
            "use_lstm": False,
            "lstm_cell_size": 64,
            "max_seq_len": 20,
            "lstm_use_prev_action": True,
            "lstm_use_prev_reward": True,
            "fcnet_hiddens": [128, 64],
            "fcnet_activation": "relu",
        }
    }
}

import warnings
from ray import tune
warnings.filterwarnings("ignore")

# --------- TRAINING -----------
stop_criteria = {
    "training_iteration": 10_000, # Set a high number for iterations
    "episode_reward_mean": -10,    # Stop when mean reward reaches 200
}

stop_criteria_PPO = {
    "training_iteration": 5_000, # Set a high number for iterations
    # "episode_reward_mean": -10,    # Stop when mean reward reaches 200
}


from ray.tune.logger import TBXLogger
from ray.tune.logger import pretty_print
from ray.tune.logger import CSVLoggerCallback
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler

search_alg = OptunaSearch(
    metric="episode_reward_mean", 
    mode="max"
)

scheduler = ASHAScheduler(
    max_t=300,              # max training iterations
    grace_period=20,        # don't stop early before this
    reduction_factor=3,
    metric="episode_reward_mean",
    mode="max"
)

results1 = tune.run(
    "SAC",
    config=config_dict1,
    stop=stop_criteria,
    verbose=1,
    local_dir="~/SAC_results",
    checkpoint_at_end=True,                      # Save a checkpoint at the end
    checkpoint_freq=5,                           # Checkpoint every 5 iterations
    keep_checkpoints_num=3,                      # Keep only top 3
    checkpoint_score_attr="episode_reward_mean", # Use max episode return for ranking
    callbacks=[CSVLoggerCallback()],
    num_samples=20,
    search_alg=search_alg,
    scheduler=scheduler,
    name="SAC_LSTM_OptunaASHA",
    log_to_file=True
)

# best configuration
# best_trial_SAC_LSTM = results1.get_best_trial(metric="episode_reward_mean", mode="max")
# Best weights
# best_checkpoint_SAC_LSTM = results1.get_best_checkpoint(
#     best_trial_SAC_LSTM,
#     metric="episode_reward_mean",
#     mode="max"
# )
# restore_path = best_checkpoint_SAC_LSTM.to_directory()
# results_restored = tune.run(
#     "SAC",
#     config=config_dict1,
#     stop=stop_criteria,
#     verbose=1,
#     name="SAC_LSTM-Experiment_restored",
#     local_dir="C:\\Users\\arminlotfy.DOE\\ray_results",
#     checkpoint_at_end=True,
#     restore=restore_path,
#     resume="AUTO",
#     checkpoint_freq=500,
#     keep_checkpoints_num=3,
#     checkpoint_score_attr="episode_reward_mean",
#     log_to_file=True,
#     callbacks=[CSVLoggerCallback()]
# )


results2 = tune.run(
    "SAC",
    config=config_dict2,
    stop=stop_criteria,
    verbose=1,
    name="SAC-Experiment",
    local_dir="~/SAC_results",
    checkpoint_at_end=True,                      # Save a checkpoint at the end
    checkpoint_freq=5,                           # Checkpoint every 5 iterations
    keep_checkpoints_num=3,                      # Keep only top 3
    checkpoint_score_attr="episode_reward_mean", # Use max episode return for ranking
    log_to_file=True,  
    callbacks=[CSVLoggerCallback()] 
)
# from rllib_contrib.td3 import TD3
# from your_local_td3_module import TD3Trainer
# from ray.rllib.algorithms.registry import register_algorithm

# register_algorithm("TD3", TD3Trainer)


# results3 = tune.run(
#     "TD3",
#     config=config_dict3,
#     stop=stop_criteria,
#     verbose=1,
#     name="TD3-Experiment",
#     local_dir="~/TD3_results",
#     checkpoint_at_end=True,                      # Save a checkpoint at the end
#     checkpoint_freq=5,                           # Checkpoint every 5 iterations
#     keep_checkpoints_num=3,                      # Keep only top 3
#     checkpoint_score_attr="episode_reward_mean", # Use max episode return for ranking
#     log_to_file=True, 
# )

# results4 = tune.run(
#     "TD3",
#     config=config_dict4,
#     stop=stop_criteria,
#     verbose=1,
#     name="TD3-Experiment",
#     local_dir="~/TD3_results",
#     checkpoint_at_end=True,                      # Save a checkpoint at the end
#     checkpoint_freq=5,                           # Checkpoint every 5 iterations
#     keep_checkpoints_num=3,                      # Keep only top 3
#     checkpoint_score_attr="episode_reward_mean", # Use max episode return for ranking
# )

results5 = tune.run(
    "PPO",
    config=config_dict5,
    stop=stop_criteria_PPO,
    verbose=1,
    name="PPO-LSTM-Experiment",
    local_dir="~/PPO_LSTM_results",
    checkpoint_at_end=True,                      # Save a checkpoint at the end
    checkpoint_freq=5,                           # Checkpoint every 5 iterations
    keep_checkpoints_num=3,                      # Keep only top 3
    checkpoint_score_attr="episode_reward_mean", # Use max episode return for ranking
    log_to_file=True,
    callbacks=[CSVLoggerCallback()]
)

results6 = tune.run(
    "PPO",
    config=config_dict6,
    stop=stop_criteria_PPO,
    verbose=1,
    name="PPO-Experiment",
    local_dir="~/PPO_results",
    checkpoint_at_end=True,                      # Save a checkpoint at the end
    checkpoint_freq=5,                           # Checkpoint every 5 iterations
    keep_checkpoints_num=3,                      # Keep only top 3
    checkpoint_score_attr="episode_reward_mean", # Use max episode return for ranking
    log_to_file=True,
    callbacks=[CSVLoggerCallback()]
)


###########################################################################################################
###########################################################################################################

best_trial_SAC_LSTM     = results1.get_best_trial(metric="episode_reward_mean", mode="max")
best_trial_SAC          = results2.get_best_trial(metric="episode_reward_mean", mode="max")
best_trial_PPO_LSTM     = results5.get_best_trial(metric="episode_reward_mean", mode="max")
best_trial_PPO          = results6.get_best_trial(metric="episode_reward_mean", mode="max")

# To get the checkpoint directory:
best_checkpoint_SAC_LSTM = results1.get_best_checkpoint(
    best_trial_SAC_LSTM ,
    metric="episode_reward_mean",
    mode="max"
)


best_checkpoint_SAC = results2.get_best_checkpoint(
    best_trial_SAC ,
    metric="episode_reward_mean",
    mode="max"
)

best_checkpoint_PPO_LSTM = results5.get_best_checkpoint(
    best_trial_PPO_LSTM ,
    metric="episode_reward_mean",
    mode="max"
)

best_checkpoint_PPO = results6.get_best_checkpoint(
    best_trial_PPO ,
    metric="episode_reward_mean",
    mode="max"
)

print("Best trial_SAC_LSTM:", best_trial_SAC_LSTM )     # name of best trial
print("Best checkpoint:", best_checkpoint_SAC_LSTM )    # location of best checkpoint

mean_reward_SAC_LSTM    = best_trial_SAC_LSTM.metric_analysis["episode_reward_mean"]["avg"]
mean_reward_SAC         = best_trial_SAC.metric_analysis["episode_reward_mean"]["avg"]
mean_reward_PPO_LSTM    = best_trial_PPO_LSTM.metric_analysis["episode_reward_mean"]["avg"]
mean_reward_PPO         = best_trial_PPO.metric_analysis["episode_reward_mean"]["avg"]

print("Best trial's mean SAC_LSTM episode reward:"  , mean_reward_SAC_LSTM)
print("Best trial's mean SAC episode reward:"       , mean_reward_SAC)
print("Best trial's mean PPO_LSTM episode reward:"  , mean_reward_PPO_LSTM)
print("Best trial's mean PPO episode reward:"       , mean_reward_PPO)


# look inside of each trial dataframe to see the results:
df = results1.trial_dataframes[best_trial_SAC_LSTM.trial_id]
print(df["episode_reward_mean"].describe())
mean_reward = df["episode_reward_mean"].mean()



########################### EValuation of trained agnet: ######################################################

# EVal Env has no randomization in selecting parameters
class MyEvalEnv(Env):
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        self.i_dc_estimate      = config.get("i_dc_estimate", np.zeros(1000))
        self.i_dc_future        = config.get("i_dc_future", np.zeros(1000))
        self.max_steps          = config.get("max_steps", 8000)
        
        self.observation_space = spaces.Box(low=np.array([-1.5, -1.5, -1.5, 0, 0, 0]), 
                                            high=np.array([1.5,  1.5,  1.5, 1, 1, 1]), 
                                            shape=(6,), dtype=np.float32
)
        
        self.action_space       = spaces.Box(low=-np.array([min_current]), 
                                            high=np.array([max_current]), shape=(1,), dtype=np.float32)
        
        self.input_current      = self.i_dc_estimate[0] #+ 10* np.random.uniform(-1, 1)
        self.output_current     = []
        self.future_current     = self.i_dc_future[0]   #+ 10* np.random.uniform(-1, 1)
        self.helper_current     = butter_lowpas_filter(self.i_dc_estimate, 0.00637, 1, order=1)

        # Deining Buffers
        self.buffer = deque(maxlen=3)
        self.current_buffer = deque(maxlen=3)
        self.std_buffer = deque(maxlen=3)
        # History
        self.dt                 = 0.1
        self.V_hist             = []
        self.SOC_hist           = []
        self.Q_hist             = []
        self.V_SC_hist          = []
        self.battery_I_hist     = []
        self.SC_I_hist          = []
        self.requested_I_hist   = []
        self.provided_I_hist    = []
        self.R_std              = []
        self.R_current          = []
        self.R_capacity         = []
        self.R_current_a        = []
        self.R_distance         = []
        self.reward             = []
        self.R_count            = []
        # Battery
        self.battery_current    = 0
        self.battery_capacity   = 3.3 + 0.1 * np.random.uniform(-1, 1)
        self.battery_SOC        = 0.9#0.60  + 0.20  * np.random.uniform(-1, 1)
        self.n_batt_par         = 22
        self.n_batt_seri        = 60
        self.battery_cell_voltage    = 2.5 + self.battery_SOC  * (4 - 2.5)
        self.battery_voltage    = self.battery_cell_voltage * self.n_batt_seri
        self.fading_coefficient = 2e-8  # alpha 2 coeff for derivative
        self.R0                 = 0.03
        self.R1                 = 0.04
        self.C1                 = 750
        # SC
        self.C_n_seri           = 7
        self.SC_current         = 0
        self.SC_voltage         = 15 #(np.random.uniform(12,16)) 
        self.SC_pack_voltage    = self.SC_voltage  * self.C_n_seri  
        self.end_counter        = self.max_steps
        self.SC_C               = 58
        self.R_esr              = 22e-3
        


        self.terminated         = False
        self.truncated          = False
        self.info               = {}
        self.step_count         = 0
        # Defining Models
        self.battery    = ECM_RC_Battery(Q_init=self.battery_capacity, alpha=self.fading_coefficient, R0=self.R0, R1=self.R1, C1=self.C1, dt=self.dt, Vmin=2.5, Vmax=4.0, SOC_init=self.battery_SOC )
        self.sc         = Supercapacitor(C=self.SC_C, R_esr=self.R_esr, V_init=self.SC_voltage , dt=self.dt)
        

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # self.start_idx          = np.random.randint(0, len(self.i_dc_estimate) - 2000)
        self.step_count         = 0
        self.input_current      = self.i_dc_estimate[self.step_count] #+ 10* np.random.uniform(-1, 1)
        self.future_current     = self.i_dc_future[self.step_count]   #+ 10* np.random.uniform(-1, 1)
        self.output_current     = []

        self.battery_current    = 0
        self.battery_capacity   = 3.3 + 0.1 * np.random.uniform(-1, 1)
        

        self.buffer.clear()
        self.current_buffer.clear()

        self.SC_C               = 58
        self.R_esr              = 22e-3
        self.C_n_seri           = 7
        self.SC_current         = 0
        
        self.terminated         = False
        self.truncated          = False
        self.info               = {}
        self.step_count         = 0
        self.R_count            = []

        #Random section
        # self.SOC_init           = np.random.uniform(0.6, 0.8)
        # self.SC_V_init          = 12 + 2 * np.random.uniform(-1, 1) # 16V is the nominal voltage of SCs
        

        self.battery    = ECM_RC_Battery(Q_init=self.battery_capacity, alpha=self.fading_coefficient, R0=self.R0, R1=self.R1, C1=self.C1, dt=self.dt, Vmin=2.5, Vmax=4.0, SOC_init=self.battery_SOC )
        self.sc         = Supercapacitor(C=self.SC_C, R_esr=self.R_esr, V_init=self.SC_voltage , dt=self.dt)

        V_bat, self.SOC, self.Q, self.V_RC      = self.battery.step(0)
        V_out, self.V_sc                        = self.sc.step(0)

        self.battery_SOC                        = self.SOC
        self.battery_cell_voltage               = V_bat
        self.battery_voltage                    = self.battery_cell_voltage * self.n_batt_seri
        self.SC_voltage                         = V_out
        self.SC_pack_voltage                    = self.SC_voltage  * self.C_n_seri 

        self.state = np.array([self.input_current/np.max(self.i_dc_estimate),
                                self.future_current/np.max(self.i_dc_estimate),
                                self.battery_voltage/252,
                                1,
                                self.battery_SOC,
                                np.clip(self.SC_pack_voltage /112,0.0,1.0)], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        self.buffer.append(float(action))
        self.battery_current    = np.mean(self.buffer) if len(self.buffer) > 2 else 0.0
        self.SC_current         = self.input_current - self.battery_current
        battery_current_cell    = self.battery_current/self.n_batt_par
        SC_current_cell         = self.SC_current/1 # We don not have parallel SCs in this case  
        
        V_bat, SOC, Q, V_RC     = self.battery.step(battery_current_cell)
        V_SC, V_sc              = self.sc.step(SC_current_cell)
        self.battery_SOC        = SOC  
        self.SC_voltage         = V_SC
        self.SC_pack_voltage    = self.SC_voltage*self.C_n_seri 
        self.battery_voltage    = V_bat*self.n_batt_seri 

        # Updating counter
        self.step_count += 1

        # Stop conditions section
        self.truncated = self.battery_SOC  <= 0.05 or self.SC_pack_voltage <= 60 or self.SC_pack_voltage >= 120 or Q <= 0 or self.battery_voltage <= 150 or self.battery_voltage >= 254 #some extra freedom
        self.terminated = self.step_count >= self.end_counter - 1  
                
        # Updating current values
        self.input_current      = self.i_dc_estimate[self.step_count] 
        self.future_current     = self.i_dc_future[self.step_count ] if self.step_count < self.end_counter else self.i_dc_estimate[-1]  

        # Append to history
        self.V_hist.append(self.battery_voltage)                          
        self.SOC_hist.append(SOC*100)                                       
        self.Q_hist.append(Q)                                                
        self.V_SC_hist.append(V_SC*self.C_n_seri)                           
        self.battery_I_hist.append(self.battery_current)                    
        self.SC_I_hist.append(self.SC_current )                                
        self.requested_I_hist.append(self.input_current)                    
        self.provided_I_hist.append(self.battery_current + self.SC_current) 

        
        # State signal section
        self.state = np.array([self.input_current/np.max(self.i_dc_estimate),
                                self.future_current/np.max(self.i_dc_estimate),
                                self.battery_voltage/252,
                                Q/self.battery_capacity,
                                self.battery_SOC,
                                np.clip(self.SC_pack_voltage/112,0.0,1.0)], dtype=np.float32)
        
        
        
        # Define reward sections:
        
        self.buffer.append(self.battery_current)
        r_std = -10*abs(np.std(self.buffer))
        self.R_std.append(r_std)

        sum_current = self.battery_current + self.SC_current
        self.output_current.append(sum_current)

        r_current = -abs(sum_current-self.input_current)
        # r_current = -10*abs(self.battery_current-self.helper_current[self.step_count-1]) #new version

        self.R_current.append(r_current)

        r_capacity = -1000 * abs(self.battery_capacity - Q)
        self.R_capacity.append(r_capacity)

        #Derivative current
        self.current_buffer.append(self.battery_current)
        r_current_a_temp = -50*abs(self.current_buffer[-1] - self.current_buffer[0]) if len(self.current_buffer) > 2 else 0
        self.std_buffer.append(r_current_a_temp)
        r_current_a = np.clip(np.mean(self.std_buffer) if len(self.std_buffer) > 2 else 0,-1000,0)

        # reward_temp = r_current + r_distance + r_capacity + r_current_a

        # reward = float(np.clip(reward_temp, -1e3, 1e4)) #if self.truncated == False else -abs((self.step_count + self.start_idx) - \
        #                                                                                         #self.end_counter) + r_current + r_capacity +  \
        #                                                                                         #+ r_current_a + r_std -100
        # reward += 10 if not self.truncated else -100
        
        reward = r_current_a
        if self.truncated :
            reward -= 1000
        reward += 10*(self.step_count / self.end_counter) # Reward for progress
        if self.terminated:
            reward += 200
        reward_cc = np.clip(10 * -((abs(self.battery_current) - abs(self.SC_current))/ (sum_current +0.1)),-1000,1) # Reward for balancing the current
        self.R_std.append(reward_cc)
        reward +=reward_cc


        self.reward.append(reward)
        # info section                                  
        # if self.truncated == False and self.terminated == False:
        #     self.info = {}
        # else:
        self.info = {
        'battery_voltage'       : self.V_hist,
        'battery_SOC'           : self.SOC_hist,
        'battery_capacity'      : self.Q_hist,
        'SC_voltage'            : self.V_SC_hist,
        'battery_I_history'     : self.battery_I_hist,
        'SC_I_history'          : self.SC_I_hist,
        'requested_I_history'   : self.requested_I_hist,
        'provided_I_history'    : self.provided_I_hist,
        "balancer"              : self.R_std,
        "r_current"             : self.R_current,
        "r_capacity"            : self.R_capacity,
        "r_current_a"           : self.R_current_a,
        "r_distance"            : self.R_distance,
        "r_counter"             : self.R_count,
        "reward"                : self.reward,
        }

        return self.state, reward, self.terminated,self.truncated, self.info

    def render(self, mode='human'):
        pass

    def close(self):
        pass

register_env("MyEvalEnv", lambda config: MyEvalEnv(config))

# from gymnasium.envs.registration import register
# register(
#     id="MyCustomEnv-v0",
#     entry_point=MyEvalEnv,
# )

# env = gym.make("MyCustomEnv-v0", config=config_dict1["env_config"])


n_episodes = 10
returns = []
# Define a low-pass Butterworth filter function

def butter_lowpas_filter(i_input, cutoff,fs,order=1):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, i_input)
    return y

filtered_i = butter_lowpas_filter(i_dc_estimate, 0.00637, 1, order=1)

action = np.zeros((len(i_dc_estimate), 1), dtype=np.float32)
action[:, 0] = filtered_i
# action[:, 1] = i_dc_estimate - filtered_i
# np.save("action.npy", action[:, 0])

plt.figure(figsize=(10, 6))
plt.plot(i_dc_estimate, label='Original Signal', color='black')
plt.plot(action[:, 0], label='Battery Signal', color='blue')
# plt.plot(action[:, 1], label='SC Signal', color='red')
plt.title('Low-pass Filtered Signal')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show()

# testing ENV
eval_config = best_trial_SAC_LSTM.config.copy()
eval_config["env"] = "MyEvalEnv"
eval_config["env_config"] = {
    "i_dc_estimate": i_dc_estimate,
    "i_dc_future": i_dc_future,
    "max_steps": 8000
}
env = MyEvalEnv(eval_config["env_config"])
obs, info = env.reset()
done = False
total_reward = 0.0
counter = 0
info_hist = {}
info_hist[("STD", "r_current", "r_capacity", "r_current_a", "r_distance", "Reward")] = {}

info_hist_rand = {}
info_hist_rand[("STD", "r_current", "r_capacity", "r_current_a", "r_distance", "Reward")] = {}
action_hist=[]

#low pass filter
while not done:
    obs, reward, done, truncated, info = env.step(action[counter,:])
    total_reward += reward
    counter +=1
    returns.append(total_reward)
    if done or truncated or counter ==8000:
        info_hist = info
        print("SOC :", info_hist['battery_SOC'][-1], "Q", info_hist['battery_capacity'][-1])
        break
env.close()

# random filter
obs, info = env.reset()
done = False
total_reward = 0.0
counter = 0
info_hist_rand = {}
info_hist_rand[("STD", "r_current", "r_capacity", "r_current_a", "r_distance", "Reward")] = {}
action_hist = np.empty((0, 2))  # For actions with 2 values (battery, sc)


while not done:
    # action1 = np.array([
    #     np.random.uniform(np.min(i_dc_estimate), np.max(i_dc_estimate))/20,
    #     np.random.uniform(np.min(i_dc_estimate), np.max(i_dc_estimate))/20
    # ])
    action1 = np.array([
        -10,
        -10
    ])
    action_hist = np.append(action_hist, [action1], axis=0)
    obs, reward, done, truncated, info = env.step(action1)
    total_reward += reward
    counter += 1
    returns.append(total_reward)
    print(f"Step {counter}, \t Reward: {reward:.3f}")
    if done or truncated:
        info_hist_rand = info
        print("SOC :", info_hist_rand['battery_SOC'][-1], "\t Done:", done, "\t Truncated:", truncated, "SC Voltage:", info_hist_rand['SC_voltage'][-1])
        break

env.close()

# action_hist = np.array(action_hist)
# print(info_hist.keys())
# print(f"Episode Return  : {total_reward}")
# print(f"STD             : {info_hist['STD']}")
# print(f"r_current       : {info_hist['r_current']}")
# print(f"r_capacity      : {info_hist['r_capacity']}")
# print(f"r_current_a     : {info_hist['r_current_a']}")
# print(f"r_distance      : {info_hist['r_distance']}")
# print(f"Reward          : {info_hist['reward']}")

plt.figure(figsize=(10, 6))
plt.plot(info_hist['battery_I_history'][:-50]      , label='battery_I_history'     , color='black')
plt.plot(info_hist['SC_I_history'][:-50]           , label='SC_I_history'          , color='red')
# plt.plot(info_hist_rand['battery_I_history'][:-50]    , label='battery_I_history_rand'   , color='blue')
# plt.plot(info_hist_rand['SC_I_history'][:-50]     , label='SC_I_history_rand'    , color='green')
# plt.plot(info_hist['STD']                   , label='STD'                   , color='brown')
# plt.plot(info_hist['r_current']             , label='r_current'             , color='pink')
# plt.plot(info_hist['r_capacity']            , label='r_capacity'            , color='orange')
# plt.plot(info_hist['r_current_a']           , label='r_current_a'           , color='cyan')
# plt.plot(info_hist['r_distance']            , label='r_distance'            , color='gold')
# plt.plot(info_hist['reward'][:-50]                  , label='reward'                , color='magenta')
# plt.plot(info_hist_rand['reward'][:-50]             , label='reward1'                , color='blue')
plt.title('Low-pass Filtered Signal')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show()    

plt.figure(figsize=(10, 6))
plt.plot(action_hist[:,0]     , label='battery'     , color='green')
plt.plot(action_hist[:,1]     , label='sc'          , color='gold')
plt.title('action Signal')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show()    

####################################Testing Battery Model ##########################################################
battery    = ECM_RC_Battery(Q_init=3.3, alpha=0.0002, R0=30e-3, R1=0.02, C1=2500, dt=1, Vmin=2.5, Vmax=4.2)
capacitor = Supercapacitor(C=500, R_esr=0.01, V_init=16, dt=1.0)
SOC1=[]
SOC2=[]
V_battery = []
V_capacitor = []
for i in range(len(action)):
    V_bat, SOC, Q, V_RC = battery.step(action[i]/22.0)
    V_out, V_SC = capacitor.step(i_dc_estimate[i]-action[i])
    print(f"\t Step {i+1}: \t SOC: {SOC}, \t Capacity: {Q}, \t V_bat: {V_bat*60}, \t V_SC: {V_SC*7}, current: {action[i,0]}")
    SOC1.append(SOC)
    SOC2.append(Q)
    V_battery.append(V_bat*60)
    V_capacitor.append(V_SC*7)    
    if V_bat <= 2:
        print("Battery SOC is too low, stopping simulation.")
        break

# for i in range(1000):
#     V_out, V_SC = capacitor.step(np.random.uniform(np.min(action[:,1]),np.max(action[:,1])))
#     print(f"\t Step {i+1}: \t SOC: {V_out}, \t Capacity: {V_SC}")
#     SOC2.append(V_SC)
#     if V_SC <= 2:
#         print("Battery SOC is too low, stopping simulation.")
#         break


# for i in range(1000):
#     V_bat, SOC, Q, V_RC = battery.step(np.random.uniform(np.min(i_dc_estimate),np.max(i_dc_estimate))/60)
#     # print(f"\t Step {i+1}: \t SOC: {SOC}, \t Capacity: {Q}")
#     SOC2.append(SOC)
#     if SOC <= 0.05:
#         print("Battery SOC is too low, stopping simulation.")
#         break

fig, ax = plt.subplots(1, 3, figsize=(10, 6))  # 1 row, 2 columns

# Plot 1
ax[0].plot(SOC1, label='SOC', color='gold')
ax[0].plot(SOC2, label='Q', color='black')
ax[0].set_title('Low-pass Filtered Signal')
ax[0].set_xlabel('Sample Number')
ax[0].set_ylabel('Amplitude')
ax[0].legend()
ax[0].grid()

# Plot 2
ax[1].plot(V_battery, label='V_battery', color='gold')
ax[1].plot(V_capacitor, label='V_capacitor', color='black')
ax[1].set_title('Low-pass Filtered Signal')
ax[1].set_xlabel('Sample Number')
ax[1].set_ylabel('Amplitude')
ax[1].legend()
ax[1].grid()

# Plot 3
ax[2].plot(action[:,0],                     label='I_battery', color='gold')
ax[2].plot(i_dc_estimate[:]-action[:,0],    label='I_capacitor', color='black')
ax[2].plot(i_dc_estimate[:],                label='requested', color='blue')
ax[2].set_title('Low-pass Filtered Signal')
ax[2].set_xlabel('Sample Number')
ax[2].set_ylabel('Amplitude')
ax[2].legend()
ax[2].grid()

# Show both plots
plt.tight_layout()
plt.show()


# Effect of ai on Q
fig, axes = plt.subplots(2, 1, figsize=(8, 6))  # 2 rows, 1 column

x = np.linspace(0, 10, 100)
axes[0].plot(action[:,0])
axes[0].set_title("Current")

axes[1].plot(i_dc_estimate)
axes[1].set_title("requested")

plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 5))
plt.plot(action[:, 0], label='I_battery', color='gold')
plt.plot(i_dc_estimate[:] - action[:, 0] , label='I_capacitor', color='black')
plt.plot(i_dc_estimate[:], label='Requested Current', color='blue')

plt.title('Low-pass Filtered Signal')
plt.xlabel('Sample Number')
plt.ylabel('Current [A]')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

################################ testing Algo ( testing and training ENV must be the same) ##################

#################Loading the trained agent from checkpoint ##########################
# import ray
# from ray.rllib.algorithms.sac import SAC
# from ray.rllib.algorithms.algorithm import Algorithm
# import faulthandler, sys

# # --- Fix for IPython/Jupyter faulthandler issue ---
# faulthandler.enable(file=sys.__stdout__)

# # --- Start Ray ---
# ray.init(ignore_reinit_error=True, log_to_driver=False)

# # --- Path to your checkpoint directory ---
# checkpoint_path = "/home/armin/ray_results/SAC-LSTM-Experimente-newReward/SAC_MyCustomEnv_566e6_00000_0_2025-07-14_16-23-00/checkpoint_000040"

# # --- Load the agent ---
# algo_SAC_LSTM = SAC.from_checkpoint(checkpoint_path)
######################################################################################

from ray.rllib.algorithms.sac import SAC
from ray.rllib.algorithms.ppo import PPO

checkpoint_path_1 = best_checkpoint_SAC_LSTM.path
algo_SAC_LSTM = SAC.from_checkpoint(checkpoint_path_1)

checkpoint_path_2 = best_checkpoint_SAC.path
algo_SAC = SAC.from_checkpoint(checkpoint_path_2)

checkpoint_path_3 = best_checkpoint_PPO_LSTM.path
algo_PPO_LSTM = PPO.from_checkpoint(checkpoint_path_3)

checkpoint_path_4 = best_checkpoint_PPO.path
algo_PPO = PPO.from_checkpoint(checkpoint_path_4)

# Prepare config for restoring
eval_config1 = best_trial_SAC_LSTM.config.copy()
eval_config1["env"] = "MyEvalEnv"
eval_config1["env_config"] = {
    "i_dc_estimate": i_dc_estimate,
    "i_dc_future": i_dc_future,
    "max_steps": len(i_dc_estimate)
}

eval_config2 = best_trial_SAC.config.copy()
eval_config2["env"] = "MyEvalEnv"
eval_config2["env_config"] = {
    "i_dc_estimate": i_dc_estimate,
    "i_dc_future": i_dc_future,
    "max_steps": len(i_dc_estimate)
}

eval_config3 = best_trial_PPO_LSTM.config.copy()
eval_config3["env"] = "MyEvalEnv"
eval_config3["env_config"] = {
    "i_dc_estimate": i_dc_estimate,
    "i_dc_future": i_dc_future,
    "max_steps": len(i_dc_estimate)
}

eval_config4 = best_trial_PPO.config.copy()
eval_config4["env"] = "MyEvalEnv"
eval_config4["env_config"] = {
    "i_dc_estimate": i_dc_estimate,
    "i_dc_future": i_dc_future,
    "max_steps": len(i_dc_estimate)
}


# Instantiate the environment manually (for step-by-step evaluation)
env1 = MyEvalEnv(eval_config1["env_config"])
env2 = MyEvalEnv(eval_config2["env_config"])
env3 = MyEvalEnv(eval_config3["env_config"])
env4 = MyEvalEnv(eval_config4["env_config"])

obs, info = env1.reset()
done = False
truncated = False
total_reward = 0.0
action_mem = np.empty((0, 1))
reward = []
counter = 0
info_hist_SAC_LSTM = {}
info_hist_SAC_LSTM[("STD", "r_current", "r_capacity", "r_current_a", "r_distance", "Reward")] = {}

info_hist_SAC = {}
info_hist_SAC[("STD", "r_current", "r_capacity", "r_current_a", "r_distance", "Reward")] = {}

info_hist_PPO_LSTM = {}
info_hist_PPO_LSTM[("STD", "r_current", "r_capacity", "r_current_a", "r_distance", "Reward")] = {}

info_hist_PPO = {}
info_hist_PPO[("battery_I_history", "SC_I_history", "r_capacity", "r_current_a", "r_distance", "Reward")] = {}

info_hist_lowpass = {}
# SAC-LSTM
while not (done or truncated):
    action = algo_SAC_LSTM.compute_single_action(obs, explore=False)
    obs, reward, done, truncated, info = env1.step(action) 
    action_mem = np.append(action_mem, [action], axis=0)
    total_reward += reward
    counter += 1
    print(f"Step {counter}, \t Reward: {reward:.3f}")
    if done or truncated:
        info_hist_SAC_LSTM = info
        print("SOC :", info_hist_SAC_LSTM['battery_SOC'][-1], "\t Done:", done, "\t Truncated:", truncated, "battery_capacity:", info_hist_SAC_LSTM['battery_capacity'][-1], "SC_voltage:", info_hist_SAC_LSTM['SC_voltage'][-1])
        break

env1.close()

# PARAMETERS MONITORING SECTION
fig, ax = plt.subplots(1, 2, figsize=(10, 6))
ax[0].plot(info_hist_SAC_LSTM['battery_SOC']         , label='battery_SOC'         , color='red')
ax[0].plot(info_hist_SAC_LSTM['battery_voltage']     , label='battery_voltage'     , color='blue')
ax[0].plot(info_hist_SAC_LSTM['SC_voltage']          , label='SC_voltage'          , color='green')

ax[1].plot(info_hist_SAC_LSTM['battery_capacity']    , label='battery_capacity'    , color='black')
fig.suptitle("Battery Monitoring Parameters", fontsize=16)
fig.supxlabel("Sample Number", fontsize=12)
fig.supylabel("Amplitude", fontsize=12)
ax[0].legend()
ax[1].legend()
ax[0].grid(True)
ax[1].grid(True)
plt.show() 
plt.close('all')

# REWARD SECTION
plt.figure(figsize=(10, 6))
plt.plot(info_hist_SAC_LSTM['balancer']                  , label='balancer'           , color='red')
# plt.plot(info_hist_SAC_LSTM['r_current']            , label='r_current'     , color='black')
plt.plot(info_hist_SAC_LSTM['r_capacity']           , label='r_capacity'    , color='blue')
plt.plot(info_hist_SAC_LSTM['r_current_a']          , label='r_current_a'   , color='green')
# plt.plot(info_hist_SAC_LSTM['r_distance']           , label='r_distance'    , color='gold')
plt.title('Reward monitoring')
# fig.suptitle('Reward monitoring')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show() 
plt.close('all')

# ACTION SECTION
plt.figure(figsize=(10, 6))
plt.plot(info_hist_SAC_LSTM['requested_I_history']   , label='requested_I_history'   , color='red')
plt.plot(info_hist_SAC_LSTM['provided_I_history']    , label='provided_I_history'    , color='green')
plt.plot(info_hist_SAC_LSTM['battery_I_history']     , label='batt'                  , color='gold')
plt.plot(info_hist_SAC_LSTM['SC_I_history']          , label='SC'                    , color='blue')
plt.title('Reward monitoring')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show() 
plt.close('all')

obs, info = env2.reset()
done = False
truncated = False
total_reward = 0.0
action_mem = np.empty((0, 1))
reward = []
counter = 0

# SAC
while not (done or truncated):
    action = algo_SAC.compute_single_action(obs, explore=False)
    obs, reward, done, truncated, info = env2.step(action) 
    action_mem = np.append(action_mem, [action], axis=0)
    total_reward += reward
    counter += 1
    print(f"Step {counter}, \t Reward: {reward:.3f}")
    if done or truncated:
        info_hist_SAC = info
        print("SOC :", info_hist_SAC['battery_SOC'][-1], "\t Done:", done, "\t Truncated:", truncated, "battery_capacity:", info_hist_SAC['battery_capacity'][-1], "SC_voltage:", info_hist_SAC['SC_voltage'][-1])
        break

env2.close()

# PARAMETERS MONITORING SECTION
fig, ax = plt.subplots(1, 2, figsize=(10, 6))
ax[0].plot(info_hist_SAC['battery_SOC']         , label='battery_SOC'         , color='red')
ax[0].plot(info_hist_SAC['battery_voltage']     , label='battery_voltage'     , color='blue')
ax[0].plot(info_hist_SAC['SC_voltage']          , label='SC_voltage'          , color='green')

ax[1].plot(info_hist_SAC['battery_capacity']    , label='battery_capacity'    , color='black')
fig.suptitle("Battery Monitoring Parameters", fontsize=16)
fig.supxlabel("Sample Number", fontsize=12)
fig.supylabel("Amplitude", fontsize=12)
ax[0].legend()
ax[1].legend()
ax[0].grid(True)
ax[1].grid(True)
plt.show() 

# REWARD SECTION
plt.figure(figsize=(10, 6))
plt.plot(info_hist_SAC['STD']                  , label='STD'           , color='red')
plt.plot(info_hist_SAC['r_current']            , label='r_current'     , color='black')
plt.plot(info_hist_SAC['r_capacity']           , label='r_capacity'    , color='blue')
plt.plot(info_hist_SAC['r_current_a']          , label='r_current_a'   , color='green')
plt.plot(info_hist_SAC['r_distance']           , label='r_distance'    , color='gold')
plt.title('Reward monitoring')
# fig.suptitle('Reward monitoring')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show() 

# ACTION SECTION
plt.figure(figsize=(10, 6))
plt.plot(info_hist_SAC['requested_I_history']   , label='requested_I_history'   , color='red')
plt.plot(info_hist_SAC['provided_I_history']    , label='provided_I_history'    , color='green')
plt.plot(info_hist_SAC['battery_I_history']     , label='batt'                  , color='gold')
plt.plot(info_hist_SAC['SC_I_history']          , label='SC'                    , color='blue')
plt.title('Reward monitoring')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show() 




# PPO LSTM
obs, info = env3.reset()
done = False
truncated = False
total_reward = 0.0
while not (done or truncated):
    action = algo_PPO_LSTM.compute_single_action(obs, explore=False)  # Do NOT overwrite algo.config
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    if done or truncated or counter ==len(i_dc_estimate):
        info_hist_PPO_LSTM = info
        break
env3.close()

# PPO
obs, info = env4.reset()
done = False
truncated = False
total_reward = 0.0
while not (done or truncated):
    action = algo_PPO.compute_single_action(obs, explore=False)  # Do NOT overwrite algo.config
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    if done or truncated or counter ==len(i_dc_estimate):
        info_hist_PPO = info
        break
env4.close()

######### Low pass filter
env1 = MyEvalEnv(eval_config1["env_config"])
obs, info = env1.reset()
done = False
truncated = False
total_reward = 0.0
action_mem = np.empty((0, 1))
reward = []
counter = 0
info_hist_lowpass = {}
counter=0

action_mem = np.empty((0, 1), dtype=np.float32)
for i in range(len(i_dc_estimate)):

    obs, reward, done, truncated, info = env1.step(float(action[i]))
    action_mem = np.append(action_mem, float(action[i]))
    total_reward += reward
    counter += 1
    print(f"Step {i+1}, \t Reward: {reward:.3f}")
    if done or truncated or counter == len(i_dc_estimate):
        info_hist_lowpass = info
        print("SOC :", info_hist_lowpass['battery_SOC'][-1],
            "\t Done:", done, "\t Truncated:", truncated,
            "battery_capacity:", info_hist_lowpass['battery_capacity'][-1],
            "SC_voltage:", info_hist_lowpass['SC_voltage'][-1])
        break

env1.close()

# PARAMETERS MONITORING SECTION
fig, ax = plt.subplots(1, 2, figsize=(10, 6))
ax[0].plot(info_hist_lowpass['battery_SOC']         , label='battery_SOC'         , color='red')
ax[0].plot(info_hist_lowpass['battery_voltage']     , label='battery_voltage'     , color='blue')
ax[0].plot(info_hist_lowpass['SC_voltage']          , label='SC_voltage'          , color='green')

ax[1].plot(info_hist_lowpass['battery_capacity']    , label='battery_capacity'    , color='black')
fig.suptitle("Battery Monitoring Parameters", fontsize=16)
fig.supxlabel("Sample Number", fontsize=12)
fig.supylabel("Amplitude", fontsize=12)
ax[0].legend()
ax[1].legend()
ax[0].grid(True)
ax[1].grid(True)
plt.show() 

print("STD:", info_hist_lowpass['STD'][-1],"\n r_current:", info_hist_lowpass['r_current'][-1],
      "\n r_capacity:"          , info_hist_lowpass['r_capacity'][-1],
      "\n r_current_a:"         , info_hist_lowpass['r_current_a'][-1],
      "\n r_distance:"          , info_hist_lowpass['r_distance'][-1],
      "\n Reward:"              , info_hist_lowpass['reward'][-1],
      "\n Battery Current:"     , info_hist_lowpass['battery_I_history'][-1],
      "\n SC Current:"          , info_hist_lowpass['SC_I_history'][-1],
      "\n Requested Current:"   , info_hist_lowpass['requested_I_history'][-1],
      "\n Provided Current:"    , info_hist_lowpass['provided_I_history'][-1])




# REWARD SECTION
fig,ax = plt.subplots(2,1,figsize=(10, 6))
# ax[0].plot(info_hist_lowpass['STD']                  , label='STD'           , color='red')
ax[0].plot(info_hist_lowpass['r_current']           , label='r_current'     , color='black')
ax[0].plot(info_hist_lowpass['r_capacity']           , label='r_capacity'    , color='blue')
ax[0].plot(info_hist_lowpass['r_current_a']          , label='r_current_a'   , color='green')
ax[0].plot(info_hist_lowpass['r_distance']           , label='r_distance'    , color='gold')
# ax[0].plot(info_hist_lowpass['r_counter']           , label='r_counter'    , color='gray')
ax[0].title('Reward monitoring')
ax[0].suptitle('Reward monitoring')
ax[0].xlabel('Sample Number')
ax[0].ylabel('Amplitude')
ax[0].legend()
ax[0].grid()

ax[1].plot(info_hist_lowpass['reward']           , label='reward'    , color='darkorange')
ax[1].title('Reward monitoring')
ax[1].suptitle('Reward monitoring')
ax[1].xlabel('Sample Number')
ax[1].ylabel('Amplitude')
ax[1].legend()
ax[1].grid(True)

plt.show() 

# ACTION SECTION
plt.figure(figsize=(10, 6))
plt.plot(info_hist_lowpass['requested_I_history']   , label='requested_I_history'   , color='red')
plt.plot(info_hist_lowpass['provided_I_history']    , label='provided_I_history'    , color='green')
plt.plot(info_hist_lowpass['battery_I_history']     , label='batt'                  , color='gold')
plt.plot(info_hist_lowpass['SC_I_history']          , label='SC'                    , color='blue')
plt.title('Reward monitoring')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show() 


################################## Plotting reward components ############################################################################################
import pandas as pd
import matplotlib.pyplot as plt

# Load progress.csv from both versions
PPO_predict             = pd.read_csv("/home/armin/ray_results/PPO-Experiment/PPO_MyCustomEnv_1a126_00000_0_2025-07-06_01-50-20/progress.csv")
PPO_LSTM_predict        = pd.read_csv("/home/armin/ray_results/PPO-LSTM-Experiment/PPO_MyCustomEnv_f0aee_00000_0_2025-07-06_01-34-52/progress.csv")
PPO_nopredict           = pd.read_csv("/home/armin/ray_results/PPO_results_Nopredict/PPO_MyCustomEnv_3c4e9_00000_0_2025-07-06_10-41-00/progress.csv")
PPO_LSTM_nopredict      = pd.read_csv("/home/armin/ray_results/PPO_LSTM_results_Nopredict/PPO_MyCustomEnv_7484d_00000_0_2025-07-06_02-28-39/progress.csv")

SAC_predict             = pd.read_csv("/home/armin/ray_results/SAC-Experiment/SAC_MyCustomEnv_24ef6_00000_0_2025-07-07_12-05-04/progress.csv")
SAC_LSTM_predict        = pd.read_csv("/home/armin/ray_results/SAC-LSTM-Experiment/SAC_MyCustomEnv_31f8f_00000_0_2025-07-07_11-51-07/progress.csv")
SAC_nopredict           = pd.read_csv("/home/armin/ray_results/SAC_results_Nopredict/SAC_MyCustomEnv_a4c9d_00000_0_2025-07-06_02-08-32/progress.csv")
SAC_LSTM_nopredict      = pd.read_csv("/home/armin/ray_results/SAC_LSTM_results_Nopredict/SAC_MyCustomEnv_05714_00000_0_2025-07-06_01-49-45/progress.csv")

TD3_predict             = pd.read_csv("/home/armin/ray_results/TD3_results_prediction/TD3_MyCustomEnv_28a48_00000_0_2025-07-06_13-25-06/progress.csv")
TD3_LSTM_predict        = pd.read_csv("/home/armin/ray_results/TD3_LSTM_results_prediction/TD3_MyCustomEnv_a6914_00000_0_2025-07-06_13-07-09/progress.csv")
TD3_nopredict           = pd.read_csv("/home/armin/ray_results/TD3_results_Nopredict/TD3_MyCustomEnv_fa02d_00000_0_2025-07-06_12-48-00/progress.csv")
TD3_LSTM_nopredict      = pd.read_csv("/home/armin/ray_results/TD3_LSTM_results_Nopredict/TD3_MyCustomEnv_718bb_00000_0_2025-07-06_12-22-43/progress.csv")

# Choose the reward metric you want (most common: 'episode_reward_mean')
reward_key = 'episode_reward_mean'  # sometimes it's 'custom_metrics/total_reward_mean'

plt.plot(PPO_predict[reward_key],       label='PPO_predict',        linewidth=2)
plt.plot(PPO_LSTM_predict[reward_key],  label='PPO_LSTM_predict',   linewidth=2)
# plt.plot(PPO_nopredict[reward_key],     label='PPO_predict',        linewidth=2)
# plt.plot(PPO_LSTM_nopredict[reward_key],label='PPO_LSTM_predict',   linewidth=2)

plt.plot(SAC_predict[reward_key],       label='SAC_predict',        linewidth=2)
plt.plot(SAC_LSTM_predict[reward_key],  label='SAC_LSTM_predict',   linewidth=2)
# plt.plot(SAC_nopredict[reward_key],     label='SAC_predict',        linewidth=2)
# plt.plot(SAC_LSTM_nopredict[reward_key],label='SAC_LSTM_predict',   linewidth=2)

# plt.plot(TD3_predict[reward_key],       label='TD3_predict',        linewidth=2)
# plt.plot(TD3_LSTM_predict[reward_key],  label='TD3_LSTM_predict',   linewidth=2)
# plt.plot(TD3_nopredict[reward_key],     label='TD3_predict',        linewidth=2)
# plt.plot(TD3_LSTM_nopredict[reward_key],label='TD3_LSTM_predict',   linewidth=2)

plt.xlabel("Training Iterations")
plt.ylabel("Mean Episode Reward")
plt.title("Reward Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
