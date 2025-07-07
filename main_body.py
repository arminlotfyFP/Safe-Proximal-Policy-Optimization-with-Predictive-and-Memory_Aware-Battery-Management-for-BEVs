
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
data1 = sio.loadmat('I_UDDS.mat')
i_dc1_estimate = data['i_dc_estimate'].transpose().flatten()
i_dc2_estimate = data1['I_UDDS'].transpose().flatten()
i_dc_estimate = np.concatenate((i_dc1_estimate, i_dc2_estimate), axis=0)
i_dc_future = np.roll(i_dc_estimate, -30)
i_dc_future_mean = np.zeros_like(i_dc_future)

for i in range(len(i_dc_future) - 30):
    i_dc_future_mean[i] = i_dc_future[i:i+30].mean()

# Define a low-pass Butterworth filter function

# def butter_lowpas_filter(i_input, cutoff,fs,order=1):
#     nyq = 0.5 * fs
#     normal_cutoff = cutoff / nyq
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     y = filtfilt(b, a, i_input)
#     return y

# filtered_i = butter_lowpas_filter(i_dc_estimate, 0.00637, 1, order=1)

# plt.figure(figsize=(10, 6))
# plt.plot(i_dc_estimate, label='Original Signal', color='blue')
# plt.plot(filtered_i, label='Filtered Signal', color='red')
# plt.title('Low-pass Filtered Signal')
# plt.xlabel('Sample Number')
# plt.ylabel('Amplitude')
# plt.legend()
# plt.grid()
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(i_dc_estimate, label='Original Signal', color='blue')
# plt.plot(i_dc_future_mean, label='Filtered Signal', color='red')
# plt.title('Low-pass Filtered Signal')
# plt.xlabel('Sample Number')
# plt.ylabel('Amplitude')
# plt.legend()
# plt.grid()
# plt.show()



# Define Battery & SC

class Supercapacitor:
    def __init__(self, C=500, R_esr=0.01, V_init=2.7, dt=1.0):
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
    def __init__(self, Q_init=3.2, alpha=0.0002, R0=0.01, R1=0.02, C1=2500, dt=1, Vmin=2.5, Vmax=4.2):
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
        self.SOC = 1.0                # Initial SOC (1.0 = 100%)
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
        if self.SOC == 0 or self.SOC == 1.0:
            # self.cycle_count += 1
            # self.Q = self.Q_init * (1 - self.alpha * self.cycle_count)
            pass
        else:
            self.Q = self.Q-self.ai*0.000_05  - 0.1*(abs(I_bat) * self.dt) / 3600 if abs(I_bat)< 3*self.Q else self.Q-self.ai*0.000_05 - 3*0.1*((abs(I_bat) * self.dt) / 3600) 
        self.Q = max(self.Q, 0.0)

        return V_bat, self.SOC, self.Q, self.V_RC

min_current = -np.min(i_dc_estimate)    #########################
max_current =  np.max(i_dc_estimate)    #########################
# Define my GymEnvironment:
# input_current, future_current, battery_current, SC_current, battery_capacity,battery_SOC, SC_voltage
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
        
        self.observation_space = spaces.Box(low=np.array([-1.5, -1.5, -1.5, -1.5, 0, 0, 0]), 
                                            high=np.array([1.5, 1.5, 1.5, 1.5, 1, 1, 1]), 
                                            shape=(7,), dtype=np.float32
)
        
        self.action_space       = spaces.Box(low=-np.array([min_current,min_current]), 
                                            high=np.array([max_current,max_current]), shape=(2,), dtype=np.float32)
        
        self.input_current      = self.i_dc_estimate[0] + 10* np.random.uniform(-1, 1)
        self.output_current     = []
        self.future_current     = self.i_dc_future[0]   + 10* np.random.uniform(-1, 1)
        

        # Deining Buffers
        self.buffer = deque(maxlen=60)
        self.current_buffer = deque(maxlen=10)

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
        self.reward             = []

        # Battery
        self.battery_current    = 0
        self.battery_capacity   = 3.2 + 0.1 * np.random.uniform(-1, 1)
        self.battery_SOC        = 0.60  + 0.20  * np.random.uniform(-1, 1)
        self.battery_voltage    = 2.5 + self.battery_SOC  * (3.1 - 2.5)
        self.n_batt_par         = 22
        self.n_batt_seri        = 60
        self.fading_coefficient = 2e-8  # alpha 2 coeff for derivative
        self.R0                 = 0.03
        self.R1                 = 0.04
        self.C1                 = 750
        # SC
        self.C_n_seri           = 7
        self.SC_current         = 0
        self.SC_voltage         = (np.random.uniform(9,16))
        self.end_counter        = self.max_steps
        self.SC_C               = 58
        self.R_esr              = 22e-3
        


        self.terminated         = False
        self.truncated          = False
        self.info               = {}
        self.step_count         = 0
        # Defining Models
        self.battery    = ECM_RC_Battery(Q_init=self.battery_capacity, alpha=self.fading_coefficient, R0=self.R0, R1=self.R1, C1=self.C1, dt=self.dt, Vmin=2.5, Vmax=4.0)
        self.sc         = Supercapacitor(C=self.SC_C, R_esr=self.R_esr, V_init=self.SC_voltage , dt=self.dt)
        

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # shuffle_indices         = np.random.permutation(len(self.i_dc_estimate))  # ############
        # self.i_dc_estimate      = self.i_dc_estimate[shuffle_indices]               #######################
        # self.i_dc_future        = self.i_dc_future[shuffle_indices]             #######################

        self.start_idx          = np.random.randint(0, len(self.i_dc_estimate) - 100)                   ############
        self.input_current      = self.i_dc_estimate[self.start_idx] + 10* np.random.uniform(-1, 1) #################
        self.future_current     = self.i_dc_future[self.start_idx]   + 10* np.random.uniform(-1, 1) #################
        self.output_current     = []

        self.battery_current    = 0
        self.battery_capacity   = 3.2 + 0.1 * np.random.uniform(-1, 1)
        self.battery_SOC        = 0.60  + 0.20  * np.random.uniform(-1, 1)
        self.battery_voltage    = 2.5 + self.battery_SOC  * (3.1 - 2.5)

        self.buffer.clear()
        self.current_buffer.clear()

        self.SC_C               = 58
        self.R_esr              = 22e-3
        self.C_n_seri           = 7
        self.SC_current         = 0
        self.SC_voltage         = (np.random.uniform(9,16))
        self.terminated         = False
        self.truncated          = False
        self.info               = {}
        self.step_count         = 0

        self.battery    = ECM_RC_Battery(Q_init=self.battery_capacity, alpha=self.fading_coefficient, R0=self.R0, R1=self.R1, C1=self.C1, dt=self.dt, Vmin=2.5, Vmax=4.0)
        self.sc         = Supercapacitor(C=self.SC_C, R_esr=self.R_esr, V_init=self.SC_voltage , dt=self.dt)

        self.state = np.array([self.input_current/np.max(self.i_dc_estimate), self.future_current/np.max(self.i_dc_estimate), self.battery_current/np.max(self.i_dc_estimate),
                            self.SC_current/np.max(self.i_dc_estimate), 1, self.battery_SOC, np.clip(self.SC_voltage/16,0.0,1.0)], dtype=np.float32)
        
        self.state = np.clip(self.state, -1e6, 1e6)
        return self.state, {}

    def step(self, action):
        self.battery_current    = np.clip(action[0], self.action_space.low[0], self.action_space.high[0])
        self.SC_current         = np.clip(action[1], self.action_space.low[1], self.action_space.high[1])
        battery_current_cell    = self.battery_current/self.n_batt_par
        SC_current_cell         = self.SC_current/1 # We do not have parallel SCs in this case  
        
        V_bat, SOC, Q, V_RC = self.battery.step(battery_current_cell)
        V_SC, V_sc          = self.sc.step(SC_current_cell)
        self.battery_SOC    = SOC  
        self.SC_voltage     = V_SC*self.C_n_seri 


        # state section
        epsilon = np.random.uniform(0, 1)
        self.input_current      = self.i_dc_estimate[self.step_count+ self.start_idx] if epsilon > 0.1 else self.i_dc_estimate[self.step_count] + 10* np.random.uniform(0, 1)
        self.future_current     = self.i_dc_future[self.step_count + self.start_idx]   if epsilon > 0.1 else self.i_dc_future[self.step_count] + 10* np.random.uniform(0, 1)

        # Append to history
        self.V_hist.append(V_bat*self.n_batt_seri)                 
        self.SOC_hist.append(SOC*100)                                       
        self.Q_hist.append(Q)                                               
        self.V_SC_hist.append(V_SC*self.C_n_seri)                           
        self.battery_I_hist.append(self.battery_current)                    
        self.SC_I_hist.append(self.SC_current )                               
        self.requested_I_hist.append(self.input_current)                    
        self.provided_I_hist.append(self.battery_current + self.SC_current)

        self.state = np.array([self.input_current/np.max(self.i_dc_estimate), self.future_current/np.max(self.i_dc_estimate), self.battery_current/np.max(self.i_dc_estimate),
                            self.SC_current/np.max(self.i_dc_estimate), Q/self.battery_capacity, self.battery_SOC, np.clip(self.SC_voltage/110,0.0,1.0)], dtype=np.float32)
        
        # Stop conditions section
        self.truncated = self.battery_SOC  <= 0.05 or self.SC_voltage <= 60 or self.SC_voltage >= 110
        self.terminated = self.step_count + self.start_idx >= self.end_counter - 1  #############################
        
        
        # Define reward sections:
        
        #STD
        self.buffer.append(self.battery_current)
        r_std = -abs(np.std(self.buffer))
        #Current
        sum_current = self.battery_current + self.SC_current
        self.output_current.append(sum_current)
        r_current = -10*abs(sum_current-self.input_current)
        #Capacity
        r_capacity = -500 * abs(self.battery_capacity - Q)
        #Derivative current
        self.current_buffer.append(self.battery_current)
        r_current_a = -abs(self.current_buffer[-1] - self.current_buffer[0]) if len(self.current_buffer) > 1 else 0
        # Distance
        r_distance = np.clip(20000*(1 / (1+ abs(self.end_counter - self.step_count))),0,10)
        # SOC??????????

        reward_temp = r_current + r_capacity + r_current_a + r_std + r_distance

        reward = float(np.clip(reward_temp, -1e3, 1e4)) if self.truncated == False else 100 * reward_temp
        self.reward.append(reward)
        assert not np.isnan(self.state).any(), "NaN in observation"
        assert not np.isnan(reward), "NaN in reward"
        assert np.isfinite(self.state).all(), "Inf in observation"
        # info section                                  #########################################
        if self.truncated != True and self.terminated != True:
            self.info = {}
        else:
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
        self.step_count += 1
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
    "log_level": "ERROR",
    "framework": "torch",   # <--- change to torch
    "num_workers": 4,
    "num_gpus": 0,          # Set to 0 if you don't have a GPU
    "num_envs_per_worker": 2,
    "actor_lr": 1e-4,
    "critic_lr": 1e-4,
    "alpha_lr": 5e-5,
    "normalize_actions": True,
    "normalize_observations": False,
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
            "use_lstm": True,
            "lstm_cell_size": 32,
            "max_seq_len": 75,
            "lstm_use_prev_action": True,
            "lstm_use_prev_reward": True,
            "fcnet_hiddens": [32, 16],
            "fcnet_activation": "relu",
            "burn_in": 5, # Number of initial steps to ignore before LSTM starts processing
        }
    }
}

config_dict2 = {
    "log_level": "ERROR",
    "framework": "torch",   # <--- change to torch
    "num_workers": 4,
    "num_gpus": 0,          # Set to 0 if you don't have a GPU
    "num_envs_per_worker": 2,
    "actor_lr": 1e-4,
    "critic_lr": 1e-4,
    "alpha_lr": 5e-5,
    "normalize_actions": True,
    "normalize_observations": False,
    "clip_rewards": False,
    "target_entropy": "auto",
    "entropy_coeff": "auto",  # Automatically adjust entropy coefficient
    "explore": False,  # Enable exploration
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
            "lstm_cell_size": 32,
            "max_seq_len": 75,
            "lstm_use_prev_action": True,
            "lstm_use_prev_reward": True,
            "fcnet_hiddens": [32, 16],
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
    "training_iteration": 500, # Set a high number for iterations
    #"episode_reward_mean": -10,    # Stop when mean reward reaches 200
}

stop_criteria_PPO = {
    "training_iteration": 500, # Set a high number for iterations
    "episode_reward_mean": -10,    # Stop when mean reward reaches 200
}


from ray.tune.logger import TBXLogger
from ray.tune.logger import pretty_print
from ray.tune.logger import CSVLoggerCallback

results1 = tune.run(
    "SAC",
    config=config_dict1,
    stop=stop_criteria,
    verbose=1,
    name="SAC-LSTM-Experiment",
    local_dir="~/SAC_LSTM_results",
    checkpoint_at_end=True,                      
    checkpoint_freq=5,                           
    keep_checkpoints_num=3,                      
    checkpoint_score_attr="episode_reward_mean", # Maximize by default
    log_to_file=True,  
    callbacks=[CSVLoggerCallback()]
)




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

best_trial_SAC_LSTM = results1.get_best_trial(metric="episode_reward_mean", mode="max")
best_trial_SAC      = results2.get_best_trial(metric="episode_reward_mean", mode="max")
best_trial_PPO_LSTM = results5.get_best_trial(metric="episode_reward_mean", mode="max")
best_trial_PPO      = results6.get_best_trial(metric="episode_reward_mean", mode="max")

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


class MyEvalEnv(Env):
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        self.i_dc_estimate      = config.get("i_dc_estimate", np.zeros(1000))
        self.i_dc_future        = config.get("i_dc_future", np.zeros(1000))
        self.max_steps          = config.get("max_steps", 8000)
        
        self.observation_space = spaces.Box(low=np.array([-1.5, -1.5, -1.5, -1.5, 0, 0, 0]), 
                                            high=np.array([1.5, 1.5, 1.5, 1.5, 1, 1, 1]), 
                                            shape=(7,), dtype=np.float32
)
        
        self.action_space       = spaces.Box(low=-np.array([min_current,min_current]), 
                                            high=np.array([max_current,max_current]), shape=(2,), dtype=np.float32)
        
        self.input_current      = self.i_dc_estimate[0] + 10* np.random.uniform(-1, 1)
        self.output_current     = []
        self.future_current     = self.i_dc_future[0]   + 10* np.random.uniform(-1, 1)
        

        # Deining Buffers
        self.buffer = deque(maxlen=60)
        self.current_buffer = deque(maxlen=10)

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
        # Battery
        self.battery_current    = 0
        self.battery_capacity   = 3.2 + 0.1 * np.random.uniform(-1, 1)
        self.battery_SOC        = 0.60  + 0.20  * np.random.uniform(-1, 1)
        self.battery_voltage    = 2.5 + self.battery_SOC  * (3.1 - 2.5)
        self.n_batt_par         = 22
        self.n_batt_seri        = 60
        self.fading_coefficient = 2e-8  # alpha 2 coeff for derivative
        self.R0                 = 0.03
        self.R1                 = 0.04
        self.C1                 = 750
        # SC
        self.C_n_seri           = 7
        self.SC_current         = 0
        self.SC_voltage         = (np.random.uniform(9,16))  ############################################
        self.end_counter        = self.max_steps
        self.SC_C               = 58
        self.R_esr              = 22e-3
        


        self.terminated         = False
        self.truncated          = False
        self.info               = {}
        self.step_count         = 0
        # Defining Models
        self.battery    = ECM_RC_Battery(Q_init=self.battery_capacity, alpha=self.fading_coefficient, R0=self.R0, R1=self.R1, C1=self.C1, dt=self.dt, Vmin=2.5, Vmax=4.0)
        self.sc         = Supercapacitor(C=self.SC_C, R_esr=self.R_esr, V_init=self.SC_voltage , dt=self.dt)
        

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.input_current      = self.i_dc_estimate[0] + 10* np.random.uniform(-1, 1)
        self.future_current     = self.i_dc_future[0]   + 10* np.random.uniform(-1, 1)
        self.output_current     = []

        self.battery_current    = 0
        self.battery_capacity   = 3.2 + 0.1 * np.random.uniform(-1, 1)
        self.battery_SOC        = 0.60  + 0.20  * np.random.uniform(-1, 1)
        self.battery_voltage    = 2.5 + self.battery_SOC  * (3.1 - 2.5)

        self.buffer.clear()
        self.current_buffer.clear()

        self.SC_C               = 58
        self.R_esr              = 22e-3
        self.C_n_seri           = 7
        self.SC_current         = 0
        self.SC_voltage         = (np.random.uniform(9,16))  ############################################
        self.terminated         = False
        self.truncated          = False
        self.info               = {}
        self.step_count         = 0

        self.battery    = ECM_RC_Battery(Q_init=self.battery_capacity, alpha=self.fading_coefficient, R0=self.R0, R1=self.R1, C1=self.C1, dt=self.dt, Vmin=2.5, Vmax=4.0)
        self.sc         = Supercapacitor(C=self.SC_C, R_esr=self.R_esr, V_init=self.SC_voltage , dt=self.dt)

        self.state = np.array([self.input_current/np.max(self.i_dc_estimate), self.future_current/np.max(self.i_dc_estimate), self.battery_current/np.max(self.i_dc_estimate),
                            self.SC_current/np.max(self.i_dc_estimate), 1, self.battery_SOC, np.clip(self.SC_voltage/16,0.0,1.0)], dtype=np.float32) ############################################
        return self.state, {}

    def step(self, action):
        self.battery_current    = np.clip(action[0],np.min(self.i_dc_estimate),np.max(self.i_dc_estimate))
        self.SC_current         = np.clip(action[1],np.min(self.i_dc_estimate),np.max(self.i_dc_estimate))
        battery_current_cell    = action[0]/self.n_batt_par
        SC_current_cell         = action[1]/1 # We don not have parallel SCs in this case  
        
        V_bat, SOC, Q, V_RC = self.battery.step(battery_current_cell)
        V_SC, V_sc = self.sc.step(SC_current_cell)
        self.battery_SOC = SOC  
        self.SC_voltage = V_SC*self.C_n_seri 

        # state section
        self.input_current      = self.i_dc_estimate[self.step_count]   # Not randomely generated here
        self.future_current     = self.i_dc_future[self.step_count]     # Not randomely generated here

        # Append to history
        self.V_hist.append(V_bat*self.n_batt_seri)                          
        self.SOC_hist.append(SOC*100)                                       
        self.Q_hist.append(Q)                                                
        self.V_SC_hist.append(V_SC*self.C_n_seri)                           
        self.battery_I_hist.append(self.battery_current)                    
        self.SC_I_hist.append(self.SC_current )                                
        self.requested_I_hist.append(self.input_current)                    
        self.provided_I_hist.append(self.battery_current + self.SC_current) 

        

        self.state = np.array([self.input_current/np.max(self.i_dc_estimate), self.future_current/np.max(self.i_dc_estimate), self.battery_current/np.max(self.i_dc_estimate),
                            self.SC_current/np.max(self.i_dc_estimate), Q/self.battery_capacity, self.battery_SOC, np.clip(self.SC_voltage/16,0.0,1.0)], dtype=np.float32)
        
        self.truncated  = self.battery_SOC  <= 0.05 or self.SC_voltage <= 60 or self.SC_voltage >= 110
        self.terminated = self.step_count >= self.end_counter - 1
        
        
        # Define reward sections:
        
        self.buffer.append(self.battery_current)
        r_std = -abs(np.std(self.buffer))
        self.R_std.append(r_std)

        sum_current = self.battery_current + self.SC_current
        self.output_current.append(sum_current)

        r_current = -10*abs(sum_current-self.input_current)
        self.R_current.append(r_current)

        r_capacity = -500 * abs(self.battery_capacity - Q)
        self.R_capacity.append(r_capacity)

        self.current_buffer.append(self.battery_current)
        r_current_a = -abs(self.current_buffer[-1] - self.current_buffer[0]) if len(self.current_buffer) > 1 else 0
        self.R_current_a.append(r_current_a)

        r_distance = np.clip(20000*(1 / (1+ abs(self.end_counter - self.step_count))),0,10)
        self.R_distance.append(r_distance)

        reward_temp = r_current + r_capacity + r_current_a + r_std + r_distance

        reward = float(np.clip(reward_temp, -1e3, 1e4)) #if self.truncated == False else 100 * reward_temp
        self.reward.append(reward)

        # info section                                  #########################################
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
        "STD"                   : self.R_std,
        "r_current"             : self.R_current,
        "r_capacity"            : self.R_capacity,
        "r_current_a"           : self.R_current_a,
        "r_distance"            : self.R_distance,
        "reward"                : self.reward,
        }
        self.step_count += 1
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

action = np.zeros((len(i_dc_estimate), 2), dtype=np.float32)
action[:, 0] = filtered_i
action[:, 1] = i_dc_estimate - filtered_i

plt.figure(figsize=(10, 6))
plt.plot(i_dc_estimate, label='Original Signal', color='black')
plt.plot(action[:, 0], label='Battery Signal', color='blue')
plt.plot(action[:, 1], label='SC Signal', color='red')
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
battery    = ECM_RC_Battery()
capacitor = Supercapacitor(C=500, R_esr=0.01, V_init=16, dt=1.0)
SOC1=[]
SOC2=[]
for i in range(len(action)):
    V_bat, SOC, Q, V_RC = battery.step(action[i,0]/60)
    # V_out, V_SC = capacitor.step(action[i,1])
    print(f"\t Step {i+1}: \t SOC: {SOC}, \t Capacity: {Q}")
    SOC1.append(SOC)
    SOC2.append(Q)
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

plt.figure(figsize=(10, 6))
# plt.plot(SOC1           , label='SOC'            , color='gold')
plt.plot(SOC2           , label='Q'              , color='black')
plt.title('Low-pass Filtered Signal')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show()  


# Effect of ai on Q
fig, axes = plt.subplots(2, 1, figsize=(8, 6))  # 2 rows, 1 column

x = np.linspace(0, 10, 100)
axes[0].plot(action[:,0]/60)
axes[0].set_title("Current")

axes[1].plot(SOC2)
axes[1].set_title("Q")

plt.tight_layout()
plt.show()


################################ testing Algo ( testing and training ENV must be the same) ##################
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
action_mem =[]
reward = []
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
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    action_mem.append(action)
    if done or truncated or counter ==len(i_dc_estimate):
        info_hist_SAC_LSTM = info
        print("Done:", done, "Truncated:", truncated)
        break
env1.close()

# SAC
obs, info = env2.reset()
done = False
truncated = False
total_reward = 0.0
while not (done or truncated):
    action = algo_SAC.compute_single_action(obs, explore=False)  # Do NOT overwrite algo.config
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    if done or truncated or counter ==len(i_dc_estimate):
        info_hist_SAC = info
        break
env2.close()

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

# Low pass filter
for i in range(len(i_dc_estimate)):

    obs, reward, done, truncated, info = env1.step(action[i,:])
    action_mem.append(action[i,:])
    total_reward += reward
    counter += 1
    print(f"Step {counter}, \t Reward: {reward:.3f}")
    if done or truncated:
        info_hist_lowpass = info
        print("SOC :", info_hist_lowpass['battery_SOC'][-1], "\t Done:", done, "\t Truncated:", truncated, "SC Voltage:", info_hist_lowpass['SC_voltage'][-1])
        break

env.close()

plt.figure(figsize=(10, 6))
plt.plot(info_hist_SAC_LSTM['STD']                  , label='STD'           , color='red')
plt.plot(info_hist_SAC_LSTM['r_current']            , label='r_current'     , color='black')
plt.plot(info_hist_SAC_LSTM['r_capacity']           , label='r_capacity'    , color='blue')
plt.plot(info_hist_SAC_LSTM['r_current_a']          , label='r_current_a'   , color='green')
plt.plot(info_hist_SAC_LSTM['r_distance']           , label='r_distance'    , color='gold')
plt.title('Reward monitoring')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show() 

plt.figure(figsize=(10, 6))
plt.plot(info_hist_lowpass['STD']                  , label='STD'           , color='red')
plt.plot(info_hist_lowpass['r_current']            , label='r_current'     , color='black')
plt.plot(info_hist_lowpass['r_capacity']           , label='r_capacity'    , color='blue')
plt.plot(info_hist_lowpass['r_current_a']          , label='r_current_a'   , color='green')
plt.plot(info_hist_lowpass['r_distance']           , label='r_distance'    , color='gold')
plt.title('Reward monitoring')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show() 

plt.figure(figsize=(10, 6))
plt.plot(action_mem[:,0]                  , label='batt'           , color='red')
plt.plot(action_mem[:,1]                  , label='SC'           , color='blue')
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
