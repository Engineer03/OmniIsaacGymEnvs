# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math

import numpy as np
import torch
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.rotations import *
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.anymal import Anymal
from omniisaacgymenvs.robots.articulations.views.anymal_view import AnymalView
from omniisaacgymenvs.tasks.utils.anymal_terrain_generator import *
from omniisaacgymenvs.utils.terrain_utils.terrain_utils import *
from pxr import UsdLux, UsdPhysics

# import logging
# logging.basicConfig(level=logging.WARNING)

# from logging import (DEBUG, INFO, basicConfig, critical, debug, error, exception, info, warning)
# basicConfig(
#         level=DEBUG, format='[{levelname:.4}] : {message}', style='{')


class AnymalTerrainTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None: #1
        ##print("anymal_terrain.py: construct AnymalTerrainTask(RLTask)\n")
        # ##print("name ", name, "\n") #name  AnymalTerrain 
        # ##print("sim_config ", sim_config, "\n") #sim_config  <omniisaacgymenvs.utils.config_utils.sim_config.SimConfig object at 0x721540398e20> 
        # ##print("env ", env, "\n") #env  <VecEnvRLGames instance> 

        self.height_samples = None
        self.custom_origins = False
        self.init_done = False
        self._env_spacing = 0.0

        self._num_observations = 188
        self._num_actions = 12

        self.count = 0

        self.update_config(sim_config)

        RLTask.__init__(self, name, env)

        self.height_points = self.init_height_points()
        self.measured_heights = None
        # joint positions offsets
        self.default_dof_pos = torch.zeros(
            (self.num_envs, 12), dtype=torch.float, device=self.device, requires_grad=False
        )
        # reward episode sums
        torch_zeros = lambda: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {
            "lin_vel_xy": torch_zeros(),
            "lin_vel_z": torch_zeros(),
            "ang_vel_z": torch_zeros(),
            "ang_vel_xy": torch_zeros(),
            "orient": torch_zeros(),
            "torques": torch_zeros(),
            "joint_acc": torch_zeros(),
            "base_height": torch_zeros(),
            "air_time": torch_zeros(),
            "collision": torch_zeros(),
            "stumble": torch_zeros(),
            "action_rate": torch_zeros(),
            "hip": torch_zeros(),
        }
        return

    def update_config(self, sim_config): #2
        ##print("anymal_terrain.py: update_config\n")
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        # ##print("self._sim_config ", self._sim_config, "\n") #<omniisaacgymenvs.utils.config_utils.sim_config.SimConfig object at 0x721540398e20> 
        ###print("self._cfg ", self._cfg, "\n") #a set of config.yaml, AnymalTerrain.yaml, AnymalTerrainPPO.yaml in this order
        ###print("self._task_cfg ", self._task_cfg, "\n") #AnymalTerrain.yaml
        #self.horizon_length = self._cfg["params"]["config"]["horizon_length"]
        ###print("self.horizon_length ", self.horizon_length,"\n")
        #breakpoint()

        # normalization
        self.lin_vel_scale = self._task_cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self._task_cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self._task_cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self._task_cfg["env"]["learn"]["dofVelocityScale"]
        self.height_meas_scale = self._task_cfg["env"]["learn"]["heightMeasurementScale"]
        self.action_scale = self._task_cfg["env"]["control"]["actionScale"]

        # reward scales
        self.rew_scales = {}
        self.rew_scales["termination"] = self._task_cfg["env"]["learn"]["terminalReward"]
        self.rew_scales["lin_vel_xy"] = self._task_cfg["env"]["learn"]["linearVelocityXYRewardScale"]
        self.rew_scales["lin_vel_z"] = self._task_cfg["env"]["learn"]["linearVelocityZRewardScale"]
        self.rew_scales["ang_vel_z"] = self._task_cfg["env"]["learn"]["angularVelocityZRewardScale"]
        self.rew_scales["ang_vel_xy"] = self._task_cfg["env"]["learn"]["angularVelocityXYRewardScale"]
        self.rew_scales["orient"] = self._task_cfg["env"]["learn"]["orientationRewardScale"]
        self.rew_scales["torque"] = self._task_cfg["env"]["learn"]["torqueRewardScale"]
        self.rew_scales["joint_acc"] = self._task_cfg["env"]["learn"]["jointAccRewardScale"]
        self.rew_scales["base_height"] = self._task_cfg["env"]["learn"]["baseHeightRewardScale"]
        self.rew_scales["action_rate"] = self._task_cfg["env"]["learn"]["actionRateRewardScale"]
        self.rew_scales["hip"] = self._task_cfg["env"]["learn"]["hipRewardScale"]
        self.rew_scales["fallen_over"] = self._task_cfg["env"]["learn"]["fallenOverRewardScale"]

        # command ranges
        self.command_x_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["yaw"]

        # base init state
        pos = self._task_cfg["env"]["baseInitState"]["pos"]
        rot = self._task_cfg["env"]["baseInitState"]["rot"]
        v_lin = self._task_cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self._task_cfg["env"]["baseInitState"]["vAngular"]
        self.base_init_state = pos + rot + v_lin + v_ang
        #print("self.base_init_state ", self.base_init_state,"\n")
        #default: [0.0, 0.0, 0.62, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 

        # default joint positions
        self.named_default_joint_angles = self._task_cfg["env"]["defaultJointAngles"]

        # other
        self.decimation = self._task_cfg["env"]["control"]["decimation"]
        self.dt = self.decimation * self._task_cfg["sim"]["dt"]
        self.max_episode_length_s = self._task_cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.push_interval = int(self._task_cfg["env"]["learn"]["pushInterval_s"] / self.dt + 0.5)
        self.Kp = self._task_cfg["env"]["control"]["stiffness"]
        self.Kd = self._task_cfg["env"]["control"]["damping"]
        self.curriculum = self._task_cfg["env"]["terrain"]["curriculum"]
        self.base_threshold = 0.2
        self.knee_threshold = 0.1

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        self._num_envs = self._task_cfg["env"]["numEnvs"] #2048 #horizon_length: 24
        #better to input num_envs from terminal, not here
        #self._num_envs = 64 ##horizon_length: 64, minibatch_size: 4096
        ###print("self._num_envs ", self._num_envs,"\n")
        #minibatch_size: 16384
        #logging.debug("Check the number of environments")
        print("self._num_envs ", self._num_envs,"\n")
        #debug("DEBUG level")
        #breakpoint()

        self._task_cfg["sim"]["default_physics_material"]["static_friction"] = self._task_cfg["env"]["terrain"][
            "staticFriction"
        ]
        self._task_cfg["sim"]["default_physics_material"]["dynamic_friction"] = self._task_cfg["env"]["terrain"][
            "dynamicFriction"
        ]
        self._task_cfg["sim"]["default_physics_material"]["restitution"] = self._task_cfg["env"]["terrain"][
            "restitution"
        ]

        self._task_cfg["sim"]["add_ground_plane"] = False

    def _get_noise_scale_vec(self, cfg): #9
        ##print("anymal_terrain.py: _get_noise_scale_vec\n")
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self._task_cfg["env"]["learn"]["addNoise"]
        noise_level = self._task_cfg["env"]["learn"]["noiseLevel"]
        noise_vec[:3] = self._task_cfg["env"]["learn"]["linearVelocityNoise"] * noise_level * self.lin_vel_scale
        noise_vec[3:6] = self._task_cfg["env"]["learn"]["angularVelocityNoise"] * noise_level * self.ang_vel_scale
        noise_vec[6:9] = self._task_cfg["env"]["learn"]["gravityNoise"] * noise_level
        noise_vec[9:12] = 0.0  # commands
        noise_vec[12:24] = self._task_cfg["env"]["learn"]["dofPositionNoise"] * noise_level * self.dof_pos_scale
        noise_vec[24:36] = self._task_cfg["env"]["learn"]["dofVelocityNoise"] * noise_level * self.dof_vel_scale
        noise_vec[36:176] = (
            self._task_cfg["env"]["learn"]["heightMeasurementNoise"] * noise_level * self.height_meas_scale
        )
        noise_vec[176:188] = 0.0  # previous actions
        return noise_vec

    def init_height_points(self): #3 #leads to set_task in vec_env_rlgames.py
        ##print("anymal_terrain.py: init_height_points\n")
        # 1mx1.6m rectangle (without center line)
        y = 0.1 * torch.tensor(
            [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], device=self.device, requires_grad=False
        )  # 10-50cm on each side
        x = 0.1 * torch.tensor(
            [-8, -7, -6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 7, 8], device=self.device, requires_grad=False
        )  # 20-80cm on each side
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _create_trimesh(self, create_mesh=True): #6
        ##print("anymal_terrain.py: _create_trimesh\n")
        self.terrain = Terrain(self._task_cfg["env"]["terrain"], num_robots=self.num_envs)
        vertices = self.terrain.vertices
        triangles = self.terrain.triangles
        position = torch.tensor([-self.terrain.border_size, -self.terrain.border_size, 0.0])
        if create_mesh:
            add_terrain_to_stage(stage=self._stage, vertices=vertices, triangles=triangles, position=position)
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        )

    def set_up_scene(self, scene) -> None: #4
        ##print("anymal_terrain.py: set_up_scene\n")
        self._stage = get_current_stage()
        self.get_terrain()
        self.get_anymal()
        super().set_up_scene(scene, collision_filter_global_paths=["/World/terrain"])
        self._anymals = AnymalView(
            prim_paths_expr="/World/envs/.*/anymal", name="anymal_view", track_contact_forces=True
            #prim_paths_expr="/World/a1/.*/anymal", name="anymal_view", track_contact_forces=True
        )
        scene.add(self._anymals)
        scene.add(self._anymals._knees)
        scene.add(self._anymals._base)

    def initialize_views(self, scene): #leads to AnymalView
        #print("anymal_terrain.py: initialize_views\n")
        # initialize terrain variables even if we do not need to re-create the terrain mesh
        self.get_terrain(create_mesh=False)

        super().initialize_views(scene)
        if scene.object_exists("anymal_view"):
            scene.remove_object("anymal_view", registry_only=True)
        if scene.object_exists("knees_view"):
            scene.remove_object("knees_view", registry_only=True)
        if scene.object_exists("base_view"):
            scene.remove_object("base_view", registry_only=True)
        #print("leads to AnymalView")
        self._anymals = AnymalView( #leads to AnymalView
            prim_paths_expr="/World/envs/.*/anymal", name="anymal_view", track_contact_forces=True
            #prim_paths_expr="/World/a1/.*/anymal", name="anymal_view", track_contact_forces=True
            #prim_paths_expr="/World/a1/*", name="anymal_view", track_contact_forces=True
            #prim_paths_expr = ["/World/a1/base", "/World/a1/trunk", "/World/a1/FL_hip", "/World/a1/FR_hip", "/World/a1/RL_hip", "/World/a1/RR_hip"],  name="anymal_view", track_contact_forces=True
        )
        scene.add(self._anymals)
        scene.add(self._anymals._knees)
        scene.add(self._anymals._base)

        #print("scence ", scene)
        #breakpoint()

    def get_terrain(self, create_mesh=True): #5
        ##print("anymal_terrain.py: get_terrain\n")
        self.env_origins = torch.zeros((self.num_envs, 3), device=self.device, requires_grad=False)
        if not self.curriculum:
            self._task_cfg["env"]["terrain"]["maxInitMapLevel"] = self._task_cfg["env"]["terrain"]["numLevels"] - 1
        self.terrain_levels = torch.randint(
            0, self._task_cfg["env"]["terrain"]["maxInitMapLevel"] + 1, (self.num_envs,), device=self.device
        )
        self.terrain_types = torch.randint(
            0, self._task_cfg["env"]["terrain"]["numTerrains"], (self.num_envs,), device=self.device
        )
        self._create_trimesh(create_mesh=create_mesh) #6
        self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)

    def get_anymal(self): #7
        ##print("anymal_terrain.py: get_anymal\n")
        anymal_translation = torch.tensor([0.0, 0.0, 0.66])
        #anymal_translation = torch.tensor([0.0, 0.0, 0.0])
        anymal_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0])
        #print("self.default_zero_env_path ", self.default_zero_env_path, "\n") #/World/envs/env_0
        anymal = Anymal(
            prim_path=self.default_zero_env_path + "/anymal",
            name="anymal",
            #prim_path=self.default_zero_env_path + "/a1",
            #name="a1",
            #prim_path=self.default_zero_env_path,
            #name="anymal",
            translation=anymal_translation,
            orientation=anymal_orientation,
        )
        
        ###print("anymal ", anymal, "\n") #anymal  <omniisaacgymenvs.robots.articulations.anymal.Anymal object at 0x7215eb3fa860>

        self._sim_config.apply_articulation_settings(
            "anymal", get_prim_at_path(anymal.prim_path), self._sim_config.parse_actor_config("anymal")
            #"a1", get_prim_at_path(anymal.prim_path), self._sim_config.parse_actor_config("a1")
        )
        anymal.set_anymal_properties(self._stage, anymal.prim)
        anymal.prepare_contacts(self._stage, anymal.prim)

        self.dof_names = anymal.dof_names
        for i in range(self.num_actions):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

    #at the very first
    def post_reset(self): #8
        print("anymal_terrain.py: post_reset\n")
        #breakpoint()
        self.base_init_state = torch.tensor(
            self.base_init_state, dtype=torch.float, device=self.device, requires_grad=False
        )

        self.timeout_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # initialize some data used later on
        self.up_axis_idx = 2
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self._task_cfg) #9
        self.commands = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )  # x vel, y vel, yaw vel, heading
        #print("self.commands ", self.commands.shape, "\n") #(num_envs, 4)
        #breakpoint()

        self.commands_scale = torch.tensor(
            [self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale],
            device=self.device,
            requires_grad=False,
        )
        self.gravity_vec = torch.tensor(
            get_axis_params(-1.0, self.up_axis_idx), dtype=torch.float, device=self.device #10
        ).repeat((self.num_envs, 1))
        self.forward_vec = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.torques = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.feet_air_time = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros((self.num_envs, 12), dtype=torch.float, device=self.device, requires_grad=False)

        for i in range(self.num_envs):
            self.env_origins[i] = self.terrain_origins[self.terrain_levels[i], self.terrain_types[i]]
        self.num_dof = self._anymals.num_dof
        self.dof_pos = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
        self.dof_vel = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
        self.base_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.base_quat = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.base_velocities = torch.zeros((self.num_envs, 6), dtype=torch.float, device=self.device)

        self.knee_pos = torch.zeros((self.num_envs * 4, 3), dtype=torch.float, device=self.device)
        self.knee_quat = torch.zeros((self.num_envs * 4, 4), dtype=torch.float, device=self.device)

        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices) #11
        self.init_done = True

    #environment reset. After this, robots are moved to the initial location
    def reset_idx(self, env_ids): #11 and if in post_physics_step function (15)
        print("anymal_terrain.py: reset_idx\n")
        #print("env_ids ", env_ids, "\n") #This env_ids is used to identify which robot I am going to use
        #With one robot, env_ids: tensor([0], device='cuda:0') 
        #breakpoint()

        #Converts env_ids to int32 type.
        indices = env_ids.to(dtype=torch.int32)

        #Generates random offsets for positions and velocities for the degrees of freedom (DOFs) of the robots being reset.
        positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        #Resets the positions and velocities of the DOFs for the specified environments.
        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
        self.dof_vel[env_ids] = velocities

        self.update_terrain_level(env_ids)
        #Resets the base position
        self.base_pos[env_ids] = self.base_init_state[0:3]
        #Add random offsets to ensure variety
        #self.base_pos[env_ids, 0:3] += self.env_origins[env_ids] #comment out for inference
        self.base_pos[env_ids, 0:2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=self.device)
        #Resets the base orientation (quaternion) and velocities.
        self.base_quat[env_ids] = self.base_init_state[3:7]
        self.base_velocities[env_ids] = self.base_init_state[7:]

        #Sets the new positions, orientations, velocities, joint positions, and joint velocities in the simulation for the specified environments.
        self._anymals.set_world_poses(
            positions=self.base_pos[env_ids].clone(), orientations=self.base_quat[env_ids].clone(), indices=indices
        )
        self._anymals.set_velocities(velocities=self.base_velocities[env_ids].clone(), indices=indices)
        self._anymals.set_joint_positions(positions=self.dof_pos[env_ids].clone(), indices=indices)
        self._anymals.set_joint_velocities(velocities=self.dof_vel[env_ids].clone(), indices=indices)


        #print("self.commands[env_ids] ", self.commands[env_ids], "\n")
        #tensor([[-0.3703, -0.5490, -0.0423, -0.8895]], device='cuda:0')

        #Randomly generates commands for x, y, and yaw (rotation around the vertical axis).
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device
        ).squeeze()
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device
        ).squeeze()
        self.commands[env_ids, 3] = torch_rand_float(
            self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device
        ).squeeze()

        #Sets small commands to zero if the norm of x and y commands is less than 0.25 ensuring only significant commands are applied.
        self.commands[env_ids] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.25).unsqueeze(
            1
        ) 

        #print("self.commands[env_ids] ", self.commands[env_ids], "\n") #self.commands[env_ids, 2] (yaw vel) does not change.  x vel, y vel, yaw vel, heading
        #Because this function is from initial state and end state, meaning from one point to another and then reset. So not necessary to change yew vel throughout one episode
        #tensor([[-0.5366, -0.4517, -0.0423, -0.6778]], device='cuda:0')
        #breakpoint()

        #desired commands (self.commands[env_ids, 2] changes every time)
        self.commands[env_ids]=torch.tensor([[0.9000, 0.0, -0.1824, 0.9000]], device='cuda:0')

        #Resets the last actions, DOF velocities, feet air time, progress buffer, and reset buffer for the specified environments.
        # set small commands to zero
        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        # fill extras
        #Updates the extras dictionary with mean episode rewards and terrain levels for the environments being reset.
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())

    def update_terrain_level(self, env_ids): #12 #leads to construction of RLGTrainer in rlgames_train.py
        ##print("anymal_terrain.py: update_terrain_level\n")
        ##print("self.terrain_levels[env_ids] ", self.terrain_levels[env_ids].shape, "\n") #(num_envs)
        ##print("self.env_origins[env_ids] ", self.env_origins[env_ids].shape, "\n") #(num_envs, 3)
        #breakpoint()
        if not self.init_done or not self.curriculum:
            # do not change on initial reset
            return
        root_pos, _ = self._anymals.get_world_poses(clone=False)
        distance = torch.norm(root_pos[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        self.terrain_levels[env_ids] -= 1 * (
            distance < torch.norm(self.commands[env_ids, :2]) * self.max_episode_length_s * 0.25
        )
        self.terrain_levels[env_ids] += 1 * (distance > self.terrain.env_length / 2)
        self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], 0) % self.terrain.env_rows
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

        ##print("self.terrain_levels[env_ids] ", self.terrain_levels[env_ids], "\n")
        ##print("self.env_origins[env_ids] ", self.env_origins[env_ids], "\n")

    def refresh_dof_state_tensors(self): #14 #keeps repeating between pre_physics_step function and post_physics_step function
        ##print("anymal_terrain.py: refresh_dof_state_tensors\n")
        self.dof_pos = self._anymals.get_joint_positions(clone=False)
        ###print("self.dof_pos ", self.dof_pos.shape, self.dof_pos, "\n") #(num_envs, 12) self._num_actions = 12. The value keeps when this function is repeated
        self.dof_vel = self._anymals.get_joint_velocities(clone=False)
        ###print("self.dof_vel ", self.dof_vel.shape, self.dof_vel, "\n") #(num_envs, 12) self._num_actions = 12. The value keeps when this function is repeated

    def refresh_body_state_tensors(self): #16
        ##print("anymal_terrain.py: refresh_body_state_tensors\n")
        self.base_pos, self.base_quat = self._anymals.get_world_poses(clone=False)
        self.base_velocities = self._anymals.get_velocities(clone=False)
        self.knee_pos, self.knee_quat = self._anymals._knees.get_world_poses(clone=False)

    #(num_envs, 12) self._num_actions = 12
    #prepare and apply actions before the physics simulation steps forward. 
    def pre_physics_step(self, actions): #13 #comes from step function in vec_env_rlgames.py
        #print("anymal_terrain.py: pre_physics_step\n")

        if not self.world.is_playing():
            return

        self.actions = actions.clone().to(self.device) #clone, exactly the same as actions. Cloning ensures the original actions tensor remains unchanged.
        ###print("self.actions ", self.actions.shape, "\n")
        for i in range(self.decimation): #decimation: 4,  Number of control action updates @ sim DT per policy DT
            if self.world.is_playing():
                torques = torch.clip(
                    self.Kp * (self.action_scale * self.actions + self.default_dof_pos - self.dof_pos)
                    - self.Kd * self.dof_vel,
                    -80.0,
                    80.0,
                )
                self._anymals.set_joint_efforts(torques) # The computed torques are applied to the robot's joints.
                self.torques = torques 
                SimulationContext.step(self.world, render=False) #The simulation steps forward by one step.
                self.refresh_dof_state_tensors() #14 #The DOF state tensors are refreshed to update the current positions and velocities.

    #Actually apply the torques to the robot and get the observation
    #process data after the physics simulation step has been completed. 
    #This function updates various state variables, computes necessary metrics, and prepares observations for the next step
    def post_physics_step(self): #15
        #print("anymal_terrain.py: post_physics_step\n")
        self.progress_buf[:] += 1 #Increment the progress buffer for each environment to track the number of steps taken.

        if self.world.is_playing():

            #Update the degrees of freedom (DOF) and body state tensors to reflect the latest state after the physics step.
            self.refresh_dof_state_tensors() #14
            self.refresh_body_state_tensors() #16

            self.common_step_counter += 1
            #If the counter reaches the push interval, apply a push to the robots to introduce disturbances or variability in the environment.
            if self.common_step_counter % self.push_interval == 0: 
                self.push_robots()

            # prepare quantities for state update
            self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.base_velocities[:, 0:3])
            self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.base_velocities[:, 3:6])
            self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0) #17

            self.check_termination() #18
            self.get_states()
            self.calculate_metrics() #19

            env_ids = self.reset_buf.nonzero(as_tuple=False).flatten() #Identify environments that need to be reset based on the reset buffer.
            if len(env_ids) > 0: 
                self.reset_idx(env_ids) 

            self.get_observations() #20
            if self.add_noise:
                self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

            self.last_actions[:] = self.actions[:]
            self.last_dof_vel[:] = self.dof_vel[:]

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def push_robots(self): #if in post_physics_step function
        ##print("anymal_terrain.py: push_robots\n")
        self.base_velocities[:, 0:2] = torch_rand_float(
            -1.0, 1.0, (self.num_envs, 2), device=self.device
        )  # lin vel x/y
        ##print("self.base_velocities[:, 0:2] ", self.base_velocities[:, 0:2], "\n")
        self._anymals.set_velocities(self.base_velocities)
        ##print("self.base_velocities[:, 0:2] ", self.base_velocities[:, 0:2], "\n")

    def check_termination(self): #18
        ##print("anymal_terrain.py: check_termination\n")
        self.timeout_buf = torch.where(
            self.progress_buf >= self.max_episode_length - 1,
            torch.ones_like(self.timeout_buf),
            torch.zeros_like(self.timeout_buf),
        )
        knee_contact = (
            torch.norm(self._anymals._knees.get_net_contact_forces(clone=False).view(self._num_envs, 4, 3), dim=-1)
            > 1.0
        )
        self.has_fallen = (torch.norm(self._anymals._base.get_net_contact_forces(clone=False), dim=1) > 1.0) | (
            torch.sum(knee_contact, dim=-1) > 1.0
        )
        self.reset_buf = self.has_fallen.clone()
        self.reset_buf = torch.where(self.timeout_buf.bool(), torch.ones_like(self.reset_buf), self.reset_buf)

    def calculate_metrics(self): #19
        ##print("anymal_terrain.py: calculate_metrics\n")
        ###print("self.commands[:, :2] ", self.commands[:, :2].shape, "\n") #(num_envs, 2)
        # velocity tracking reward
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        rew_lin_vel_xy = torch.exp(-lin_vel_error / 0.25) * self.rew_scales["lin_vel_xy"]
        rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25) * self.rew_scales["ang_vel_z"]

        ###print("self.base_ang_vel[:, :2] ", self.base_ang_vel[:, :2].shape, "\n") #(num_envs, 2)
        # other base velocity penalties
        rew_lin_vel_z = torch.square(self.base_lin_vel[:, 2]) * self.rew_scales["lin_vel_z"]
        rew_ang_vel_xy = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * self.rew_scales["ang_vel_xy"]

        ###print("self.projected_gravity[:, :2] ", self.projected_gravity[:, :2].shape, "\n") #(num_envs, 2)
        # orientation penalty
        rew_orient = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * self.rew_scales["orient"]

        ###print("self.base_pos[:, 2] ", self.base_pos[:, 2].shape, "\n") #(num_envs)
        # base height penalty
        rew_base_height = torch.square(self.base_pos[:, 2] - 0.52) * self.rew_scales["base_height"]

        # torque penalty
        rew_torque = torch.sum(torch.square(self.torques), dim=1) * self.rew_scales["torque"]

        # joint acc penalty
        rew_joint_acc = torch.sum(torch.square(self.last_dof_vel - self.dof_vel), dim=1) * self.rew_scales["joint_acc"]

        # fallen over penalty
        rew_fallen_over = self.has_fallen * self.rew_scales["fallen_over"]

        # action rate penalty
        rew_action_rate = (
            torch.sum(torch.square(self.last_actions - self.actions), dim=1) * self.rew_scales["action_rate"]
        )

        # cosmetic penalty for hip motion
        rew_hip = (
            torch.sum(torch.abs(self.dof_pos[:, 0:4] - self.default_dof_pos[:, 0:4]), dim=1) * self.rew_scales["hip"]
        )

        # total reward
        self.rew_buf = (
            rew_lin_vel_xy
            + rew_ang_vel_z
            + rew_lin_vel_z
            + rew_ang_vel_xy
            + rew_orient
            + rew_base_height
            + rew_torque
            + rew_joint_acc
            + rew_action_rate
            + rew_hip
            + rew_fallen_over
        )
        ###print("self.rew_buf ", self.rew_buf, "\n")
        self.rew_buf = torch.clip(self.rew_buf, min=0.0, max=None) #convert minus values into 0.0
        ###print("self.rew_buf ", self.rew_buf.shape, "\n") #(num_envs)

        # add termination reward
        self.rew_buf += self.rew_scales["termination"] * self.reset_buf * ~self.timeout_buf

        # log episode reward sums
        self.episode_sums["lin_vel_xy"] += rew_lin_vel_xy
        self.episode_sums["ang_vel_z"] += rew_ang_vel_z
        self.episode_sums["lin_vel_z"] += rew_lin_vel_z
        self.episode_sums["ang_vel_xy"] += rew_ang_vel_xy
        self.episode_sums["orient"] += rew_orient
        self.episode_sums["torques"] += rew_torque
        self.episode_sums["joint_acc"] += rew_joint_acc
        self.episode_sums["action_rate"] += rew_action_rate
        self.episode_sums["base_height"] += rew_base_height
        self.episode_sums["hip"] += rew_hip

        #breakpoint()

    def get_observations(self): #20 #leads to _process_data function in vec_env_rlgames.py
        #print("anymal_terrain.py: get_observations\n")
        self.measured_heights = self.get_heights() #21
        ###print("self.measured_heights ", self.measured_heights.shape, "\n") #(num_envs, 140)
        heights = (
            torch.clip(self.base_pos[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.0) * self.height_meas_scale
        )

        # ##print("self.base_lin_vel ", self.base_lin_vel.shape, "\n") #(num_envs, 3)
        # ##print("self.base_ang_vel ", self.base_ang_vel.shape, "\n") #(num_envs, 3)
        # ##print("self.projected_gravity ", self.projected_gravity.shape, "\n") #(num_envs, 3)
        # ##print("self.commands[:, :3] ", self.commands[:, :3].shape, "\n") #(num_envs, 3)
        # ##print("self.dof_pos ", self.dof_pos.shape, "\n") #(num_envs, 12) 12 = 4 * 3
        # ##print("self.dof_vel ", self.dof_vel.shape, "\n") #(num_envs, 12)
        # ##print("heights ", heights.shape, "\n") #(num_envs, 140)
        # ##print("self.actions ", self.actions.shape, "\n") #(num_envs, 12) 12 = 3 * 4

        ##print("self.commands[:, :3] ", self.commands[:, :3], "\n") #(num_envs, 4)
        ##print("self.actions ", self.actions, "\n")
        ##print("self.commands ", self.commands.shape, "\n")
        #breakpoint()

        self.obs_buf = torch.cat(
            (
                self.base_lin_vel * self.lin_vel_scale, #(num_envs, 3)
                self.base_ang_vel * self.ang_vel_scale, #(num_envs, 3)
                self.projected_gravity, #(num_envs, 3) 
                #For inference (1, 188), scaled commmands are located in 10th, 11th and 12th
                self.commands[:, :3] * self.commands_scale, #(num_envs, 3) #x vel, y vel, yaw vel
                self.dof_pos * self.dof_pos_scale, #(num_envs, 12)
                self.dof_vel * self.dof_vel_scale, #(num_envs, 12)
                heights, #(num_envs, 140)
                self.actions, #(num_envs, 12)
            ),
            dim=-1,
        )

        #print("self.obs_buf  ", self.obs_buf, "\n") #(num_envs, 188) #(1, 188) for inference
        #print("self.commands ", self.commands) #tensor([[-0.3703, -0.5490, -0.1824, -0.8895]], device='cuda:0')
        #print("scaled commands ", self.commands[:, :3] * self.commands_scale) #tensor([[-0.7406, -1.0981, -0.0486]], device='cuda:0')

    def get_ground_heights_below_knees(self):
        ##print("anymal_terrain.py: get_ground_heights_below_knees\n")
        points = self.knee_pos.reshape(self.num_envs, 4, 3)
        points += self.terrain.border_size
        points = (points / self.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py + 1]
        heights = torch.min(heights1, heights2)
        breakpoint()
        return heights.view(self.num_envs, -1) * self.terrain.vertical_scale
    

    def get_ground_heights_below_base(self):
        ##print("anymal_terrain.py: get_ground_heights_below_base\n")
        points = self.base_pos.reshape(self.num_envs, 1, 3)
        points += self.terrain.border_size
        points = (points / self.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py + 1]
        heights = torch.min(heights1, heights2)
        breakpoint()
        return heights.view(self.num_envs, -1) * self.terrain.vertical_scale
    

    def get_heights(self, env_ids=None): #21
        ##print("anymal_terrain.py: get_heights\n")
        if env_ids:
            points = quat_apply_yaw( #22
                self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]
            ) + (self.base_pos[env_ids, 0:3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (
                self.base_pos[:, 0:3]
            ).unsqueeze(1)

        points += self.terrain.border_size
        points = (points / self.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]

        heights2 = self.height_samples[px + 1, py + 1]
        heights = torch.min(heights1, heights2)

        return heights.view(self.num_envs, -1) * self.terrain.vertical_scale


@torch.jit.script
def quat_apply_yaw(quat, vec): #22
    ##print("anymal_terrain.py: quat_apply_yaw\n")
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, 1:3] = 0.0
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


@torch.jit.script
def wrap_to_pi(angles): #17
    ##print("anymal_terrain.py: wrap_to_pi\n")
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


def get_axis_params(value, axis_idx, x_value=0.0, dtype=float, n_dims=3): #10
    ##print("anymal_terrain.py: get_axis_params\n")
    """construct arguments to `Vec` according to axis index."""
    zs = np.zeros((n_dims,))
    assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
    zs[axis_idx] = 1.0
    params = np.where(zs == 1.0, value, zs)
    params[0] = x_value
    return list(params.astype(dtype))




# import math

# import numpy as np
# import torch
# from omni.isaac.core.simulation_context import SimulationContext
# from omni.isaac.core.utils.prims import get_prim_at_path
# from omni.isaac.core.utils.stage import get_current_stage
# from omni.isaac.core.utils.torch.rotations import *
# from omniisaacgymenvs.tasks.base.rl_task import RLTask
# from omniisaacgymenvs.robots.articulations.anymal import Anymal
# from omniisaacgymenvs.robots.articulations.views.anymal_view import AnymalView
# from omniisaacgymenvs.tasks.utils.anymal_terrain_generator import *
# from omniisaacgymenvs.utils.terrain_utils.terrain_utils import *
# from pxr import UsdLux, UsdPhysics


# class AnymalTerrainTask(RLTask):
#     def __init__(self, name, sim_config, env, offset=None) -> None:

#         self.height_samples = None
#         self.custom_origins = False
#         self.init_done = False
#         self._env_spacing = 0.0

#         self._num_observations = 188
#         self._num_actions = 12

#         self.update_config(sim_config)

#         RLTask.__init__(self, name, env)

#         self.height_points = self.init_height_points()
#         self.measured_heights = None
#         # joint positions offsets
#         self.default_dof_pos = torch.zeros(
#             (self.num_envs, 12), dtype=torch.float, device=self.device, requires_grad=False
#         )
#         # reward episode sums
#         torch_zeros = lambda: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
#         self.episode_sums = {
#             "lin_vel_xy": torch_zeros(),
#             "lin_vel_z": torch_zeros(),
#             "ang_vel_z": torch_zeros(),
#             "ang_vel_xy": torch_zeros(),
#             "orient": torch_zeros(),
#             "torques": torch_zeros(),
#             "joint_acc": torch_zeros(),
#             "base_height": torch_zeros(),
#             "air_time": torch_zeros(),
#             "collision": torch_zeros(),
#             "stumble": torch_zeros(),
#             "action_rate": torch_zeros(),
#             "hip": torch_zeros(),
#         }
#         return

#     def update_config(self, sim_config):
#         self._sim_config = sim_config
#         self._cfg = sim_config.config
#         self._task_cfg = sim_config.task_config

#         # normalization
#         self.lin_vel_scale = self._task_cfg["env"]["learn"]["linearVelocityScale"]
#         self.ang_vel_scale = self._task_cfg["env"]["learn"]["angularVelocityScale"]
#         self.dof_pos_scale = self._task_cfg["env"]["learn"]["dofPositionScale"]
#         self.dof_vel_scale = self._task_cfg["env"]["learn"]["dofVelocityScale"]
#         self.height_meas_scale = self._task_cfg["env"]["learn"]["heightMeasurementScale"]
#         self.action_scale = self._task_cfg["env"]["control"]["actionScale"]

#         # reward scales
#         self.rew_scales = {}
#         self.rew_scales["termination"] = self._task_cfg["env"]["learn"]["terminalReward"]
#         self.rew_scales["lin_vel_xy"] = self._task_cfg["env"]["learn"]["linearVelocityXYRewardScale"]
#         self.rew_scales["lin_vel_z"] = self._task_cfg["env"]["learn"]["linearVelocityZRewardScale"]
#         self.rew_scales["ang_vel_z"] = self._task_cfg["env"]["learn"]["angularVelocityZRewardScale"]
#         self.rew_scales["ang_vel_xy"] = self._task_cfg["env"]["learn"]["angularVelocityXYRewardScale"]
#         self.rew_scales["orient"] = self._task_cfg["env"]["learn"]["orientationRewardScale"]
#         self.rew_scales["torque"] = self._task_cfg["env"]["learn"]["torqueRewardScale"]
#         self.rew_scales["joint_acc"] = self._task_cfg["env"]["learn"]["jointAccRewardScale"]
#         self.rew_scales["base_height"] = self._task_cfg["env"]["learn"]["baseHeightRewardScale"]
#         self.rew_scales["action_rate"] = self._task_cfg["env"]["learn"]["actionRateRewardScale"]
#         self.rew_scales["hip"] = self._task_cfg["env"]["learn"]["hipRewardScale"]
#         self.rew_scales["fallen_over"] = self._task_cfg["env"]["learn"]["fallenOverRewardScale"]

#         # command ranges
#         self.command_x_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
#         self.command_y_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
#         self.command_yaw_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["yaw"]

#         # base init state
#         pos = self._task_cfg["env"]["baseInitState"]["pos"]
#         rot = self._task_cfg["env"]["baseInitState"]["rot"]
#         v_lin = self._task_cfg["env"]["baseInitState"]["vLinear"]
#         v_ang = self._task_cfg["env"]["baseInitState"]["vAngular"]
#         self.base_init_state = pos + rot + v_lin + v_ang

#         # default joint positions
#         self.named_default_joint_angles = self._task_cfg["env"]["defaultJointAngles"]

#         # other
#         self.decimation = self._task_cfg["env"]["control"]["decimation"]
#         self.dt = self.decimation * self._task_cfg["sim"]["dt"]
#         self.max_episode_length_s = self._task_cfg["env"]["learn"]["episodeLength_s"]
#         self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
#         self.push_interval = int(self._task_cfg["env"]["learn"]["pushInterval_s"] / self.dt + 0.5)
#         self.Kp = self._task_cfg["env"]["control"]["stiffness"]
#         self.Kd = self._task_cfg["env"]["control"]["damping"]
#         self.curriculum = self._task_cfg["env"]["terrain"]["curriculum"]
#         self.base_threshold = 0.2
#         self.knee_threshold = 0.1

#         for key in self.rew_scales.keys():
#             self.rew_scales[key] *= self.dt

#         self._num_envs = self._task_cfg["env"]["numEnvs"]

#         self._task_cfg["sim"]["default_physics_material"]["static_friction"] = self._task_cfg["env"]["terrain"][
#             "staticFriction"
#         ]
#         self._task_cfg["sim"]["default_physics_material"]["dynamic_friction"] = self._task_cfg["env"]["terrain"][
#             "dynamicFriction"
#         ]
#         self._task_cfg["sim"]["default_physics_material"]["restitution"] = self._task_cfg["env"]["terrain"][
#             "restitution"
#         ]

#         self._task_cfg["sim"]["add_ground_plane"] = False

#     def _get_noise_scale_vec(self, cfg):
#         noise_vec = torch.zeros_like(self.obs_buf[0])
#         self.add_noise = self._task_cfg["env"]["learn"]["addNoise"]
#         noise_level = self._task_cfg["env"]["learn"]["noiseLevel"]
#         noise_vec[:3] = self._task_cfg["env"]["learn"]["linearVelocityNoise"] * noise_level * self.lin_vel_scale
#         noise_vec[3:6] = self._task_cfg["env"]["learn"]["angularVelocityNoise"] * noise_level * self.ang_vel_scale
#         noise_vec[6:9] = self._task_cfg["env"]["learn"]["gravityNoise"] * noise_level
#         noise_vec[9:12] = 0.0  # commands
#         noise_vec[12:24] = self._task_cfg["env"]["learn"]["dofPositionNoise"] * noise_level * self.dof_pos_scale
#         noise_vec[24:36] = self._task_cfg["env"]["learn"]["dofVelocityNoise"] * noise_level * self.dof_vel_scale
#         noise_vec[36:176] = (
#             self._task_cfg["env"]["learn"]["heightMeasurementNoise"] * noise_level * self.height_meas_scale
#         )
#         noise_vec[176:188] = 0.0  # previous actions
#         return noise_vec

#     def init_height_points(self):
#         # 1mx1.6m rectangle (without center line)
#         y = 0.1 * torch.tensor(
#             [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], device=self.device, requires_grad=False
#         )  # 10-50cm on each side
#         x = 0.1 * torch.tensor(
#             [-8, -7, -6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 7, 8], device=self.device, requires_grad=False
#         )  # 20-80cm on each side
#         grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

#         self.num_height_points = grid_x.numel()
#         points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
#         points[:, :, 0] = grid_x.flatten()
#         points[:, :, 1] = grid_y.flatten()
#         return points

#     def _create_trimesh(self, create_mesh=True):
#         self.terrain = Terrain(self._task_cfg["env"]["terrain"], num_robots=self.num_envs)
#         vertices = self.terrain.vertices
#         triangles = self.terrain.triangles
#         position = torch.tensor([-self.terrain.border_size, -self.terrain.border_size, 0.0])
#         if create_mesh:
#             add_terrain_to_stage(stage=self._stage, vertices=vertices, triangles=triangles, position=position)
#         self.height_samples = (
#             torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
#         )

#     def set_up_scene(self, scene) -> None:
#         self._stage = get_current_stage()
#         self.get_terrain()
#         self.get_anymal()
#         super().set_up_scene(scene, collision_filter_global_paths=["/World/terrain"])
#         self._anymals = AnymalView(
#             prim_paths_expr="/World/envs/.*/anymal", name="anymal_view", track_contact_forces=True
#         )
#         scene.add(self._anymals)
#         scene.add(self._anymals._knees)
#         scene.add(self._anymals._base)

#     def initialize_views(self, scene):
#         # initialize terrain variables even if we do not need to re-create the terrain mesh
#         self.get_terrain(create_mesh=False)

#         super().initialize_views(scene)
#         if scene.object_exists("anymal_view"):
#             scene.remove_object("anymal_view", registry_only=True)
#         if scene.object_exists("knees_view"):
#             scene.remove_object("knees_view", registry_only=True)
#         if scene.object_exists("base_view"):
#             scene.remove_object("base_view", registry_only=True)
#         self._anymals = AnymalView(
#             prim_paths_expr="/World/envs/.*/anymal", name="anymal_view", track_contact_forces=True
#         )
#         scene.add(self._anymals)
#         scene.add(self._anymals._knees)
#         scene.add(self._anymals._base)

#     def get_terrain(self, create_mesh=True):
#         self.env_origins = torch.zeros((self.num_envs, 3), device=self.device, requires_grad=False)
#         if not self.curriculum:
#             self._task_cfg["env"]["terrain"]["maxInitMapLevel"] = self._task_cfg["env"]["terrain"]["numLevels"] - 1
#         self.terrain_levels = torch.randint(
#             0, self._task_cfg["env"]["terrain"]["maxInitMapLevel"] + 1, (self.num_envs,), device=self.device
#         )
#         self.terrain_types = torch.randint(
#             0, self._task_cfg["env"]["terrain"]["numTerrains"], (self.num_envs,), device=self.device
#         )
#         self._create_trimesh(create_mesh=create_mesh)
#         self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)

#     def get_anymal(self):
#         anymal_translation = torch.tensor([0.0, 0.0, 0.66])
#         anymal_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0])
#         anymal = Anymal(
#             prim_path=self.default_zero_env_path + "/anymal",
#             name="anymal",
#             translation=anymal_translation,
#             orientation=anymal_orientation,
#         )
#         self._sim_config.apply_articulation_settings(
#             "anymal", get_prim_at_path(anymal.prim_path), self._sim_config.parse_actor_config("anymal")
#         )
#         anymal.set_anymal_properties(self._stage, anymal.prim)
#         anymal.prepare_contacts(self._stage, anymal.prim)

#         self.dof_names = anymal.dof_names
#         for i in range(self.num_actions):
#             name = self.dof_names[i]
#             angle = self.named_default_joint_angles[name]
#             self.default_dof_pos[:, i] = angle

#     def post_reset(self):
#         self.base_init_state = torch.tensor(
#             self.base_init_state, dtype=torch.float, device=self.device, requires_grad=False
#         )

#         self.timeout_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

#         # initialize some data used later on
#         self.up_axis_idx = 2
#         self.common_step_counter = 0
#         self.extras = {}
#         self.noise_scale_vec = self._get_noise_scale_vec(self._task_cfg)
#         self.commands = torch.zeros(
#             self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
#         )  # x vel, y vel, yaw vel, heading
#         self.commands_scale = torch.tensor(
#             [self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale],
#             device=self.device,
#             requires_grad=False,
#         )
#         self.gravity_vec = torch.tensor(
#             get_axis_params(-1.0, self.up_axis_idx), dtype=torch.float, device=self.device
#         ).repeat((self.num_envs, 1))
#         self.forward_vec = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float, device=self.device).repeat(
#             (self.num_envs, 1)
#         )
#         self.torques = torch.zeros(
#             self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
#         )
#         self.actions = torch.zeros(
#             self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
#         )
#         self.last_actions = torch.zeros(
#             self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
#         )
#         self.feet_air_time = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
#         self.last_dof_vel = torch.zeros((self.num_envs, 12), dtype=torch.float, device=self.device, requires_grad=False)

#         for i in range(self.num_envs):
#             self.env_origins[i] = self.terrain_origins[self.terrain_levels[i], self.terrain_types[i]]
#         self.num_dof = self._anymals.num_dof
#         self.dof_pos = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
#         self.dof_vel = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
#         self.base_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
#         self.base_quat = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
#         self.base_velocities = torch.zeros((self.num_envs, 6), dtype=torch.float, device=self.device)

#         self.knee_pos = torch.zeros((self.num_envs * 4, 3), dtype=torch.float, device=self.device)
#         self.knee_quat = torch.zeros((self.num_envs * 4, 4), dtype=torch.float, device=self.device)

#         indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
#         self.reset_idx(indices)
#         self.init_done = True

#     def reset_idx(self, env_ids):
#         indices = env_ids.to(dtype=torch.int32)

#         positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
#         velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

#         self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
#         self.dof_vel[env_ids] = velocities

#         self.update_terrain_level(env_ids)
#         self.base_pos[env_ids] = self.base_init_state[0:3]
#         self.base_pos[env_ids, 0:3] += self.env_origins[env_ids]
#         self.base_pos[env_ids, 0:2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=self.device)
#         self.base_quat[env_ids] = self.base_init_state[3:7]
#         self.base_velocities[env_ids] = self.base_init_state[7:]

#         self._anymals.set_world_poses(
#             positions=self.base_pos[env_ids].clone(), orientations=self.base_quat[env_ids].clone(), indices=indices
#         )
#         self._anymals.set_velocities(velocities=self.base_velocities[env_ids].clone(), indices=indices)
#         self._anymals.set_joint_positions(positions=self.dof_pos[env_ids].clone(), indices=indices)
#         self._anymals.set_joint_velocities(velocities=self.dof_vel[env_ids].clone(), indices=indices)

#         self.commands[env_ids, 0] = torch_rand_float(
#             self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device
#         ).squeeze()
#         self.commands[env_ids, 1] = torch_rand_float(
#             self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device
#         ).squeeze()
#         self.commands[env_ids, 3] = torch_rand_float(
#             self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device
#         ).squeeze()
#         self.commands[env_ids] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.25).unsqueeze(
#             1
#         )  # set small commands to zero

#         self.last_actions[env_ids] = 0.0
#         self.last_dof_vel[env_ids] = 0.0
#         self.feet_air_time[env_ids] = 0.0
#         self.progress_buf[env_ids] = 0
#         self.reset_buf[env_ids] = 1

#         # fill extras
#         self.extras["episode"] = {}
#         for key in self.episode_sums.keys():
#             self.extras["episode"]["rew_" + key] = (
#                 torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
#             )
#             self.episode_sums[key][env_ids] = 0.0
#         self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())

#     def update_terrain_level(self, env_ids):
#         if not self.init_done or not self.curriculum:
#             # do not change on initial reset
#             return
#         root_pos, _ = self._anymals.get_world_poses(clone=False)
#         distance = torch.norm(root_pos[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
#         self.terrain_levels[env_ids] -= 1 * (
#             distance < torch.norm(self.commands[env_ids, :2]) * self.max_episode_length_s * 0.25
#         )
#         self.terrain_levels[env_ids] += 1 * (distance > self.terrain.env_length / 2)
#         self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], 0) % self.terrain.env_rows
#         self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

#     def refresh_dof_state_tensors(self):
#         self.dof_pos = self._anymals.get_joint_positions(clone=False)
#         self.dof_vel = self._anymals.get_joint_velocities(clone=False)

#     def refresh_body_state_tensors(self):
#         self.base_pos, self.base_quat = self._anymals.get_world_poses(clone=False)
#         self.base_velocities = self._anymals.get_velocities(clone=False)
#         self.knee_pos, self.knee_quat = self._anymals._knees.get_world_poses(clone=False)

#     def pre_physics_step(self, actions):
#         if not self.world.is_playing():
#             return

#         self.actions = actions.clone().to(self.device)
#         for i in range(self.decimation):
#             if self.world.is_playing():
#                 torques = torch.clip(
#                     self.Kp * (self.action_scale * self.actions + self.default_dof_pos - self.dof_pos)
#                     - self.Kd * self.dof_vel,
#                     -80.0,
#                     80.0,
#                 )
#                 self._anymals.set_joint_efforts(torques)
#                 self.torques = torques
#                 SimulationContext.step(self.world, render=False)
#                 self.refresh_dof_state_tensors()

#     def post_physics_step(self):
#         self.progress_buf[:] += 1

#         if self.world.is_playing():

#             self.refresh_dof_state_tensors()
#             self.refresh_body_state_tensors()

#             self.common_step_counter += 1
#             if self.common_step_counter % self.push_interval == 0:
#                 self.push_robots()

#             # prepare quantities
#             self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.base_velocities[:, 0:3])
#             self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.base_velocities[:, 3:6])
#             self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
#             forward = quat_apply(self.base_quat, self.forward_vec)
#             heading = torch.atan2(forward[:, 1], forward[:, 0])
#             self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0)

#             self.check_termination()
#             self.get_states()
#             self.calculate_metrics()

#             env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
#             if len(env_ids) > 0:
#                 self.reset_idx(env_ids)

#             self.get_observations()
#             if self.add_noise:
#                 self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

#             self.last_actions[:] = self.actions[:]
#             self.last_dof_vel[:] = self.dof_vel[:]

#         return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

#     def push_robots(self):
#         self.base_velocities[:, 0:2] = torch_rand_float(
#             -1.0, 1.0, (self.num_envs, 2), device=self.device
#         )  # lin vel x/y
#         self._anymals.set_velocities(self.base_velocities)

#     def check_termination(self):
#         self.timeout_buf = torch.where(
#             self.progress_buf >= self.max_episode_length - 1,
#             torch.ones_like(self.timeout_buf),
#             torch.zeros_like(self.timeout_buf),
#         )
#         knee_contact = (
#             torch.norm(self._anymals._knees.get_net_contact_forces(clone=False).view(self._num_envs, 4, 3), dim=-1)
#             > 1.0
#         )
#         self.has_fallen = (torch.norm(self._anymals._base.get_net_contact_forces(clone=False), dim=1) > 1.0) | (
#             torch.sum(knee_contact, dim=-1) > 1.0
#         )
#         self.reset_buf = self.has_fallen.clone()
#         self.reset_buf = torch.where(self.timeout_buf.bool(), torch.ones_like(self.reset_buf), self.reset_buf)

#     def calculate_metrics(self):
#         # velocity tracking reward
#         lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
#         ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
#         rew_lin_vel_xy = torch.exp(-lin_vel_error / 0.25) * self.rew_scales["lin_vel_xy"]
#         rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25) * self.rew_scales["ang_vel_z"]

#         # other base velocity penalties
#         rew_lin_vel_z = torch.square(self.base_lin_vel[:, 2]) * self.rew_scales["lin_vel_z"]
#         rew_ang_vel_xy = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * self.rew_scales["ang_vel_xy"]

#         # orientation penalty
#         rew_orient = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * self.rew_scales["orient"]

#         # base height penalty
#         rew_base_height = torch.square(self.base_pos[:, 2] - 0.52) * self.rew_scales["base_height"]

#         # torque penalty
#         rew_torque = torch.sum(torch.square(self.torques), dim=1) * self.rew_scales["torque"]

#         # joint acc penalty
#         rew_joint_acc = torch.sum(torch.square(self.last_dof_vel - self.dof_vel), dim=1) * self.rew_scales["joint_acc"]

#         # fallen over penalty
#         rew_fallen_over = self.has_fallen * self.rew_scales["fallen_over"]

#         # action rate penalty
#         rew_action_rate = (
#             torch.sum(torch.square(self.last_actions - self.actions), dim=1) * self.rew_scales["action_rate"]
#         )

#         # cosmetic penalty for hip motion
#         rew_hip = (
#             torch.sum(torch.abs(self.dof_pos[:, 0:4] - self.default_dof_pos[:, 0:4]), dim=1) * self.rew_scales["hip"]
#         )

#         # total reward
#         self.rew_buf = (
#             rew_lin_vel_xy
#             + rew_ang_vel_z
#             + rew_lin_vel_z
#             + rew_ang_vel_xy
#             + rew_orient
#             + rew_base_height
#             + rew_torque
#             + rew_joint_acc
#             + rew_action_rate
#             + rew_hip
#             + rew_fallen_over
#         )
#         self.rew_buf = torch.clip(self.rew_buf, min=0.0, max=None)

#         # add termination reward
#         self.rew_buf += self.rew_scales["termination"] * self.reset_buf * ~self.timeout_buf

#         # log episode reward sums
#         self.episode_sums["lin_vel_xy"] += rew_lin_vel_xy
#         self.episode_sums["ang_vel_z"] += rew_ang_vel_z
#         self.episode_sums["lin_vel_z"] += rew_lin_vel_z
#         self.episode_sums["ang_vel_xy"] += rew_ang_vel_xy
#         self.episode_sums["orient"] += rew_orient
#         self.episode_sums["torques"] += rew_torque
#         self.episode_sums["joint_acc"] += rew_joint_acc
#         self.episode_sums["action_rate"] += rew_action_rate
#         self.episode_sums["base_height"] += rew_base_height
#         self.episode_sums["hip"] += rew_hip

#     def get_observations(self):
#         self.measured_heights = self.get_heights()
#         heights = (
#             torch.clip(self.base_pos[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.0) * self.height_meas_scale
#         )
#         self.obs_buf = torch.cat(
#             (
#                 self.base_lin_vel * self.lin_vel_scale,
#                 self.base_ang_vel * self.ang_vel_scale,
#                 self.projected_gravity,
#                 self.commands[:, :3] * self.commands_scale,
#                 self.dof_pos * self.dof_pos_scale,
#                 self.dof_vel * self.dof_vel_scale,
#                 heights,
#                 self.actions,
#             ),
#             dim=-1,
#         )

#     def get_ground_heights_below_knees(self):
#         points = self.knee_pos.reshape(self.num_envs, 4, 3)
#         points += self.terrain.border_size
#         points = (points / self.terrain.horizontal_scale).long()
#         px = points[:, :, 0].view(-1)
#         py = points[:, :, 1].view(-1)
#         px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
#         py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

#         heights1 = self.height_samples[px, py]
#         heights2 = self.height_samples[px + 1, py + 1]
#         heights = torch.min(heights1, heights2)
#         return heights.view(self.num_envs, -1) * self.terrain.vertical_scale

#     def get_ground_heights_below_base(self):
#         points = self.base_pos.reshape(self.num_envs, 1, 3)
#         points += self.terrain.border_size
#         points = (points / self.terrain.horizontal_scale).long()
#         px = points[:, :, 0].view(-1)
#         py = points[:, :, 1].view(-1)
#         px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
#         py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

#         heights1 = self.height_samples[px, py]
#         heights2 = self.height_samples[px + 1, py + 1]
#         heights = torch.min(heights1, heights2)
#         return heights.view(self.num_envs, -1) * self.terrain.vertical_scale

#     def get_heights(self, env_ids=None):
#         if env_ids:
#             points = quat_apply_yaw(
#                 self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]
#             ) + (self.base_pos[env_ids, 0:3]).unsqueeze(1)
#         else:
#             points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (
#                 self.base_pos[:, 0:3]
#             ).unsqueeze(1)

#         points += self.terrain.border_size
#         points = (points / self.terrain.horizontal_scale).long()
#         px = points[:, :, 0].view(-1)
#         py = points[:, :, 1].view(-1)
#         px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
#         py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

#         heights1 = self.height_samples[px, py]

#         heights2 = self.height_samples[px + 1, py + 1]
#         heights = torch.min(heights1, heights2)

#         return heights.view(self.num_envs, -1) * self.terrain.vertical_scale


# @torch.jit.script
# def quat_apply_yaw(quat, vec):
#     quat_yaw = quat.clone().view(-1, 4)
#     quat_yaw[:, 1:3] = 0.0
#     quat_yaw = normalize(quat_yaw)
#     return quat_apply(quat_yaw, vec)


# @torch.jit.script
# def wrap_to_pi(angles):
#     angles %= 2 * np.pi
#     angles -= 2 * np.pi * (angles > np.pi)
#     return angles


# def get_axis_params(value, axis_idx, x_value=0.0, dtype=float, n_dims=3):
#     """construct arguments to `Vec` according to axis index."""
#     zs = np.zeros((n_dims,))
#     assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
#     zs[axis_idx] = 1.0
#     params = np.where(zs == 1.0, value, zs)
#     params[0] = x_value
#     return list(params.astype(dtype))
