import numpy as np
import os
import torch

from gym import spaces

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.base.vec_task import VecTask

from typing import Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

class WalkFromEnergyMin(VecTask):
    def plot_actions(self, actions):
        return
        counts, bins = np.histogram(actions.cpu().flatten().numpy())
        plt.stairs(counts, bins)

        plt.show()

    def __init__(
            self, cfg, rl_device, sim_device, graphics_device_id, headless,
            virtual_screen_capture, force_render):

        self.plt_data = []

        self.cfg = cfg

        self._load_config()

        self._read_active_joints()

        super().__init__(
            config=self.cfg, rl_device=rl_device, sim_device=sim_device,
            graphics_device_id=graphics_device_id, headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render)

        self.dt = self.sim_params.dt

        self.num_actions = self.cfg["env"]["numActions"]

        self._prepare_sim()
        self._prepare_tensors()

        self.actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float,
            device=self.device, requires_grad=False)

        self.act_space = spaces.Box(
            np.zeros(self.num_actions), np.ones(self.num_actions) * 1.)

        self.reset_idx(torch.arange(self.num_envs, device=self.device))


    def create_sim(self):
        self.up_axis_idx = 2 # Z up
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine,
            self.sim_params)
        self._create_ground_plane()
        self._create_envs(
            self.num_envs, self.cfg["env"]["envSpacing"],
            int(np.sqrt(self.num_envs)))

        if self.randomize:
            self.apply_randomizations(self.randomization_params)


    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.gym.add_ground(self.sim, plane_params)


    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/ester/urdf/ester.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg["env"]["urdfAsset"]["defaultDofDriveMode"]
        asset_options.collapse_fixed_joints = False
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = False
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.disable_gravity = False
        asset_options.thickness = 0.01

        ester_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(ester_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(ester_asset)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.initial_pos[:3])

        body_names = self.gym.get_asset_rigid_body_names(ester_asset)
        self.dof_names = self.gym.get_asset_dof_names(ester_asset)
        # We fail the episode if anything other than feet or shin touch the
        # ground
        contact_fail_names = [
            s for s in body_names if "lower" not in s and "foot" not in s]
        foot_names = [s for s in body_names if "foot" in s]

        self.contact_fail_indices = torch.zeros(
            len(contact_fail_names), dtype=torch.long, device=self.device,
            requires_grad=False)
        self.foot_contact_sensor_indices = torch.zeros(
            len(foot_names), dtype=torch.long, device=self.device,
            requires_grad=False)
        self.base_index = 0

        dof_props = self.gym.get_asset_dof_properties(ester_asset)
        for i in range(self.num_dof):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            dof_props["stiffness"][i] = self.cfg["env"]["control"]["stiffness"]
            dof_props["damping"][i] = self.cfg["env"]["control"]["damping"]

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        self.ester_handles = []
        self.envs = []

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(
                self.sim, env_lower, env_upper, num_per_row)
            ester_handle = self.gym.create_actor(
                env_ptr, ester_asset, start_pose, "ester", i, 1, 0)
            self.gym.set_actor_dof_properties(
                env_ptr, ester_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, ester_handle)
            self.envs.append(env_ptr)
            self.ester_handles.append(ester_handle)

        for i in range(len(contact_fail_names)):
            self.contact_fail_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.ester_handles[0], contact_fail_names[i])
        for i in range(len(foot_names)):
            self.foot_contact_sensor_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.ester_handles[0], foot_names[i])

        self.base_index = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.ester_handles[0], "base")


    def pre_physics_step(self, actions):
        self.plot_actions(actions)
        self.actions = actions.clone().to(self.device)
        # Actions range from 0 to 1 so split them between the joint limits
        targets = self.default_dof_pos.clone()
        targets[self.dof_active] = self.action_scale[self.dof_active] * self.actions.flatten()+ self.dof_lower_limit[self.dof_active]

        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(targets))

        # store previous states before update
        self.prev_root_states = self.root_states.clone()
        self.prev_dof_vels = self.dof_vel.clone()

        if self.random_external_force:
            rand = torch.rand(self.num_envs, device=self.device) < self.force_toggle_chance
            self.apply_external_forces[rand] = ~self.apply_external_forces[rand]
            force_on = torch.logical_and(rand, self.apply_external_forces)
            force_off = torch.logical_and(rand, ~self.apply_external_forces)

            self.external_forces_body[force_on] = (
                torch.rand(3, device=self.device) - 0.5) * 2 * self.force_strength
            self.external_force_pos_body[force_on] = (
                torch.rand(3, device=self.device) - 0.5) * 2 * self.force_pos_range
            self.external_forces_body[force_off] = 0
            self.external_force_pos_body[force_off] = 0

            self.gym.apply_rigid_body_force_at_pos_tensors(
                self.sim, gymtorch.unwrap_tensor(self.external_forces),
                gymtorch.unwrap_tensor(self.external_force_pos),
                gymapi.LOCAL_SPACE)


    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)


    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_ester_reward(
            # tensors
            self.root_states,
            self.foot_states,
            self.commands,
            self.torques,
            self.dof_vel,
            (self.dof_vel - self.prev_dof_vel) / self.dt,
            self.contact_forces,
            self.contact_fail_indices,
            self.progress_buf,
            # dict
            self.rew_scales,
            # other
            self.base_index,
            self.max_episode_length,
        )


    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.obs_buf[:] = compute_ester_observations(
            # tensors
            self.root_states,
            self.prev_root_states,
            self.commands,
            self.dof_pos,
            self.default_dof_pos,
            self.dof_vel,
            self.torques,
            self.contact_forces,
            self.foot_contact_sensor_indices,
            self.gravity_vec,
            self.actions,
            # scales
            self.lin_vel_scale,
            self.lin_acc_scale,
            self.ang_vel_scale,
            self.dof_pos_scale,
            self.dof_vel_scale,
            self.torques_scale,
            self.dt
        )


    def reset_idx(self, env_ids):
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        positions = self.default_dof_pos[env_ids]
        velocities = torch.zeros(
            len(env_ids), self.num_dof, dtype=torch.float,
            device=self.device, requires_grad=False)

        if self.randomize:
            # 99.99997 % of values between 0 and 1
            positions = torch.randn(
                len(env_ids), self.num_dof, dtype=torch.float,
                device=self.device, requires_grad=False) * 0.1 + 0.5
            positions = self.dof_lower_limit + positions * (self.dof_upper_limit - self.dof_lower_limit)
            velocities = torch_rand_float(
                -0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = positions
        self.dof_vel[env_ids] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.initial_root_states),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.commands_x[env_ids] = torch_rand_float(
            self.command_x_range[0], self.command_x_range[1],
            (len(env_ids), 1), device=self.device).squeeze()
        self.commands_y[env_ids] = torch_rand_float(
            self.command_y_range[0], self.command_y_range[1],
            (len(env_ids), 1), device=self.device).squeeze()
        self.commands_yaw[env_ids] = torch_rand_float(
            self.command_yaw_range[0], self.command_yaw_range[1],
            (len(env_ids), 1), device=self.device).squeeze()

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1


    def _load_config(self):
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        self.initial_pos = self.cfg["env"]["initialPos"]

        self.random_external_force = self.cfg["env"]["randomForce"]["apply"]
        self.force_toggle_chance = self.cfg["env"]["randomForce"]["toggle_chance"]
        self.force_strength = self.cfg["env"]["randomForce"]["strength"]
        self.force_pos_range = self.cfg["env"]["randomForce"]["posRange"]

        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]
        self.named_joint_upper_limits = self.cfg["env"]["dofUpperLimits"]
        self.named_joint_lower_limits = self.cfg["env"]["dofLowerLimits"]

        self.command_x_range = self.cfg["env"]["cmdVelRanges"]["lin_x"]
        self.command_y_range = self.cfg["env"]["cmdVelRanges"]["lin_y"]
        self.command_yaw_range = self.cfg["env"]["cmdVelRanges"]["ang_z"]

        self.rew_scales = {}
        self.rew_scales["lin_vel"] = self.cfg["env"]["learn"]["linVelRewardScale"]
        self.rew_scales["ang_vel"] = self.cfg["env"]["learn"]["angVelRewardScale"]
        self.rew_scales["energy"] = self.cfg["env"]["learn"]["energyRewardScale"]
        self.rew_scales["acc"] = self.cfg["env"]["learn"]["accRewardScale"]
        self.rew_scales["feet_raised"] = self.cfg["env"]["learn"]["feetRaisedRewardScale"]

        self.lin_vel_scale = self.cfg["env"]["learn"]["linVelScale"]
        self.lin_acc_scale = self.cfg["env"]["learn"]["linAccScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angVelScale"]

        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPosScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelScale"]
        self.torques_scale = self.cfg["env"]["learn"]["dofEffScale"]


    def _read_active_joints(self):
        self.named_active_joints = self.cfg["env"]["activeJoints"]
        num_actions = 0
        self.num_joints = 0
        for name in self.named_active_joints:
            self.num_joints += 1
            if self.named_active_joints[name]:
                num_actions += 1
        self.cfg["env"]["numActions"] = num_actions
        self.cfg["env"]["numObservations"] = self.cfg["env"]["numObs"] + num_actions


    def _prepare_sim(self):
        self.dt = self.sim_params.dt
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        if self.viewer != None:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)


    def _prepare_tensors(self):
        actor_root_state = self.gym.acquire_actor_root_state_tensor(
            self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rbd_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(
            self.sim)
        torques = self.gym.acquire_dof_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.prev_root_states = self.root_states.clone()
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(
            self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(
            self.num_envs, self.num_dof, 2)[..., 1]
        self.prev_dof_vel = torch.zeros_like(self.dof_vel)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.torques = gymtorch.wrap_tensor(torques).view(
            self.num_envs, self.num_dof)
        self.rigid_body_states = gymtorch.wrap_tensor(rbd_tensor).view(
            self.num_envs, -1, 13) # shape: num_envs, num_bodies, 13
        self.foot_states = self.rigid_body_states[:, self.foot_contact_sensor_indices, :]

        self.commands = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device)
        self.commands_x = self.commands.view(self.num_envs, 3)[..., 0]
        self.commands_y = self.commands.view(self.num_envs, 3)[..., 1]
        self.commands_yaw = self.commands.view(self.num_envs, 3)[..., 2]

        self.default_dof_pos = torch.zeros_like(
            self.dof_pos, dtype=torch.float, device=self.device,
            requires_grad=False)
        self.dof_upper_limit = torch.zeros_like(
            self.dof_pos, dtype=torch.float, device=self.device,
            requires_grad=False)
        self.dof_lower_limit = torch.zeros_like(
            self.dof_pos, dtype=torch.float, device=self.device,
            requires_grad=False)
        self.dof_active = torch.zeros_like(
            self.dof_pos, dtype=torch.bool, device=self.device,
            requires_grad=False)

        for i in range(self.num_joints):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            active = self.named_active_joints[name]
            self.default_dof_pos[:, i] = angle
            upper = self.named_joint_upper_limits[name]
            lower = self.named_joint_lower_limits[name]
            self.dof_upper_limit[:, i] = upper
            self.dof_lower_limit[:, i] = lower
            self.dof_active[:, i] = active

        self.action_scale = self.dof_upper_limit - self.dof_lower_limit

        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:, :3] = to_torch(
            self.initial_pos, device=self.device, requires_grad=False)
        self.initial_root_states[:, 7] = 1
        self.gravity_vec = to_torch(
            get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))

        self.apply_external_forces = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device,
            requires_grad=False)
        self.external_forces = torch.zeros(
            self.num_envs * self.num_bodies, 3, dtype=torch.float,
            device=self.device, requires_grad=False)
        self.external_forces_body = self.external_forces[0::self.num_bodies, :]
        self.external_force_pos = torch.zeros_like(self.external_forces)
        self.external_force_pos_body = self.external_force_pos[0::self.num_bodies, :]

        self.reset_idx(torch.arange(self.num_envs, device=self.device))


@torch.jit.script
def compute_ester_reward(
    # tensors
    root_states,
    foot_states,
    commands,
    torques,
    dof_vel,
    dof_acc,
    contact_forces,
    contact_fail_indices,
    episode_lengths,
    # dict
    rew_scales,
    # other
    base_index,
    max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, float], int, int) -> Tuple[Tensor, Tensor]
    base_quat = root_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10])
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13])

    # velocity tracking reward
    lin_vel_error = torch.sum(
        torch.square(commands[:, :2] - base_lin_vel[:, :2]), dim=1)
    ang_vel_error = torch.square(commands[:, 2] - base_ang_vel[:, 2])
    rew_lin_vel = torch.exp(-lin_vel_error/0.25) * rew_scales["lin_vel"]
    rew_ang_vel = torch.exp(-ang_vel_error/0.25) * rew_scales["ang_vel"]

    # energy penalty
    rew_energy = torch.sum(torch.square(torques * dof_vel), dim=1) * rew_scales["energy"]

    # acc penalty
    rew_acc = torch.sum(torch.square(dof_acc), dim=1) * rew_scales["acc"]

    # feet lifted off ground penalty - penalise based on z position
    rew_feet = torch.sum(foot_states[:, :, 2], dim=1) * rew_scales["feet_raised"]

    total_reward = rew_lin_vel + rew_ang_vel + rew_energy + rew_acc + rew_feet
    total_reward = torch.clip(total_reward, 0., None)

    # check for reset conditions
    reset = torch.norm(contact_forces[:, base_index, :], dim=1) > 1.
    reset = reset | torch.any(
        torch.norm(contact_forces[:, contact_fail_indices, :], dim=2) > 1.,
        dim=1)
    timeout = episode_lengths >= max_episode_length
    reset = reset | timeout

    return total_reward.detach(), reset


@torch.jit.script
def compute_ester_observations(
    root_states,
    prev_root_states,
    commands,
    dof_pos,
    default_dof_pos,
    dof_vel,
    torques,
    contact_forces,
    foot_contact_sensor_indices,
    gravity_vec,
    actions,
    lin_vel_scale,
    lin_acc_scale,
    ang_vel_scale,
    dof_pos_scale,
    dof_vel_scale,
    torques_scale,
    dt
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float, float, float, float) -> Tensor
    base_quat = root_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10])
    prev_lin_vel = quat_rotate_inverse(prev_root_states[:, 3:7], prev_root_states[:, 7:10])
    base_lin_acc = ((base_lin_vel - prev_lin_vel) / dt) * lin_acc_scale
    base_lin_vel *= lin_vel_scale
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13]) * ang_vel_scale
    projected_gravity = quat_rotate(base_quat, gravity_vec)
    dof_pos_scaled = (dof_pos - default_dof_pos) * dof_pos_scale
    commands_scaled = commands*torch.tensor([lin_vel_scale, lin_vel_scale, ang_vel_scale], requires_grad=False, device=commands.device)

    foot_contacts = torch.any(contact_forces[:, foot_contact_sensor_indices, :] > 1., 2)

    obs = torch.cat((base_lin_vel,
                     base_lin_acc,
                     base_ang_vel,
                     projected_gravity,
                     commands_scaled,
                     dof_pos_scaled,
                     dof_vel*dof_vel_scale,
                     torques*torques_scale,
                     foot_contacts,
                     actions),
                     dim=1)

    return obs
