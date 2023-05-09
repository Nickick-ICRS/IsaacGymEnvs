import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.gymtorch import *
from isaacgym.torch_utils import *

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask


class TwoWheelWalkingRobot(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]
        self.max_lin_vel = self.cfg["env"]["maxLinVel"]
        self.max_ang_vel = self.cfg["env"]["maxAngVel"]
        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.dof_torque_scale = self.cfg["env"]["dofTorqueScale"]
        self.lin_vel_rew_scale = self.cfg["env"]["linVelRewScale"]
        self.ang_vel_rew_scale = self.cfg["env"]["angVelRewScale"]
        self.up_rew_scale = self.cfg["env"]["upRewScale"]
        self.actions_cost_scale = self.cfg["env"]["actionsCost"]
        self.energy_cost_scale = self.cfg["env"]["energyCost"]
        self.joints_at_limit_cost_scale = self.cfg["env"]["jointsAtLimitCost"]
        self.termination_height = self.cfg["env"]["terminationHeight"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.cfg["env"]["numObservations"] = 47
        self.cfg["env"]["numActions"] = 8

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(50.0, 25.0, 2.4)
            cam_target = gymapi.Vec3(45.0, 25.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        torques = self.gym.acquire_dof_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:, 7:13] = 0  # set lin_vel and ang_vel to 0

        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.dof_torque = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_dof)
        self.initial_dof_pos = torch.zeros_like(self.dof_pos, device=self.device, dtype=torch.float)
        zero_tensor = torch.tensor([0.0], device=self.device)
        self.initial_dof_pos[:, self.has_limits] = torch.where(
            self.dof_limits_lower > zero_tensor, self.dof_limits_lower,
            torch.where(
                self.dof_limits_upper < zero_tensor,
                self.dof_limits_upper,
                self.initial_dof_pos[:, self.has_limits]))
        self.initial_dof_vel = torch.zeros_like(self.dof_vel, device=self.device, dtype=torch.float)

        self.cmd_vel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.tgt_cmd_vel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        self.dt = self.cfg["sim"]["dt"]
        self.max_lin_acc = self.cfg["env"]["maxLinAcc"] * self.dt
        self.max_ang_acc = self.cfg["env"]["maxAngAcc"] * self.dt

    def create_sim(self):
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        print(f'num envs {self.num_envs} env spacing {self.cfg["env"]["envSpacing"]}')
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')

        asset_file = self.cfg["env"]["asset"]["assetFileName"]

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        # Note - DOF mode is set in the MJCF file and loaded by Isaac Gym
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.angular_damping = 0.0

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*get_axis_params(1.05, 2))

        self.robot_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []
        vel_limits = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            robot_handle = self.gym.create_actor(
                env_ptr, robot_asset, start_pose, "robot", i, 1, 0)

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, robot_handle, j, gymapi.MESH_VISUAL,
                    gymapi.Vec3(0.97, 0.38, 0.06))

            self.envs.append(env_ptr)
            self.robot_handles.append(robot_handle)

        dof_prop = self.gym.get_actor_dof_properties(env_ptr, robot_handle)
        self.has_limits = torch.tensor(
            dof_prop['hasLimits'], dtype=torch.bool, device=self.device)
        for j in range(self.num_dof):
            if dof_prop['hasLimits'][j]:
                dof_prop['driveMode'][j] = gymapi.DOF_MODE_POS
                # TODO: regularly update this
                dof_prop['stiffness'][j] = self.cfg["env"]["control"]["stiffness_pos"]
                dof_prop['damping'][j] = self.cfg["env"]["control"]["damping_pos"]
                if dof_prop['lower'][j] > dof_prop['upper'][j]:
                    self.dof_limits_lower.append(dof_prop['upper'][j])
                    self.dof_limits_upper.append(dof_prop['lower'][j])
                else:
                    self.dof_limits_lower.append(dof_prop['lower'][j])
                    self.dof_limits_upper.append(dof_prop['upper'][j])
            else:
                dof_prop['driveMode'][j] = gymapi.DOF_MODE_VEL
                dof_prop['stiffness'][j] = 0
                dof_prop['damping'][j] = self.cfg["env"]["control"]["damping_vel"]
                vel_limits.append(dof_prop['velocity'][j])

        self.vel_limits = torch.tensor(vel_limits, dtype=torch.float, device=self.device)

        for env_ptr, robot_handle in zip(self.envs, self.robot_handles):
            self.gym.set_actor_dof_properties(env_ptr, robot_handle, dof_prop)

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_reward(
            self.obs_buf,
            self.reset_buf,
            self.progress_buf,
            self.actions,
            self.cmd_vel,
            self.root_states,
            self.lin_vel_rew_scale,
            self.ang_vel_rew_scale,
            self.up_rew_scale,
            self.actions_cost_scale,
            self.energy_cost_scale,
            self.joints_at_limit_cost_scale,
            self.termination_height,
            self.max_episode_length
        )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.obs_buf[:] = compute_observations(
            self.root_states,
            self.cmd_vel,
            self.dof_pos,
            self.dof_vel,
            self.dof_torque,
            self.has_limits,
            self.dof_limits_lower,
            self.dof_limits_upper,
            self.max_lin_vel,
            self.max_ang_vel,
            self.dof_vel_scale,
            self.dof_torque_scale,
            self.actions)

    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        positions = torch_rand_float(-0.2, 0.2, (len(env_ids), sum(self.has_limits)), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)
        cmd_vel_xs = torch_rand_float(-self.max_lin_vel, self.max_lin_vel, (len(env_ids), 1), device=self.device).squeeze()
        cmd_vel_zs = torch_rand_float(-self.max_ang_vel, self.max_ang_vel, (len(env_ids), 1), device=self.device).squeeze()

        self.dof_pos[env_ids][:, self.has_limits] = tensor_clamp(
            self.initial_dof_pos[env_ids][:, self.has_limits] + positions,
            self.dof_limits_lower,
            self.dof_limits_upper)
        self.dof_vel[env_ids] = velocities
        self.cmd_vel[env_ids, 0] = cmd_vel_xs
        self.cmd_vel[env_ids, 2] = cmd_vel_zs

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.initial_root_states),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        positions = torch.zeros_like(self.actions)
        velocities = torch.zeros_like(self.actions)
        positions[:, self.has_limits] = self.dof_limits_lower + self.actions[:, self.has_limits] * (self.dof_limits_upper - self.dof_limits_lower)
        velocities[:, ~self.has_limits] = self.actions[:, ~self.has_limits] * self.vel_limits * 2 - self.vel_limits

        pos_tensor = gymtorch.unwrap_tensor(positions)
        vel_tensor = gymtorch.unwrap_tensor(velocities)

        self.gym.set_dof_position_target_tensor(self.sim, pos_tensor)
        self.gym.set_dof_velocity_target_tensor(self.sim, vel_tensor)

    def post_physics_step(self):
        self.progress_buf += 1

        # Each episode is up to 1000 steps - after 100 start occaisionally
        # changing cmd_vel
        env_ids = self.progress_buf > 100
        change = torch.logical_and(
            torch_rand_float(0., 1., (self.num_envs, 1), device=self.device).squeeze() < 0.01,
            env_ids)
        if torch.any(change):
            self.tgt_cmd_vel[change, 0] = torch_rand_float(
                -self.max_lin_vel, self.max_lin_vel, (torch.sum(change), 1), device=self.device).squeeze()
            self.tgt_cmd_vel[change, 2] = torch_rand_float(
                -self.max_ang_vel, self.max_ang_vel, (torch.sum(change), 1), device=self.device).squeeze()

        # Prevent instantaneous cmd_vel changes
        cmd_vel_err = torch.abs(self.tgt_cmd_vel[:, 2] - self.cmd_vel[:, 2]) > self.max_lin_acc
        sign = self.tgt_cmd_vel[:, 2] - self.cmd_vel[:, 2] >= 0
        self.cmd_vel[:, 2][cmd_vel_err] += self.max_lin_acc * sign[cmd_vel_err]
        self.cmd_vel[:, 2][~cmd_vel_err] = self.tgt_cmd_vel[:, 2][~cmd_vel_err]
        cmd_vel_err = torch.abs(self.tgt_cmd_vel[:, 0] - self.cmd_vel[:, 0]) > self.max_lin_acc
        sign = self.tgt_cmd_vel[:, 0] - self.cmd_vel[:, 0] >= 0
        self.cmd_vel[:, 0][cmd_vel_err] += self.max_lin_acc * sign[cmd_vel_err]
        self.cmd_vel[:, 0][~cmd_vel_err] = self.tgt_cmd_vel[:, 0][~cmd_vel_err]

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_actor_root_state_tensor(self.sim)

            points = []
            colors = []
            for i in range(self.num_envs):
                origin = self.gym.get_env_origin(self.envs[i])
                pose = self.root_states[i, :].cpu().numpy()
                p = [origin.x + pose[0], origin.y + pose[1], origin.z + pose[2]]
                x, y, z, w = (pose[3], pose[4], pose[5], pose[6])
                yaw = np.arctan2(2*(z*w + y*x), -1 + 2*(w*w + x*x + y*y))
                points.append([p[0], p[1], p[2], p[0], p[1], p[2]])
                colors.append([0.97, 0.1, 0.06])
                num_pts = 10
                for k in range(num_pts):
                    theta = self.cmd_vel[i, 2].cpu().numpy() * k / num_pts + yaw
                    R = np.array([[np.cos(theta), -np.sin(theta)],
                                  [np.sin(theta),  np.cos(theta)]])
                    v = R * self.cmd_vel[i, :2].cpu().numpy()
                    p = [points[-1][3], points[-1][4], points[-1][5]]
                    points.append(
                        [p[0]          , p[1]          , p[2],
                         p[0] + v[0, 0], p[1] + v[1, 0], p[2]])
                    colors.append([0.97, 0.1, 0.06])
            self.gym.add_lines(self.viewer, None, self.num_envs * 2, points, colors)

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_reward(
    obs_buf,
    reset_buf,
    progress_buf,
    actions,
    cmd_vel,
    root_states,
    lin_vel_rew_scale,
    ang_vel_rew_scale,
    up_rew_scale,
    actions_cost_scale,
    energy_cost_scale,
    joints_at_limit_cost_scale,
    termination_height,
    max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor]


    base_quat = root_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10])
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13])

    lin_vel_error = torch.sum(torch.square(cmd_vel[:, :2] - base_lin_vel[:, :2]), dim=1)
    ang_vel_error = torch.square(cmd_vel[:, 2] - base_ang_vel[:, 2])

    rew_vel_xy = torch.exp(-lin_vel_error/0.25) * lin_vel_rew_scale
    rew_vel_z  = torch.exp(-ang_vel_error/0.25) * ang_vel_rew_scale

    # aligning up axis of robot and environment
    # TODO: Confirm it's #10
    rew_up = obs_buf[:, 16] * up_rew_scale

    # obs_buf shapes: 1, 3, 3, 3, 4, 3, num_dofs(8), num_dofs(8), num_dofs(8), num_dofs(8)
    # z pos, cmd_vel, lin_vel, ang_vel, orientation, up_vec, pos, vel, torque, actions
    # energy penalty for movement
    actions_cost = torch.sum(obs_buf[:, 31:39] ** 2, dim=-1)
    electricity_cost = torch.sum(torch.abs(obs_buf[:, 23:31] * obs_buf[:, 31:39]), dim=-1)
    dof_at_limit_cost = torch.sum(obs_buf[:, 17:23] > 0.99, dim=-1) + torch.sum(obs_buf[:, 17:23] < 0.01, dim=-1)

    total_reward = rew_up + rew_vel_xy + rew_vel_z -\
        actions_cost_scale * actions_cost - energy_cost_scale * electricity_cost - dof_at_limit_cost * joints_at_limit_cost_scale

    # adjust reward for fallen agents
    total_reward = torch.where(obs_buf[:, 0] < termination_height, torch.zeros_like(total_reward), total_reward)

    # reset agents
    reset = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)

    return total_reward, reset


@torch.jit.script
def compute_observations(
    root_states,
    cmd_vel,
    dof_pos,
    dof_vel,
    dof_torque,
    has_limits,
    dof_limits_lower,
    dof_limits_upper,
    max_lin_vel,
    max_ang_vel,
    dof_vel_scale,
    dof_torque_scale,
    actions,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float, Tensor) -> Tensor

    torso_position = root_states[:, 0:3]
    torso_rotation = root_states[:, 3:7]
    velocity = quat_rotate_inverse(torso_rotation, root_states[:, 7:10])
    ang_velocity = quat_rotate_inverse(torso_rotation, root_states[:, 10:13])

    up_vec = torch.zeros_like(torso_position)
    up_vec[:, 2] = 1;
    up_vec = quat_rotate_inverse(torso_rotation, up_vec)

    dof_pos_scaled = unscale(
        dof_pos[:, has_limits], dof_limits_lower, dof_limits_upper)

    # obs_buf shapes: 1, 3, 3, 3, 4, 3, num_lim_dofs(6), num_dofs(8), num_dofs(8), num_dofs(8)
    # z pos, cmd_vel, lin_vel, ang_vel, orientation, up_vec, pos, vel, torque, actions
    cmd_vel_scaled = torch.zeros_like(cmd_vel)
    cmd_vel_scaled += cmd_vel
    cmd_vel_scaled[:, :2] /= max_lin_vel
    cmd_vel_scaled[:, 2] /= max_ang_vel
    obs = torch.cat(
        (torso_position[:, 2].view(-1, 1), cmd_vel, velocity, ang_velocity,
         torso_rotation, up_vec, dof_pos_scaled, dof_vel * dof_vel_scale,
         dof_torque * dof_torque_scale, actions), dim=-1)

    return obs 
