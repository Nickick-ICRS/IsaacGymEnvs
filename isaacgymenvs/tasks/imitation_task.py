import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from .base.vec_task import VecTask

from typing import Tuple, Dict

class ImitationTask(VecTask):
    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        
        self.cfg = cfg

        # imitation
        self.motion_filenames = self.cfg["task"]["motionFileNames"]
        self.enable_cycle_sync = self.cfg["task"]["enableCycleSync"]
        self.ref_state_init_prop = self.cfg["task"]["refStateInitProb"]
        self.enable_rand_init_time = self.cfg["task"]["enableRandInitTime"]

        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.action_scale = self.cfg["env"]["learn"]["actionScale"]

        # reward scales
        self.rew_scale = {}
        self.rew_scale["pose"] = self.cfg["task"]["poseRewardScale"]
        self.rew_scale["vel"] = self.cfg["task"]["velocityRewardScale"]
        self.rew_scale["effector"] = self.cfg["task"]["endEffectorRewardScale"]
        self.rew_scale["rootPose"] = self.cfg["task"]["rootPoseRewardScale"]
        self.rew_scale["rootVel"] = self.cfg["task"]["rootVelocityRewardScale"]
        self.rew_scale["global"] = self.cfg["task"]["globalRewardScale"]
        
        # error scales
        self.err_scale = {}
        self.err_scale["pose"] = self.cfg["task"]["poseErrorScale"]
        self.err_scale["vel"] = self.cfg["task"]["velocityErrorScale"]
        self.err_scale["effector"] = self.cfg["task"]["endEffectorErrorScale"]
        self.err_scale["effectorHeight"] = self.cfg["task"]["endEffectorHeightErrorScale"]
        self.err_scale["rootPose"] = self.cfg["task"]["rootPoseErrorScale"]
        self.err_scale["rootVel"] = self.cfg["task"]["rootVelocityErrorScale"]

        # randomization
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        # command ranges
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]

        # plane params
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        state = pos + rot + v_lin + v_ang

        self.base_init_state = state

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        # TODO: confirm
        self.cfg["env"]["numObservations"] = 48
        self.cfg["env"]["numActions"] = 12

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)

        # other
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

        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        torques = self.gym.acquire_dof_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_dof)

        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_y = self.commands.view(self.num_envs, 3)[..., 1]
        self.commands_x = self.commands.view(self.num_envs, 3)[..., 0]
        self.commands_yaw = self.commands.view(self.num_envs, 3)[..., 2]
        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)

        for i in range(self.cfg["env"]["numActions"]):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

        # initialize some data used later on
        self.extras = {}
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:] = to_torch(self.base_init_state, device=self.device, requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the first
        # sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        seld.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "urdf/anymal_c/urdf/anymal.urdf"
        robot_name = "anymal"

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.colapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False

        robot_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        extremity_name = "SHANK" if asset_options.collapse_fixed_joints else "FOOT"
        feet_names = [s for s in body_names if extremity_name in s]
        self.feet_indices = torch.zeros(
            len(feet_names), dtype=torch.long, device=self.device,
            requires_grad=False)
        knee_names = [s for s in body_names if "THIGH" in s]
        self.knee_indices = torch.zeros(
            len(knee_names), dtype=torch.long, device=self.device,
            requires_grad=False)
        self.base_index = 0

        dof_props = self.gym.get_asset_dof_properties(robot_asset)
        for i in range(self.num_dof):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][i] = self.cfg["env"]["control"]["stiffness"]
            dof_props['damping'][i] = self.cfg["env"]["control"]["damping"]

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.robot_handles = []
        self.envs = []

        # create env instances
        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(
                self.sim, env_lower, env_upper, num_per_row)
            robot_handle = self.gym.create_actor(
                env_ptr, robot_asset, start_pose, robot_name, i, 1, 0)
            self.gym.set_actor_dof_properties(
                env_ptr, robot_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, robot_handle)
            self.envs.append(env_ptr)
            self.robot_handles.append(robot_handle)

        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.robot_handles[0], feet_names[i])
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.robot_handles[0], knee_names[i])

        self.base_index = self.gym.find_actor_rigid_body_handle(
            self.envs[0]. self.robot_handles[0]. "base")

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        targets = self.action_scale * self.actions + self.default_dof_pos
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(targets))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

    def compute_rewards(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_robot_reward(
            # tensors
            self.root_states,
            self.commands,
            self.torques,
            self.contact_forces,
            self.knee_indices,
            self.progress_buf,
            # dict
            self.rew_scales,
            # other
            self.base_index,
            self.max_episode_length
        )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        
        self.obs_buf[:] = compute_robot_observations(
            # tensors
            self.root_states,
            self.commands,
            self.dof_pos,
            self.default_dof_pos,
            self.dof_vel,
            self.gravity_vec,
            self.actions,
            # scales
            self.lin_vel_scale,
            self.ang_vel_scale,
            self.dof_pos_scale,
            self.dof_vel_scale
        )

    def reset_idx(self, env_ids):
        # randomization happens at reset time on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        positions_offset = torch_rand_float(
            0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch.rand_float(
            -0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)
        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
        self.dof_vel[env_ids] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.initial_root_states),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.commands_x[env_ids] = torch_rand_float(
            self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1),
            device=self.device).squeeze()
        self.commands_y[env_ids] = torch_rand_float(
            self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1),
            device=self.device).squeeze()
        self.commands_yaw[env_ids] = torch_rand_float(
            self.command_yaw_range[0], self.command_yaw_range[1],
            (len(env_ids), 1), device=self.device).squeeze()

        # Load the reference motions to learn
        self.ref_motions = self._load_ref_motions(self.ref_motion_filenames)
        self._reset_ref_motion(env_ids)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

    def _load_ref_motions(self, filenames):
        # TODO
        pass

    def _reset_ref_motion(self, env_ids):
        pass

    def _update_ref_motion(self, env_ids):
        pass


# jit functions

@torch.jit.script
def compute_robot_reward(
    # tensors
    root_states,
    mimic_states,
    joint_states, # reshaped to (num_actors, num_dofs, 2)
    mimic_joint_states, # reshaped to (num_actors, num_dofs, 2)
    knee_indicies,
    episode_lengths,
    # dict
    rew_scales,
    err_scales,
    # other
    base_index,
    max_episode_length
):
    dof_diff = joint_states - mimic_joint_states;
    # TODO: Check that this does position [0] and velocity [1] separately
    dof_err = torch.square(dof_diff)
    pose_err = torch.sum(dof_err[:, :, 0], dim=1)
    vel_err = torch.sum(dof_err[:, :, 1], dim=1)

    # 0,1,2 pos, 3,4,5,6 quat, 7,8,9 linvel, 10,11,12 angvel
    base_quat = root_states[:, 3:7]
    root_pos_diff = root_states[:, 0:3] - mimic_root_states[:, 0:3]
    root_pos_err = torch.sum(torch.square(root_pos_diff), dim=1)
    root_rot_diff = quat_unit(quat_mul(
        base_quat, 3:7], quat_conjugate(mimic_root_states[:, 3:7])))
    root_rot_diff_ang = get_euler_xyz(root_rot_diff)
    root_rot_err = torch.sum(torch.square(root_rot_diff_angle), dim=1)
    root_pose_err = root_pos_err + 0.5 * root_rot_err

    root_lin_vel_diff = root_states[:, 7:10] - mimic_root_states[:, 7:10]
    root_lin_vel_err = torch.sum(torch.square(root_lin_vel_diff), dim=1)
    root_ang_vel_diff = root_states[:, 10:] - mimic_root_states[:, 10:]
    root_ang_vel_err = torch.sum(torch.square(root_ang_vel_diff), dim=1)
    root_vel_err = root_lin_vel_err + 0.1 * root_ang_vel_err

    # Apply weights to individual rewards
    pose_reward = torch.exp(pose_err * -err_scales["pose"])
    velocity_reward = torch.exp(vel_err * -err_scales["vel"])
    #end_effector_reward = torch.exp(end_effector_err * -err_scales["effector"])
    root_pose_reward = torch.exp(root_pose_err * -err_scales["rootPose"])
    root_velocity_reward = torch.exp(root_vel_err * -err_scales["rootVel"])

    # Finalise reward calculation
    total_reward = pose_reward * rew_scales["pose"] \
                 + velocity_reward * rew_scales["vel"] \
    #             + end_effector_reward * rew_scales["effector"] \
                 + root_pose_reward * rew_scales["rootPose"] \
                 + root_velocity_reward * rew_scales["rootVel"]
    total_reward = torch.clip(total_reward * rew_scales["global"], 0., None)

    # End training if we collapse or time out
    reset = torch.norm(contact_forces[:, base_index, :], dim=1) > 1.
    reset = reset | torch.any(torch.norm(contact_forces[:, knee_indices, :], dim=2) > 1., dim=1)
    time_out = episode_lengths > max_episode_length  # no terminal reward for time-outs
    reset = reset | time_out

    return total_reward.detach(), reset


@torch.jit.script
def compute_robot_observations(
        root_states, commands, dof_pos, default_dof_pos, dof_vel, gravity_vec,
        actions, lin_vel_scale, ang_vel_scale, dof_pos_scale, dof_vel_scale

        # tensors
        root_states,
        ref_root_states
        # dicts
        motion_t0s,
        motions,
        # other
        timestep,
        tar_steps
        ):
    # TODO
    tar_poses = []


    for motion, i in enumerate(motions):
        t = motion_t0s[i]
        ref_pos = gymapi.Vec3(ref_root_states[i, 0:3])
        quat = gymapi.Quat(root_states[i, 3:7])
        rot_dir = quat.rotate(gymapi.Vec3(1, 0, 0))
        heading = torch.arctan2(rot_dir.y, rot_dir.x)
        # Apply obs noise?
        inv_heading_quat = gymapi.Quat.from_euler_zyx(-heading, 0, 0)

        for step in tar_steps:
            tar_t = motion_t0s[i] + step * timestep
            
            tar_pose = calc_ref_pose(tar_t, motion)

            p = motion.get_frame_root_pos(tar_pose)
            r = motion.get_frame_root_rot(tar_pose)
            tar_root_pos = gymapi.Vec3(p[0], p[1], p[2])
            tar_root_quat = gymapi.Quat(r[0], r[1], r[2], r[3])

            tar_root_pos -= ref_pos
            tar_root_pos = inv_heading_quat.rotate(tar_root_pos)
            tar_root_quat = (inv_heading_quat * tar_root_quat).normalize()

            p = np.array([tar_root_pos.x, tar_root_pos.y, tar_root_pos.z])
            r = np.array([tar_root_rot.x, tar_root_rot.y, tar_root_rot.z, tar_root_rot.w]) 
            motion.set_frame_root_pos(p, tar_pose)
            motion.set_frame_root_rot(r, tar_pose)

