import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.base.vec_task import VecTask

from typing import Tuple, Dict

class LearnFromMotionCapture(VecTask):
    
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        
        self.cfg = cfg

        # normalization parameters
        self.lin_vel_scale = self.cfg["env"]["learn"]["linVelScale"]
        self.lin_acc_scale = self.cfg["env"]["learn"]["linAccScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angVelScale"]

        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPosScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelScale"]
        self.torques_scale = self.cfg["env"]["learn"]["dofEffScale"]
        self.action_scale = self.cfg["env"]["learn"]["actionScale"]

        # reward scales
        self.rew_scales = {}
        self.rew_scales["chainError"] = self.cfg["env"]["learn"]["chainErrorRewardScale"]
        self.rew_scales["footError"] = self.cfg["env"]["learn"]["footErrorRewardScale"]
#        self.rew_scales["pos"] = self.cfg["env"]["learn"]["posRewardScale"]
        self.rew_scales["energy"] = self.cfg["env"]["learn"]["energyRewardScale"]

        # randomization
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        # command ranges
        self.command_x_range = self.cfg["env"]["cmdVelRanges"]["lin_x"]
        self.command_y_range = self.cfg["env"]["cmdVelRanges"]["lin_y"]
        self.command_yaw_range = self.cfg["env"]["cmdVelRanges"]["ang_z"]

        # plane params
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        # base initialisation state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLin"]
        v_ang = self.cfg["env"]["baseInitState"]["vAng"]
        state = pos + rot + v_lin + v_ang

        self.base_init_state = state

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        self.cfg["env"]["numObservations"] = 65
        self.cfg["env"]["numActions"] = 16

        # random external force parameters
        self.force_toggle_chance = self.cfg["env"]["force"]["toggleChance"] # per step
        self.force_strength = self.cfg["env"]["force"]["maxStrength"] # per axis
        self.force_pos_range = self.cfg["env"]["force"]["posRange"] # diameter

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

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

        # create motion capture data storage and variables
        self.current_frames = torch.zeros(self.num_envs, dtype=torch.float64, device=self.device)
        self.mocap_data = self.load_mocap_data(
            self.cfg["env"]["mocap"]["path"],
            self.cfg["env"]["mocap"]["trimStart"],
            self.cfg["env"]["mocap"]["trimEnd"],
            self.cfg["env"]["mocap"]["scale"])
        self.time_per_frame = self.cfg["env"]["mocap"]["timePerFrame"]
        self.reset_on_loop = self.cfg["env"]["mocap"]["resetOnLoop"]
        self.frames = torch.zeros((self.num_envs, self.mocap_data.shape[1], 3), dtype=torch.float, device=self.device)
        self.frame_rate_multipliers = torch.ones((self.num_envs), dtype=torch.float, device=self.device)

        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        actor_link_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        torques = self.gym.acquire_dof_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.link_states = gymtorch.wrap_tensor(actor_link_state).view(self.num_envs, -1, 13)
        self.prev_root_states = self.root_states.clone()
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_dof)

        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)

        for i in range(self.cfg["env"]["numActions"]):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

        # additional data used later on
        self.extras = {}
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:] = to_torch(self.base_init_state, device=self.device, requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device = self.device, requires_grad=False)

        # to randomly apply external forces to the robot during training
        self.apply_external_forces = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.external_forces = torch.zeros(self.num_envs * self.num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.external_forces_body = self.external_forces[0::self.num_bodies, :]
        self.external_force_pos = torch.zeros_like(self.external_forces)
        self.external_force_pos_body = self.external_force_pos[0::self.num_bodies, :]

        self.reset_idx(torch.arange(self.num_envs, device=self.device))


    def create_sim(self):
        self.up_axis_idx = 2 # Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing apply once immediately on startup before the first simulation step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "urdf/ester/urdf/ester.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg["env"]["urdfAsset"]["defaultDofDriveMode"]
        asset_options.collapse_fixed_joints = False
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.disable_gravity = False
        asset_options.thickness = 0.01

        ester_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(ester_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(ester_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        body_names = self.gym.get_asset_rigid_body_names(ester_asset)
        self.dof_names = self.gym.get_asset_dof_names(ester_asset)
        contact_fail_names = [s for s in body_names if "lower" not in s and "foot" not in s] # all non-lower leg/foot rigid bodies
        foot_names = [s for s in body_names if "foot" in s]
        self.contact_fail_indices = torch.zeros(len(contact_fail_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.foot_contact_sensor_indices = torch.zeros(len(foot_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.base_index = 0

        # fl, fr, rl, rr, spine
        self.ester_chain_indices = [[6, 7, 8], [10, 11, 12], [14, 15, 16], [18, 19, 20], [3, 0, 4]]
        self.mocap_chain_indices = [[7, 8, 9, 10], [12, 13, 14, 15], [17, 18, 19], [21, 22, 23], [0, 1, 2, 3]]
        self.ester_foot_indices = torch.tensor([8, 12, 16, 20], dtype=torch.long, device=self.device)
        self.mocap_foot_indices = torch.tensor([10, 15, 19, 23], dtype=torch.long, device=self.device)
         

        dof_props = self.gym.get_asset_dof_properties(ester_asset)
        for i in range(self.num_dof):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][i] = self.cfg["env"]["control"]["stiffness"]
            dof_props['damping'][i] = self.cfg["env"]["control"]["damping"]

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.ester_handles = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            ester_handle = self.gym.create_actor(env_ptr, ester_asset, start_pose, "ester", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, ester_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, ester_handle)
            self.envs.append(env_ptr)
            self.ester_handles.append(ester_handle)

        for i in range(len(contact_fail_names)):
            self.contact_fail_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.ester_handles[0], contact_fail_names[i])
        for i in range(len(foot_names)):
            self.foot_contact_sensor_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.ester_handles[0], foot_names[i])

        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.ester_handles[0], "base")


    def load_mocap_data(self, path, trim_start, trim_end, scale):
        # path relative to IsaacGymEnvs/isaacgymenvs/data
        # this file is in IsaacGymEnvs/isaacgymenvs/tasks
        filedir = os.path.join(os.getcwd(), os.path.dirname(__file__))
        full_path = os.path.join(filedir, '../data', path)

        data = np.loadtxt(full_path, delimiter=',', dtype=float)
        data = np.reshape(data, [-1, int(data.shape[1] / 3), 3])

        # axis in weird order
        tmp = np.copy(data[:, :, 2])
        data[:, :, 2] = data[:, :, 1]
        data[:, :, 1] = -tmp

        data *= scale

        return torch.from_numpy(data[trim_start:trim_end])

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        targets = self.action_scale * self.actions + self.default_dof_pos
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))

        # store prev root states now before update
        self.prev_root_states = self.root_states.clone()

        # apply random external force
        rand = torch.rand(self.num_envs, device=self.device) < self.force_toggle_chance
        # toggle forces if selected
        self.apply_external_forces[rand] = ~self.apply_external_forces[rand]
        force_on = torch.logical_and(rand, self.apply_external_forces)
        force_off = torch.logical_and(rand, ~self.apply_external_forces)

        self.external_forces_body[force_on] = (torch.rand(3, device=self.device) - 0.5) * 2 * self.force_strength
        self.external_force_pos_body[force_on] = (torch.rand(3, device=self.device) - 0.5) * self.force_pos_range
        self.external_forces_body[force_off] = 0
        self.external_force_pos_body[force_off] = 0

        # (actually) apply random external force
        self.gym.apply_rigid_body_force_at_pos_tensors(self.sim, gymtorch.unwrap_tensor(self.external_forces), gymtorch.unwrap_tensor(self.external_force_pos), gymapi.LOCAL_SPACE)


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
            self.link_states,
            self.frames,
            self.torques,
            self.contact_forces,
            self.contact_fail_indices,
            self.progress_buf,
            self.ester_foot_indices,
            self.mocap_foot_indices,
            # list
            self.ester_chain_indices,
            self.mocap_chain_indices,
            # dict
            self.rew_scales,
            # other
            self.base_index,
            self.max_episode_length,
        )

        self.calculate_avg_torques(self.torques, self.reset_buf)
        self.reset_buf |= self.get_next_frames()

    
    def calculate_avg_torques(self, torques, reset):
        try: self.torque_avgs
        except AttributeError:
            self.torque_avgs = torch.zeros_like(torques, device=self.device, requires_grad=False)
            self.robot_iteration = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)

        self.torque_avgs += torques
        self.robot_iteration += 1

        if torch.any(reset == True):
            summed_torques = self.torque_avgs[reset == True]
            summed_torques = (summed_torques.t() / self.robot_iteration[reset == True]).t()
            print(summed_torques)

            self.torque_avgs[reset == True] = 0
            self.robot_iteration[reset == True] = 0


    def get_next_frames(self):
        self.current_frames += self.frame_rate_multipliers / self.time_per_frame
        reset = np.zeros(self.num_envs, dtype=np.bool)
        # wrap frames if at end of mocap_data
        remainder = self.current_frames % 1
        overflow = self.current_frames >= self.mocap_data.shape[0]
        self.current_frames[overflow] = remainder[overflow]
        if self.reset_on_loop:
            reset[overflow] = 1
        current_frame = torch.floor(self.current_frames).long()
        next_frame = (current_frame + 1) % self.mocap_data.shape[0]
        frame_1 = self.mocap_data[current_frame]
        frame_2 = self.mocap_data[next_frame]
        self.frames = torch.lerp(frame_1, frame_2, remainder[:, None, None])

        return reset


    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.obs_buf[:] = compute_ester_observations(
            # tensors
            self.root_states,
            self.prev_root_states,
            self.dof_pos,
            self.default_dof_pos,
            self.dof_vel,
            self.torques,
            self.contact_forces,
            self.foot_contact_sensor_indices,
            self.gravity_vec,
            self.actions,
            self.frame_rate_multipliers,
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
        # randomization happens at reset time
        if self.randomize:
                self.apply_randomizations(self.randomization_params)

        positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.frame_rate_multipliers[env_ids] = torch.rand((len(env_ids)), device=self.device, dtype=torch.float) * 1.5 + 0.5 # 0.5 < X < 2
        self.get_next_frames()

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids]# * positions_offset
        self.dof_vel[env_ids] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.initial_root_states),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1


## jit functions for actual calculation of reward and obs


@torch.jit.script
# splits a single chain (e.g. left leg) into points for each robot or
# mocap frame instance
def split_into_points(
    # int
    num_points,
    # Tensor (num_envs, num_links, 3)
    links,
    # Tensor (length_of_chain)
    chain):
    # type: (int, Tensor, Tensor) -> Tensor
    # create output array
    num_envs = links.shape[0]
    r_num_envs = torch.arange(start=0, end=num_envs)
    points = torch.zeros((num_envs, num_points, 3), dtype=torch.float, device=links.device)

    # find the axis from the bottom to the top of the chains
    # shape: (num_envs, 3)
    axis = links[:, chain[-1]] - links[:, chain[0]]
    dist = torch.linalg.norm(axis, dim=1)
    axis = torch.nn.functional.normalize(axis, dim=1)

    # n points => n-1 slices
    slice_dist = dist / (num_points - 1)

    chain_it = torch.zeros(num_envs, dtype=torch.long)
    chain_axis_full = links[r_num_envs, chain[chain_it + 1]] - links[r_num_envs, chain[chain_it]]
    chain_axis = torch.nn.functional.normalize(chain_axis_full, dim=1)
    chain_mag = torch.linalg.norm(chain_axis_full, dim=1)

    # dot each axis vector
    axis_dot = (axis * chain_axis).sum(dim=1)

    # skip any initial axis that are 90 degrees as we can't handle
    while torch.any(torch.abs(axis_dot) < 1e-12):
        chain_it[torch.abs(axis_dot) < 1e-12] += 1
        chain_axis_full = links[r_num_envs, chain[chain_it + 1]] - links[r_num_envs, chain[chain_it]]
        chain_axis = torch.nn.functional.normalize(chain_axis_full, dim=1)
        chain_mag = torch.linalg.norm(chain_axis_full, dim=1)
        axis_dot = (axis * chain_axis).sum(dim=1)
    
    # find D in the quadrilateral ABDC where AB is along the central axis
    # and CD is along the chain axis
    A = links[:, chain[0]]
    C = links[:, chain[0]]

    points[:, 0, :] = C

    # initial point and final point added separately
    for i in range(num_points - 2):
        B = A + axis * slice_dist[:, None]
        CD = chain_axis * (torch.linalg.norm(B - A, dim=1) / axis_dot)[:, None]
        D = C + CD

        # If we extend past the current link we need to find the next link
        # in the chain
        overshot = torch.linalg.norm(D - links[r_num_envs, chain[chain_it]]) > chain_mag
        if torch.any(overshot):
            # take key slices of overshot envs
            o_links = links[overshot]
            o_chain_it = chain_it[overshot]
            o_axis = axis[overshot]
            o_chain_axis_full = chain_axis_full[overshot]
            o_chain_axis = chain_axis[overshot]
            o_axis_dot = axis_dot[overshot]
            o_A = A[overshot]
            o_B = B[overshot]
            o_C = C[overshot]
            o_D = D[overshot]

            r_overshot = torch.arange(start=0, end=len(overshot))
            mD = o_links[r_overshot, chain[o_chain_it+1]]
            mB = o_A + o_axis * (torch.linalg.norm(mD - o_C) * o_axis_dot)[:, None]

            o_chain_it += 1
            o_chain_axis_full = o_links[r_overshot, chain[o_chain_it+1]] - o_links[r_overshot, chain[o_chain_it]]
            o_chain_axis = torch.nn.functional.normalize(o_chain_axis_full, dim=1)
            chain_mag[overshot] = torch.linalg.norm(o_chain_axis_full, dim=1)
            o_axis_dot = (o_axis * o_chain_axis).sum(dim=1)

            mDD = o_chain_axis * (torch.linalg.norm(o_B - mB, dim=1) / o_axis_dot)[:, None]

            o_D = mD + mDD

        # store points
        points[:, i+1, :] = D
        A = B
        C = D

    points[:, -1, :] = links[:, chain[-1]]

    return points


@torch.jit.script
def herons_formula(A, B, C):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    # calculate area of triangles with heron's formula
    a = torch.linalg.norm(A - B, dim=1)
    b = torch.linalg.norm(B - C, dim=1)
    c = torch.linalg.norm(C - A, dim=1)
    s = (a + b + c) / 2
    return torch.sqrt(s * (s-a) * (s-b) * (s-c))


@torch.jit.script
def integrate_err(ester_link_pos, ester_chain, mocap_link_pos, mocap_chain):
    # type: (Tensor, List[int], Tensor, List[int]) -> Tensor
    """
    @param ester_link_pos (num_envs, num_links, 3) -> positions of links
    @param ester_chain (num_links_in_chain) -> idx of links
    @param mocap_link_pos (num_envs, num_pts, 3) -> positions of mocap
    @param mocap_chain (num_links_in_chain) -> idx of links
    
    @returns integral between ester_chain and mocap_chain relative to each
             root
    """
    num_pts = int(20)
    ester_points = split_into_points(num_pts, ester_link_pos, torch.tensor(ester_chain))
    mocap_points = split_into_points(num_pts, mocap_link_pos, torch.tensor(mocap_chain))

    sum_area = torch.zeros((ester_link_pos.shape[0]), device=ester_link_pos.device, dtype=torch.float32)
    for i in range(num_pts - 1):
        # find the area of the quadrilateral ABDC
        A = ester_points[:, i]
        B = ester_points[:, i+1]
        C = mocap_points[:, i]
        D = mocap_points[:, i+1]

        # split into triangles ABC and BCD to find area
        area = herons_formula(A, B, C) + herons_formula(B, C, D)
        sum_area += area

    return sum_area


@torch.jit.script
def compute_ester_reward(
    # tensors
    link_states, # (num_envs, num_links, 13)
    mocap_frames, # (num_envs, num_links, 3)
    torques,
    contact_forces,
    contact_fail_indices,
    episode_lengths,
    ester_foot_indices,
    mocap_foot_indices,
    # list
    ester_chain_indices, # (num_chains, len_chain)
    mocap_chain_indices, # (num_chains, len_chain)
    # dict
    rew_scales,
    # other
    base_index,
    max_episode_length,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, List[List[int]], List[List[int]], dict[str, float], int, int) -> Tuple[Tensor, Tensor]
    int_err = torch.zeros((link_states.shape[0]), dtype=torch.float32, device=link_states.device)
    for ester_chain, mocap_chain in zip(ester_chain_indices, mocap_chain_indices):
        int_err += integrate_err(link_states[:, :, :3], ester_chain, mocap_frames, mocap_chain)

    chain_rew = torch.exp(-int_err)

    foot_rew = torch.exp(-torch.sum(torch.linalg.norm(link_states[:, ester_foot_indices, :3] - mocap_frames[:, mocap_foot_indices, :], dim=-1), dim=-1))

    energy_rew = torch.exp(-torch.sum(torch.square(torques), dim=-1)/0.25)

    total_reward = rew_scales["chainError"] * chain_rew + \
                   rew_scales["footError"] * foot_rew + \
                   rew_scales["energy"] * energy_rew
    total_reward = torch.clip(total_reward, 0., None)

    reset = torch.norm(contact_forces[:, base_index, :], dim=1) > 1.
    # if non-foot contact forces exist we fell over
    reset = reset | torch.any(torch.norm(contact_forces[:, contact_fail_indices, :], dim=2) > 1., dim=1)
    # if we timeout the episode is over
    timeout = episode_lengths >= max_episode_length -1
    reset = reset | timeout

    return total_reward.detach(), reset


@torch.jit.script
def compute_ester_observations(
    root_states,
    prev_root_states,
    dof_pos,
    default_dof_pos,
    dof_vel,
    torques,
    contact_forces,
    foot_contact_sensor_indices,
    gravity_vec,
    actions,
    frame_rate_multipliers,
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

    foot_contacts = torch.any(contact_forces[:, foot_contact_sensor_indices, :] > 1., 2)

    # rescale to -1 to 1 and set shape to (X, 1)
    scaled_frame_rate_multipliers = 2 * (frame_rate_multipliers[:, None] - 0.5) / 1.5 - 1

    obs = torch.cat((scaled_frame_rate_multipliers,
                     base_lin_vel,
                     base_lin_acc,
                     base_ang_vel,
                     projected_gravity,
                     dof_pos_scaled,
                     dof_vel*dof_vel_scale,
    #                 torques*torques_scale,
                     foot_contacts,
                     actions),
                     dim=1)

    return obs
