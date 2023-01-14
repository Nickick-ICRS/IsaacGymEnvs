import numpy as np
import math
import os
from isaacgym import gymapi, gymutil
import yaml

def load_conf(conf):
    with open(conf, "r") as f:
        config = yaml.load(f)
        return config

def clamp (x, min_value, max_value):
    return max(min(x, max_value), min_value)

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")

args = gymutil.parse_arguments(
    description="Animate robot exploration space",
    custom_parameters=[
        {"name": "--config", "type": str, "default": '', "help": "Config file from which to visualise robot exploration space"}
    ])

if args.config == '':
    print("*** Please specify config file with --confix X")
    quit()

cfg = load_conf(os.path.join(
    os.path.join(root, "isaacgymenvs/cfg/task"), args.config))

gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 60.0
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()


plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

asset_root = os.path.join(root, "assets")
asset_file = "urdf/ester/urdf/ester.urdf"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = False
asset_options.use_mesh_materials = True

print("Loading asset '{}' fromt '{}'".format(asset_file, asset_root))
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

dof_names = gym.get_asset_dof_names(asset)
dof_props = gym.get_asset_dof_properties(asset)

num_dofs = gym.get_asset_dof_count(asset)
dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)

dof_types = [gym.get_asset_dof_type(asset, i) for i in range(num_dofs)]
dof_positions = dof_states['pos']

lower_limits = dof_props['lower']
upper_limits = dof_props['upper']

defaults = np.zeros(num_dofs)
conf_uppers = np.zeros(num_dofs)
conf_lowers = np.zeros(num_dofs)
is_active = np.zeros(num_dofs)
speeds = np.zeros(num_dofs)


for i in range(num_dofs):
    dof_positions[i] = defaults[i]
    speeds[i] = 2 * clamp(
        2 * (conf_uppers[i] - conf_lowers[i]),
        0.25 * math.pi,
        3.0 * math.pi)

    conf_uppers[i] = cfg["env"]["dofUpperLimits"][dof_names[i]]
    conf_lowers[i] = cfg["env"]["dofLowerLimits"][dof_names[i]]
    is_active[i] = cfg["env"]["activeJoints"][dof_names[i]]

    if conf_lowers[i] < lower_limits[i]:
        print("** WARNING: DOF {}'s config lower limit is {} which is lower than the URDF lower limit of {}".format(
            dof_names[i], conf_lowers[i], lower_limits[i]))
    if conf_uppers[i] > upper_limits[i]:
        print("** WARNING: DOF {}'s config upper limit is {} which is higher than the URDF upper limit of {}".format(
            dof_names[i], conf_uppers[i], upper_limits[i]))

num_envs = 36
num_per_row = 6
spacing = 1
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

cam_pos = gymapi.Vec3(6.2, 2.0, 6)
cam_target = gymapi.Vec3(2, -1, 3)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

envs = []
actor_handles = []

print("Creating {} envs".format(num_envs))
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 1.32, 0.0)
    pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

    actor_handle = gym.create_actor(env, asset, pose, "actor", i, 1)
    actor_handles.append(actor_handle)

    gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)

ANIM_SEEK_LOWER = 1
ANIM_SEEK_UPPER = 2
ANIM_SEEK_DEFAULT = 3
ANIM_FINISHED = 4

anim_state = ANIM_SEEK_LOWER
current_dof = 0
while True:
    if is_active[current_dof]:
        break

    current_dof = (current_dof + 1) % num_dofs
    if current_dof == 0:
        print("*** No dofs are active, unable to animate")
        quit()
print("Animating DOF {} ('{}')".format(current_dof, dof_names[current_dof]))

while not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    speed = speeds[current_dof]

    if anim_state == ANIM_SEEK_LOWER:
        dof_positions[current_dof] -= speed * dt
        if dof_positions[current_dof] <= conf_lowers[current_dof]:
            dof_positions[current_dof] = conf_lowers[current_dof]
            anim_state = ANIM_SEEK_UPPER
    elif anim_state == ANIM_SEEK_UPPER:
        dof_positions[current_dof] += speed * dt
        if dof_positions[current_dof] >= conf_uppers[current_dof]:
            dof_positions[current_dof] = conf_uppers[current_dof]
            anim_state = ANIM_SEEK_DEFAULT
    elif anim_state == ANIM_SEEK_DEFAULT:
        dof_positions[current_dof] -= speed * dt
        if dof_positions[current_dof] <= defaults[current_dof]:
            dof_positions[current_dof] = defaults[current_dof]
            anim_state = ANIM_FINISHED
    elif anim_state == ANIM_FINISHED:
        dof_positions[current_dof] = defaults[current_dof]
        current_dof = (current_dof + 1) % num_dofs
        if is_active[current_dof]:
            anim_state = ANIM_SEEK_LOWER
            print("Animating DOF {} ('{}')".format(
                current_dof, dof_names[current_dof]))

    gym.clear_lines(viewer)

    for i in range(num_envs):
        gym.set_actor_dof_states(envs[i], actor_handles[i], dof_states, gymapi.STATE_POS)

        dof_handle = gym.get_actor_dof_handle(
            envs[i], actor_handles[i], current_dof)
        frame = gym.get_dof_frame(envs[i], dof_handle)

        p1 = frame.origin
        p2 = frame.origin + frame.axis * 0.7
        color = gymapi.Vec3(1.0, 0.0, 0.0)
        gymutil.draw_line(p1, p2, color, gym, viewer, envs[i])

    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    gym.sync_frame_time(sim)

print("Done")
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
