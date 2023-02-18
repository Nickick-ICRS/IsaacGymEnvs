import torch
import numpy as np
import os

class JointDataReplayer:
    def __init__(
        self, joint_data=None, random_start=False, loop=True, filename=None,
        num_parallels=1, device=None
    ):
        """
        Class to play back motion capture data that has been remapped to
        the robot joint angles. Interpolates between frames.

        @param joint_data The entire data to playback (np.array, shape 
                          (num_frames, num_joints))
        @param random_start Whether to start from a random frame (bool)
        @param loop Whether to loop forever or only play once. (bool)
        @param filename Filename to load or store data to. If joint_data is
                        also set, only used for saving data. (string)
        """
        self.loop = loop
        assert device is not None
        self.device = device
        self.current_frames = torch.zeros(
            num_parallels, dtype=torch.float, device=self.device,
            requires_grad=False)
        self.filename = filename
        if joint_data is None:
            assert filename is not None, "Cannot initialise without joint data or a file to load from"
            joint_data = np.loadtxt(filename)

        self.random_start = random_start
        if random_start:
            self.current_frames = torch.rand(
                num_parallels, dtype=torch.float, device=self.device,
                requires_grad=False) * joint_data.shape[0]

        # We assume the final frame is the first frame, so add if it isn't
        if np.linalg.norm(joint_data[0] - joint_data[-1]) > 1e-4:
            joint_data = np.append(joint_data, [joint_data[0]], axis=0)

        # shape = (num_parallels, num_frames, num_joints)
        self.joint_data = torch.zeros(
            (num_parallels, joint_data.shape[0], joint_data.shape[1]),
            dtype=torch.float, device=self.device, requires_grad=False)

        self.ret_frames = torch.zeros(
            (num_parallels, self.joint_data.shape[-1]), dtype=torch.float,
            device=self.device, requires_grad=False)

        self.max_frames = joint_data.shape[0] - 1


    def next_frame(self, dt, fps, reset=None):
        """
        Get the next joint angle frame. Interpolates.
        @param dt The time that has elapsed since the last call to
                  next_frame
        @param fps The "normal" fps of the data. Can be used to control
                   playback speed (shape: num_parallels)
        """
# TODO: Process reset
        done = False
        # negative dt is invalid
        dt = max(0, dt)
        self.current_frames += fps * dt
        if reset is not None:
            if self.random_start:
                self.current_frames[reset] = torch.rand(
                    reset.shape[0], dtype=torch.float, device=self.device,
                    requires_grad=False) * self.joint_data.shape[1]
            else:
                self.current_frames[reset] = 0

        overflow = self.current_frames >= self.max_frames
        done = torch.zeros_like(fps, dtype=torch.bool)
        if torch.any(overflow):
            if not self.loop:
                self.current_frames[overflow] = self.max_frames
                done[overflow] = True
                self.ret_frames[overflow] = self.joint_data[overflow][:, self.max_frames - 1]
            else:
                self.current_frames[overflow] -= self.max_frames
                # signal we looped if they care
                done[overflow] = True

        if torch.any(~overflow):
            prev_frames = torch.floor(
                self.current_frames[~overflow]).type(torch.long)
            next_frames = prev_frames + 1
            remainders = self.current_frames[~overflow] - prev_frames

            self.ret_frames[~overflow] = self._interpolate_frames(
                prev_frames, next_frames, remainders, ~overflow)
        return self.ret_frames, done


    def _interpolate_frames(
        self, prev_frame_ids, next_frame_ids, remainders, parallels
    ):
        remainders = torch.clamp(remainders, 0, 1)

        n = prev_frame_ids.shape[0]
        A = torch.arange(n)
        prev_frames = self.joint_data[parallels][A, prev_frame_ids]
        next_frames = self.joint_data[parallels][A, next_frame_ids]

        deltas = next_frames - prev_frames
        return prev_frames + deltas * remainders.reshape(n, 1)
