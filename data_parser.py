from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
# from utils import smpl_to_adt



#########################
###### ADT Dataset ######
#########################
class ADT(Dataset):

    NUM_BODY_JOINTS = 21
    NUM_HAND_JOINTS = 15

    def __init__(self, sequence_path, 
                 use_hands=False,
                 dtype=torch.float32,
                 model_type='smplx',
                 joints_to_ign=None,
                 adt_format='adt51',
                 **kwargs):
        super(ADT, self).__init__()

        self.use_hands = use_hands
        self.model_type = model_type
        self.dtype = dtype
        self.joints_to_ign = joints_to_ign
        self.adt_format = adt_format

        self.num_joints = (self.NUM_BODY_JOINTS +
                           2 * self.NUM_HAND_JOINTS * use_hands)  # 21 + 2*15

        self.skeleton_data = self.read_adt_skeleton_sequence(sequence_path)
        self.cnt = 0
    
    def read_adt_skeleton_sequence(self, sequence_path):
        # load ADT data path
        if sequence_path.split("_")[-1][0] == "M":
            selected_skeleton_file = "Skeleton_T.json"
        else:
            selected_skeleton_file = "Skeleton_C.json"
        skeleton_path = os.path.join(sequence_path, selected_skeleton_file)
        with open(skeleton_path) as f:
            json_data = json.load(f)
        raw_skeleton_data = json_data["frames"]
        num_frames = len(raw_skeleton_data)
        print(f"Load {num_frames} frames from {skeleton_path}")
        # read each frame's raw 3D keypoints
        skeleton_data = []
        for frame_idx in range(len(raw_skeleton_data)):
            raw_keypoints_3d = np.array(raw_skeleton_data[frame_idx]["joints"])
            skeleton_data.append(raw_keypoints_3d)
        skeleton_data = np.array(skeleton_data)
        return skeleton_data
    
    def get_model2data(self):
        if self.adt_format.lower() == 'adt51':
            if self.model_type == 'smplx':
                # ['Skeleton', 'Ab', 'Chest', 'Neck', 'Head', 'LShoulder', 'LUArm', 'LFArm', 'LHand',  'RShoulder', 
                # 'RUArm', 'RFArm', 'RHand',  'LThigh', 'LShin', 'LFoot', 'LToe', 'RThigh', 'RShin', 'RFoot', 'RToe']
                body_mapping = np.array([0, 3, 9, 12, 15, 13, 16, 18, 20, 14, 
                                        17, 19, 21, 1, 4, 7, 60, 2, 5, 8, 63], dtype=np.int32)  # 21
                mapping = [body_mapping]
                if self.use_hands:
                    # 'LHand', 'LThumb1', 'LThumb2', 'LThumb3', 'LIndex1', 'LIndex2', 'LIndex3', 'LMiddle1',
                    #  'LMiddle2', 'LMiddle3', 'LRing1', 'LRing2', 'LRing3', LPinky1', 'LPinky2', 'LPinky3', 
                    lhand_mapping = np.array([20, 37, 38, 39, 25, 26, 27,
                                            28, 29, 30, 34, 35, 36, 
                                            31, 32, 33], dtype=np.int32)  # 16
                    # 'RHand', 'RThumb1', 'RThumb2', 'RThumb3', 'RIndex1', 'RIndex2', 'RIndex3', 'RMiddle1', 
                    #  'RMiddle2', 'RMiddle3', 'RRing1', 'RRing2', 'RRing3', 'RPinky1', 'RPinky2', 'RPinky3',
                    rhand_mapping = np.array([21, 52, 53, 54, 40, 41, 42, 
                                            43, 44, 45, 49, 50, 51, 
                                            46, 47, 48], dtype=np.int32)  # 16

                    mapping += [lhand_mapping, rhand_mapping]
                return np.concatenate(mapping)
        else:
            raise ValueError('Unknown joint format: {}'.format(self.adt_format))

    def get_joint_weights(self):
        # The weights for the joint terms in the optimization
        optim_weights = np.ones(self.num_joints + 2 * self.use_hands,
                                dtype=np.float32)

        # Neck, Left and right hip
        # These joints are ignored because SMPL has no neck joint and the
        # annotation of the hips is ambiguous.
        if self.joints_to_ign is not None and -1 not in self.joints_to_ign:
            optim_weights[self.joints_to_ign] = 0.
        return torch.tensor(optim_weights, dtype=self.dtype)

    def __len__(self):
        return self.skeleton_data.shape[0]

    def __getitem__(self, idx):
        keypoints_frame = self.skeleton_data[idx]
        return self.read_item(keypoints_frame)

    def read_item(self, keypoints_frame):
        body_keypoints = np.vstack((keypoints_frame[0:9], keypoints_frame[24:28], keypoints_frame[43:]), dtype=np.float32)
        if self.use_hands:
            left_hand_keyp = keypoints_frame[8:24].astype(np.float32)
            right_hand_keyp = keypoints_frame[27:43].astype(np.float32)
            body_keypoints = np.concatenate(
                [body_keypoints, left_hand_keyp, right_hand_keyp], axis=0)
        body_keypoints = np.expand_dims(body_keypoints, axis=0)  # 1 x num_joints x 3
        return body_keypoints

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.cnt >= self.skeleton_data.shape[0]:
            raise StopIteration

        keypoints_frame = self.skeleton_data[self.cnt]
        self.cnt += 1

        return self.read_item(keypoints_frame)
    



class CustomDataset(Dataset):

    NUM_BODY_JOINTS = 21
    NUM_HAND_JOINTS = 15

    def __init__(self, sequence_path, 
                 use_hands=False,
                 dtype=torch.float32,
                 model_type='smplx',
                 joints_to_ign=None,
                 adt_format='adt51',
                 **kwargs):
        super(CustomDataset, self).__init__()

        self.use_hands = use_hands
        self.model_type = model_type
        self.dtype = dtype
        self.joints_to_ign = joints_to_ign
        self.adt_format = adt_format

        self.num_joints = (self.NUM_BODY_JOINTS +
                           2 * self.NUM_HAND_JOINTS * use_hands)  # 21 + 2*15

        self.skeleton_data = self.read_adt_skeleton_sequence(sequence_path)
        self.cnt = 0
    
    def read_adt_skeleton_sequence(self, sequence_path):
        # load custom data path: take ADT format for convenience
        selected_skeleton_file = "skeletons.json"  # "skeletons.json"
        # selected_skeleton_file = "filtered_skeletons.json"   # "skeletons.json"
        skeleton_path = os.path.join(sequence_path, selected_skeleton_file)
        json_data = []
        with open(skeleton_path, 'r') as f:
            for line in f:
                json_data.append(json.loads(line))
        num_frames = len(json_data)
        print(f"Load {num_frames} frames from {skeleton_path}")
        # read each frame's raw 3D keypoints
        skeleton_data = []
        for frame_idx in range(num_frames):
            raw_keypoints_3d = np.array(json_data[frame_idx]["joints"]).reshape(-1, 3)
            skeleton_data.append(raw_keypoints_3d)
        skeleton_data = np.array(skeleton_data)
        return skeleton_data
    
    def get_model2data(self):
        if self.adt_format.lower() == 'adt51':
            if self.model_type == 'smplx':
                # ['Skeleton', 'Ab', 'Chest', 'Neck', 'Head', 'LShoulder', 'LUArm', 'LFArm', 'LHand',  'RShoulder', 
                # 'RUArm', 'RFArm', 'RHand',  'LThigh', 'LShin', 'LFoot', 'LToe', 'RThigh', 'RShin', 'RFoot', 'RToe']
                body_mapping = np.array([0, 3, 9, 12, 15, 13, 16, 18, 20, 14, 
                                        17, 19, 21, 1, 4, 7, 60, 2, 5, 8, 63], dtype=np.int32)  # 21
                mapping = [body_mapping]
                if self.use_hands:
                    # 'LHand', 'LThumb1', 'LThumb2', 'LThumb3', 'LIndex1', 'LIndex2', 'LIndex3', 'LMiddle1',
                    #  'LMiddle2', 'LMiddle3', 'LRing1', 'LRing2', 'LRing3', LPinky1', 'LPinky2', 'LPinky3', 
                    lhand_mapping = np.array([20, 37, 38, 39, 25, 26, 27,
                                            28, 29, 30, 34, 35, 36, 
                                            31, 32, 33], dtype=np.int32)  # 16
                    # 'RHand', 'RThumb1', 'RThumb2', 'RThumb3', 'RIndex1', 'RIndex2', 'RIndex3', 'RMiddle1', 
                    #  'RMiddle2', 'RMiddle3', 'RRing1', 'RRing2', 'RRing3', 'RPinky1', 'RPinky2', 'RPinky3',
                    rhand_mapping = np.array([21, 52, 53, 54, 40, 41, 42, 
                                            43, 44, 45, 49, 50, 51, 
                                            46, 47, 48], dtype=np.int32)  # 16

                    mapping += [lhand_mapping, rhand_mapping]
                return np.concatenate(mapping)
        else:
            raise ValueError('Unknown joint format: {}'.format(self.adt_format))

    def get_joint_weights(self):
        # The weights for the joint terms in the optimization
        optim_weights = np.ones(self.num_joints + 2 * self.use_hands,
                                dtype=np.float32)

        # Neck, Left and right hip
        # These joints are ignored because SMPL has no neck joint and the
        # annotation of the hips is ambiguous.
        if self.joints_to_ign is not None and -1 not in self.joints_to_ign:
            optim_weights[self.joints_to_ign] = 0.
        return torch.tensor(optim_weights, dtype=self.dtype)

    def __len__(self):
        return self.skeleton_data.shape[0]

    def __getitem__(self, idx):
        keypoints_frame = self.skeleton_data[idx]
        return self.read_item(keypoints_frame)

    def read_item(self, keypoints_frame):
        body_keypoints = np.vstack((keypoints_frame[0:9], keypoints_frame[24:28], keypoints_frame[43:]), dtype=np.float32)
        if self.use_hands:
            left_hand_keyp = keypoints_frame[8:24].astype(np.float32)
            right_hand_keyp = keypoints_frame[27:43].astype(np.float32)
            body_keypoints = np.concatenate(
                [body_keypoints, left_hand_keyp, right_hand_keyp], axis=0)
        body_keypoints = np.expand_dims(body_keypoints, axis=0)  # 1 x num_joints x 3
        return body_keypoints

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.cnt >= self.skeleton_data.shape[0]:
            raise StopIteration

        keypoints_frame = self.skeleton_data[self.cnt]
        self.cnt += 1

        return self.read_item(keypoints_frame)
    


class OwnSkeletonDataset(Dataset):
    """CSV skeleton format with 21 body joints and one frame per row."""

    NUM_BODY_JOINTS = 21

    JOINT_NAMES = {
        0: 'pelvis',
        1: 'R_hip',
        2: 'R_knee',
        3: 'R_ankle',
        4: 'L_hip',
        5: 'L_knee',
        6: 'L_ankle',
        7: 'spine1',
        8: 'spine2',
        9: 'spine3',
        10: 'neck',
        11: 'head',
        12: 'head_end',
        13: 'R_shoulder_inner',
        14: 'R_shoulder',
        15: 'R_elbow',
        16: 'R_wrist',
        17: 'L_shoulder_inner',
        18: 'L_shoulder',
        19: 'L_elbow',
        20: 'L_wrist',
    }

    CONNECTIONS = [
        (0, 1), (1, 2), (2, 3),
        (0, 4), (4, 5), (5, 6),
        (0, 7), (7, 8), (8, 9),
        (9, 10), (10, 11), (11, 12),
        (10, 13), (13, 14), (14, 15), (15, 16),
        (10, 17), (17, 18), (18, 19), (19, 20),
    ]

    def __init__(self, sequence_path,
                 dtype=torch.float32,
                 model_type='smplx',
                 joints_to_ign=None,
                 skeleton_scale=None,
                 **kwargs):
        super(OwnSkeletonDataset, self).__init__()

        self.dtype = dtype
        self.model_type = model_type
        self.joints_to_ign = joints_to_ign
        # The provided CSV appears centimeter-like; SMPL-X is meter-scale.
        self.skeleton_scale = 0.01 if skeleton_scale is None else skeleton_scale

        self.skeleton_path = self.resolve_skeleton_path(sequence_path)
        self.sequence_name = os.path.splitext(os.path.basename(self.skeleton_path))[0]
        self.skeleton_data = self.read_skeleton_sequence(self.skeleton_path)
        self.cnt = 0

    def resolve_skeleton_path(self, sequence_path):
        if os.path.isfile(sequence_path):
            return sequence_path

        csv_files = sorted(
            item for item in os.listdir(sequence_path)
            if item.lower().endswith('.csv')
        )
        if not csv_files:
            raise FileNotFoundError(
                f'No CSV skeleton file found in {sequence_path}'
            )
        if len(csv_files) > 1:
            print(f'Found multiple CSV files in {sequence_path}; using {csv_files[0]}')
        return os.path.join(sequence_path, csv_files[0])

    def read_skeleton_sequence(self, skeleton_path):
        raw_data = np.genfromtxt(skeleton_path, delimiter=',', dtype=np.float32)
        if raw_data.ndim == 1:
            raw_data = raw_data.reshape(1, -1)

        # If a CSV header is present, genfromtxt returns NaNs for that row.
        raw_data = raw_data[~np.isnan(raw_data).all(axis=1)]

        expected_cols = 1 + self.NUM_BODY_JOINTS * 3
        if raw_data.shape[1] != expected_cols:
            raise ValueError(
                f'Expected {expected_cols} columns: frame index + '
                f'{self.NUM_BODY_JOINTS}*3 xyz values. Got {raw_data.shape[1]} '
                f'in {skeleton_path}.'
            )

        joints = raw_data[:, 1:].reshape(-1, self.NUM_BODY_JOINTS, 3)
        joints = joints * self.skeleton_scale
        print(
            f'Load {joints.shape[0]} frames from {skeleton_path} '
            f'with scale {self.skeleton_scale}'
        )
        return joints.astype(np.float32)

    def get_model2data(self):
        if self.model_type != 'smplx':
            raise ValueError('OwnSkeletonDataset currently supports model_type=smplx')

        # Input order:
        # pelvis, R hip/knee/ankle, L hip/knee/ankle, spine1/2/3,
        # neck, head, head_end, R collar/shoulder/elbow/wrist,
        # L collar/shoulder/elbow/wrist.
        return np.array([
            0,   # pelvis
            2,   # R_hip
            5,   # R_knee
            8,   # R_ankle
            1,   # L_hip
            4,   # L_knee
            7,   # L_ankle
            3,   # spine1
            6,   # spine2
            9,   # spine3
            12,  # neck
            15,  # head
            15,  # head_end has no direct SMPL-X joint; ignore or downweight it
            14,  # R_shoulder_inner / right collar
            17,  # R_shoulder
            19,  # R_elbow
            21,  # R_wrist
            13,  # L_shoulder_inner / left collar
            16,  # L_shoulder
            18,  # L_elbow
            20,  # L_wrist
        ], dtype=np.int32)

    def get_joint_weights(self):
        optim_weights = np.ones(self.NUM_BODY_JOINTS, dtype=np.float32)

        # SMPL-X has a head joint but no separate head-end joint.
        optim_weights[12] = 0.0

        if self.joints_to_ign is not None and -1 not in self.joints_to_ign:
            optim_weights[self.joints_to_ign] = 0.0
        return torch.tensor(optim_weights, dtype=self.dtype)

    def __len__(self):
        return self.skeleton_data.shape[0]

    def __getitem__(self, idx):
        return self.read_item(self.skeleton_data[idx])

    def read_item(self, keypoints_frame):
        return np.expand_dims(keypoints_frame.astype(np.float32), axis=0)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.cnt >= self.skeleton_data.shape[0]:
            raise StopIteration

        keypoints_frame = self.skeleton_data[self.cnt]
        self.cnt += 1
        return self.read_item(keypoints_frame)
