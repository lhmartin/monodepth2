# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

import torchvision
import torch
from data_helper import UnlabeledDataset, LabeledDataset

from .mono_dataset import MonoDataset


class ProjectDataset(MonoDataset):
    """Class for Project dataset loader
    """
    def __init__(self, *args, **kwargs):
        super(ProjectDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (306, 256)
        self.num_to_cam = { 0 : "CAM_BACK_LEFT",
                          1 : "CAM_BACK_RIGHT",
                          2 : "CAM_BACK",
                          3 : "CAM_FRONT_LEFT",
                          4 : "CAM_FRONT_RIGHT",
                          5 : "CAM_FRONT",
                          }

        transform = torchvision.transforms.ToTensor()

        unlabeled_scene_index = np.arange(134)
        # The scenes from 106 - 133 are labeled
# You should divide the labeled_scene_index into two subsets (training and validation)
        test_set = np.array([125, 113, 117, 122, 133])
        unlabeled_scene_index = np.delete(unlabeled_scene_index, test_set)

        unlabeled_trainset = UnlabeledDataset(image_folder="../../data", scene_index=unlabeled_scene_index, first_dim='sample', transform=transform)
        trainloader = torch.utils.data.DataLoader(unlabeled_trainset, batch_size=1, shuffle=False, num_workers=2)

        self.loader = trainloader

    def check_depth(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.
        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:
            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.
        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.
        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        print("CALLED __getitem__")

        print("self.frame_idxs")
        print(self.frame_idxs)

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        folder = ""

        print("HERE")

        for i in self.frame_idxs:
            print( "i = ", i)
            inputs[("color", i, -1)] = self.get_color(folder, index + i * 6, "", do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        #for i in self.frame_idxs:
        #    del inputs[("color", i, -1)]
        #    del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs


    def get_color(self, folder, frame_index, side, do_flip):
        from torchvision import transforms
        print("Frame index = ", frame_index)
        print(frame_index/6)

        color = self.loader.dataset[int(frame_index/6)][(frame_index % 6)]


        color = transforms.ToPILImage()(color)
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def check_depth(self):
        return False

    def __len__(self):

        print("CALLED __len__")
        return len(self.loader) * 6
