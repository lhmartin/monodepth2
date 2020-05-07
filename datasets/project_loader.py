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
import random

import torchvision
import torch
from data_helper import UnlabeledDataset, LabeledDataset
from torchvision import transforms

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

        self.full_res_shape = (256, 256)
        self.num_to_cam = { 0 : "CAM_BACK_LEFT",
                          1 : "CAM_BACK_RIGHT",
                          2 : "CAM_BACK",
                          3 : "CAM_FRONT_LEFT",
                          4 : "CAM_FRONT_RIGHT",
                          5 : "CAM_FRONT",
                          }
        self.intrinsics = {'CAM_FRONT_LEFT': [[879.03824732/5, 0.0, 613.17597314/5, 0],
                            [0.0, 879.03824732/4, 524.14407205/4 , 0],
                            [0.0, 0.0, 1.0, 0],
                                         [0, 0, 0, 1]],
                            'CAM_FRONT': [[882.61644117/5, 0.0, 621.63358525/5, 0],
                            [0.0, 882.61644117/4, 524.38397862/4, 0],
                            [0.0, 0.0, 1.0, 0],
                                         [0, 0, 0, 1]],
                            'CAM_FRONT_RIGHT': [[880.41134027/5, 0.0, 618.9494972/5, 0],
                            [0.0, 880.41134027/4, 521.38918482/4, 0],
                            [0.0, 0.0, 1.0, 0],
                                         [0, 0, 0, 1]],
                            'CAM_BACK_LEFT': [[881.28264688/4, 0.0, 612.29732111/5, 0],
                            [0.0, 881.28264688/4, 521.77447199/4, 0],
                            [0.0, 0.0, 1.0, 0],
                                         [0, 0, 0, 1]],
                            'CAM_BACK': [[882.93018422/5, 0.0, 616.45479905/5, 0],
                            [0.0, 882.93018422/4, 528.27123027/4, 0],
                            [0.0, 0.0, 1.0, 0],
                                         [0, 0, 0, 1]],
                            'CAM_BACK_RIGHT': [[881.63835671/5, 0.0, 607.66308183/5, 0],
                            [0.0, 881.63835671/4, 525.6185326/4, 0],
                            [0.0, 0.0, 1.0, 0],
                                         [0, 0, 0, 1]]}

        transform = transforms.Compose(
                [transforms.Resize([256, 256]),
                transforms.ToTensor()])

        unlabeled_scene_index = np.arange(134)
        # The scenes from 106 - 133 are labeled
        # You should divide the labeled_scene_index into two subsets (training and validation)
        test_set = np.array([125, 113, 117, 122, 133])
        scene_index = np.delete(unlabeled_scene_index, test_set)
        print(scene_index)
        print(len(scene_index))
        if self.filenames == "val":
            scene_index = test_set

        unlabeled_trainset = UnlabeledDataset(image_folder= "../data", scene_index=scene_index, first_dim='sample', transform=transform)
        loader = torch.utils.data.DataLoader(unlabeled_trainset, batch_size=1, shuffle=False, num_workers=2)

        self.loader = loader

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


        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        folder = ""


        for i in self.frame_idxs:
            inputs[("color", i, -1)] = self.get_color(folder, index, "", do_flip, i)



        K = self.intrinsics[self.num_to_cam[index % 6]]
        K = np.array(K)
        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        K = self.intrinsics[self.num_to_cam[index % 6]]
        K = np.array(K)

        inv_K = np.linalg.pinv(K)

        inputs[("K", 0)] = torch.from_numpy(K).float()
        inputs[("inv_K", 0)] = torch.from_numpy(inv_K).float()


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


    def get_color(self, folder, frame_index, side, do_flip, dif):

        black =  torch.Tensor(np.zeros((3,256,256), dtype=float))
        if frame_index == 97524:
            color = black
        elif int(int(frame_index//6 % 126)) == 125 and dif == 1:
            color = black
        elif int(int(frame_index//6 % 126)) == 1 and dif == -1:
            color = black
        else:
            frame_index += 6*dif
            color = self.loader.dataset[frame_index//6][int(frame_index % 6)]
            #print(type(color))


        color = transforms.ToPILImage()(color)
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def check_depth(self):
        return False

    def __len__(self):

        return len(self.loader) * 6
