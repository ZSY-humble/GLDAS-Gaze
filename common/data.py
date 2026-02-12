import torch
import torchvision.transforms as T
from . import utils
from detectron2.data.detection_utils import read_image
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import scipy.ndimage as filters
import numpy as np
from os.path import join, dirname
import warnings
import torch.utils.data
import torch.multiprocessing

class FFN_IRL(Dataset):
    """FFN_IRL dataset for visual search."""

    def __init__(self, root_dir, initial_fix, img_info, annos, transform, pa, catIds, coco_annos=None):
        self.img_info = img_info
        self.root_dir = root_dir
        self.img_dir = join(root_dir, 'images')
        self.transform = transform
        self.pa = pa
        self.bboxes = annos
        self.initial_fix = initial_fix
        self.catIds = catIds
        self.coco_helper = None
        self.fv_tid = len(self.catIds)

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, idx):
        imgId = self.img_info[idx]
        cat_name, img_name, condition = imgId.split('*')
        bbox = self.bboxes.get(imgId, None)

        if cat_name == 'none':
            im_path = "{}/{}".format(self.img_dir, img_name)
        else:
            c = cat_name.replace(' ', '_')
            im_path = "{}/{}/{}".format(self.img_dir, c, img_name)

        im = Image.open(im_path)
        im_tensor = self.transform(im)

        if bbox is not None:
            coding = utils.multi_hot_coding(bbox, self.pa.patch_size, self.pa.patch_num)
            coding = torch.from_numpy(coding).view(1, -1)
        else:
            coding = torch.zeros(1, self.pa.patch_count)

        action_mask = np.zeros(
            self.pa.patch_num[0] * self.pa.patch_num[1],
            dtype=np.uint8
        )

        is_fv = condition == 'freeview'
        ret = {
            'task_id': self.fv_tid if is_fv else self.catIds[cat_name],
            'img_name': img_name,
            'cat_name': cat_name,
            'im_tensor': im_tensor,
            'label_coding': coding,
            'condition': condition,
            'action_mask': torch.from_numpy(action_mask)
        }
        return ret


class GLDAS_Human_Gaze(Dataset):
    def __init__(self, root_dir, fix_labels, bbox_annos, scene_annos, pa, transform,
                 catIds, blur_action=False, acc_foveal=True, coco_annos=None):
        self.root_dir = root_dir
        self.img_dir = join(root_dir, 'images')
        self.pa = pa
        self.transform = transform
        self.to_tensor = T.ToTensor()
        self.fix_labels = list(
            filter(lambda x: len(x[3]) <= pa.max_traj_length, fix_labels))
        self.catIds = catIds
        self.blur_action = blur_action
        self.acc_foveal = acc_foveal
        self.bboxes = bbox_annos
        self.coco_helper = None
        self.fv_tid = len(self.catIds)

    def __len__(self):
        return len(self.fix_labels)

    def __getitem__(self, idx):
        img_name, cat_name, condition, fixs, action, is_last, sid, dura = self.fix_labels[idx]
        imgId = cat_name + '_' + img_name
        bbox = self.bboxes.get(imgId, None)

        if cat_name == 'none':
            im_path = "{}/{}".format(self.img_dir, img_name)
        else:
            c = cat_name.replace(' ', '_')
            im_path = "{}/{}/{}".format(self.img_dir, c, img_name)

        im = Image.open(im_path)
        im_tensor = self.transform(im.copy())
        assert im_tensor.shape[-1] == self.pa.im_w and im_tensor.shape[-2] == self.pa.im_h, "wrong image size."

        IOR_weight_map = np.zeros((self.pa.im_h, self.pa.im_w), dtype=np.float32)
        IOR_weight_map += 1

        scanpath_length = len(fixs)
        if scanpath_length == 0:
            fixs = [(0, 0)]

        fixs = fixs + [fixs[-1]] * (self.pa.max_traj_length - len(fixs))
        is_padding = torch.zeros(self.pa.max_traj_length)
        is_padding[scanpath_length:] = 1

        fixs_tensor = torch.FloatTensor(fixs)
        fixs_tensor /= torch.FloatTensor([self.pa.im_w + 1, self.pa.im_h + 1])

        next_fixs_tensor = fixs_tensor.clone()
        if not is_last:
            x, y = utils.action_to_pos(action, [1, 1], [self.pa.im_w, self.pa.im_h])
            next_fix = torch.FloatTensor([x, y]) / torch.FloatTensor([self.pa.im_w, self.pa.im_h])
            next_fixs_tensor[scanpath_length:] = next_fix

        target_fix_map = np.zeros(self.pa.im_w * self.pa.im_h, dtype=np.float32)
        if not is_last:
            target_fix_map[action] = 1
            target_fix_map = target_fix_map.reshape(self.pa.im_h, -1)
            target_fix_map = filters.gaussian_filter(
                target_fix_map, sigma=self.pa.target_fix_map_sigma)
            target_fix_map /= target_fix_map.max()
        else:
            target_fix_map = target_fix_map.reshape(self.pa.im_h, -1)

        is_fv = condition == 'freeview'
        ret = {
            "task_id": self.fv_tid if is_fv else self.catIds[cat_name],
            "is_freeview": is_fv,
            "true_state": im_tensor,
            "target_fix_map": target_fix_map,
            "true_action": torch.tensor([action], dtype=torch.long),
            'img_name': img_name,
            'task_name': cat_name,
            'normalized_fixations': fixs_tensor,
            'next_normalized_fixations': next_fixs_tensor,
            'is_TP': condition == 'present',
            'is_last': is_last,
            'is_padding': is_padding,
            'true_or_fake': 1.0,
            'IOR_weight_map': IOR_weight_map,
            'scanpath_length': scanpath_length,
            'duration': dura,
            'subj_id': sid - 1
        }
        return ret
