import json
import os
import random

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF

from .data_utils import get_params, read_image


class VFHQDataset(data.Dataset):
    def __init__(
        self,
        root_dir,
        split,
        size=256,
        is_train=True,
        evaluate_all=True,
        data_type="two",
        repeat=1,
        use_jpg=False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.size = size
        self.is_train = is_train
        self.evaluate_all = evaluate_all
        self.data_type = data_type
        if not os.path.exists(root_dir):
            print(f"{root_dir} does not exist")
            root_dir = "./data/VFHQ_datasets_extracted"

        if split == "train":
            self.video_dir = os.path.join(root_dir, "VFHQ-Train", "extracted_cropped_face_results")
            self.files_names_json = os.path.join(
                root_dir, "VFHQ-Train", "extracted_cropped_face_results_file_names.json"
            )
        else:
            self.video_dir = os.path.join(root_dir, "VFHQ-Test", "extracted_cropped_face_results")
            self.files_names_json = os.path.join(
                root_dir, "VFHQ-Test", "extracted_cropped_face_results_file_names.json"
            )

        if use_jpg:
            print("use jpg .....")
            self.video_dir = self.video_dir.replace(
                "extracted_cropped_face_results", "extracted_cropped_face_results_jpg"
            )
            self.files_names_json = self.files_names_json.replace(
                "extracted_cropped_face_results_file_names", "extracted_cropped_face_results_file_names_jpg"
            )

        with open(self.files_names_json, "r") as f:
            self.data_dict = json.load(f)

        self.clips = self.data_dict["clips"]
        if not self.is_train and not self.evaluate_all:
            self.clips = self.clips[:32]
        if repeat > 1:
            self.clips = self.clips * repeat

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        clip_name = self.clips[index]
        video_name = self.data_dict[clip_name]["video_name"]
        frame_names = self.data_dict[clip_name]["frames"]

        T = len(frame_names)

        if self.is_train:
            frame_ids = np.random.choice(T, replace=True, size=2)
            frame_path_1 = os.path.join(self.video_dir, video_name, clip_name, frame_names[frame_ids[0]])
            frame_path_2 = os.path.join(self.video_dir, video_name, clip_name, frame_names[frame_ids[1]])
            frame_path_1 = frame_path_1.strip()
            frame_path_2 = frame_path_2.strip()

            frame_1 = read_image(frame_path_1, self.size)
            frame_2 = read_image(frame_path_2, self.size)

            if random.random() < 0.5:
                source, driving = frame_1, frame_2
            else:
                source, driving = frame_2, frame_1
            if random.random() < 0.5:
                source = TF.hflip(source)
                driving = TF.hflip(driving)

            brightness, contrast, saturation, hue = get_params()
            source = TF.adjust_brightness(source, brightness)
            driving = TF.adjust_brightness(driving, brightness)
            source = TF.adjust_contrast(source, contrast)
            driving = TF.adjust_contrast(driving, contrast)
            source = TF.adjust_saturation(source, saturation)
            driving = TF.adjust_saturation(driving, saturation)
            source = TF.adjust_hue(source, hue)
            driving = TF.adjust_hue(driving, hue)

            if "norm" in self.data_type:
                source = TF.normalize(source, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                driving = TF.normalize(driving, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            data_sample = {
                "source": source,
                "driving": driving,
            }
            return data_sample

        else:
            if not self.evaluate_all:
                frame_names = frame_names[:64]
            frame_paths = [
                os.path.join(self.video_dir, video_name, clip_name, frame_name) for frame_name in frame_names
            ]
            frame_paths = [frame_path.strip() for frame_path in frame_paths]
            frames = [read_image(frame_path, self.size) for frame_path in frame_paths]
            video = torch.stack(frames, dim=0)

            if "norm" in self.data_type:
                video = video * 2.0 - 1.0

            out_clip_name = clip_name + ".mp4"
            data_sample = {
                "video": video,
                "out_clip_name": out_clip_name,
            }
            return data_sample


class VFHQReconVisDataset(data.Dataset):
    def __init__(
        self,
        root_dir,
        selected_indies=None,
        size=256,
        data_type="two",
        part_idx=0,
        part_num=1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.size = size
        self.data_type = data_type
        print(f"root_dir {root_dir}")

        self.video_dir = os.path.join(root_dir, "VFHQ-Test", "extracted_cropped_face_results")
        self.files_names_json = os.path.join(root_dir, "VFHQ-Test", "extracted_cropped_face_results_file_names.json")

        with open(self.files_names_json, "r") as f:
            self.data_dict = json.load(f)

        clips = self.data_dict["clips"]
        if selected_indies is not None:
            clips = [clips[i] for i in selected_indies]

        self.clips = clips[part_idx::part_num]

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        clip_name = self.clips[index]
        video_name = self.data_dict[clip_name]["video_name"]
        frame_names = self.data_dict[clip_name]["frames"]

        T = len(frame_names)

        frame_paths = [os.path.join(self.video_dir, video_name, clip_name, frame_name) for frame_name in frame_names]
        frame_paths = [frame_path.strip() for frame_path in frame_paths]
        frames = [read_image(frame_path, self.size) for frame_path in frame_paths]
        video = torch.stack(frames, dim=0)

        if "norm" in self.data_type:
            video = video * 2.0 - 1.0

        # out_clip_name = clip_name + ".mp4"
        return video, clip_name
