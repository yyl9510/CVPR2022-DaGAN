import random

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

from cv2 import COLOR_BGR2GRAY, COLOR_BGR2RGB, COLOR_RGB2BGR, cvtColor, imread
from PIL import Image
from skimage import img_as_float32, img_as_ubyte


def get_params(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
    # augmentation_params:
    # flip_param:
    #     horizontal_flip: True
    #     time_flip: True
    # jitter_param:
    #     brightness: 0.1
    #     contrast: 0.1
    #     saturation: 0.1
    #     hue: 0.1
    if brightness > 0:
        brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
    else:
        brightness_factor = None

    if contrast > 0:
        contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
    else:
        contrast_factor = None

    if saturation > 0:
        saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
    else:
        saturation_factor = None

    if hue > 0:
        hue_factor = random.uniform(-hue, hue)
    else:
        hue_factor = None
    return brightness_factor, contrast_factor, saturation_factor, hue_factor


def crop_square_tensor(img, size=256, interpolation=TF.InterpolationMode.BICUBIC):
    h, w = img.shape[1:]  # C, H, W
    if h == size and w == size:
        return img
    min_size = np.amin([h, w])

    # Centralize and crop
    crop_img = img[
        :, int(h / 2 - min_size / 2) : int(h / 2 + min_size / 2), int(w / 2 - min_size / 2) : int(w / 2 + min_size / 2)
    ]
    resized = TF.resize(crop_img, (size, size), interpolation=interpolation, antialias=True)

    return resized


def crop_square_video_tensor(video, size=256, interpolation=TF.InterpolationMode.BICUBIC):
    # video: T, C, H, W
    T, C, H, W = video.shape
    min_size = np.amin([H, W])

    # Centralize and crop
    crop_video = video[
        :,
        :,
        int(H / 2 - min_size / 2) : int(H / 2 + min_size / 2),
        int(W / 2 - min_size / 2) : int(W / 2 + min_size / 2),
    ]
    resize_images = [
        TF.resize(crop_img, (size, size), interpolation=interpolation, antialias=True) for crop_img in crop_video
    ]
    resized = torch.stack(resize_images, dim=0)

    return resized


def crop_square_ndarray(img, size=256, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]
    if h == size and w == size:
        return img
    min_size = np.amin([h, w])

    # Centralize and crop
    crop_img = img[
        int(h / 2 - min_size / 2) : int(h / 2 + min_size / 2),
        int(w / 2 - min_size / 2) : int(w / 2 + min_size / 2),
    ]
    if h >= size:
        resized = cv2.resize(crop_img, (size, size), interpolation=interpolation)
    else:
        resized = cv2.resize(crop_img, (size, size), interpolation=cv2.INTER_CUBIC)

    return resized


def crop_square_pil(img, size=256):
    w, h = img.size
    if h == size and w == size:
        return img
    min_size = np.amin([w, h])
    crop_img = img.crop(
        (
            int(w / 2 - min_size / 2),
            int(h / 2 - min_size / 2),
            int(w / 2 + min_size / 2),
            int(h / 2 + min_size / 2),
        )
    )
    if h >= size:
        interpolation = Image.ANTIALIAS
    else:
        interpolation = Image.BICUBIC
    resized = crop_img.resize((size, size), resample=interpolation)
    return resized


def pad_square(img, size=256):
    h, w = img.shape[:2]
    if h == size and w == size:
        return img
    max_size = np.amax([h, w])
    pad_img = np.zeros((max_size, max_size, 3), dtype=np.uint8)
    pad_img[
        int(max_size / 2 - h / 2) : int(max_size / 2 + h / 2), int(max_size / 2 - w / 2) : int(max_size / 2 + w / 2)
    ] = img
    if max_size >= size:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC
    resized = cv2.resize(pad_img, (size, size), interpolation=interpolation)
    return resized


def crop_square_ndarray_upside(img, size=256, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]
    min_size = np.amin([h, w])

    # Centralize and crop
    crop_img = img[
        h - min_size : h,
        int(w / 2 - min_size / 2) : int(w / 2 + min_size / 2),
    ]
    if h >= size:
        resized = cv2.resize(crop_img, (size, size), interpolation=interpolation)
    else:
        resized = cv2.resize(crop_img, (size, size), interpolation=cv2.INTER_CUBIC)

    return resized


def crop_square(img, size=256):
    if isinstance(img, np.ndarray):
        return crop_square_ndarray(img, size=size)
    elif isinstance(img, torch.Tensor):
        return crop_square_tensor(img, size=size)
    elif isinstance(img, Image.Image):
        return crop_square_pil(img, size=size)


def crop_rect(img, stride=32):
    h, w = img.shape[:2]
    if h % stride == 0 and w % stride == 0:
        return img
    new_h = h // stride * stride
    new_w = w // stride * stride
    start_h = int((h - new_h) / 2)
    start_w = int((w - new_w) / 2)
    crop_img = img[start_h : start_h + new_h, start_w : start_w + new_w]

    return crop_img


def resize_short(img, size=256):
    h, w = img.shape[:2]
    if h == size and w == size:
        return img
    if h > size:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC
    new_w = int(w / (h / size))
    resized = cv2.resize(img, (new_w, size), interpolation=interpolation)
    return resized


def read_image(image_path, size=256, to_cuda=False, pad_sq=False, crop_sq=True, crop_re=False, resize_sht=False):
    img = imread(image_path)
    if resize_sht:
        img = resize_short(img, size=size)
    if pad_sq:
        img = pad_square(img, size=size)
    if crop_sq:
        img = crop_square(img, size=size)
    if crop_re:
        img = crop_rect(img)
    img = img_as_float32(cvtColor(img, COLOR_BGR2RGB))
    img = np.array(img, dtype="float32").transpose((2, 0, 1))
    img = torch.from_numpy(img)
    if to_cuda:
        img = img.unsqueeze(0).cuda()

    return img


def center_crop_and_resize(img, target_height, target_width):
    # Calculate the aspect ratio of the target dimensions
    target_aspect = target_width / target_height

    # Calculate source image aspect ratio
    source_height, source_width = img.shape[0], img.shape[1]
    source_aspect = source_width / source_height

    # Determine dimensions of the crop based on the aspect ratio
    if source_aspect > target_aspect:
        # Crop width to match the target aspect ratio
        crop_width = int(source_height * target_aspect)
        crop_height = source_height
    else:
        # Crop height to match the target aspect ratio
        crop_width = source_width
        crop_height = int(source_width / target_aspect)

    # Calculate starting coordinates (x, y) for the crop
    x = (source_width - crop_width) // 2
    y = (source_height - crop_height) // 2

    # Crop the image
    cropped_img = img[y : y + crop_height, x : x + crop_width]

    if crop_width < target_width:
        interp = cv2.INTER_CUBIC
    elif crop_width > target_width:
        interp = cv2.INTER_AREA
    # Resize the cropped image to the target dimensions
    resized_img = cv2.resize(cropped_img, (target_width, target_height), interpolation=interp)

    return resized_img


def read_image_vsr(image_path, img_h=160, img_w=320, to_cuda=False):
    img = imread(image_path)
    img = center_crop_and_resize(img, img_h, img_w)
    img = img_as_float32(cvtColor(img, COLOR_BGR2RGB))
    img = np.array(img, dtype="float32").transpose((2, 0, 1))
    img = torch.from_numpy(img)
    if to_cuda:
        img = img.unsqueeze(0).cuda()

    return img


def resize_short_and_crop(img, frame_shape):
    h, w = img.shape[:2]
    if h == frame_shape[0] and w == frame_shape[1]:
        return img

    th, tw = frame_shape[:2]
    if h < w:
        new_h = th
        new_w = int(w * new_h / h)
    else:
        new_w = tw
        new_h = int(h * new_w / w)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    if new_h > th:
        start_h = int((new_h - th) / 2)
        start_w = 0
    elif new_w > tw:
        start_h = 0
        start_w = int((new_w - tw) / 2)
    elif new_h < th or new_w < tw:
        img = np.zeros((th, tw, 3), dtype=np.uint8)
        start_h = int((th - new_h) / 2)
        start_w = int((tw - new_w) / 2)
        img[start_h : start_h + new_h, start_w : start_w + new_w] = resized
        assert img.shape[:2] == frame_shape[:2], f"{img.shape[:2]} != {frame_shape[:2]}"
        return img

    crop_img = resized[start_h : start_h + th, start_w : start_w + tw]
    assert crop_img.shape[:2] == frame_shape[:2], f"{crop_img.shape[:2]} != {frame_shape[:2]}"

    return crop_img


def read_image_any_shape(image_path, frame_shape=(384, 672), to_cuda=False):
    img = imread(image_path)
    img = resize_short_and_crop(img, frame_shape)
    img = img_as_float32(cvtColor(img, COLOR_BGR2RGB))
    img = np.array(img, dtype="float32").transpose((2, 0, 1))
    img = torch.from_numpy(img)
    if to_cuda:
        img = img.unsqueeze(0).cuda()

    return img


def crop_square_mask(msk, size=256, interpolation=cv2.INTER_NEAREST):
    h, w = msk.shape[:2]
    if h == size and w == size:
        return msk
    min_size = np.amin([h, w])

    # Centralize and crop
    crop_img = msk[
        int(h / 2 - min_size / 2) : int(h / 2 + min_size / 2),
        int(w / 2 - min_size / 2) : int(w / 2 + min_size / 2),
    ]
    resized = cv2.resize(crop_img, (size, size), interpolation=interpolation)
    return resized


def read_mask(mask_path, size=256, to_cuda=False, to_tensor=True):
    msk = imread(mask_path)
    msk = crop_square_mask(msk, size=size)
    msk = cvtColor(msk, COLOR_BGR2GRAY)[:, :, np.newaxis]
    msk = np.array(msk).transpose((2, 0, 1))
    if to_tensor:
        msk = torch.from_numpy(msk)
    if to_cuda:
        msk = msk.unsqueeze(0).cuda()

    return msk


def cvt_image(image, size=256, to_cuda=False):
    img = crop_square(image, size=size)
    img = img_as_float32(cvtColor(img, COLOR_BGR2RGB))
    img = np.array(img, dtype="float32").transpose((2, 0, 1))
    img = torch.from_numpy(img)
    if to_cuda:
        img = img.unsqueeze(0).cuda()

    return img


def image_to_numpy(image):
    if isinstance(image, np.ndarray):
        return image
    elif isinstance(image, torch.Tensor):
        out_img = np.transpose(image.data.cpu().numpy(), [0, 2, 3, 1])[0]
        out_img = img_as_ubyte(out_img)

        out_img = cvtColor(out_img, COLOR_RGB2BGR)
        return out_img
