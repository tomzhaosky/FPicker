"""
File: data/augmentations.py
Author: Tingyin Zhao @ SGroup, EE, Tsinghua
Date: 2025-12-23
Description: Implements a suite of data augmentation techniques,
such as random cropping, flipping, and photometric distortions.
"""

import cv2
import numpy as np
from numpy import random


def compute_intersection(boxes1, boxes2):
    """
    Compute the intersection area between two sets of boxes.
    Args:
        boxes1: Bounding boxes, Shape: [N, 4]
        boxes2: Single bounding box, Shape: [4]
    """
    max_xy = np.minimum(boxes1[:, 2:], boxes2[2:])
    min_xy = np.maximum(boxes1[:, :2], boxes2[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def compute_iou(boxes1, boxes2):
    """
    Compute the Jaccard Overlap (IoU) of two sets of boxes.
    Args:
        boxes1: Multiple bounding boxes, Shape: [N, 4]
        boxes2: Single bounding box, Shape: [4]
    Return:
        iou: Shape: [N]
    """
    inter = compute_intersection(boxes1, boxes2)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])
    union = area1 + area2 - inter
    return inter / union


class Compose(object):
    """Composes several augmentations together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class ConvertFromInts(object):
    """Convert image from integer to float32."""

    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class Normalize(object):
    """Normalize the image with mean and std."""

    def __init__(self, mean=None, std=None):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image /= 255.0
        image -= self.mean
        image /= self.std
        return image, boxes, labels


class ToAbsoluteCoords(object):
    """Convert relative coordinates (0-1) to absolute pixel coordinates."""

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height
        return image, boxes, labels


class ToPercentCoords(object):
    """Convert absolute pixel coordinates to relative coordinates (0-1)."""

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height
        return image, boxes, labels


class Resize(object):
    """Resize the image to a fixed size."""

    def __init__(self, size=512):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size, self.size))
        return image, boxes, labels


class RandomSaturation(object):
    """Randomly adjust saturation in HSV space."""

    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "upper must be >= lower."
        assert self.lower >= 0, "lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)
        return image, boxes, labels


class RandomHue(object):
    """Randomly adjust hue in HSV space."""

    def __init__(self, delta=18.0):
        assert 0.0 <= delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomLightingNoise(object):
    """Randomly swap image channels."""

    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor(object):
    """Convert image color space between BGR and HSV."""

    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast(object):
    """Randomly adjust image contrast."""

    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "upper must be >= lower."
        assert self.lower >= 0, "lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness(object):
    """Randomly adjust image brightness."""

    def __init__(self, delta=32):
        assert 0.0 <= delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class RandomSampleCrop(object):
    """
    Randomly crop the image and adjust bounding boxes.
    Strategy:
        1. Original image
        2. Sample patch with min IoU constraints (0.1, 0.3, 0.7, 0.9)
        3. Random sampling
    """

    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # Use index to select mode to avoid numpy warning about ragged sequences
            mode_idx = random.randint(0, len(self.sample_options))
            mode = self.sample_options[mode_idx]

            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # Max trials (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # Aspect ratio constraint between 0.5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # Convert to integer rect [x1, y1, x2, y2]
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])

                # Calculate IoU between the crop and gt boxes
                overlap = compute_iou(boxes, rect)

                if len(boxes) == 0:
                    max_overlap = 0.0
                else:
                    max_overlap = overlap.max()

                # Check min and max overlap constraints
                if min_iou is not None and max_overlap < min_iou:
                    continue

                if max_iou is not None and max_overlap > max_iou:
                    continue

                # Crop image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]

                # Keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # Mask in boxes where center is inside the crop
                mask_left_top = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
                mask_right_bottom = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
                mask = mask_left_top * mask_right_bottom

                if not np.any(mask):
                    continue

                # Filter valid boxes and labels
                current_boxes = boxes[mask, :].copy()
                current_labels = labels[mask]

                # Clip boxes to crop boundaries
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                current_boxes[:, :2] -= rect[:2]  # Adjust to new coordinate system

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels


class Expand(object):
    """Expand the image canvas (zoom out)."""

    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, channels = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)

        expand_image = np.zeros(
            (int(height * ratio), int(width * ratio), channels),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
        int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels


class RandomMirror(object):
    """Horizontal flip."""

    def __call__(self, image, boxes, labels):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, labels


class RandomInvert(object):
    """Vertical flip (Inverted)."""

    def __call__(self, image, boxes, labels):
        height, _, _ = image.shape
        if random.randint(2):
            image = image[::-1, :]
            boxes = boxes.copy()
            boxes[:, 1::2] = height - boxes[:, 3::-2]
        return image, boxes, labels


class RandomRotate(object):
    """Rotate image 90 degrees."""

    def __call__(self, image, boxes, labels):
        height, width, _ = image.shape
        if random.randint(2):
            image_b = image[:, :, 0].T
            image_g = image[:, :, 1].T
            image_r = image[:, :, 2].T
            image = np.stack([image_b, image_g, image_r], axis=2)

            boxes = boxes.copy()
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            # Rotate coordinates
            boxes = np.stack([y1, x1, y2, x2], axis=1)

        return image, boxes, labels


class SwapChannels(object):
    """Reorder image channels."""

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    """Apply a sequence of photometric distortions."""

    def __init__(self):
        self.distortions = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)

        # Apply distortions in random order
        if random.randint(2):
            distort = Compose(self.distortions[:-1])
        else:
            distort = Compose(self.distortions[1:])

        im, boxes, labels = distort(im, boxes, labels)
        return im, boxes, labels


class SSDAugmentation(object):
    """Standard SSD Data Augmentation Pipeline."""

    def __init__(self, size=512, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
        self.mean = mean
        self.size = size
        self.std = std
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            Normalize(self.mean, self.std)
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)


class ColorAugmentation(object):
    """Augmentation Pipeline with only color distortions (no cropping/expansion)."""

    def __init__(self, size=512, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
        self.mean = mean
        self.size = size
        self.std = std
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            PhotometricDistort(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            Normalize(self.mean, self.std)
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)
