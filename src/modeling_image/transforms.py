"""Deterministic preprocessing (letterbox onto a white canvas, default 256x768) and optional
mild augmentations for the Kangaroo images. No transform distorts the image by resizing it to a
square, and no content is cropped.

`CONTENT_FIT_FRACTION < 1.0` leaves a small white margin around the scaled image (90% fill).
The margin was chosen heuristically so that the optional mild augmentations
(`build_augmentation()`: rotation <= 2 degrees, translation <= 2%, scaling 0.95-1.05) rarely move
content off the canvas; it is not a mathematical guarantee for every input shape. The
configuration used in the paper applies NO random augmentation (deterministic letterbox +
ImageNet normalisation only).
"""

from __future__ import annotations

from PIL import Image
from torchvision import transforms as T

CANVAS_HEIGHT = 256
CANVAS_WIDTH = 768
CONTENT_FIT_FRACTION = 0.90  # safety margin for augmentations (see module docstring)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class LetterboxToCanvas:
    """Scales the image proportionally (keeping the aspect ratio) so that it fits entirely within
    `fit_fraction * (height, width)` and centres it on a white `height x width` canvas.
    Nothing is cropped and no content is cut off."""

    def __init__(
        self, height: int = CANVAS_HEIGHT, width: int = CANVAS_WIDTH, fit_fraction: float = CONTENT_FIT_FRACTION
    ) -> None:
        self.height = height
        self.width = width
        self.fit_fraction = fit_fraction

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        target_w = self.width * self.fit_fraction
        target_h = self.height * self.fit_fraction
        scale = min(target_w / img.width, target_h / img.height)
        new_w = max(1, round(img.width * scale))
        new_h = max(1, round(img.height * scale))
        resized = img.resize((new_w, new_h), Image.BILINEAR)

        canvas = Image.new("RGB", (self.width, self.height), (255, 255, 255))
        offset_x = (self.width - new_w) // 2
        offset_y = (self.height - new_h) // 2
        canvas.paste(resized, (offset_x, offset_y))
        return canvas


class GrayscaleTo3Channel:
    """Converts to grayscale and replicates it into 3 identical channels (optional variant)."""

    def __call__(self, img: Image.Image) -> Image.Image:
        return img.convert("L").convert("RGB")


def build_augmentation() -> T.Compose:
    """Mild augmentations (optional, not used in the paper): rotation <= 2 degrees, translation
    <= 2%, scaling 0.95-1.05, brightness/contrast +-10%. `fill=255` keeps the background white.
    Applied AFTER the letterbox, within the `CONTENT_FIT_FRACTION` margin."""
    return T.Compose(
        [
            T.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.95, 1.05), fill=255),
            T.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    )


def build_transform(*, grayscale: bool, augment: bool) -> T.Compose:
    """Builds the full transform: letterbox -> [grayscale->3ch] -> [mild augmentations]
    -> ToTensor -> ImageNet normalisation."""
    steps: list = [LetterboxToCanvas()]
    if grayscale:
        steps.append(GrayscaleTo3Channel())
    if augment:
        steps.append(build_augmentation())
    steps.append(T.ToTensor())
    steps.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return T.Compose(steps)
