import numpy as np
from PIL import Image, ImageEnhance
import torch

TARGET_SIZE = (224, 224)

def resize_crop_and_mask(crop, mask, target_size=TARGET_SIZE):
    """Resize the image and its matching binary mask."""
    mask = np.asarray(mask)

    if mask.shape != (crop.height, crop.width):
        raise ValueError("Image and mask dimensions do not match.")

    resized_image = crop.convert("RGB").resize(
        target_size,
        resample=Image.Resampling.BILINEAR,
    )

    mask_image = Image.fromarray(
        (mask > 0).astype(np.uint8) * 255
    )

    resized_mask = np.array(
        mask_image.resize(
            target_size,
            resample=Image.Resampling.NEAREST,
        )
    )

    return resized_image, (resized_mask > 0).astype(np.uint8)


def augment_training_sample(image, mask, rng):
    """Apply matching spatial augmentation to an image–mask pair."""
    if mask.shape != (image.height, image.width):
        raise ValueError("Image and mask dimensions do not match.")

    image = image.copy()
    mask = mask.copy()

    flipped = rng.random() < 0.5

    if flipped:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask = np.fliplr(mask).copy()

    brightness = rng.uniform(0.9, 1.1)
    contrast = rng.uniform(0.9, 1.1)

    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)

    return image, mask, {
        "flipped": flipped,
        "brightness": brightness,
        "contrast": contrast,
    }
    
    
IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406], dtype=torch.float32
).view(3, 1, 1)

IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225], dtype=torch.float32
).view(3, 1, 1)


def image_to_tensor(image):
    # RGB image: [height, width, channels], values 0–1.
    pixels = np.array(image.convert("RGB"), dtype=np.float32) / 255.0

    # Rearrange to [channels, height, width].
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).contiguous()

    # Normalize each channel.
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def mask_to_tensor(mask):
    # Boolean mask: True = annotated damage.
    # No ImageNet normalization is applied.
    return torch.from_numpy(
        np.array(mask, dtype=bool, copy=True)
    )