import random
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
from preprocess_utils import *


class CarDDCropDataset(Dataset):
    def __init__(self, manifest, annotation_data, split):
        self.samples = (
            manifest[manifest["split"] == split]
            .reset_index(drop=True)
            .copy()
        )
        self.split = split

        self.coco = COCO()
        self.coco.dataset = annotation_data[split]
        self.coco.createIndex()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples.iloc[index]

        try:
            annotation = self.coco.anns[
                int(sample["annotation_id"])
            ]

            box = tuple(
                int(sample[key])
                for key in ["left", "top", "right", "bottom"]
            )
            left, top, right, bottom = box

            # Extract the image crop.
            with Image.open(sample["source_path"]) as image:
                crop = image.convert("RGB").crop(box)

            # Extract the selected instance's matching mask.
            full_mask = self.coco.annToMask(annotation)
            mask = full_mask[top:bottom, left:right]

            # Apply the same spatial resizing to both.
            crop, mask = resize_crop_and_mask(crop, mask)

            if not mask.any():
                raise ValueError("Empty mask after resizing")

            # Random augmentation is enabled only for training.
            if self.split == "train":
                crop, mask, _ = augment_training_sample(
                    crop, mask, rng=random
                )

            return {
                "image": image_to_tensor(crop),
                "label": torch.tensor(
                    int(sample["class_index"]),
                    dtype=torch.long,
                ),
                "mask": mask_to_tensor(mask),
                "sample_id": sample["sample_id"],
                "image_id": int(sample["image_id"]),
                "annotation_id": int(sample["annotation_id"]),
            }

        except Exception as error:
            raise RuntimeError(
                f"Failed to prepare {sample['sample_id']}: {error}"
            ) from error