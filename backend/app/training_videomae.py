import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import numpy as np
import json
import os
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sys import exit
from training import ASLVideoTensorDataset

from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification, TrainingArguments, Trainer
# import pytorchvideo.data
from pytorchvideo.data import LabeledVideoDataset

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)

DIR = "/u50/quyumr/archive"
TRAIN_DIR = "/u50/quyumr/archive/train_output_json_video_avi"
TEST_DIR = "/u50/quyumr/archive/test_output_json_video_avi"
VALIDATION_DIR = "/u50/quyumr/archive/validation_output_json_video_avi"
JSON_PATH = "/u50/quyumr/archive/WLASL_v0.3.json"

# 

def generate_class_integer_mappings(directory: str, mappings_exist: bool = False, json_path: str = JSON_PATH, max_classes = None) :
    """
    Generate integer -> class and class -> integer mapping and load them.
    If mappings already exist, they should not be re-generated
    directory: is the output directory if mappings exist; is the input 
        directory when mappings do exist
    """
    if mappings_exist :
        with open(f"{directory}/idx_to_class.json", 'r') as f:
            idx_to_class = json.load(f)
        with open(f"{directory}/class_to_idx.json", 'r') as f:
            class_to_idx = json.load(f)
    else :
        with open(json_path, "r") as f:
            data = json.load(f)

        # get glosses and filter them down if desired
        all_glosses = sorted([entry["gloss"] for entry in data])
        if max_classes is not None:
            glosses = all_glosses[:max_classes]  # only keep the first max_classes glosses
        else:
            glosses = all_glosses

        class_to_idx = {g: i for i, g in enumerate(glosses)}
        with open(f"{directory}/class_to_idx.json", 'w') as f :
            json.dump(class_to_idx, f, indent=2)
        print("{class: idx} dictionary created")

        idx_to_class = {i: g for i, g in enumerate(glosses)}
        with open(f"{directory}/idx_to_class.json", 'w') as f :
            json.dump(idx_to_class, f, indent=2)
        print("{idx: class} dictionary created")

    return idx_to_class, class_to_idx

idx_to_class, class_to_idx = generate_class_integer_mappings(DIR, mappings_exist=True, json_path=JSON_PATH)
exit()

# model init
model_ckpt = "MCG-NJU/videomae-base"
image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
model = VideoMAEForVideoClassification.from_pretrained(
    model_ckpt,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)

mean = image_processor.image_mean
std = image_processor.image_std
if "shortest_edge" in image_processor.size:
    height = width = image_processor.size["shortest_edge"]
else:
    height = image_processor.size["height"]
    width = image_processor.size["width"]
resize_to = (height, width)

num_frames_to_sample = model.config.num_frames
sample_rate = 4
fps = 30
clip_duration = num_frames_to_sample * sample_rate / fps

train_transform = Compose(
    [
        UniformTemporalSubsample(num_frames_to_sample),
        Lambda(lambda x: x / 255.0),
        Normalize(mean, std),
        RandomShortSideScale(min_size=256, max_size=320),
        RandomCrop(resize_to),
        RandomHorizontalFlip(p=0.5),
    ]
)

test_dataset = LabeledVideoDataset(
    data_path="/u50/quyumr/archive/test_output_json_video_avi/",
    clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
    decode_audio=False,
    transform=train_transform,
)
