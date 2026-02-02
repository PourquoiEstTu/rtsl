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

TRAIN_DIR = "/u50/quyumr/archive/train_output_json_video_avi"
TEST_DIR = "/u50/quyumr/archive/test_output_json_video_avi"
VALIDATION_DIR = "/u50/quyumr/archive/validation_output_json_video_avi"
JSON_PATH = "/u50/quyumr/archive/WLASL_v0.3.json"

# 

def generate_class_integer_mappings(mappings_exist: bool = False, json_path: str = JSON_PATH) :
    """
    Generate integer -> class and class -> integer mapping and load them.
    If mappings already exist, they should not be re-generated
    """
    if not mappings_exist :
        train = ASLVideoTensorDataset(TRAIN_DIR, json_path, split="train", max_classes=None)
        test = ASLVideoTensorDataset(TEST_DIR, json_path, split="test", max_classes=None)
        validation = ASLVideoTensorDataset(VALIDATION_DIR, json_path, split="val", max_classes=None)

    with open(f"{TRAIN_DIR}/idx_to_class.json", 'r') as f:
        train_idx_to_class = json.load(f)
    with open(f"{TRAIN_DIR}/class_to_idx.json", 'r') as f:
        train_class_to_idx = json.load(f)

    with open(f"{TEST_DIR}/idx_to_class.json", 'r') as f:
        test_idx_to_class = json.load(f)
    with open(f"{TEST_DIR}/class_to_idx.json", 'r') as f:
        test_class_to_idx = json.load(f)

    with open(f"{VALIDATION_DIR}/idx_to_class.json", 'r') as f:
        validation_idx_to_class = json.load(f)
    with open(f"{VALIDATION_DIR}/class_to_idx.json", 'r') as f:
        validation_class_to_idx = json.load(f)
    return train_idx_to_class, train_class_to_idx, test_idx_to_class, test_class_to_idx, validation_idx_to_class, validation_class_to_idx

train_idx_to_class, train_class_to_idx, test_idx_to_class, test_class_to_idx, validation_idx_to_class, validation_class_to_idx = generate_class_integer_mappings(mappings_exist=True, json_path=JSON_PATH)
# exit()

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
