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
import shutil
# from training import ASLVideoTensorDataset
# 
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification, TrainingArguments, Trainer

from transformers import TrainingArguments, Trainer
import pytorchvideo
from pytorchvideo.data import LabeledVideoDataset
import evaluate

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
# print(idx_to_class)
# exit()

def ucf_file_hierarchy_conversion(directory: str, train_dir: str, test_dir: str, val_dir: str, json_path: str = JSON_PATH) :
    """
        Turns the training, test, and val videos into the UCF101 dataset's.
        The hierarchy looks like this:
        train/
           class1/
               video_1.mp4
               video_2.mp4
               ...
           class2/
               video_1.mp4
               video_2.mp4
               ...
           ...
        val/
           class1/
               video_1.mp4
               video_2.mp4
               ...
           ...
        test/
           class1/
               video_1.mp4
               video_2.mp4
               ...
        directory: A path to a directory that has the training, test, and val
          videos in separate directories as specified in the remaining 
          function parameters
    """
    if not os.path.exists(json_path) :
        raise IOError(f"JSON file does not exist at {json_path}")
    if not os.path.exists(f"{directory}/ucf_hierarchy") :
        os.makedirs(f"{directory}/ucf_hierarchy")
        os.makedirs(f"{directory}/ucf_hierarchy/train")
        os.makedirs(f"{directory}/ucf_hierarchy/test")
        os.makedirs(f"{directory}/ucf_hierarchy/val")
    with open(json_path, 'r') as f :
        data = json.load(f)
    for entry in data : 
        print(f"Moving videos for gloss [{entry['gloss']}]")
        current_train_gloss_dir = f"{directory}/ucf_hierarchy/train/{entry['gloss']}"
        current_test_gloss_dir = f"{directory}/ucf_hierarchy/test/{entry['gloss']}"
        current_val_gloss_dir = f"{directory}/ucf_hierarchy/val/{entry['gloss']}"
        os.makedirs(current_train_gloss_dir, exist_ok=True)
        os.makedirs(current_test_gloss_dir,  exist_ok=True)
        os.makedirs(current_val_gloss_dir,   exist_ok=True)
        for instance in entry["instances"] :
            if instance["split"] == "train" and os.path.exists(f"{TRAIN_DIR}/{instance['video_id']}.avi") :
                shutil.copy2(f"{train_dir}/{instance['video_id']}.avi", current_train_gloss_dir)
            if instance["split"] == "test" and os.path.exists(f"{TEST_DIR}/{instance['video_id']}.avi"):
                shutil.copy2(f"{test_dir}/{instance['video_id']}.avi", current_test_gloss_dir)
            if instance["split"] == "val" and os.path.exists(f"{VALIDATION_DIR}/{instance['video_id']}.avi") :
                shutil.copy2(f"{val_dir}/{instance['video_id']}.avi", current_val_gloss_dir)
            
# ucf_file_hierarchy_conversion(DIR, TRAIN_DIR, TEST_DIR, VALIDATION_DIR)
# exit()

def collate_fn(examples):
    # permute to (num_frames, num_channels, height, width)
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

# model init
print("===================================================")
model_ckpt = "MCG-NJU/videomae-base"
image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
model = VideoMAEForVideoClassification.from_pretrained(
    model_ckpt,
    label2id=class_to_idx,
    id2label=idx_to_class,
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

print("===================================================\n")

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

test_dataset = pytorchvideo.data.LabeledVideoDataset(
    labeled_video_paths="/u50/quyumr/archive/test_output_json_video_avi/",
    clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
    decode_audio=False,
    transform=train_transform,
)
# print(test_dataset.__next__())

model_name = model_ckpt.split("/")[-1]
new_model_name = f"{model_name}-finetuned-on-test"
num_epochs = 4
batch_size = 10

# args = TrainingArguments(
#     new_model_name,
#     remove_unused_columns=False,
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     learning_rate=5e-5,
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     warmup_steps=1,
#     logging_steps=10,
#     load_best_model_at_end=True,
#     metric_for_best_model="accuracy",
#     push_to_hub=False,
#     max_steps=(test_dataset.num_videos // batch_size) * num_epochs,
# )
# 
# trainer = Trainer(
#     model,
#     args,
#     train_dataset=test_dataset,
#     eval_dataset=test_dataset,
#     processing_class=image_processor,
#     compute_metrics=compute_metrics,
#     data_collator=collate_fn,
# )
# train_results = trainer.train()
