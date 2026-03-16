import json
import torch

from configs import Config
from tgcn_model import GCN_muti_att

def get_model(parent_dir, num_classes):
    checkpoint_path = f"{parent_dir}/rtsl/backend/models/checkpoints/asl{num_classes}/pytorch_model.bin"
    config_path = f"{parent_dir}/rtsl/backend/models/configs/asl{num_classes}.ini"
    config = Config(config_path)
    
    # initalize model
    model = GCN_muti_att(
        input_feature=config.num_samples * 2,  # 50 * 2 = 100
        hidden_feature=config.hidden_size,      # 256
        num_class= num_classes,
        p_dropout=config.drop_p,               # 0.3
        num_stage=config.num_stages            # 24
    )

    # load weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu', )
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval() # put in evaluation mode

    return model

def get_labels(parent_dir, num_classes):
    labels_path = f"{parent_dir}/rtsl/backend/data_splits/{num_classes}/class_to_idx.json"
    with open(labels_path, 'r') as f:
        labels = [w for w in json.load(f)]
    return labels
