import torch
from tgcn_model import GCN_muti_att
from configs import Config
import json
# from transformers import AutoModel, AutoModelForSequenceClassification, PreTrainedModel

def generate_class_integer_mappings(directory: str, mappings_exist: bool, json_path: str, max_classes = None) :
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

idx_to_class, class_to_idx = generate_class_integer_mappings("/home/pourquoi/repos/rtsl/backend/app/", mappings_exist=False, json_path="/home/pourquoi/repos/rtsl/backend/app/all.json")

# checkpoint_path = "/home/pourquoi/repos/rtsl/backend/app/checkpoints/asl1000/pytorch_model.bin"
# config_path =     "/home/pourquoi/repos/rtsl/backend/app/checkpoints/asl1000/config.ini"
# config = Config(config_path)
#
# # initialize model
# model = GCN_muti_att(
#     input_feature=config.num_samples * 2,  # 50 * 2 = 100
#     hidden_feature=config.hidden_size,      # 256
#     num_class=2000,
#     p_dropout=config.drop_p,               # 0.3
#     num_stage=config.num_stages            # 24
# )
#
# # Load weights
# checkpoint = torch.load(checkpoint_path, map_location='cpu', )
# state_dict = checkpoint.get('state_dict', checkpoint)
# model.load_state_dict(state_dict, strict=False)
# model.eval()
