from load_from_huggingface import load_tgcn_from_hf
import torch
from tgcn_model import GCN_muti_att
from configs import Config
from huggingface_hub import hf_hub_download

# Load the model
repo_id = "your-username/tgcn-wlasl"  # Replace with your repo
model, config = load_tgcn_from_hf(repo_id, model_size="asl2000")

# Model is ready for inference
model.eval()

# Download and load checkpoint
checkpoint_path = hf_hub_download(
    repo_id="your-username/tgcn-wlasl",
    filename="checkpoints/asl2000/pytorch_model.bin"
)

config_path = hf_hub_download(
    repo_id="your-username/tgcn-wlasl",
    filename="checkpoints/asl2000/config.ini"
)

# Load config
config = Config(config_path)

# Initialize model
model = GCN_muti_att(
    input_feature=config.num_samples * 2,  # 50 * 2 = 100
    hidden_feature=config.hidden_size,      # 256
    num_class=2000,
    p_dropout=config.drop_p,               # 0.3
    num_stage=config.num_stages            # 24
)

# Load weights
checkpoint = torch.load(checkpoint_path, map_location='cpu')
state_dict = checkpoint.get('state_dict', checkpoint)
model.load_state_dict(state_dict, strict=False)
model.eval()

# Inference
# Input shape: (batch_size, 55, num_samples * 2)
# Example: (1, 55, 100) for 50 frames with x,y coordinates
x = torch.randn(1, 55, 100)  # Example input
output = model(x)
predictions = torch.softmax(output, dim=1)
