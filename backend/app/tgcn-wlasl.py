import torch
from tgcn_model import GCN_muti_att
from configs import Config
# from transformers import AutoModel, AutoModelForSequenceClassification, PreTrainedModel

checkpoint_path = "/u50/quyumr/rtsl/backend/app/checkpoints/asl1000/pytorch_model.bin"
config_path = "/u50/quyumr/rtsl/backend/app/checkpoints/asl1000/config.ini"
config = Config(config_path)

# initialize model
model = GCN_muti_att(
    input_feature=config.num_samples * 2,  # 50 * 2 = 100
    hidden_feature=config.hidden_size,      # 256
    num_class=2000,
    p_dropout=config.drop_p,               # 0.3
    num_stage=config.num_stages            # 24
)

# Load weights
checkpoint = torch.load(checkpoint_path, map_location='cpu', )
state_dict = checkpoint.get('state_dict', checkpoint)
model.load_state_dict(state_dict, strict=False)
model.eval()


