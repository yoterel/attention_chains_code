import torch
import torch.nn as nn
import sys
DINO_V2_REPO_PATH = "../dinov2"
sys.path.append(DINO_V2_REPO_PATH)
try:
    from dinov2.layers.attention import MemEffAttention
except ImportError:
    print("Could not import dinov2. Verify that the DINO_V2_REPO_PATH points to the dinov2 repository.")
from xformers.ops import unbind

# Define a recursive monkey patching function
def replace_attn(module):
    module_output = module
    if isinstance(module, MemEffAttention):
        module_output = CustomAttention(module.num_heads,
                                        module.scale, 
                                        module.qkv,
                                        module.proj)
    for name, child in module.named_children():
        module_output.add_module(name, replace_attn(child))
    del module

    return module_output

class CustomAttention(nn.Module):
    def __init__(self, num_heads, scale, qkv, proj):
        super().__init__()
        self.num_heads = num_heads
        self.scale = scale
        dim = int((self.scale**-2) * self.num_heads)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)
        # Dummy layers for hooking
        self.dummy_before_softmax = DummyLayer()
        self.dummy_after_softmax = DummyLayer()
        # initialize
        self.qkv.load_state_dict(qkv.state_dict())
        self.proj.load_state_dict(proj.state_dict())
    
    def forward(self, x, attn_bias=None):

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = self.memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def memory_efficient_attention(self, query, key, value, attn_bias=None):
        scale = 1.0 / query.shape[-1] ** 0.5
        query = query * scale
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attn = query @ key.transpose(-2, -1)
        if attn_bias is not None:
            attn = attn + attn_bias
        attn = self.dummy_before_softmax(attn)
        attn = attn.softmax(-1)
        attn = self.dummy_after_softmax(attn)
        attn = nn.functional.dropout(attn, 0.0)
        attn = attn @ value
        return attn.transpose(1, 2).contiguous()


class DummyLayer(nn.Module):
    def forward(self, x):
        return x


def load_pretrained(arch):
    torch.hub.set_dir("./models/dinov2")
    BACKBONE_SIZE = arch.split("_")[1]
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    backbone_arch = backbone_archs[BACKBONE_SIZE]
    backbone_name = f"dinov2_{backbone_arch}_reg"
    print(f"Loading DINOv2 model with backbone: {backbone_name}")
    model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    return model