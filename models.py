import os
import torch
import torch.nn as nn
import torchvision.models as models
import timm
from SHViT import build_shvit, load_finetuned_monster, load_finetuned_triple_monster
os.environ['TORCH_HOME'] = 'model_cache'

class L0Mask(nn.Module):
    def __init__(self, num_heads, temp=2./3., limit_l=-0.1, limit_r=1.1):
        super().__init__()
        self.num_heads = num_heads
        self.temp = temp
        self.limit_l = limit_l
        self.limit_r = limit_r
        self.log_alpha = nn.Parameter(torch.Tensor(num_heads))

    def forward(self, training=True):
        if training:
            u = torch.rand_like(self.log_alpha)
            s = torch.sigmoid((self.log_alpha + torch.log(u) - torch.log(1 - u)) / self.temp)
            s_bar = s * (self.limit_r - self.limit_l) + self.limit_l
            z = torch.clamp(s_bar, min=0.0, max=1.0)
        else:
            s = torch.sigmoid(self.log_alpha)
            s_bar = s * (self.limit_r - self.limit_l) + self.limit_l
            z = torch.clamp(s_bar, min=0.0, max=1.0)
            z = (z > 0.0).float()
        return z

class PrunableAttention(nn.Module):
    def __init__(self, original_attn):
        super().__init__()
        self.num_heads = original_attn.num_heads
        self.scale = original_attn.scale
        self.qkv = original_attn.qkv
        self.attn_drop = original_attn.attn_drop
        self.proj = original_attn.proj
        self.proj_drop = original_attn.proj_drop
        self.l0_mask = L0Mask(self.num_heads)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v)
        mask = self.l0_mask(training=self.training).view(1, self.num_heads, 1, 1)
        x = x * mask
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def get_model():
    # Get environment variables
    model_name = os.environ.get('MODEL_NAME', 'resnet50')
    checkpoint_path = os.environ.get('CHECKPOINT_PATH', '')
    is_pruned = os.environ.get('IS_PRUNED', 'false').lower() == 'true'
    print(f"Requesting Model: {model_name}")

    # Logic to handle local checkpoints
    use_pretrained = True
    kwargs = {}
    
    if checkpoint_path and os.path.exists(checkpoint_path) and not is_pruned:
        print(f"Loading local weights from: {checkpoint_path}")
        use_pretrained = False # Disable download
        kwargs['checkpoint_path'] = checkpoint_path

    if 'shvit' in model_name:
        if not checkpoint_path:
            raise ValueError(f"SHViT models require CHECKPOINT_PATH env var.")
        
        try:
            model = build_shvit(model_name, checkpoint_path)
            model.cuda()
            model.eval()

            if hasattr(model, 'head') and hasattr(model.head, 'fuse'):
                print("Fusing BN_Linear head into standard Linear layer for evaluation compatibility...")
                model.head = model.head.fuse()
            
            return model
        except Exception as e:
            print(f"Error loading SHViT: {e}")
            raise e
    elif 'double' in model_name:
        try:
            if not checkpoint_path or not os.path.exists(checkpoint_path):
                raise ValueError(f"Valid CHECKPOINT_PATH required for doublehead. Got: {checkpoint_path}")
            model = load_finetuned_monster(checkpoint_path)
            
            if hasattr(model, 'head') and hasattr(model.head, 'fuse'):
                print("Fusing BN_Linear head into standard Linear layer for evaluation compatibility...")
                model.head = model.head.fuse()

            return model.cuda().eval()
            
        except Exception as e:
            print(f"Error loading DoubleHead: {e}")
            raise e
    elif 'triple' in model_name:
        try:
            if not checkpoint_path or not os.path.exists(checkpoint_path):
                raise ValueError(f"Valid CHECKPOINT_PATH required for doublehead. Got: {checkpoint_path}")
            model = load_finetuned_triple_monster(checkpoint_path)
            
            if hasattr(model, 'head') and hasattr(model.head, 'fuse'):
                print("Fusing BN_Linear head into standard Linear layer for evaluation compatibility...")
                model.head = model.head.fuse()

            return model.cuda().eval()
            
        except Exception as e:
            print(f"Error loading DoubleHead: {e}")
            raise e
    elif 'vit' in model_name or 'patch' in model_name:
        try:
            print("Trying to download model!----")
            # model = timm.create_model(model_name, pretrained=(not is_pruned), **kwargs)

            model = timm.create_model(model_name, pretrained=(False), **kwargs)
            print("Model downloaded!----")
            if is_pruned:
                print("Patching ViT with PrunableAttention layers...")
                for block in model.blocks:
                    block.attn = PrunableAttention(block.attn)
                
                if checkpoint_path and os.path.exists(checkpoint_path):
                    print(f"Loading pruned weights from: {checkpoint_path}")
                    state_dict = torch.load(checkpoint_path, map_location='cuda')
                    model.load_state_dict(state_dict)
                else:
                    raise ValueError("IS_PRUNED is true but CHECKPOINT_PATH is invalid.")
            
            '''elif checkpoint_path and os.path.exists(checkpoint_path):
                state_dict = torch.load(checkpoint_path, map_location='cuda')
                model.load_state_dict(state_dict)'''
            
            return model.cuda().eval()
        except Exception as e:
            print(f"Error loading timm model {model_name}: {e}")
            raise e
    else:
        # Fallback for standard torchvision models (ResNet)
        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            model = models.resnet50(weights=weights)
            model.eval()
            return model
        except:
             model = models.resnet50(pretrained=True)
             model.eval()
             return model
def generate_torchvision_models(list_of_models):
    list_of_models['torchvision-model-0'] = lambda num_classes : torchvision.models.densenet121(
        DenseNet121_Weights(DenseNet121_Weights.IMAGENET1K_V1)
    )
    list_of_models['torchvision-model-1'] = lambda num_classes : torchvision.models.densenet201(
        DenseNet201_Weights(DenseNet201_Weights.IMAGENET1K_V1)
    )
    list_of_models['torchvision-model-2'] = lambda num_classes : torchvision.models.densenet161(
        DenseNet161_Weights(DenseNet161_Weights.IMAGENET1K_V1)
    )
    list_of_models['torchvision-model-3'] = lambda num_classes : torchvision.models.densenet169(
        DenseNet169_Weights(DenseNet169_Weights.IMAGENET1K_V1)
    )
    list_of_models['torchvision-model-4'] = lambda num_classes : torchvision.models.resnet152(
        ResNet152_Weights(ResNet152_Weights.IMAGENET1K_V1)
    )
    list_of_models['torchvision-model-5'] = lambda num_classes : torchvision.models.resnet101(
        ResNet101_Weights(ResNet101_Weights.IMAGENET1K_V1)
    )  
    list_of_models['torchvision-model-6'] = lambda num_classes : torchvision.models.resnext50_32x4d(
        ResNeXt50_32X4D_Weights(ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
    )
    list_of_models['torchvision-model-7'] = lambda num_classes : torchvision.models.resnext101_32x8d(
        ResNeXt101_32X8D_Weights(ResNeXt101_32X8D_Weights.IMAGENET1K_V1)
    ) 
    list_of_models['torchvision-model-8'] = lambda num_classes : torchvision.models.wide_resnet101_2(
        Wide_ResNet101_2_Weights(Wide_ResNet101_2_Weights.IMAGENET1K_V1)
    )
    list_of_models['torchvision-model-9'] = lambda num_classes : torchvision.models.wide_resnet50_2(
        Wide_ResNet50_2_Weights(Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    )
    list_of_models['torchvision-model-10'] = lambda num_classes : torchvision.models.regnet_x_32gf(
        RegNet_X_32GF_Weights(RegNet_X_32GF_Weights.IMAGENET1K_V1)
    )
    list_of_models['torchvision-model-11'] = lambda num_classes : torchvision.models.regnet_x_16gf(
        RegNet_X_16GF_Weights(RegNet_X_16GF_Weights.IMAGENET1K_V1)
    )
    list_of_models['torchvision-model-12'] = lambda num_classes : torchvision.models.resnet50(
        ResNet50_Weights(ResNet50_Weights.IMAGENET1K_V1)
    )
    
model_list = {}

generate_torchvision_models(model_list)
