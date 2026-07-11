import torch
import torch.nn as nn
import os.path as osp
import os
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler
import numpy as np
from tqdm import tqdm
from dassl.utils import mkdir_if_missing
import pandas as pd
import matplotlib.pyplot as plt
from trainers.classification.base_learner import VLBaseLearner
import torch.nn.functional as F

# Import standard CLIP models
from models.clip import clip
from models.clip.model import convert_weights
from .coop import load_clip_to_cpu

# Import MedCLIP if available
try:
    from .coop_medclip import load_medclip_to_cpu
    HAS_MEDCLIP = True
except ImportError:
    HAS_MEDCLIP = False

# Import BioMedCLIP if available
try:
    from .coop_biomedclip import load_clip_to_cpu as load_biomedclip_to_cpu
    from models.open_clip import get_tokenizer
    HAS_BIOMEDCLIP = True
except ImportError:
    HAS_BIOMEDCLIP = False

# Normalized dataset names to templates mapping
CUSTOM_TEMPLATES = {
    "digestpath": "An H&E image patch of {} tissue.",
    "pannuke": "An H&E image patch of {} skin tissue.",
    "covid": "A chest X-ray image of {} patient",
    "rsna18": "A chest X-ray image of {} patient",
    "kathercolon": "An H&E image of {}.",  
}

@TRAINER_REGISTRY.register()
class ZeroshotCLIP(VLBaseLearner):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        model_name = cfg.MODEL.NAME
        
        print(f"Loading {model_name} (backbone: {cfg.MODEL.BACKBONE.NAME})")
        
        # Choose the right loader based on model type
        if model_name == "medclip" and HAS_MEDCLIP:
            self.model_type = "medclip"
            clip_model = load_medclip_to_cpu(cfg)
        elif model_name == "biomedclip" and HAS_BIOMEDCLIP:
            self.model_type = "biomedclip"
            clip_model = load_biomedclip_to_cpu(cfg)
        else:
            # Default to standard CLIP, PLIP, or QuiltNet
            self.model_type = model_name.lower()  # "plip", "quiltnet", or "clip"
            clip_model = load_clip_to_cpu(cfg)
            
        clip_model.to(self.device)
        
        # Get template for the dataset - normalize dataset name to lowercase
        dataset_name = cfg.DATASET.NAME.lower()
        temp = CUSTOM_TEMPLATES.get(dataset_name, "An image of {}.")
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        
        # Process prompts based on model type
        if self.model_type == "medclip":
            # MedCLIP text encoding
            tokenized_prompts = clip_model.text_model.tokenizer(
                prompts, 
                padding='max_length', 
                max_length=25, 
                truncation=True, 
                return_tensors='pt'
            )
            tokenized_prompts = {k: v.to(self.device) for k, v in tokenized_prompts.items()}
            
            with torch.no_grad():
                # Process through text model
                output = clip_model.text_model.model(**tokenized_prompts)
                last_hidden_states = torch.stack([
                    output['hidden_states'][1], 
                    output['hidden_states'][2], 
                    output['hidden_states'][-1]
                ])
                # Pooling operation
                embed = last_hidden_states.permute(1,0,2,3).mean(2).mean(1)
                # Project to get final text features
                text_features = clip_model.text_model.projection_head(embed)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
            # Store for later use
            self.tokenized_prompts = tokenized_prompts
        
        elif self.model_type == "biomedclip":
            # BioMedCLIP text encoding using the original approach
            tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
            context_length = 256
            tokenized_prompts = tokenizer(prompts, context_length=context_length)
            tokenized_prompts = tokenized_prompts.to(self.device)
            
            # Creating attention mask
            attention_mask = (tokenized_prompts != clip_model.text.config.pad_token_id).long()
            attention_mask = attention_mask.to(self.device)
            
            with torch.no_grad():
                # Process text features
                x = clip_model.text.transformer(input_ids=tokenized_prompts, attention_mask=attention_mask)
                pooled = clip_model.text.pooler(x, attention_mask)
                text_features = clip_model.text.proj(pooled)
                text_features = F.normalize(text_features, dim=-1)
                
            # Store for later use
            self.tokenized_prompts = tokenized_prompts
            self.attention_mask = attention_mask
            
        else:
            # Standard CLIP tokenization
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
            tokenized_prompts = tokenized_prompts.to(self.device)
            
            with torch.no_grad():
                text_features = clip_model.encode_text(tokenized_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
            # Store tokenized prompts
            self.tokenized_prompts = tokenized_prompts
                
        # Store text features and model
        self.text_features = text_features
        self.clip_model = clip_model
        
    def model_inference(self, image):
        # Process image features based on model type
        with torch.no_grad():
            if self.model_type == "medclip":
                # MedCLIP image encoding
                image_features = self.clip_model.vision_model(image)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
            elif self.model_type == "biomedclip":
                # BioMedCLIP image encoding 
                image_features = self.clip_model.visual(image)
                image_features = F.normalize(image_features, dim=-1)
                
            else:
                # Standard CLIP image encoding
                image_features = self.clip_model.encode_image(image)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
            # Compute logits
            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ self.text_features.t()
            
        return logits, image_features, self.text_features