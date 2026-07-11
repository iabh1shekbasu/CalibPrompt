import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from models.clip import clip
import torch.nn.functional as F

from tqdm import tqdm
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from trainers.classification.base_learner import VLBaseLearner
from models.clip import clip
from models.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .losses import LossRegistry

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    model_name = cfg.MODEL.NAME if hasattr(cfg.MODEL, 'NAME') else 'clip'
    
    if model_name == 'clip':
        # Original CLIP loading
        print("\n\n\nUsing CLIP\n\n\n")
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)
    elif model_name == 'plip':
        # PLIP path
        print("\n\n\nUsing PLIP\n\n\n")
        model_path = osp.join(cfg.MODEL_ROOT, "plip", 'plip_vit_b32.pt')
    elif model_name == 'quiltnet':
        # QuiltNet path
        print("\n\n\nUsing QuiltNet\n\n\n")
        model_path = osp.join(cfg.MODEL_ROOT, "quiltnet", 'quiltnet_b32.pt')
    else:
        raise ValueError(f"Model '{model_name}' not supported. Choose 'clip' or 'plip' or 'quiltnet")
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {
        "trainer": 'CoOp',
        "vision_depth": 0,
        "language_depth": 0, 
        "vision_ctx": 0,
        "language_ctx": 0
    }
    
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION if hasattr(cfg.TRAINER.COOP, "CLASS_TOKEN_POSITION") else "end"

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

# Add CustomCLIP class for prompt learning
# Add CustomCLIP class for prompt learning
class CustomCLIP(nn.Module): # Changed for ECCV
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        # loss configuration
        self.enabled_losses = cfg.TRAINER.COOP.LOSS.ENABLED_LOSSES
        # Store cfg for loss functions
        self.cfg = cfg
        # Initialize weights in LossRegistry
        LossRegistry.init_weights(cfg)

    def compute_losses(self, logits, label, text_features=None, zero_shot_logits=None, zero_shot_text_features=None):
        """
        Compute all enabled losses and return their weighted sum
        
        Args:
            logits: Model predictions (batch_size, num_classes)
            label: Ground truth labels (batch_size,)
            text_features: Text embeddings from prompt learner
            zero_shot_logits: Zero-shot model predictions for ECCV losses
            zero_shot_text_features: Zero-shot text features for TEXT_MOMENT_MATCHING
        
        Returns:
            losses: Dictionary containing individual and total losses
        """
        losses = {}
        total_loss = 0.0
        
        # Handle the case where ECCV losses are enabled but zero_shot_logits are not provided
        eccv_losses_requested = set(['ECCV_PENALTY', 'ECCV_ZS']).intersection(self.enabled_losses)
        active_losses = self.enabled_losses.copy()
        
        # If ECCV losses are requested but zero_shot_logits is None, remove them from active losses
        if eccv_losses_requested and zero_shot_logits is None:
            for loss_name in eccv_losses_requested:
                active_losses.remove(loss_name)
            print(f"\nWARNING: Requested ECCV losses {eccv_losses_requested} disabled because zero_shot_logits not provided!")
        
        # Add verification for ECCV losses when they are provided
        if eccv_losses_requested and zero_shot_logits is not None and not hasattr(self, 'eccv_verified'):
            print(f"\nECCV losses using zero-shot logits with shape: {zero_shot_logits.shape}")
            self.eccv_verified = True
        
        # Process all active losses
        for loss_name in active_losses:
            loss_fn = LossRegistry.get_loss(loss_name)
            if loss_fn is not None:
                # Handle different types of losses
                if loss_name == 'AS':
                    # Feature-based loss
                    if text_features is not None:
                        loss_value = loss_fn(text_features)
                    else:
                        continue
                elif loss_name in ['FL', 'LS', 'SMAC']:
                    # Losses that need config parameters
                    loss_value = loss_fn(logits, label, cfg=self.cfg)
                elif loss_name in ['ECCV_PENALTY', 'ECCV_ZS']:
                    # Losses that need zero-shot logits
                    loss_value = loss_fn(logits, label, zero_shot_logits=zero_shot_logits)
                elif loss_name == 'TEXT_MOMENT_MATCHING':
                    # Needs both tuned and zero-shot text features
                    if text_features is None or zero_shot_text_features is None:
                        print(f"WARNING: Required features missing for {loss_name}")
                        continue
                    loss_value = loss_fn(logits, label, text_features=text_features, 
                                        zero_shot_text_features=zero_shot_text_features)
                elif loss_name == 'MARGIN_MEAN_VAR':
                    # Direct use with logits and labels
                    loss_value = loss_fn(logits, label)
                else:
                    # Standard losses
                    loss_value = loss_fn(logits, label)
                    
                # Weight and accumulate loss
                weight = LossRegistry.get_weight(loss_name)
                losses[f'{loss_name}_loss'] = loss_value
                total_loss += weight * loss_value
                
        losses['loss'] = total_loss
        return losses

    def forward(self, image, label=None, zero_shot_logits=None, zero_shot_text_features=None):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        if self.prompt_learner.training:
            return self.compute_losses(logits, label, text_features=text_features,
                                      zero_shot_logits=zero_shot_logits,
                                      zero_shot_text_features=zero_shot_text_features)

        return logits, image_features, text_features

@TRAINER_REGISTRY.register()
class CoOp(VLBaseLearner): # Changed for ECCV
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]
        
        # Set default for class token position if not present
        if not hasattr(cfg.TRAINER.COOP, "CLASS_TOKEN_POSITION"):
            cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"
            
        # Check if CSC exists
        if not hasattr(cfg.TRAINER.COOP, "CSC"):
            cfg.TRAINER.COOP.CSC = False

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building CustomCLIP with learnable prompts (CoOp)")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        
        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
                
        # Only optimize the prompt learner
        params_to_optimize = self.model.prompt_learner
        model_to_register = "prompt_learner"
        component_to_register = self.model.prompt_learner

        if getattr(cfg.MODEL, 'INIT_WEIGHTS', None) and hasattr(self.model, 'prompt_learner'):
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        
        # Build optimizer for the appropriate parameters
        self.optim = build_optimizer(params_to_optimize, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model(model_to_register, component_to_register, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Check if we need the zero-shot model (for ECCV losses or TEXT_MOMENT_MATCHING)
        if hasattr(cfg.TRAINER.COOP, 'LOSS') and hasattr(cfg.TRAINER.COOP.LOSS, 'ENABLED_LOSSES'):
            need_zs_model = any(loss in cfg.TRAINER.COOP.LOSS.ENABLED_LOSSES for loss in ['ECCV_PENALTY', 'ECCV_ZS', 'TEXT_MOMENT_MATCHING'])
        else:
            need_zs_model = False
        
        # Initialize zero-shot model only if needed
        if need_zs_model:
            print("\n" + "="*80)
            print("INITIALIZING ZERO-SHOT MODEL FOR CALIBRATION LOSSES")
            print("="*80)
            
            from trainers.classification.zsclip import ZeroshotCLIP
            self.zeroshot_model = ZeroshotCLIP(cfg)
            self.zeroshot_model.cfg = self.cfg
            self.zeroshot_model.dm = self.dm
            self.zeroshot_model.device = self.device
            
            # Add debug info before building model
            print(f"Dataset: {self.cfg.DATASET.NAME}")
            print(f"Model type: {self.cfg.MODEL.NAME}")
            print(f"Backbone: {self.cfg.MODEL.BACKBONE.NAME}")
            
            # Build the zero-shot model
            self.zeroshot_model.build_model()
            
            # After building, verify the prompts and template
            from trainers.classification.zsclip import CUSTOM_TEMPLATES
            dataset_name = self.cfg.DATASET.NAME.lower()
            template = CUSTOM_TEMPLATES.get(dataset_name, "An image of {}.")
            print(f"\nTemplate used: '{template}'")
            print(f"Zero-shot model type: {getattr(self.zeroshot_model, 'model_type', 'standard')}")
            print("="*80 + "\n")
        else:
            self.zeroshot_model = None
            print("\n" + "="*80)
            print("SKIPPING ZERO-SHOT MODEL INITIALIZATION (NOT NEEDED FOR ENABLED LOSSES)")
            print("="*80 + "\n")

        # Multi-GPU handling
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            # For prompt learning, only parallelize the text encoder
            self.model.text_encoder = nn.DataParallel(self.model.text_encoder)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        # Get the list of enabled losses
        enabled_losses = model.enabled_losses if not hasattr(model, 'module') else model.module.enabled_losses
        
        # Check if we need zero-shot features
        need_zs_logits = any(loss in enabled_losses for loss in ['ECCV_PENALTY', 'ECCV_ZS'])
        need_zs_text_features = 'TEXT_MOMENT_MATCHING' in enabled_losses
        
        # Only compute zero-shot features if needed
        zs_logits = None
        zs_text_features = None
        
        if (need_zs_logits or need_zs_text_features) and self.zeroshot_model is not None:
            with torch.no_grad():
                if need_zs_logits:
                    zs_logits, _, _ = self.zeroshot_model.model_inference(image)
                    
                    # Verify zero-shot logits in first training batch
                    if self.batch_idx == 0 and self.epoch == 0:
                        print("\n" + "="*80)
                        print("VERIFYING ZERO-SHOT LOGITS FOR ECCV LOSSES")
                        print(f"Zero-shot logits shape: {zs_logits.shape}")
                        print(f"Min/Max values: {zs_logits.min().item():.4f}/{zs_logits.max().item():.4f}")
                        
                        # Print sample predictions
                        sample_idx = 0
                        print(f"\nSample image zero-shot logits:")
                        for i, logit in enumerate(zs_logits[sample_idx]):
                            if i < 5 or i > zs_logits.shape[1] - 5:  # Just print first and last 5 classes
                                class_name = self.dm.dataset.classnames[i]
                                print(f"  Class '{class_name}': {logit.item():.4f}")
                        print("="*80 + "\n")
                
                if need_zs_text_features:
                    # Get the text features from zero-shot model for TEXT_MOMENT_MATCHING
                    if hasattr(self.zeroshot_model, 'text_features'):
                        zs_text_features = self.zeroshot_model.text_features
                    else:
                        print("WARNING: Zero-shot model lacks text_features attribute")
                        zs_text_features = None

        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                losses = model(image, label, zero_shot_logits=zs_logits, 
                            zero_shot_text_features=zs_text_features)
            optim.zero_grad()
            scaler.scale(losses['loss']).backward()
            scaler.step(optim)
            scaler.update()
        else:
            losses = model(image, label, zero_shot_logits=zs_logits,
                        zero_shot_text_features=zs_text_features)
            optim.zero_grad()
            losses['loss'].backward()
            optim.step()
        
        # Build loss summary to handle all enabled losses
        loss_summary = {'loss': losses['loss'].item()}
        
        for loss_name in enabled_losses:
            loss_key = f'{loss_name}_loss'
            if loss_key in losses:
                loss_summary[loss_key] = losses[loss_key].item()

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)