import os.path as osp
import os
import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

# Import VLBaseLearner instead of TrainerX
from trainers.classification.base_learner import VLBaseLearner
from models.open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
from .losses import LossRegistry  # Import LossRegistry for loss computations


def load_clip_to_cpu(cfg):
    if cfg.MODEL.NAME == 'biomedclip':
        print("\n\nUsing BioMedCLIP ...\n\n")
        model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        model.eval()   
    else:
        raise ValueError(f"Model {cfg.MODEL.NAME} not found. Supported models are: biomedclip")                    
    return model


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype

        tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

        # ctx_dim = medclip_model.text_model.projection_head.weight.shape[0]
        ctx_dim = 768
        clip_imsize = 224 # BioMedCLIP's default image size

        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        
        if ctx_init:
            raise NotImplementedError("This part is not yet implemented.")
            # use given words to initialize context vectors

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

        # Choose to either use random or pre-trained context initialization
        print("\n\nUsing Random Context Initialization\n\n")
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        # print("\n\nUsing Pre-trained Context Initialization\n\n")

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(tokenizer.tokenizer.encode(name))-2 for name in classnames]   # [CLS] and [SEP] are not counted
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        context_length = 256
        prompts_tokens = tokenizer(prompts, context_length=context_length)
        self.prompts_attention_mask = (prompts_tokens != clip_model.text.config.pad_token_id).long()

        with torch.no_grad():
            prompts_tokens_embeddings = clip_model.text.transformer.embeddings(input_ids=prompts_tokens).type(dtype) # [n_cls, 256, 768]

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", prompts_tokens_embeddings[:, :1, :])  # CLS
        self.register_buffer("token_suffix", prompts_tokens_embeddings[:, 1 + n_ctx :, :])  # CLASS_NAMES_TOKENS, SEP, PAD

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION
        
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
        else:
            raise ValueError

        return prompts, self.prompts_attention_mask


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.text_model = clip_model.text
       
    def forward(self, prompts_embeddings, prompts_attention_mask, normalize=False):
        out = self.text_model.transformer(inputs_embeds=prompts_embeddings, attention_mask=prompts_attention_mask)
        pooled_out = self.text_model.pooler(out, prompts_attention_mask)
        projected = self.text_model.proj(pooled_out)
        return F.normalize(projected, dim=-1) if normalize else projected
    

class ImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.vision_model = clip_model.visual

    def forward(self, image, normalize=False):
        features = self.vision_model(image)
        return F.normalize(features, dim=-1) if normalize else features


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()
        
        clip_model.dtype = clip_model.visual.head.proj.weight.dtype

        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.image_encoder = ImageEncoder(clip_model)
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.device = device

        cfg.defrost()
        cfg.DTYPE = str(self.dtype).split(".")[1]
        cfg.DEVICE = str(self.device)
        cfg.freeze()

        # Loss configuration - get enabled losses from config or default to CE
        self.enabled_losses = getattr(cfg.TRAINER.COOP, "LOSS", {}).get("ENABLED_LOSSES", ["CE"])
        # Store cfg for loss functions
        self.cfg = cfg
        # Initialize weights in LossRegistry if it has this method
        if hasattr(LossRegistry, "init_weights"):
            LossRegistry.init_weights(cfg)

        # self.normalize = torchvision.transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
     
    def compute_losses(self, logits, label, features=None, zero_shot_logits=None):
        """
        Compute all enabled losses and return their weighted sum
        
        Args:
            logits: Model predictions (batch_size, num_classes)
            label: Ground truth labels (batch_size,)
            features: Optional feature embeddings for feature-based losses
            zero_shot_logits: Zero-shot model predictions for ECCV losses
        
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
                    if features is not None:
                        loss_value = loss_fn(features)
                    else:
                        continue
                elif loss_name in ['FL', 'LS', 'SMAC']:
                    # Losses that need config parameters
                    loss_value = loss_fn(logits, label, cfg=self.cfg)
                elif loss_name in ['ECCV_PENALTY', 'ECCV_ZS']:
                    # Losses that need zero-shot logits
                    loss_value = loss_fn(logits, label, zero_shot_logits=zero_shot_logits)
                else:
                    # Standard losses
                    loss_value = loss_fn(logits, label)
                    
                # Weight and accumulate loss
                weight = LossRegistry.get_weight(loss_name)
                losses[f'{loss_name}_loss'] = loss_value
                total_loss += weight * loss_value
                
        losses['loss'] = total_loss
        return losses

    def forward(self, image, label=None, zero_shot_logits=None):
        image_features = self.image_encoder(image.type(self.dtype), normalize=True)
        
        prompts_embeddings, prompts_attention_mask = self.prompt_learner()
        text_features = self.text_encoder(prompts_embeddings, prompts_attention_mask.to(self.device), normalize=True)

        logits = self.logit_scale.exp() * image_features @ text_features.t()

        if self.prompt_learner.training and label is not None:
            return self.compute_losses(logits, label, text_features, zero_shot_logits=zero_shot_logits)

        return logits, image_features, text_features


@TRAINER_REGISTRY.register()
class CoOp_BioMedCLIP(VLBaseLearner):
    """Context Optimization (CoOp) for BioMedCLIP.

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]
        
        # Set default loss configuration if not present
        if not hasattr(cfg.TRAINER.COOP, "LOSS"):
            cfg.defrost()
            cfg.TRAINER.COOP.LOSS = type('', (), {})()
            cfg.TRAINER.COOP.LOSS.ENABLED_LOSSES = ["CE"]  # Default to CrossEntropy
            cfg.freeze()

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(f"Loading BioMedCLIP ...")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, self.device)

        print("\n\nTurning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
                # print(f"Not Learnable: {name}")
            else: 
                print(f"Learnable: {name}")
        print("\n\n")   

        if hasattr(cfg.MODEL, 'INIT_WEIGHTS') and cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        self.register_model("coop", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Check if we need the zero-shot model (for ECCV losses)
        if hasattr(cfg.TRAINER.COOP, 'LOSS') and hasattr(cfg.TRAINER.COOP.LOSS, 'ENABLED_LOSSES'):
            need_zs_model = any(loss in cfg.TRAINER.COOP.LOSS.ENABLED_LOSSES for loss in ['ECCV_PENALTY', 'ECCV_ZS'])
        else:
            need_zs_model = False
        
        # Initialize zero-shot model only if needed
        if need_zs_model:
            print("\n" + "="*80)
            print("INITIALIZING ZERO-SHOT MODEL FOR ECCV CALIBRATION LOSSES")
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

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()

        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    # Add model_inference method - required for VLBaseLearner's test method
    def model_inference(self, input):
        """Helper method to run model inference - required by VLBaseLearner."""
        return self.model(input)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)  # image: [B, C, H, W]
        
        model = self.model
        optim = self.optim
        scaler = self.scaler

        # Get the list of enabled losses
        enabled_losses = model.enabled_losses if not hasattr(model, 'module') else model.module.enabled_losses
        
        # Check if we need zero-shot logits (only for ECCV losses)
        need_zs_logits = any(loss in enabled_losses for loss in ['ECCV_PENALTY', 'ECCV_ZS'])
        
        # Only compute zero-shot logits if needed
        zs_logits = None
        if need_zs_logits and hasattr(self, 'zeroshot_model') and self.zeroshot_model is not None:
            with torch.no_grad():
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

        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            raise NotImplementedError("AMP is not yet supported.")
        else:
            losses = model(image, label, zero_shot_logits=zs_logits)
            
            # Handle the case where losses is a dictionary or a single loss value
            if isinstance(losses, dict):
                loss = losses['loss']
            else:
                loss = losses
                losses = {'loss': loss}
            
            optim.zero_grad()
            loss.backward()
            optim.step()

        # Build loss summary
        loss_summary = {'loss': loss.item()}
        
        # Add individual loss components to summary if available
        if isinstance(losses, dict):
            for k, v in losses.items():
                if k != 'loss' and isinstance(v, torch.Tensor):
                    loss_summary[k] = v.item()

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        """Parse batch for training - overriding parent method."""
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
        
    def parse_batch_test(self, batch):
        """Parse batch for testing."""
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