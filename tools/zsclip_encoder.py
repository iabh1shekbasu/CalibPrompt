from models.clip import clip
import torch

def build_clip_templates(dataset_name):

    CUSTOM_TEMPLATES = {
       "KatherColon": "An H&E image of {}.",  
       "DigestPath": "An H&E image patch of {} tissue.",
       "PanNuke": "An H&E image patch of {} skin tissue.",
        "Covid": "A chest X-ray image of {} patient",
        "RSNA18": "A chest X-ray image of {} patient",
    }
    return CUSTOM_TEMPLATES[dataset_name]


def build_zsclip(backbone_name):

    # load zero-shot CLIP model
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {"trainer": 'ZeroshotCLIP',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    
    clip_model = clip.build_model(state_dict or model.state_dict(), design_details)
    # clip_model.cuda()

    return clip_model

