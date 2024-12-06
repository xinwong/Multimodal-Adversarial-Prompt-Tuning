import torch
import torch.nn as nn

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.model import convert_weights

from .advcoop import load_clip_to_cpu
from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT

from tqdm import tqdm
from torchvision import transforms
import torchattacks
from autoattack import AutoAttack

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}

def load_clip_to_cpu_TeCoA(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    TeCoA_state_dict = torch.load("/path/to/TeCoAmodel_best.pth.tar", map_location="cpu")
    model.visual.load_state_dict(TeCoA_state_dict['vision_encoder_state_dict'], strict=False)

    design_details = {"trainer": 'CoOp',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class CLIPWrapper(nn.Module):
    def __init__(self, clip_model, text_features):
        super().__init__()
        self.clip_model = clip_model
        self.text_features = text_features
        self.logit_scale = clip_model.logit_scale.exp()
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

    def forward(self, image):
        image = self.normalize(image)
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = self.logit_scale * image_features @ self.text_features.t()
        return logits


@TRAINER_REGISTRY.register()
class ZeroshotCLIP(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        TeCoA = False
        print("TeCoA: {}".format(TeCoA))
        
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        if TeCoA:
            clip_model = load_clip_to_cpu_TeCoA(cfg)
        else:
            clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model
        self.model = CLIPWrapper(clip_model, text_features)

    # def model_inference(self, image):
    #     image_features = self.clip_model.encode_image(image)
    #     image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    #     logit_scale = self.clip_model.logit_scale.exp()
    #     logits = logit_scale * image_features @ self.text_features.t()
    #     return logits

    def before_adv_test(self, attack='PGD', eps=4/255, alpha=1/255, steps=10):
        r"""
        Arguments:
            eps (float): maximum perturbation. (Default: 4/255)
            alpha (float): step size. (Default: 1/255)
            steps (int): number of steps. (Default: 10)
            random_start (bool): using random initialization of delta. (Default: True)
        """

        if attack == 'auto':
            attacker = AutoAttack(self.model, norm='Linf', eps=eps, version='standard')
        elif attack == 'pgd':
            attacker = torchattacks.PGD(self.model,
                        eps=eps,
                        alpha=alpha,
                        steps=steps,
                        random_start=True)
        elif attack == 'di':
            attacker = torchattacks.DIFGSM(self.model,
                        eps=eps,
                        alpha=alpha,
                        steps=steps)
        else:
            raise ValueError(f"Unknown attack: {attack}")
        
        # If inputs were normalized, then
        # attacker.set_normalization_used(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

        self.adv_test_pkl = torch.empty(size=[len(self.test_loader.dataset), 3, 224, 224])

        for batch_idx, batch in enumerate(tqdm(self.test_loader)):
            input, label = self.parse_batch_test(batch)
            # print(input.max(), input.min())
            if attack == 'auto':
                adv_input = attacker.run_standard_evaluation(input, label)
            else:
                adv_input = attacker(input, label)
            with torch.no_grad():
                self.adv_test_pkl[batch_idx * self.test_loader.batch_size: (batch_idx + 1) * self.test_loader.batch_size] = adv_input.detach().cpu()


@TRAINER_REGISTRY.register()
class ZeroshotCLIP2(ZeroshotCLIP):
    """Prompt ensembling."""

    # templates = IMAGENET_TEMPLATES
    templates = IMAGENET_TEMPLATES_SELECT

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        for params in clip_model.parameters():
            params.requires_grad_(False)

        # add custom-made prompt
        if cfg.DATASET.NAME != "ImageNet":
            self.templates += [CUSTOM_TEMPLATES[cfg.DATASET.NAME]]

        num_temp = len(self.templates)
        print(f"Prompt ensembling (n={num_temp})")

        mean_text_features = 0
        for i, temp in enumerate(self.templates):
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features = mean_text_features + text_features
        mean_text_features = mean_text_features / num_temp
        mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)

        self.text_features = mean_text_features
        self.clip_model = clip_model
