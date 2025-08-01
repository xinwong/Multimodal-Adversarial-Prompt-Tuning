import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import os
import torchattacks
from autoattack import AutoAttack
from trades import trades_loss
from tqdm import tqdm
from torch.utils.data import TensorDataset
from torchvision import transforms

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'IVLP_MoE',
                      "vision_depth": cfg.TRAINER.MoEAdvIVLP.PROMPT_DEPTH_VISION,
                      "language_depth": cfg.TRAINER.MoEAdvIVLP.PROMPT_DEPTH_TEXT, "vision_ctx": cfg.TRAINER.MoEAdvIVLP.N_CTX_VISION,
                      "language_ctx": cfg.TRAINER.MoEAdvIVLP.N_CTX_TEXT,
                      "num_experts": cfg.TRAINER.MoEMoEAdvIVLP.NUM_EXPERTS
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


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        # Make sure Language depth >= 1
        assert cfg.TRAINER.MoEAdvIVLP.PROMPT_DEPTH_TEXT >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch  "
        n_ctx = cfg.TRAINER.MoEAdvIVLP.N_CTX_TEXT
        ctx_init = cfg.TRAINER.MoEAdvIVLP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print(f"Adversarial Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.MoEAdvIVLP.N_CTX_VISION}")
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

    def forward(self, image, label=None):
        image = self.normalize(image)

        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features = self.image_encoder(image.type(self.dtype))

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)

        return logits


@TRAINER_REGISTRY.register()
class MoEAdvIVLP(TrainerX):
    def before_adv_test(self, attack='pgd', eps=1/255, alpha=1/255, steps=100):
        r"""
        Arguments:
            eps (float): maximum perturbation. (Default: 4/255)
            alpha (float): step size. (Default: 1/255)
            steps (int): number of steps. (Default: 10)
            random_start (bool): using random initialization of delta. (Default: True)
        """
        dataset_save_path = os.path.join(self.cfg.OUTPUT_DIR, self.cfg.DATASET.NAME + "_adv_dataset.pkl")
        print(attack)

        if os.path.isfile(dataset_save_path):
            self.adv_test_pkl, _ = torch.load(dataset_save_path).tensors
            return
        
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
        elif attack == 'cwa':
            # 使用transferattack库进行迁移攻击
            import transferattack
            # 获取attack参数
            model_name = ['resnet18','resnet101', 'densenet121']
            targeted = False
            # 创建攻击器
            attacker = transferattack.load_attack_class(attack)(
                model_name=model_name, 
                targeted=targeted
            )
            # # 应用攻击生成对抗样本
            # perturbations = attacker(images, labels)
            # # 限制扰动并应用
            # noise = torch.clamp(perturbations, -eps, eps)
            # images_adv = images + noise
            # images_adv = torch.clamp(images_adv, 0, 1)
            
            # return images_adv

        elif attack == 'ags':
            # 使用transferattack库进行迁移攻击
            import transferattack
            # 获取attack参数
            model_name = "ags_coco"
            targeted = False
            # 创建攻击器
            attacker = transferattack.load_attack_class(attack)(
                model_name=model_name, 
                targeted=targeted
            )
            # # 应用攻击生成对抗样本
            # perturbations = attacker(images, labels)
            # # 限制扰动并应用
            # noise = torch.clamp(perturbations, -eps, eps)
            # images_adv = images + noise
            # images_adv = torch.clamp(images_adv, 0, 1)
        else:
            raise ValueError(f"Unknown attack: {attack}")
        
        # If inputs were normalized, then
        # attacker.set_normalization_used(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

        self.adv_test_pkl = torch.empty(size=[len(self.test_loader.dataset), 3, 224, 224])
        all_labels = torch.empty(size=[len(self.test_loader.dataset)])

        for batch_idx, batch in enumerate(tqdm(self.test_loader)):
            input, label = self.parse_batch_test(batch)
            if attack == 'auto':
                adv_input = attacker.run_standard_evaluation(input, label)
            elif attack == "cwa" or attack =="ags":
                black_box_eps = 8.0/255 
                # 应用攻击生成对抗样本
                perturbations = attacker(input, label)
                # 限制扰动并应用
                noise = torch.clamp(perturbations, -black_box_eps, black_box_eps)
                adv_input = input + noise
                adv_input = torch.clamp(adv_input, 0, 1)            
            else:
                adv_input = attacker(input, label)
            with torch.no_grad():
                start_idx = batch_idx * self.test_loader.batch_size
                end_idx = start_idx + input.size(0)
                self.adv_test_pkl[start_idx: end_idx] = adv_input.detach().cpu()
                all_labels[start_idx: end_idx] = label.detach().cpu()

        # adv_test_dataset = TensorDataset(self.adv_test_pkl, all_labels)

        # torch.save(adv_test_dataset, dataset_save_path)

        # print(f"Saving to: {dataset_save_path}")

    @torch.no_grad()
    def test_adv(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        array_to_pkl = self.adv_test_pkl
        print(f"Evaluate on the *{split}* set")
        
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            _, label = self.parse_batch_test(batch)
            adv_input = array_to_pkl[batch_idx * data_loader.batch_size: (batch_idx + 1) * data_loader.batch_size]
            adv_output = self.model_inference(adv_input.to(label.device))
            self.evaluator.process(adv_output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def check_cfg(self, cfg):
        assert cfg.TRAINER.MoEAdvIVLP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.MoEAdvIVLP.PREC == "fp32" or cfg.TRAINER.MoEAdvIVLP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        # name_to_update = "prompt_learner"
        names_to_update = ["prompt_learner", "expert_prompts", "gate"] # Names of parameters to be updated. We need to train the MoE layers.

        for name, param in self.model.named_parameters():
            # if name_to_update not in name:
            if not any(key in name for key in names_to_update):
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.MoEAdvIVLP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model  = self.model
        optim  = self.optim
        scaler = self.scaler

        eps    = self.cfg.AT.TRAIN.EPS / 255.0
        alpha  = self.cfg.AT.TRAIN.ALPHA / 255.0
        steps  = self.cfg.AT.TRAIN.STEPS
        loss_type = self.cfg.AT.TRAIN.AT_LOSS_TYPE

        attacker = torchattacks.PGD(self.model,
                                    eps=eps,
                                    alpha=alpha,
                                    steps=steps,
                                    random_start=True)
        # attacker.set_normalization_used(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]) # If inputs were normalized, then

        N = image.size(0)
        if loss_type == "clean":
            com_images = image
        elif loss_type == "adv_full":
            adv_image  = attacker(image, label)
            com_images = adv_image
        elif loss_type == "adv_half":
            adv_image  = attacker(image[N//2:], label[N//2:])
            com_images = torch.cat([image[:N//2], adv_image], dim=0)
        else:
            raise ValueError(f"Invalid loss type: {loss_type}")

        prec = self.cfg.TRAINER.MoEAdvIVLP.PREC
        if prec == "amp":
            with autocast():
                loss = model(com_images, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(com_images, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

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
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
