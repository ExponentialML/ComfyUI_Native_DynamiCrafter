import os
import torch
import comfy
import yaml
import folder_paths
import random

from einops import rearrange
from comfy import model_base, model_management, model_detection, latent_formats, model_sampling
from .lvdm.modules.networks.openaimodel3d import UNetModel as DynamiCrafterUNetModel

from .utils.model_utils import DynamiCrafterBase, DYNAMICRAFTER_CONFIG, VIDEOCRAFTER_CONFIG, \
    load_image_proj_dict, load_dynamicrafter_dict, get_image_proj_model, load_vae_dict

from .utils.utils import get_models_directory

MODEL_DIR= "dynamicrafter_models"
MODEL_DIR_PATH = os.path.join(folder_paths.models_dir, MODEL_DIR)


def handle_cfg(x, num_video_frames):
    use_cfg = x.shape[0] > num_video_frames
    num_video_frames *= 2 if use_cfg else 1
    batch_size = 1 if not use_cfg else 2

    return num_video_frames, use_cfg, batch_size

# There is probably a better way to do this, but with the apply_model callback, this seems necessary.
# The model gets wrapped around a CFG Denoiser class, and handles the conditioning parts there.
# We cannot access it, so we must find the conditioning according to how ComfyUI handles it.
def get_conditioning_pair(c_crossattn, use_cfg: bool):
    if not use_cfg:
        return c_crossattn

    conditioning_group = []

    for i in range(c_crossattn.shape[0]):
        # Get the positive and negative conditioning.
        positive_idx = i + 1
        negative_idx = i

        if positive_idx >= c_crossattn.shape[0]:
            break

        if not torch.equal(c_crossattn[[positive_idx]], c_crossattn[[negative_idx]]):
            conditioning_group = [
                c_crossattn[[positive_idx]], 
                c_crossattn[[negative_idx]]
            ]
            break

    if len(conditioning_group) == 0:
        raise ValueError("Could not get the appropriate conditioning group.")

    return torch.cat(conditioning_group)

def get_conditioning_args(in_dict: dict, required_args: list ):
    out_args = []
    for k in required_args:
        del_key = False
        if isinstance(k, tuple):
            if k[-1] == 'del':
                del_key = True
            k = k[0]
        out_args.append(in_dict.get(k, None))
        if del_key:
            del in_dict[k]
        return out_args

def expand_frames_tensor(tens: torch.Tensor, max_frames: int):
    """
    Expands the frame tensors by repeating the last tensor to the desired length.
    """
    return torch.cat(
            [tens] + [tens[-1:]] * abs(tens.shape[0] - max_frames)
        )[:max_frames]

def create_advanced_options( 
        fps_as_max_random: bool = False, 
        all_frames_as_context: bool = False, 
        force_start_frame: bool = False,
        reference_frames = None
    ):
    return {
            "fps_as_max_random": fps_as_max_random,
            "all_frames_as_context": all_frames_as_context,
            "force_start_frame": force_start_frame,
            "reference_frames": reference_frames,
        }  

def build_advanced_options(advanced_options: dict = None):
    if advanced_options is not None:
        return advanced_options.values()  
    else:
        return create_advanced_options().values()

def load_model_dicts(model_path: str, videocrafter_mode: False):
    model_state_dict = comfy.utils.load_torch_file(model_path)
    dynamicrafter_dict = load_dynamicrafter_dict(model_state_dict)

    if not videocrafter_mode:
        image_proj_dict = load_image_proj_dict(model_state_dict)
    else:
        image_proj_dict = None
        
    return dynamicrafter_dict, image_proj_dict

def get_prediction_type(is_eps: bool, model_config, video_crafter_mode: bool):
    if video_crafter_mode:
        model_config.unet_config["image_cross_attention_scale_learnable"] = False
        return model_base.ModelType.EPS

    if not is_eps and "image_cross_attention_scale_learnable" in model_config.unet_config.keys():
            model_config.unet_config["image_cross_attention_scale_learnable"] = False

    return model_base.ModelType.EPS if is_eps else model_base.ModelType.V_PREDICTION

def handle_model_management(dynamicrafter_dict: dict, model_config):
    parameters = comfy.utils.calculate_parameters(dynamicrafter_dict, "model.diffusion_model.")
    load_device = model_management.get_torch_device()
    unet_dtype = model_management.unet_dtype(
        model_params=parameters, 
        supported_dtypes=model_config.supported_inference_dtypes
    )
    manual_cast_dtype = model_management.unet_manual_cast(
        unet_dtype, 
        load_device, 
        model_config.supported_inference_dtypes
    )
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)
    inital_load_device = model_management.unet_inital_load_device(parameters, unet_dtype)
    offload_device = model_management.unet_offload_device()

    return load_device, inital_load_device

def check_leftover_keys(state_dict: dict):
    left_over = state_dict.keys()
    if len(left_over) > 0:
        print("left over keys:", left_over)

def get_fps_cond(fps: int =16, random_fps: bool = False):
    fs = torch.tensor([fps], dtype=torch.long,  device=model_management.intermediate_device())
    return fs if not random_fps else fps

class LatentConditionProcessor(object):
    def __init__(self, model, clip_vision, vae, image_proj_model=None, scale_latents=False):
        self.model = model
        self.clip_vision = clip_vision
        self.vae = vae
        self.image_proj_model = image_proj_model
        self.scale_latents = scale_latents

    def get_latent_emb(
        self, 
        images=None,
        c_concat=None, 
        norm_latent: bool = True, 
        norm_image: bool = False,
        frame_wise_emb: bool = False,
        strict: bool = False,
        all_frames: bool = False
    ):  
        # Slower, but complies with the vae processing and normalization.
        # c_concat is assumed to be an encoded latent natively through Comfy
        if strict and c_concat is not None:
            images = self.vae.decode(c_concat)
            c_concat = self.vae.encode(images[:, :, :, :3])
            images = images[:1, :, :, :3]
        else:
            if c_concat is None and images is not None:
                c_concat = self.vae.encode(images[:, :, :, :3])
            
            if images is None and c_concat is not None:
                images = self.vae.decode(c_concat[:, :, :, :3])

        if self.image_proj_model is not None:
            if norm_image:
                images = (images / 255.) * 2 - 1.0     
            
            encoded_images = []

            for idx in range(images.shape[0]):
                encoded_image = self.clip_vision.encode_image(images[[idx]])['last_hidden_state']
                encoded_images.append(encoded_image)

                if not all_frames and idx == 0:
                    break

            img_embs = self.image_proj_model(torch.cat(encoded_images))
            del encoded_images
            
        else:
            img_embs = None

        if self.scale_latents:
            if norm_latent and c_concat is None:
                vae_process_input = vae.process_input
                vae.process_input = lambda image: 2 * image - 1
                c_concat = vae.encode(images[:, :, :, :3])
                vae.process_input = vae_process_input
            c_concat = self.model.model.process_latent_in(c_concat) * 1.3
        else:
            c_concat = self.model.model.process_latent_in(c_concat)

        return c_concat, img_embs

class ReferenceProcessor:
    def __init__(
        self, 
        c_concat_ref: torch.Tensor = None, 
        img_emb_ref: torch.Tensor = None, 
        current_step: int = 0, 
        start_step: int = 1, 
        end_step: int = 20,
        strength: float = .25
    ):
        self.c_concat_ref = c_concat_ref
        self.img_emb_ref = img_emb_ref
        self.current_step = current_step
        self.start_step = start_step
        self.end_step = end_step
        self.strength = strength

    def step(self):
        self.current_step += 1

    def process(self):
        c_concat_ref = None
        img_emb_ref = None

        if self.current_step >= self.start_step and \
            self.current_step <= self.end_step:
            c_concat_ref = self.c_concat_ref
            img_emb_ref = self.img_emb_ref

        return c_concat_ref, img_emb_ref

class DynamiCrafterReferenceProcessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "start_step": ("INT", {"default": 1}),
                "end_step": ("INT", {"default": 20}),
                "strength": ("FLOAT", {
                    "default": 0.25, 
                    "min": 0., 
                    "max": 1.,
                    "step": 1e-2
                })
            },
        }
        
    CATEGORY = "Native_DynamiCrafter/Conditioning"
    RETURN_TYPES = ("DYNAMI_REFERENCE_PROCESSOR", )
    RETURN_NAMES = ("reference_frames", )

    FUNCTION = "initialize_reference_processor"
    
    def __init__(self):
        self.images = None
        self.start_step = None
        self.end_step = None
        self.strength = None

    def preprocess(self, model, vae, clip_vision, image_proj_model):
        c_concat_ref = vae.encode(self.images[:, :, :, :3])
        frame_wise_img_ref = []
        
        for i in range(self.images.shape[0]):
            encoded_image_ref = clip_vision.encode_image(self.images[[i]])['last_hidden_state']
            frame_wise_img_ref.append(encoded_image_ref)

        image_emb_ref = image_proj_model(torch.cat(frame_wise_img_ref))
        c_concat_ref = model.model.process_latent_in(c_concat_ref)
        
        reference_processor = ReferenceProcessor(
            c_concat_ref, 
            image_emb_ref,
            start_step=self.start_step,
            end_step=self.end_step,
            strength=self.strength
        )
        return reference_processor

    def initialize_reference_processor(self, images, start_step, end_step, strength):
        keys = ("images", "start_step", "end_step", "strength")
        values = (images, start_step, end_step, strength)

        for k, v in zip(keys, values):
            setattr(self, k, v)

        return ({"preprocess": self.preprocess}, )

def validate_forwardable_latent(latent, c_concat, num_video_frames, use_cfg):
    check_no_cfg = latent.shape[0] != num_video_frames
    check_with_cfg = latent.shape[0] != (num_video_frames * 2)

    latent_batch_size = latent.shape[0] if not use_cfg else latent.shape[0] // 2
    num_frames = num_video_frames if not use_cfg else num_video_frames // 2

    if all([check_no_cfg, check_with_cfg]):
        raise ValueError(
            "Please make sure your latent inputs match the number of frames in the DynamiCrafter Processor."
            f"Got a latent batch size of ({latent_batch_size}) with number of frames being ({num_frames})."
        )
    
    latent_h, latent_w = latent.shape[-2:]
    c_concat_h, c_concat_w = c_concat.shape[-2:]

    if not all([latent_h == c_concat_h, latent_w == c_concat_w]):
        raise ValueError(
            "Please make sure that your input latent and image frames are the same height and width.",
            f"Image Size: {c_concat_w * 8}, {c_concat_h * 8}, Latent Size: {latent_h * 8}, {latent_w * 8}"
        )

class DynamiAdvancedOpts:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "fps_as_max_random": ("BOOLEAN", {"default": False}),
                "all_frames_as_context": ("BOOLEAN", {"default": False}),
                "force_start_frame": ("BOOLEAN", {"default": False}),
                           },
            "optional": {
                "reference_frames": ("DYNAMI_REFERENCE_PROCESSOR", ),
            }
        }
    CATEGORY = "Native_DynamiCrafter/Options"
    RETURN_TYPES = ("DYNAMI_ADVANCED_OPTS", )
    RETURN_NAMES = ("advanced_options", )

    FUNCTION = "get_advanced_options"


    def get_advanced_options(
        self, 
        fps_as_max_random, 
        all_frames_as_context, 
        force_start_frame,
        reference_frames: DynamiCrafterReferenceProcessor.preprocess = None
    ):

        return ({
        "options": create_advanced_options(
            fps_as_max_random, 
            all_frames_as_context, 
            force_start_frame, 
            reference_frames
        )
        },)

class DynamiCrafterProcessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "clip_vision": ("CLIP_VISION", ),
                "vae": ("VAE", ),
                "image_proj_model": ("IMAGE_PROJ_MODEL", ),
                "images": ("IMAGE", ),
                "use_interpolate": ("BOOLEAN", {"default": False}),
                "img_cfg": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 1000}),
                "fps": ("INT", {"default": 15, "min": 1, "max": 1000, "step": 1}, ),
                "frames": ("INT", {"default": 16}),
                "scale_latents": ("BOOLEAN", {"default": False}),
            },
            "optional": {"advanced_options": ("DYNAMI_ADVANCED_OPTS", ) }
        }
        
    CATEGORY = "Native_DynamiCrafter/Processing"
    RETURN_TYPES = ("MODEL", "LATENT", "LATENT", )
    RETURN_NAMES = ("model", "empty_latent", "latent_img", )

    FUNCTION = "process_image_conditioning"

    def __init__(self):
        self.model_patcher = None

 
    def get_autoregressive_concat(self, transformer_options: dict, c_concat, img_embs):
        if 'c_concat_ar' in transformer_options and transformer_options['c_concat_ar'] is not None:
            c_concat = transformer_options['c_concat_ar']
            c_concat, img_embs =  self.latent_condition_processor.get_latent_emb(c_concat=c_concat, norm_image=True)
        return c_concat, img_embs

    # apply_model, {"input": input_x, "timestep": timestep_, "c": c, "cond_or_uncond": cond_or_uncond}
    def _forward(self, *args):
        transformer_options = self.model_patcher.model_options['transformer_options']
        conditioning = transformer_options['conditioning']
        apply_model = args[0]
        emb_sz = 1

        # Forward_dict = args[1]
        x, t, model_in_kwargs, _ =  args[1].values()

        if 'uncond_img' in conditioning.keys():
            del conditioning['uncond_img']
            
        c_concat, c_concat_ref ,img_embs, img_emb_uncond, img_cfg, fs, num_video_frames  = conditioning.values()

        c_concat, img_embs = self.get_autoregressive_concat(transformer_options, c_concat, img_embs)
        
        num_video_frames, use_cfg, batch_size = handle_cfg(x, num_video_frames)
        
        if use_cfg:
            c_concat = torch.cat([c_concat] * 2)

        validate_forwardable_latent(x, c_concat, num_video_frames, use_cfg)

        x_in, c_concat = map(lambda xc: rearrange(xc, '(b t) c h w -> b c t h w', b=batch_size), (x, c_concat))

        # We always assume video, so there will always be batched conditionings.
        c_crossattn = get_conditioning_pair(model_in_kwargs.pop('c_crossattn'), use_cfg)
        c_crossattn = c_crossattn[:2] if use_cfg else c_crossattn[:1]
        context_in = c_crossattn
        
        if use_cfg:
            img_emb_uncond = conditioning['image_emb_uncond']
            img_embs = torch.cat([img_embs, img_emb_uncond])

        
        if img_embs.shape[0] == num_video_frames:
            emb_sz = num_video_frames // 2 if batch_size > 1 else num_video_frames

        if c_concat_ref is not None:
            process_ref = c_concat_ref.process()
            strength = c_concat_ref.strength

            if all([ref is not None for ref in process_ref]):
                ir = process_ref[1].to(c_concat.device, c_concat.dtype)

                if img_embs.shape[0] == 2:
                    if ir.shape[0] > img_embs.shape[0]:
                        emb_sz = ir.shape[0] 
                        img_embs_cond = torch.cat([img_embs[:1]] * ir.shape[0])
                        img_embs_uncond = torch.zeros_like(img_embs_cond)
                        img_embs = torch.cat([img_embs_cond, img_embs_uncond], dim=0)
                        
                    else:
                        ir = torch.cat([ir] * 2)
                else:
                    if ir.shape[0] > img_embs.shape[0]:
                        # The img_emb reference should always be conditional, so no multiples for CFG.
                        emb_sz = ir.shape[0] 
                        
                        max_emb_sz = img_embs.shape[0]
                        img_emb_pad = [img_embs[-1:]] * (ir.shape[0] - max_emb_sz)
                        img_embs_cond= torch.cat([img_embs[:max_emb_sz]] + img_emb_pad)[:emb_sz]
                        img_embs_uncond = torch.zeros_like(img_embs_cond)

                        img_embs = torch.cat([img_embs, img_embs_uncond])

                _ie, _ir = map(lambda t: rearrange(t, 'b s l -> (b s) l'), (img_embs[:emb_sz], ir[:emb_sz]))
                sim = torch.nn.CosineSimilarity(dim=1)(_ie, _ir).unsqueeze(1)
                _ie = _ie * sim
                _ie = rearrange(_ie, '(b s) l -> b s l', b=emb_sz, s=img_embs.shape[1])
                ir[:emb_sz] = _ie

                del _ie
                del _ir

                img_embs[:emb_sz] = ir[:emb_sz] * strength + (1. - strength) * img_embs[:emb_sz]
 
            c_concat_ref.step()

        if isinstance(fs, int):
            fs = torch.tensor(
                [random.randint(3, fs)], 
                dtype=torch.long, 
                device=model_management.intermediate_device()
            )     
        fs = torch.cat([fs] * x_in.shape[0])
        
        outs = []
        for i in range(batch_size):
            model_in_kwargs['transformer_options']['cond_idx'] = i
            img_emb = img_embs[:emb_sz] if i == 0 else img_embs[-emb_sz:]

            x_out = apply_model(
                    x_in[[i]], 
                    t=torch.cat([t[:1]]),
                    context_in=context_in[[i]],
                    c_crossattn=c_crossattn, 
                    cc_concat=c_concat[[i]], # "cc" is to handle naming conflict with apply_model wrapper. 
                    # We want to handle this in the UNet forward.
                    num_video_frames=num_video_frames // 2 if use_cfg else num_video_frames, 
                    img_emb=img_emb,
                    fs=fs[[i]],
                    **model_in_kwargs
                )
            outs.append(x_out)
        
        if img_cfg > 1.0 and use_cfg:
            model_in_kwargs['transformer_options']['cond_idx'] = 0
            uncond_img = apply_model(
                x_in[[0]],
                t=torch.cat([t[:1]]),
                context_in=torch.zeros_like(context_in[[0]]),
                c_crossattn=torch.zeros_like(c_crossattn),
                cc_concat=c_concat[[0]],
                num_video_frames=num_video_frames // 2 if use_cfg else num_video_frames,
                img_emb=img_embs[:emb_sz],
                fs=fs[[0]],
                **model_in_kwargs 
            )
            uncond_img = rearrange(uncond_img, 'b c t h w -> (b t) c h w')

            self.model_patcher.model_options\
                ['transformer_options']['conditioning']['uncond_img'] = uncond_img

        x_out = torch.cat(list(reversed(outs)))
        x_out = rearrange(x_out, 'b c t h w -> (b t) c h w')

        return x_out

    def cfg_with_img_cond(self, args):
        conditioning = self.model_patcher.model_options["transformer_options"]["conditioning"]
        img_cfg = conditioning["img_cfg"]
        uncond_img = conditioning.get("uncond_img", None)
        uncond_pred, cond_pred, cond_scale = args["uncond"], args["cond"], args["cond_scale"]

        if img_cfg > 1.0 and uncond_img is not None:
            uncond_img = args["input"] - uncond_img
            cfg_result = uncond_pred + img_cfg * (uncond_img - uncond_pred) + cond_scale * (cond_pred - uncond_img)
        else:
            cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale

        return cfg_result

    def assign_forward_args(
        self, 
        c_concat, 
        image_emb, 
        image_emb_uncond, 
        img_cfg,
        fs, 
        frames,
        c_concat_ref: ReferenceProcessor = None
    ):
        self.model_patcher.model_options['transformer_options']['conditioning'] = {
            "c_concat": c_concat,
            "c_concat_ref": c_concat_ref,
            "image_emb": image_emb ,
            'image_emb_uncond': image_emb_uncond,
            "img_cfg": img_cfg,
            "fs": fs,
            "num_video_frames": frames,
        }

    def process_image_conditioning(
        self, 
        model, 
        clip_vision, 
        vae, 
        image_proj_model, 
        images, 
        use_interpolate,
        img_cfg,
        fps: int,
        frames: int,
        scale_latents: bool,
        advanced_options: dict = {}
    ):

        fps_as_max_random, all_frames_as_context, \
            force_start_frame, reference_frames = build_advanced_options(advanced_options.get("options", None))

        self.model_patcher = model
        self.latent_condition_processor = LatentConditionProcessor(
            model, 
            clip_vision, 
            vae, 
            image_proj_model, 
            scale_latents
        )

        encoded_latent = vae.encode(images[:, :, :, :3])
        c_concat, image_emb = self.latent_condition_processor.get_latent_emb(images, all_frames=all_frames_as_context)
        _, image_emb_uncond = self.latent_condition_processor.get_latent_emb(torch.zeros_like(images), all_frames=all_frames_as_context)
        fs = get_fps_cond(fps, fps_as_max_random)

        model.set_model_unet_function_wrapper(self._forward)
        model.set_model_sampler_cfg_function(self.cfg_with_img_cond)

        if reference_frames is not None:
            preprocess = reference_frames['preprocess']
            reference_frames = preprocess(model, vae, clip_vision, image_proj_model)
 
        used_interpolate_processing = False

        if use_interpolate and frames > 16:
            raise ValueError("Interpolation mode limits frames to 16.Consider using autoregressive approach for long videos.")
        if encoded_latent.shape[0] == 1:
            c_concat = torch.cat([c_concat] * frames, dim=0)[:frames]
        
            if use_interpolate:
                mask = torch.zeros_like(c_concat)
                mask[:1] = c_concat[:1]
                mask[-1:] = c_concat[:1]
                c_concat = mask
                used_interpolate_processing = True

                print("One frame was provided with use_interpolate set to True. Using start frame for both start and end.")
        else:
            if use_interpolate:
                if c_concat.shape[0] > 2:
                    print(f"To use interpolation mode,please make sure to pass a batch of exactly two images.")
                    frame_one = c_concat[[0]]

                    if c_concat.shape[0] == 1:
                        frame_two = frame_one.clone()
                    else:
                        frame_two = c_concat[[-1]]

                    c_concat = torch.cat([frame_one, frame_two])

                input_frame_count = c_concat.shape[0]

                # We're just padding to the same type an size of the concat
                masked_frames = torch.zeros_like(torch.cat([c_concat[:1]] * frames))[:frames]

                # Start frame
                masked_frames[:1] = c_concat[:1]

                end_frame_idx = -1

                # End frame
                masked_frames[-1:] = c_concat[[end_frame_idx]]
    
                c_concat = masked_frames
                used_interpolate_processing = True

                print(f"Using interpolation mode with {input_frame_count} frames.")

            if c_concat.shape[0] < frames and not used_interpolate_processing:

                if all_frames_as_context:
                    c_concat = expand_frames_tensor(c_concat, frames)
                else:
                    print(
                        "Multiple images found, but interpolation mode is unset. Using the first frame as condition.",
                    )
                    c_concat = torch.cat([c_concat[:1]] * frames)

        c_concat = c_concat[:frames]
        
        if encoded_latent.shape[0] == 1:
            encoded_latent = torch.cat([encoded_latent] * frames)[:frames]

        if encoded_latent.shape[0] < frames and encoded_latent.shape[0] != 1:
            encoded_latent = expand_frames_tensor(encoded_latent, frames)
        
        # We could store this as a state in this Node Class Instance, but to prevent any weird edge cases,
        # this should always be passed through the 'stateless' way, and let ComfyUI handle the transformer_options state.
        self.assign_forward_args(c_concat, image_emb, image_emb_uncond, img_cfg, fs, frames, reference_frames)
        
        out_zeros, out = {"samples": torch.zeros_like(c_concat)}, {"samples": encoded_latent}

        if force_start_frame:
            mask = out_zeros['samples'][:, :1, ...].clone()
            mask[1:] = 1.
            mask[:1] = 0. # Add a little noise to initialize generation.
            out.update({"noise_mask": mask}) 

        return (model, out_zeros, out)

class DynamiCrafterLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": (get_models_directory(os.listdir(MODEL_DIR_PATH)), ),
                "videocrafter_mode": ("BOOLEAN", {"default": False})
            },
        }
        
    CATEGORY = "Native_DynamiCrafter/Loaders"
    RETURN_TYPES = ("MODEL", "IMAGE_PROJ_MODEL", )
    RETURN_NAMES = ("model", "image_proj_model", )
    FUNCTION = "load_dynamicrafter"
     
    def load_dynamicrafter(self, model_path, videocrafter_mode):
        model_path = os.path.join(MODEL_DIR_PATH, model_path)
        
        if os.path.exists(model_path):
            dynamicrafter_dict, image_proj_dict = load_model_dicts(model_path, videocrafter_mode)
            CONFIG = DYNAMICRAFTER_CONFIG if not videocrafter_mode else VIDEOCRAFTER_CONFIG
            model_config = DynamiCrafterBase(CONFIG)

            dynamicrafter_dict, is_eps = model_config.process_dict_version(state_dict=dynamicrafter_dict)
            MODEL_TYPE = get_prediction_type(is_eps, model_config, videocrafter_mode)

            load_device, inital_load_device = handle_model_management(dynamicrafter_dict, model_config)

            model = model_base.BaseModel(
                model_config, 
                model_type=MODEL_TYPE, 
                device=inital_load_device, 
                unet_model=DynamiCrafterUNetModel
            )

            if not videocrafter_mode:
                image_proj_model = get_image_proj_model(image_proj_dict)
                model.load_model_weights(dynamicrafter_dict, "model.diffusion_model.")
            else:
                sd = {k.replace("model.diffusion_model.", ""): v for k, v in dynamicrafter_dict.items()}
                model.diffusion_model.load_state_dict(sd, strict=False)
                dynamicrafter_dict = {}
                image_proj_model = None
        
            check_leftover_keys(dynamicrafter_dict)

            model_patcher = comfy.model_patcher.ModelPatcher(
                model, 
                load_device=load_device, 
                offload_device=model_management.unet_offload_device(), 
                current_device=inital_load_device
            )

            model_patcher.model.diffusion_model.videocrafter_mode = videocrafter_mode
    
        return (model_patcher, image_proj_model, )

class VideoCrafterProcessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "frames": ("INT", {"default": 16, "min": 1, "max": 133333337}),
                "fps": ("INT", {"default": 15, "min": 1, "max": 1000, "step": 1}, ),
            }
        }
        
    CATEGORY = "Native_DynamiCrafter/Processing"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    FUNCTION = "process_fps"
    
    def __init__(self):
        self.model = None

    def _forward(self, *args):
        apply_model = args[0]

        # forward_dict
        fd = args[1]
        
        x, t, model_in_kwargs, cond_or_uncond =  fd['input'], fd['timestep'], fd['c'], fd['cond_or_uncond']
        num_video_frames = self.model.model_options['transformer_options']['num_video_frames']
        
        num_video_frames, use_cfg, batch_size = handle_cfg(x, num_video_frames)

        x = rearrange(x, '(b t) c h w -> b c t h w', b=batch_size)
    
        c_crossattn = get_conditioning_pair(model_in_kwargs.pop('c_crossattn'), use_cfg)
        
        c_crossattn = c_crossattn[:2] if use_cfg else c_crossattn[:1]
        outs = []

        for i in range(batch_size):
            model_in_kwargs['transformer_options']['cond_idx'] = i
            nvs = num_video_frames // 2 if batch_size > 1 else num_video_frames
            x_out = apply_model(
                x[[i]], 
                torch.cat([t[:1]]), 
                c_crossattn=torch.cat([c_crossattn[[i]]] * nvs), 
                num_video_frames=nvs, 
                **model_in_kwargs
            )
            outs.append(x_out)

        x_out = torch.cat(list(reversed(outs)))
        x_out = rearrange(x_out, 'b c t h w -> (b t) c h w')

        return x_out

    def process_fps(self, model, fps, frames):
        fs_t = torch.tensor([fps], dtype=torch.long, device=model_management.intermediate_device())
        model.model_options['transformer_options']['fs_t'] = fs_t
        model.model_options['transformer_options']['num_video_frames'] = frames
        model.set_model_unet_function_wrapper(self._forward)
        self.model = model
        return (model, )

NODE_CLASS_MAPPINGS = {
    "DynamiCrafterLoader": DynamiCrafterLoader,
    "DynamiCrafterProcessor": DynamiCrafterProcessor,
    "DynamiCrafterReferenceProcessor": DynamiCrafterReferenceProcessor,
    "VideoCrafterProcessor": VideoCrafterProcessor,
    "DynamiAdvancedOpts": DynamiAdvancedOpts
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DynamiCrafterLoader": "Load a DynamiCrafter Checkpoint",    
    "DynamiCrafterProcessor": "Apply DynamiCrafter",
    "DynamiCrafterReferenceProcessor": "Apply Reference Frames [EXPERIMENTAL]",
    "VideoCrafterProcessor": "Apply VideoCrafter",
    "DynamiAdvancedOpts": "Dynamicrafter Advanced Options"
}
