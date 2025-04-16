
import gc
import copy
import cv2
import os
import numpy as np
import torch
import torchvision
from einops import repeat
from PIL import Image, ImageFilter
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UniPCMultistepScheduler,
    LCMScheduler,
)
from diffusers.schedulers import TCDScheduler
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from transformers import AutoTokenizer, PretrainedConfig

from libs.unet_motion_model import MotionAdapter, UNetMotionModel
from libs.brushnet_CA import BrushNetModel
from libs.unet_2d_condition import UNet2DConditionModel
from diffueraser.pipeline_diffueraser import StableDiffusionDiffuEraserPipeline


checkpoints = {
    "2-Step": ["pcm_{}_smallcfg_2step_converted.safetensors", 2, 0.0],
    "4-Step": ["pcm_{}_smallcfg_4step_converted.safetensors", 4, 0.0],
    "8-Step": ["pcm_{}_smallcfg_8step_converted.safetensors", 8, 0.0],
    "16-Step": ["pcm_{}_smallcfg_16step_converted.safetensors", 16, 0.0],
    "Normal CFG 4-Step": ["pcm_{}_normalcfg_4step_converted.safetensors", 4, 7.5],
    "Normal CFG 8-Step": ["pcm_{}_normalcfg_8step_converted.safetensors", 8, 7.5],
    "Normal CFG 16-Step": ["pcm_{}_normalcfg_16step_converted.safetensors", 16, 7.5],
    "LCM-Like LoRA": [
        "pcm_{}_lcmlike_lora_converted.safetensors",
        4,
        0.0,
    ],
}

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")

def resize_frames(frames, size=None):    
    if size is not None:
        out_size = size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        frames = [f.resize(process_size) for f in frames]
    else:
        out_size = frames[0].size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        if not out_size == process_size:
            frames = [f.resize(process_size) for f in frames]
        
    return frames

def read_mask(validation_mask, fps, n_total_frames, img_size, mask_dilation_iter, frames):
    cap = cv2.VideoCapture(validation_mask)
    if not cap.isOpened():
        print("Error: Could not open mask video.")
        exit()
    mask_fps = cap.get(cv2.CAP_PROP_FPS)
    if mask_fps != fps:
        cap.release()
        raise ValueError("The frame rate of all input videos needs to be consistent.")

    masks = []
    masked_images = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:  
            break
        if(idx >= n_total_frames):
            break
        mask = Image.fromarray(frame[...,::-1]).convert('L')
        if mask.size != img_size:
            mask = mask.resize(img_size, Image.NEAREST)
        mask = np.asarray(mask)
        m = np.array(mask > 0).astype(np.uint8)
        m = cv2.erode(m,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                    iterations=1)
        m = cv2.dilate(m,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                    iterations=mask_dilation_iter)

        mask = Image.fromarray(m * 255)
        masks.append(mask)

        masked_image = np.array(frames[idx])*(1-(np.array(mask)[:,:,np.newaxis].astype(np.float32)/255))
        masked_image = Image.fromarray(masked_image.astype(np.uint8))
        masked_images.append(masked_image)

        idx += 1
    cap.release()

    return masks, masked_images

def read_priori(priori, fps, n_total_frames, img_size):
    cap = cv2.VideoCapture(priori)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    priori_fps = cap.get(cv2.CAP_PROP_FPS)
    if priori_fps != fps:
        cap.release()
        raise ValueError("The frame rate of all input videos needs to be consistent.")

    prioris=[]
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        if(idx >= n_total_frames):
            break
        img = Image.fromarray(frame[...,::-1])
        if img.size != img_size:
            img = img.resize(img_size)
        prioris.append(img)
        idx += 1
    cap.release()

    os.remove(priori) # remove priori 

    return prioris

def read_video(validation_image, video_length, nframes, max_img_size):
    vframes, aframes, info = torchvision.io.read_video(filename=validation_image, pts_unit='sec', end_pts=video_length) # RGB
    fps = info['video_fps']
    n_total_frames = int(video_length * fps)
    n_clip = int(np.ceil(n_total_frames/nframes))

    frames = list(vframes.numpy())[:n_total_frames]
    frames = [Image.fromarray(f) for f in frames]
    max_size = max(frames[0].size)
    if(max_size<256):
        raise ValueError("The resolution of the uploaded video must be larger than 256x256.")
    if(max_size>4096):
        raise ValueError("The resolution of the uploaded video must be smaller than 4096x4096.")
    if max_size>max_img_size:
        ratio = max_size/max_img_size
        ratio_size = (int(frames[0].size[0]/ratio),int(frames[0].size[1]/ratio))
        img_size = (ratio_size[0]-ratio_size[0]%8, ratio_size[1]-ratio_size[1]%8)
        resize_flag=True
    elif (frames[0].size[0]%8==0) and (frames[0].size[1]%8==0):
        img_size = frames[0].size
        resize_flag=False
    else:
        ratio_size = frames[0].size
        img_size = (ratio_size[0]-ratio_size[0]%8, ratio_size[1]-ratio_size[1]%8)
        resize_flag=True
    if resize_flag:
        frames = resize_frames(frames, img_size)
        img_size = frames[0].size

    return frames, fps, img_size, n_clip, n_total_frames


class DiffuEraser:
    def __init__(
            self, device, base_model_path, vae_path, diffueraser_path, revision=None,
            ckpt="Normal CFG 4-Step", mode="sd15", loaded=None):
        self.device = device

        ## load model
        self.vae = AutoencoderKL.from_pretrained(vae_path)
        self.noise_scheduler = DDPMScheduler.from_pretrained(base_model_path, 
                subfolder="scheduler",
                prediction_type="v_prediction",
                timestep_spacing="trailing",
                rescale_betas_zero_snr=True
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
                    base_model_path,
                    subfolder="tokenizer",
                    use_fast=False,
                )
        text_encoder_cls = import_model_class_from_model_name_or_path(base_model_path,revision)
        self.text_encoder = text_encoder_cls.from_pretrained(
                base_model_path, subfolder="text_encoder"
            )
        self.brushnet = BrushNetModel.from_pretrained(diffueraser_path, subfolder="brushnet")
        self.unet_main = UNetMotionModel.from_pretrained(
            diffueraser_path, subfolder="unet_main",
        )

        ## set pipeline
        self.pipeline = StableDiffusionDiffuEraserPipeline.from_pretrained(
            base_model_path,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet_main,
            brushnet=self.brushnet
        ).to(self.device, torch.float16)
        try:
            self.pipeline.enable_model_cpu_offload()
            print("--- Enabled Model CPU Offload ---")
        except AttributeError:
             print("--- Model CPU Offload not directly supported by this pipeline version/structure ---")
        self.pipeline.scheduler = UniPCMultistepScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.set_progress_bar_config(disable=True)

        self.noise_scheduler = UniPCMultistepScheduler.from_config(self.pipeline.scheduler.config)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)

        ## use PCM
        self.ckpt = ckpt
        PCM_ckpts = checkpoints[ckpt][0].format(mode)
        self.guidance_scale = checkpoints[ckpt][2]
        if loaded != (ckpt + mode):
            self.pipeline.load_lora_weights(
                "weights/PCM_Weights", weight_name=PCM_ckpts, subfolder=mode
            )
            loaded = ckpt + mode

            if ckpt == "LCM-Like LoRA":
                self.pipeline.scheduler = LCMScheduler()
            else:
                self.pipeline.scheduler = TCDScheduler(
                    num_train_timesteps=1000,
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    timestep_spacing="trailing",
                )
        self.num_inference_steps = checkpoints[ckpt][1]
        self.guidance_scale = 0

    # def __init__(
    #         self, device, base_model_path, vae_path, diffueraser_path, revision=None,
    #         ckpt="Normal CFG 4-Step", mode="sd15", loaded=None): # `loaded` tracks if PCM loaded to avoid reloading
    #     self.device = device
    #     print(f"--- Initializing DiffuEraser on device: {self.device} ---")

    #     print("--- Attempting to configure PyTorch SDP backends ---")
    #     if torch.cuda.is_available():
    #         torch_version = torch.__version__
    #         print(f"PyTorch CUDA available. Torch version: {torch_version}")
    #         # try:
    #         #     # Try importing flash_attn to confirm installation and get version
    #         #     import flash_attn
    #         #     flash_attn_installed = True
    #         #     flash_version = getattr(flash_attn, '__version__', 'N/A')
    #         #     print(f"Flash Attention package imported successfully (Version: {flash_version}).")
    #         # except ImportError:
    #         #     flash_attn_installed = False
    #         #     print("Flash Attention package not found or import failed.")

    #         # # Enable Flash SDP backend (supported in PyTorch >= 2.0)
    #         # # This tells PyTorch's SDPA dispatcher to prioritize flash-attn if available
    #         # if hasattr(torch.backends.cuda, "enable_flash_sdp"):
    #         #     try:
    #         #         # The boolean argument enables/disables it
    #         #         torch.backends.cuda.enable_flash_sdp(True)
    #         #         print("[INFO] Enabled PyTorch Flash SDP backend.")
    #         #     except Exception as e:
    #         #         # Catch potential errors during enabling
    #         #         print(f"[WARN] Could not enable PyTorch Flash SDP backend: {e}")
    #         # else:
    #         #     # Log if the attribute doesn't exist (older PyTorch version)
    #         #     print("[INFO] torch.backends.cuda.enable_flash_sdp not available (likely PyTorch < 2.0).")

    #         # # Enable Memory Efficient SDP backend
    #         # # This is another dispatcher setting that can use flash-attn, xformers, or other optimized kernels
    #         # if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
    #         #     try:
    #         #         torch.backends.cuda.enable_mem_efficient_sdp(True)
    #         #         print("[INFO] Enabled PyTorch Memory-Efficient SDP backend.")
    #         #     except Exception as e:
    #         #         print(f"[WARN] Could not enable PyTorch Memory-Efficient SDP backend: {e}")
    #         # else:
    #         #     print("[INFO] torch.backends.cuda.enable_mem_efficient_sdp not available.")
    #     else:
    #         # Log if CUDA is not detected
    #         print("[WARN] CUDA not available, cannot enable GPU SDP backends.")
    #     print("--- SDP backend configuration attempt finished ---")
    #     # <<< --- END: Add Flash Attention / SDPA Backend Encouragement --- >>>

    #     print(f"--- Using base: {base_model_path}, VAE: {vae_path}, DiffuEraser weights: {diffueraser_path} ---")
    #     print(f"--- PCM Checkpoint mode: {ckpt}, SD mode: {mode} ---")

    #     # 1. Load individual components
    #     print("--- Loading VAE ---")
    #     vae = AutoencoderKL.from_pretrained(vae_path, revision=revision)
    #     self.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    #     self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)

    #     print("--- Loading Tokenizer ---")
    #     tokenizer = AutoTokenizer.from_pretrained(
    #                 base_model_path, subfolder="tokenizer", revision=revision, use_fast=False,
    #             )
    #     print("--- Loading Text Encoder ---")
    #     text_encoder_cls = import_model_class_from_model_name_or_path(base_model_path, revision)
    #     text_encoder = text_encoder_cls.from_pretrained(
    #             base_model_path, subfolder="text_encoder", revision=revision
    #         )
    #     print("--- Loading BrushNet ---")
    #     brushnet = BrushNetModel.from_pretrained(diffueraser_path, subfolder="brushnet")
    #     print("--- Loading UNet Main (Motion Model) ---")
    #     unet_main = UNetMotionModel.from_pretrained(
    #         diffueraser_path, subfolder="unet_main",
    #     )
    #     # 2. Configure and initialize the default scheduler (UniPC)
    #     print("--- Configuring Default Scheduler (UniPCMultistepScheduler) ---")
    #     scheduler_config_path = os.path.join(base_model_path, "scheduler")
    #     try:
    #          # Load config dict from base model scheduler settings
    #          scheduler_config_dict = UniPCMultistepScheduler.load_config(scheduler_config_path, revision=revision)
    #          print(f"--- Loaded scheduler config from: {scheduler_config_path} ---")
    #          # Initialize UniPC using the loaded config
    #          default_scheduler = UniPCMultistepScheduler.from_config(scheduler_config_dict)
    #          # Initialize the noise scheduler instance (used in forward)
    #          self.noise_scheduler = UniPCMultistepScheduler.from_config(scheduler_config_dict)
    #          print(f"--- Initialized default UniPCMultistepScheduler and noise_scheduler ---")
    #     except Exception as e:
    #          print(f"Error loading scheduler config from {scheduler_config_path}: {e}. Cannot continue.")
    #          raise e
    #     # 3. Create the main pipeline instance, passing components
    #     print("--- Creating StableDiffusionDiffuEraserPipeline ---")
    #     self.pipeline = StableDiffusionDiffuEraserPipeline.from_pretrained(
    #         base_model_path, # Base path often needed for pipeline config even with components provided
    #         vae=vae,
    #         text_encoder=text_encoder,
    #         tokenizer=tokenizer,
    #         unet=unet_main,
    #         brushnet=brushnet,
    #         scheduler=default_scheduler, # Pass the initialized default scheduler
    #         safety_checker=None,
    #         feature_extractor=None,
    #         requires_safety_checker=False,
    #         revision=revision,
    #     )
    #     # 4. Move pipeline to device and set dtype
    #     print(f"--- Moving pipeline to {self.device} with dtype torch.float16 ---")
    #     self.pipeline.to(self.device, dtype=torch.float16)
    #     self.pipeline.set_progress_bar_config(disable=True)

    #     # 5. Enable optimizations (place after .to(device))
    #     optimizations_applied = []
    #     try:
    #         self.pipeline.enable_model_cpu_offload()
    #         optimizations_applied.append("Model CPU Offload")
    #     except (AttributeError, NotImplementedError):
    #         print("--- Warning: Model CPU Offload not available/implemented ---")
    #     try:
    #         self.pipeline.enable_attention_slicing("max")
    #         optimizations_applied.append("Max Attention Slicing")
    #     except (AttributeError, NotImplementedError):
    #         print("--- Warning: Attention Slicing not available/implemented ---")
    #     try:
    #         self.pipeline.unet.enable_gradient_checkpointing()
    #         optimizations_applied.append("UNet Gradient Checkpointing")
    #     except (AttributeError, NotImplementedError):
    #         print("--- Warning: UNet Gradient Checkpointing not available/implemented ---")
    #     try:
    #         self.pipeline.brushnet.enable_gradient_checkpointing()
    #         optimizations_applied.append("BrushNet Gradient Checkpointing")
    #     except (AttributeError, NotImplementedError):
    #          print("--- Warning: BrushNet Gradient Checkpointing not available/implemented ---")
    #     print(f"--- Enabled Optimizations: {', '.join(optimizations_applied) if optimizations_applied else 'None'} ---")

    #     # 6. Handle PCM weights loading and potential scheduler override
    #     self.ckpt = ckpt
    #     self.num_inference_steps = 20 # Default steps
    #     self.guidance_scale = 0.0    # Default guidance

    #     if ckpt in checkpoints:
    #         pcm_filename = checkpoints[ckpt][0].format(mode)
    #         pcm_path = os.path.join("weights/PCM_Weights", pcm_filename)
    #         self.guidance_scale = checkpoints[ckpt][2] # Use PCM guidance
    #         num_inf_steps_pcm = checkpoints[ckpt][1]   # Use PCM steps

    #         print(f"--- Preparing to load PCM LoRA: {pcm_filename} ---")
    #         if loaded != (ckpt + mode):
    #             if os.path.exists(pcm_path):
    #                 print(f"--- Loading PCM LoRA weights from: {pcm_path} ---")
    #                 # Load LoRA weights into the pipeline
    #                 self.pipeline.load_lora_weights(
    #                      os.path.dirname(pcm_path),
    #                      weight_name=os.path.basename(pcm_path)
    #                 )
    #                 # Update state tracker if you have one passed via `loaded` parameter
    #                 # loaded = ckpt + mode # This assignment would need `loaded` to be managed externally or made self.loaded
    #             else:
    #                  print(f"Error: PCM weight file not found at {pcm_path}. Skipping LoRA load.")

    #         # Override scheduler based on PCM type AFTER pipeline is created and LoRA loaded
    #         current_scheduler_config_dict = self.pipeline.scheduler.config
    #         if ckpt == "LCM-Like LoRA":
    #             print(f"--- Overriding Scheduler: Using LCMScheduler ---")
    #             self.pipeline.scheduler = LCMScheduler.from_config(current_scheduler_config_dict)
    #             self.noise_scheduler = LCMScheduler.from_config(current_scheduler_config_dict) # Sync noise scheduler
    #         elif "Step" in ckpt:
    #             print(f"--- Overriding Scheduler: Using TCDScheduler ---")
    #             self.pipeline.scheduler = TCDScheduler.from_config(current_scheduler_config_dict)
    #             self.noise_scheduler = TCDScheduler.from_config(current_scheduler_config_dict) # Sync noise scheduler
    #         # else: Keep the default UniPC scheduler

    #         self.num_inference_steps = num_inf_steps_pcm # Set steps based on PCM config
    #         print(f"--- Using PCM config: guidance scale={self.guidance_scale}, inference steps={self.num_inference_steps} ---")
    #     else:
    #          print(f"Warning: PCM checkpoint '{ckpt}' not found. Using default guidance={self.guidance_scale}, steps={self.num_inference_steps}.")

    #     print(f"--- Final Active Pipeline Scheduler: {type(self.pipeline.scheduler).__name__} ---")
    #     print(f"--- DiffuEraser Initialization Complete ---")


    def forward(self, validation_image, validation_mask, priori, output_path,
                max_img_size = 1280, video_length=2, mask_dilation_iter=4,
                nframes=22, # Keep this default or set slightly lower (e.g., 16, 18) if needed
                seed=None, revision = None, guidance_scale=None, blended=True):
        validation_prompt = ""
        guidance_scale_final = self.guidance_scale if guidance_scale is None else guidance_scale

        if (max_img_size<256 or max_img_size>1920): # Increased upper limit slightly based on code analysis
            raise ValueError("The max_img_size must be between 256 and 1920.")

        ################ read input video ################
        print(f"--- Reading video: {validation_image} ---")
        frames, fps, img_size, n_clip, n_total_frames = read_video(validation_image, video_length, nframes, max_img_size)
        video_len = len(frames)
        print(f"--- Video read: {video_len} frames, FPS: {fps}, Size: {img_size}, Target nframes: {nframes} ---")

        ################     read mask    ################
        print(f"--- Reading mask: {validation_mask} ---")
        validation_masks_input, validation_images_input = read_mask(validation_mask, fps, video_len, img_size, mask_dilation_iter, frames)
        print(f"--- Mask read: {len(validation_masks_input)} masks, {len(validation_images_input)} masked images ---")

        ################    read priori   ################
        print(f"--- Reading priori: {priori} ---")
        prioris = read_priori(priori, fps, n_total_frames, img_size)
        print(f"--- Priori read: {len(prioris)} frames ---")

        ## recheck frame counts and trim lists to the minimum common length
        n_total_frames = min(len(frames), len(validation_masks_input), len(prioris))
        if(n_total_frames < nframes): # Check against the actual nframes used
            # Raise error or adjust nframes? Let's raise for clarity.
            raise ValueError(f"The effective video duration ({n_total_frames} frames) is shorter than nframes ({nframes}). Processing requires at least nframes.")
        if n_total_frames < 22 and nframes >= 22: # Retain original check related to minimum practical length?
             print(f"Warning: Effective video duration ({n_total_frames} frames) is less than 22, but proceeding with nframes={nframes}.")
             # Maybe adjust nframes downwards here if needed? e.g. nframes = n_total_frames

        validation_masks_input = validation_masks_input[:n_total_frames]
        validation_images_input = validation_images_input[:n_total_frames]
        frames = frames[:n_total_frames]
        prioris = prioris[:n_total_frames]
        real_video_length = n_total_frames # Use this consistent length
        print(f"--- Adjusted lists to effective length: {real_video_length} frames ---")

        # Resize frames (ensure consistent size before processing)
        # Note: resize_frames in diffueraser.py seems basic, just resizes if needed.
        # Consider if the original full-res frames are needed later or if resized is sufficient.
        prioris = resize_frames(prioris, img_size)
        validation_masks_input = resize_frames(validation_masks_input, img_size)
        validation_images_input = resize_frames(validation_images_input, img_size)
        resized_frames = resize_frames(frames, img_size) # Keep original frames if needed for compose?
        resized_frames_ori = copy.deepcopy(resized_frames) # Keep a copy of input frames for final composition
        validation_masks_input_ori = copy.deepcopy(validation_masks_input) # Keep copy of original masks for final composition

        # Get final target dimensions
        tar_width, tar_height = resized_frames[0].size
        print(f"--- Final processing dimensions: {tar_width}x{tar_height} ---")

        ##############################################
        # DiffuEraser inference
        ##############################################
        print("--- Starting DiffuEraser inference pipeline ---")
        if seed is None:
            generator = None
            print("--- Using random generator ---")
        else:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            print(f"--- Using generator with seed: {seed} ---")

        ################  Prepare Priori Latents (Streamed VAE Encoding) ################
        latents_list = []
        # Adjust batch size based on VRAM availability during VAE. 4 is conservative.
        num_vae_batch = 4
        vae_module = self.pipeline.vae # Get VAE module reference

        print(f"--- Starting VAE encoding for {len(prioris)} priori frames in batches of {num_vae_batch} ---")
        vae_original_device = vae_module.device # Remember original device (might be CPU if offloaded)

        try:
            # Ensure VAE is on GPU for encoding
            vae_module.to(self.device)
            print(f"--- Moved VAE to {self.device} for encoding ---")

            with torch.no_grad():
                for i in range(0, len(prioris), num_vae_batch):
                    batch_pil = prioris[i : i + num_vae_batch]
                    current_batch_size = len(batch_pil) # Store the actual size of this batch

                    # Preprocess only the current batch
                    batch_tensor = [self.image_processor.preprocess(img, height=tar_height, width=tar_width).to(dtype=torch.float32) for img in batch_pil]
                    batch_tensor = torch.cat(batch_tensor).to(device=self.device, dtype=torch.float16) # Process in float16

                    # Encode the batch
                    batch_latents = vae_module.encode(batch_tensor).latent_dist.sample()
                    # Move to CPU immediately after use to minimize VRAM holding time
                    latents_list.append(batch_latents.cpu())

                    # Optional: Print progress less frequently
                    if i % (num_vae_batch * 10) == 0: # Check index i, not rely on batch_pil existence later
                         # Calculate end frame index correctly
                         end_frame_index = i + current_batch_size
                         print(f"--- VAE encoded batch up to frame {end_frame_index} ---") # Use calculated index

                    # Clean up batch tensors AFTER they are no longer needed (including for the print)
                    del batch_pil, batch_tensor, batch_latents


            # Concatenate latents on CPU
            latents = torch.cat(latents_list, dim=0)
            print(f"--- VAE Encoding complete, created latents tensor shape: {latents.shape} on CPU ---")
            del latents_list # Free CPU memory
            gc.collect()

            # Move the final, full latents tensor to GPU for the pipeline
            latents = latents.to(self.device)
            latents = latents * vae_module.config.scaling_factor # Apply scaling factor using the vae object
            print(f"--- Moved full latents tensor to {self.device} ---")

        finally:
            # Ensure VAE is returned to its original device (likely CPU due to offload)
            # or explicitly offload if it wasn't automatically handled.
            vae_module.to(vae_original_device)
            print(f"--- Returned VAE to {vae_original_device} ---")
            # Explicitly clear cache after potential large tensor operations and moves
            torch.cuda.empty_cache()
            gc.collect()
        ################ Prepare Noise ################
        shape = (
            nframes, # Base shape uses nframes
            self.pipeline.unet.config.in_channels, # Get channels from unet config
            tar_height // self.vae_scale_factor,
            tar_width // self.vae_scale_factor
        )
        # Determine dtype
        if hasattr(self.pipeline, 'text_encoder') and self.pipeline.text_encoder is not None:
            prompt_embeds_dtype = self.pipeline.text_encoder.dtype
        elif hasattr(self.pipeline, 'unet') and self.pipeline.unet is not None:
            prompt_embeds_dtype = self.pipeline.unet.dtype
        else:
            prompt_embeds_dtype = torch.float16 # Default fallback

        print(f"--- Preparing base noise for shape {shape} with dtype {prompt_embeds_dtype} ---")
        noise_pre = randn_tensor(shape, device=self.device, dtype=prompt_embeds_dtype, generator=generator)

        # Repeat noise pattern if video is longer than nframes pattern
        # Calculate necessary repeats based on real_video_length and nframes
        n_repeats_needed = (real_video_length + nframes - 1) // nframes
        print(f"--- Repeating noise pattern {n_repeats_needed} times for {real_video_length} frames ---")
        # Use einops repeat correctly: repeat the *temporal* dimension 't'
        # The pattern is (t c h w), we want to repeat t -> (repeat t)
        noise = repeat(noise_pre, "t c h w -> (repeat t) c h w", repeat=n_repeats_needed)
        # Trim excess noise to match the exact video length
        noise = noise[:real_video_length, ...]
        print(f"--- Final noise tensor shape: {noise.shape} ---")

        ################ Timesteps for Noise Addition ################
        # This seems to be adding noise only at timestep 0 for the 'add_noise' step later.
        # This is likely related to how the specific scheduler (UniPC, TCD, LCM) expects initial latents.
        # Keep as is unless scheduler documentation suggests otherwise.
        timesteps_add_noise = torch.tensor([0], device=self.device)
        timesteps_add_noise = timesteps_add_noise.long()

        ################ Pre-inference (Corrected Logic) ################
        latents_pre_out=None # Initialize
        sample_index=None    # Initialize

        # Condition uses the actual nframes value
        if real_video_length > nframes * 2:
            print(f"--- Running Pre-inference using nframes={nframes} on sampled frames ---")
            ## Sample indices based on nframes
            step = real_video_length / nframes # Use real_video_length for sampling step
            # Ensure we don't sample more indices than available frames or the desired nframes
            num_samples = min(nframes, real_video_length)
            sample_index = [int(i * step) for i in range(num_samples)]
            # Ensure indices are within bounds
            sample_index = [idx for idx in sample_index if idx < real_video_length]
            num_samples = len(sample_index) # Update num_samples after bounds check

            print(f"--- Sampled {num_samples} indices for pre-inference: {sample_index[:10]}... ---") # Print first 10

            # Gather inputs for the sampled frames
            validation_masks_input_pre = [validation_masks_input[i] for i in sample_index]
            validation_images_input_pre = [validation_images_input[i] for i in sample_index]

            # Safely gather latents for sampled indices
            if latents.shape[0] > max(sample_index):
                 latents_pre = torch.stack([latents[i] for i in sample_index])
            else:
                 print("ERROR: Not enough latents generated for pre-inference sampling.")
                 latents_pre = None # Indicate failure/skip

            if latents_pre is not None: # Only proceed if latents were sampled correctly
                # Ensure noise_pre matches the number of frames being processed (num_samples)
                if noise_pre.shape[0] != num_samples:
                     print(f"Warning: Adjusting noise_pre shape from {noise_pre.shape[0]} to {num_samples} for pre-inference")
                     # Use the first 'num_samples' noises from the base pattern
                     noise_pre_adjusted = noise_pre[:num_samples]
                else:
                     noise_pre_adjusted = noise_pre

                ## Add noise using the sampled latents and adjusted noise
                noisy_latents_pre = self.noise_scheduler.add_noise(latents_pre, noise_pre_adjusted, timesteps_add_noise)
                latents_pre = noisy_latents_pre # Use noisy latents as input to pipeline

                # Run the pipeline on the sampled subset
                print(f"--- Calling pipeline for pre-inference ({num_samples} frames) ---")
                with torch.no_grad():
                    # Pass the actual number of frames being processed
                    pipeline_output = self.pipeline(
                        num_frames=num_samples, # <<< Pass correct number of frames
                        prompt=validation_prompt,
                        images=validation_images_input_pre,
                        masks=validation_masks_input_pre,
                        num_inference_steps=self.num_inference_steps,
                        generator=generator,
                        guidance_scale=guidance_scale_final,
                        latents=latents_pre, # Start from noisy latents
                        output_type="latent" # Get latents directly if possible
                    )
                    # Check if output is dictionary (newer Diffusers) or tuple
                    if isinstance(pipeline_output, dict) and 'latents' in pipeline_output:
                         latents_pre_out = pipeline_output.latents
                    elif isinstance(pipeline_output, DiffuEraserPipelineOutput): # Check for the specific output class
                         latents_pre_out = pipeline_output.latents
                    else:
                         # Fallback or assume it's just the latents tensor if not dict/specific class
                         latents_pre_out = pipeline_output
                         print("Warning: Pre-inference pipeline output format unexpected, assuming latents tensor.")

                print(f"--- Pre-inference pipeline call complete ---")
                torch.cuda.empty_cache()

                # Decode latents and update original tensors
                if latents_pre_out is not None and latents_pre_out.shape[0] == num_samples:
                    print(f"--- Decoding {num_samples} pre-inference latents ---")
                    # Define local decode function to handle VAE device management
                    def decode_latents_local(lats_to_decode, dtype):
                        vae = self.pipeline.vae
                        original_dev = vae.device
                        try:
                            vae.to(self.device)
                            lats_to_decode = 1 / vae.config.scaling_factor * lats_to_decode.to(self.device)
                            vid = []
                            decode_batch_size = 4 # Decode in smaller batches if needed
                            for t_idx in range(0, lats_to_decode.shape[0], decode_batch_size):
                                batch_lats = lats_to_decode[t_idx : t_idx + decode_batch_size]
                                vid.append(vae.decode(batch_lats.to(dtype)).sample)
                            vid = torch.concat(vid, dim=0)
                            return vid.float().cpu() # Return to CPU
                        finally:
                            vae.to(original_dev) # Return VAE to original device

                    with torch.no_grad():
                       video_tensor_temp = decode_latents_local(latents_pre_out, dtype=torch.float16)
                       images_pre_out = self.image_processor.postprocess(video_tensor_temp, output_type="pil") # Output PIL images
                    del video_tensor_temp
                    torch.cuda.empty_cache()
                    print(f"--- Decoded pre-inference images ---")

                    ## ! --- ONLY UPDATE LATENTS --- !
                    print(f"--- Updating main latents tensor ONLY with {len(sample_index)} pre-inference results ---") # Adjusted log
                    for i, index in enumerate(sample_index): # Loop over correct indices
                        if i < latents_pre_out.shape[0] and index < latents.shape[0]: # Removed check for images_pre_out length
                            # Update latents (move pre-inf latent back to GPU for assignment)
                            latents[index] = latents_pre_out[i].to(self.device)

                            # --- REMOVED THE FOLLOWING LINES ---
                            # validation_images_input[index] = images_pre_out[i]
                            # resized_frames[index] = images_pre_out[i]
                            # validation_masks_input[index] = black_image
                            # --- END REMOVAL ---
                        else:
                            print(f"Warning: Index mismatch during pre-inference LATENT update (i={i}, index={index}). Skipping update for this index.")
                    # del images_pre_out # Delete decoded images if not needed elsewhere (they aren't here)
                    # if 'images_pre_out' in locals(): del images_pre_out # Safer deletion
                    # torch.cuda.empty_cache() # Optional cache clearing

                else: # If latents_pre_out was None or mismatch
                     print("--- Skipping pre-inference result application due to missing/mismatched latents_pre_out ---")
                     latents_pre_out = None # Ensure it's None

            else: # latents_pre was None or pipeline failed
                 print("--- Skipping pre-inference pipeline call due to sampling or other issues ---")
                 latents_pre_out = None # Ensure it's None

        else: # real_video_length <= nframes * 2
            print("--- Skipping Pre-inference step (video too short or condition not met) ---")
            latents_pre_out=None
            sample_index=None

        gc.collect()
        torch.cuda.empty_cache()

        ################  Main Frame-by-frame inference  ################
        print(f"--- Starting main inference loop using nframes={nframes} ---")
        ## Add noise to the potentially updated latents tensor
        # Use the *full* noise tensor corresponding to real_video_length
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps_add_noise)
        latents = noisy_latents # Start main inference from this noisy state

        # Clean up original noise tensor if no longer needed
        del noise, noise_pre, noisy_latents
        gc.collect()
        torch.cuda.empty_cache()

        print(f"--- Calling main pipeline for {real_video_length} frames ---")
        with torch.no_grad():
            pipeline_output = self.pipeline(
                num_frames=nframes, # Pass the chunk size parameter
                prompt=validation_prompt,
                images=validation_images_input, # Use potentially updated images
                masks=validation_masks_input,   # Use potentially updated masks
                num_inference_steps=self.num_inference_steps,
                generator=generator,
                guidance_scale=guidance_scale_final,
                latents=latents, # Start from noisy latents
                output_type="pil" # Get PIL images directly
            )
            # Check output format
            if isinstance(pipeline_output, dict) and 'frames' in pipeline_output:
                images = pipeline_output.frames
            elif isinstance(pipeline_output, DiffuEraserPipelineOutput):
                images = pipeline_output.frames
            elif isinstance(pipeline_output, tuple): # Older format?
                images = pipeline_output[0]
                print("Warning: Main pipeline output format unexpected (tuple), assuming frames.")
            else: # Assume direct frame output if not dict/tuple
                 images = pipeline_output
                 print("Warning: Main pipeline output format unexpected, assuming frames.")

        # Ensure the number of output frames matches expected length
        if len(images) < real_video_length:
            print(f"Warning: Pipeline returned {len(images)} frames, expected {real_video_length}. Padding or error handling might be needed.")
            # Handle discrepancy if necessary (e.g., repeat last frame, raise error)
        images = images[:real_video_length] # Trim if pipeline returned more for some reason
        print(f"--- Main inference pipeline call complete, received {len(images)} frames ---")

        # Clean up large latent tensor
        del latents
        gc.collect()
        torch.cuda.empty_cache()

        ################ Compose Final Video ################
        print(f"--- Composing final video using original masks and frames ---")
        binary_masks = validation_masks_input_ori # Use the original masks read from file
        mask_blurreds = []
        if blended:
            print(f"--- Applying blur blending to masks ---")
            # Consider optimizing GaussianBlur if it becomes a bottleneck (e.g., batching with kornia/cupy)
            for i in range(len(binary_masks)):
                mask_np = np.array(binary_masks[i])
                # Ensure mask is single channel for blur
                if mask_np.ndim == 3:
                    mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)
                # Apply blur
                mask_blurred_np = cv2.GaussianBlur(mask_np, (21, 21), 0) / 255.0
                # Combine original mask with blurred mask (ensure binary_mask calculation is correct)
                # Original intention might be: blend = mask * blur + (1-mask) * identity => use blur amount based on mask proximity
                # Current calculation: 1 - (1 - mask/255) * (1 - blurred) -> check logic if results look odd
                binary_mask_np = 1.0 - (1.0 - mask_np / 255.0) * (1.0 - mask_blurred_np)
                mask_blurreds.append(Image.fromarray((binary_mask_np * 255).astype(np.uint8)))
            binary_masks = mask_blurreds # Use the blurred versions for composition

        comp_frames = []
        # Use the original resized frames (before pre-inference updates) for non-masked areas
        frames_for_composition = resized_frames_ori
        print(f"--- Composing {len(images)} generated frames with original frames ---")
        for i in range(len(images)):
            # Ensure mask is broadcastable (H, W, 1)
            mask_np = np.expand_dims(np.array(binary_masks[i]), axis=2) / 255.0 # Normalize mask
            # Ensure frames are numpy arrays
            img_generated_np = np.array(images[i]).astype(np.uint8)
            img_original_np = np.array(frames_for_composition[i]).astype(np.uint8)

            # Blend: generated * mask + original * (1 - mask)
            img_comp_np = (img_generated_np * mask_np + img_original_np * (1.0 - mask_np)).astype(np.uint8)
            comp_frames.append(Image.fromarray(img_comp_np))

        del images, frames_for_composition, binary_masks, mask_blurreds # Free memory
        gc.collect()

        ################ Write Video ################
        print(f"--- Writing final video to: {output_path} ---")
        if not comp_frames:
             print("Error: No frames to write to video!")
             return None # Or raise error

        default_fps = fps
        # Use frame size from the composed frames
        output_size = comp_frames[0].size
        # Ensure fourcc is compatible
        fourcc = cv2.VideoWriter_fourcc(*"mp4v") # Common codec
        writer = cv2.VideoWriter(output_path, fourcc, default_fps, output_size)

        if not writer.isOpened():
             print(f"Error: Could not open video writer for {output_path}")
             return None

        for f_idx, frame_pil in enumerate(comp_frames):
            # Convert PIL Image to BGR numpy array for OpenCV
            img_np_bgr = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
            writer.write(img_np_bgr)
            # Optional: Print progress
            # if f_idx % 50 == 0:
            #      print(f"--- Wrote frame {f_idx} ---")

        writer.release()
        print(f"--- Video writing complete ---")
        ################################

        return output_path
            



