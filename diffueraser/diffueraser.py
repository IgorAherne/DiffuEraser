
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
from diffueraser.pipeline_diffueraser import StableDiffusionDiffuEraserPipeline, DiffuEraserPipelineOutput


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


def read_video(validation_image, max_video_length, nframes, max_img_size):
    """
    Reads video frames using OpenCV, calculates dimensions, and performs necessary resizing.
    Args:
        validation_image (str): Path to the input video file.
        max_video_length (float or None): Maximum video length in seconds to read.
                                          None or <= 0 means read the whole video.
        nframes (int): Target number of frames per processing chunk (used for n_clip).
        max_img_size (int): Maximum dimension (width or height) allowed after potential resizing.
    Returns:
        tuple: (frames, fps, img_size, n_clip, n_total_frames)
            frames (list[PIL.Image]): List of video frames (RGB).
            fps (float): Frames per second of the video.
            img_size (tuple[int, int]): Final (width, height) of frames after potential resizing
                                         (guaranteed divisible by 8).
            n_clip (int): Number of clips the video is conceptually divided into.
            n_total_frames (int): The actual number of frames read and processed.
    """
    print(f"--- Reading video using OpenCV: {validation_image} ---")
    cap = cv2.VideoCapture(validation_image)
    if not cap.isOpened():
        raise IOError(f"Error opening video file: {validation_image}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        # Attempt to get frame count and duration for a fallback calculation
        total_cv_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Note: CAP_PROP_POS_MSEC goes to end, but duration isn't directly available reliably
        # If frame count is valid, use a common default or raise error
        if total_cv_frames > 0:
            print(f"Warning: Could not determine FPS via CAP_PROP_FPS for video {validation_image}. Using fallback 30.0 fps.")
            fps = 30.0 # Or raise error: raise ValueError("Could not determine FPS")
        else:
            # If both FPS and frame count are invalid, it's likely unreadable
            cap.release()
            raise ValueError(f"Could not determine FPS or Frame Count for video: {validation_image}")
    frames = []
    frame_count = 0
    # Calculate max frames to read based on duration limit (if provided)
    max_frames_to_read = float('inf') # Default to reading all frames
    if max_video_length is not None and max_video_length > 0:
        # Calculate limit, ensure it's at least 1 frame if duration is very short but > 0
        max_frames_to_read = max(1, int(max_video_length * fps))
        print(f"--- Reading frames up to limit: {max_frames_to_read} (from max_video_length={max_video_length}, fps={fps}) ---")
    else:
        print(f"--- Reading all available frames (no max_video_length specified) ---")

    while True:
        ret, frame_bgr = cap.read()
        # Stop if no frame returned OR frame count limit reached
        if not ret or frame_count >= max_frames_to_read:
            break
        # Convert BGR (OpenCV default) to RGB for PIL
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
        frame_count += 1
    cap.release() # Release the video capture object

    if not frames:
        raise ValueError(f"No frames could be read from video file: {validation_image}")

    n_total_frames = len(frames) # The ACTUAL number of frames successfully read
    print(f"--- Successfully read {n_total_frames} frames ---")

    # Logic for Size Calculation & Resizing
    # Calculate n_clip based on the *actual* number of frames read and the target chunk size
    n_clip = int(np.ceil(n_total_frames / nframes))

    # Determine initial size and check constraints
    initial_width, initial_height = frames[0].size
    max_dimension = max(initial_width, initial_height)

    if max_dimension < 256:
        raise ValueError(f"Video resolution ({initial_width}x{initial_height}) is too small. Minimum dimension must be >= 256.")
    if max_dimension > 4096: # Check against a reasonable upper limit
        print(f"Warning: Video resolution ({initial_width}x{initial_height}) exceeds 4096. Might lead to high memory usage.")
        # raise ValueError("The resolution of the uploaded video must be smaller than 4096x4096.") # Or just warn

    # Determine target size (img_size) ensuring divisibility by 8
    target_width, target_height = initial_width, initial_height
    resize_flag = False

    if max_dimension > max_img_size:
        # Scale down if the largest dimension exceeds max_img_size
        ratio = max_dimension / max_img_size
        target_width = int(initial_width / ratio)
        target_height = int(initial_height / ratio)
        print(f"--- Max dimension {max_dimension} > {max_img_size}. Scaling down to target ~{target_width}x{target_height} ---")
        resize_flag = True

    # Ensure target dimensions are divisible by 8, adjusting width/height calculated so far
    final_width = max(8, target_width - (target_width % 8)) # Ensure at least 8px
    final_height = max(8, target_height - (target_height % 8)) # Ensure at least 8px

    # Set resize_flag if dimensions changed due to divisibility requirement OR scaling
    if final_width != initial_width or final_height != initial_height:
        resize_flag = True

    img_size = (final_width, final_height) # Final target size, divisible by 8
    
    if resize_flag: # Perform resizing if needed
        print(f"--- Resizing frames from ({initial_width}x{initial_height}) to {img_size} ---")
        frames = resize_frames(frames, img_size)
        # Verify size after resize (optional but good)
        if frames[0].size != img_size:
            print(f"Warning: Frame size after resize {frames[0].size} does not match target {img_size}. Using actual size.")
            img_size = frames[0].size
    else:
        print(f"--- No frame resizing needed. Using original size: {img_size} ---")
    # --- Return Values ---
    # Return the list of PIL frames, calculated fps, final img_size, n_clip, and actual n_total_frames
    return frames, fps, img_size, n_clip, n_total_frames


class DiffuEraser:
    def __init__(
            self, device, base_model_path, vae_path, diffueraser_path, revision=None,
            ckpt="Normal CFG 4-Step", mode="sd15", loaded=None):
        self.device = device

        ## load model
        self.vae = AutoencoderKL.from_pretrained(vae_path)
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
        self.unet_main = UNetMotionModel.from_pretrained(diffueraser_path, subfolder="unet_main")

        ## set pipeline
        self.pipeline = StableDiffusionDiffuEraserPipeline.from_pretrained(
            base_model_path,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet_main,
            brushnet=self.brushnet,
        )

        if(self.device != torch.device("cpu")):
            self.brushnet.to(self.device, torch.float16)
            self.unet_main.to(self.device, torch.float16)
            self.pipeline.to(self.device, torch.float16) 
            try:
                self.pipeline.enable_model_cpu_offload()
                print("--- Enabled Model CPU Offload ---")
            except AttributeError:
                print("--- Model CPU Offload not directly supported by this pipeline version/structure ---")

        try:
            # Check if xformers is installed and pipeline supports it
            if hasattr(self.pipeline, "enable_xformers_memory_efficient_attention"):
                 self.pipeline.enable_xformers_memory_efficient_attention()
                 print("--- Enabled xFormers memory-efficient attention ---")
            else:
                 print("--- xFormers not available or not supported by this pipeline version ---")
        except Exception as e:
            print(f"--- Failed to enable xFormers: {e}. Using default attention. ---")

        #  Explicit Gradient Checkpointing
        # print("--- Explicitly enabling gradient checkpointing for UNet and BrushNet ---")
        # if hasattr(self.pipeline.unet, "enable_gradient_checkpointing"):
        #     self.pipeline.unet.enable_gradient_checkpointing()
        # if hasattr(self.pipeline.brushnet, "enable_gradient_checkpointing"):
        #     self.pipeline.brushnet.enable_gradient_checkpointing()

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
    

    def forward(self, validation_image, validation_mask, priori, output_path,
                max_img_size = 1280, max_video_length=10, mask_dilation_iter=4,
                nframes=22, # Keep this default or set slightly lower (e.g., 16, 18) if needed
                seed=None, revision = None, guidance_scale=None, blended=True):
        validation_prompt = "background scene" #MODIF
        guidance_scale_final = self.guidance_scale if guidance_scale is None else guidance_scale

        if (max_img_size<256 or max_img_size>1920): # Increased upper limit slightly based on code analysis
            raise ValueError("The max_img_size must be between 256 and 1920.")

        ################ read input video ################
        print(f"--- Reading video: {validation_image} ---")
        frames, fps, img_size, n_clip, n_total_frames = read_video(validation_image, max_video_length, nframes, max_img_size)
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
            raise ValueError(f"The effective video duration ({n_total_frames} frames) is shorter than nframes ({nframes}). Processing requires at least nframes.")
        if n_total_frames < 22 and nframes >= 22: # Retain original check related to minimum practical length?
             print(f"Warning: Effective video duration ({n_total_frames} frames) is less than 22, but proceeding with nframes={nframes}.")

        validation_masks_input = validation_masks_input[:n_total_frames]
        validation_images_input = validation_images_input[:n_total_frames]
        frames = frames[:n_total_frames]
        prioris = prioris[:n_total_frames]
        real_video_length = n_total_frames # Use this consistent length
        print(f"--- Adjusted lists to effective length: {real_video_length} frames ---")

        # Resize frames (ensure consistent size before processing)
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
        num_vae_batch = 8 # Adjust based on VRAM
        vae_module = self.pipeline.vae

        print(f"--- Starting VAE encoding for {len(prioris)} priori frames in batches of {num_vae_batch} ---")
        vae_original_device = vae_module.device

        try:
            vae_module.to(self.device)
            print(f"--- Moved VAE to {self.device} for encoding ---")
            wanted_dtype = torch.float32 if self.device==torch.device("cpu") else torch.float16

            with torch.no_grad():
                for i in range(0, len(prioris), num_vae_batch):
                    batch_pil = prioris[i : i + num_vae_batch]
                    current_batch_size = len(batch_pil)
                    batch_tensor = [self.image_processor.preprocess(img, height=tar_height, width=tar_width).to(dtype=torch.float32) for img in batch_pil]
                    batch_tensor = torch.cat(batch_tensor).to(device=self.device, dtype=wanted_dtype)

                    batch_latents = vae_module.encode(batch_tensor).latent_dist.sample()
                    latents_list.append(batch_latents.cpu())

                    if i % (num_vae_batch * 10) == 0:
                         end_frame_index = i + current_batch_size
                         print(f"--- VAE encoded batch up to frame {end_frame_index} ---")

                    del batch_pil, batch_tensor, batch_latents

            latents = torch.cat(latents_list, dim=0)
            print(f"--- VAE Encoding complete, created latents tensor shape: {latents.shape} on CPU ---")
            del latents_list
            gc.collect()

            latents = latents.to(self.device)
            latents = latents * vae_module.config.scaling_factor
            print(f"--- Moved full latents tensor to {self.device} ---")

        finally:
            vae_module.to(vae_original_device)
            print(f"--- Returned VAE to {vae_original_device} ---")
            torch.cuda.empty_cache()
            gc.collect()

        ################ Determine Noise Dtype ################
        if hasattr(self.pipeline, 'text_encoder') and self.pipeline.text_encoder is not None:
            prompt_embeds_dtype = self.pipeline.text_encoder.dtype
        elif hasattr(self.pipeline, 'unet') and self.pipeline.unet is not None:
            prompt_embeds_dtype = self.pipeline.unet.dtype
        else:
            prompt_embeds_dtype = torch.float32 if self.device==torch.device("cpu") else torch.float16 # Default fallback

        ################ Prepare Noise and Run Pre-inference ################
        # Call the new helper method
        noise, timesteps_add_noise = self._prepare_noise_and_run_pre_inference(
            nframes=nframes,
            real_video_length=real_video_length,
            tar_height=tar_height,
            tar_width=tar_width,
            prompt_embeds_dtype=prompt_embeds_dtype,
            generator=generator,
            latents=latents,  # Pass the latents tensor (modified in-place)
            validation_masks_input=validation_masks_input,
            validation_images_input=validation_images_input,
            validation_prompt=validation_prompt,
            guidance_scale_final=guidance_scale_final
        )
        # 'latents' tensor is now potentially updated by the pre-inference step within the helper method

        ################  Main Frame-by-frame inference  ################
        print(f"--- Starting main inference loop using nframes={nframes} ---")
        ## Add noise to the potentially updated latents tensor
        # Use the 'noise' and 'timesteps_add_noise' returned by the helper method
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps_add_noise)
        latents_for_main_inference = noisy_latents # Start main inference from this noisy state

        # Clean up original noise tensor (returned by helper) and the original latents
        del noise, latents, noisy_latents
        gc.collect()
        torch.cuda.empty_cache()

        print(f"--- Calling main pipeline for {real_video_length} frames ---")
        with torch.no_grad():
            pipeline_output = self.pipeline(
                num_frames=nframes, # Pass the chunk size parameter
                prompt=validation_prompt,
                images=validation_images_input, # Use original full list
                masks=validation_masks_input,   # Use original full list
                num_inference_steps=self.num_inference_steps,
                generator=generator,
                guidance_scale=guidance_scale_final,
                latents=latents_for_main_inference, # Start from noisy latents
                output_type="pil" # Get PIL images directly
            )
            # Check output format
            if isinstance(pipeline_output, dict) and 'frames' in pipeline_output:
                images = pipeline_output.frames
            elif isinstance(pipeline_output, DiffuEraserPipelineOutput):
                images = pipeline_output.frames
            elif isinstance(pipeline_output, tuple) and len(pipeline_output)>0: # Older format?
                images = pipeline_output[0]
                print("Warning: Main pipeline output format unexpected (tuple), assuming frames.")
            else: # Assume direct frame output if not dict/tuple/known object
                 images = pipeline_output
                 print("Warning: Main pipeline output format unexpected, assuming frames.")

        # Ensure the number of output frames matches expected length
        if not isinstance(images, list) or len(images) < real_video_length:
             # Handle cases where output is not a list or length is wrong
             print(f"Error/Warning: Pipeline returned unexpected output type ({type(images)}) or incorrect frame count ({len(images) if isinstance(images, list) else 'N/A'}). Expected {real_video_length}. Attempting to recover if possible.")
             # Add recovery logic here if needed, e.g., raise error, pad frames, etc.
             # For now, we'll just proceed and trim/error later if needed.
             if not isinstance(images, list): images = [] # Prevent errors later if not a list
        images = images[:real_video_length] # Trim if pipeline returned more
        print(f"--- Main inference pipeline call complete, processed {len(images)} frames ---")

        # Clean up large latent tensor used for main inference
        del latents_for_main_inference
        gc.collect()
        torch.cuda.empty_cache()

        ################ Compose Final Video ################
        print(f"--- Composing final video using original masks and frames ---")
        binary_masks = validation_masks_input_ori # Use the original masks read from file
        mask_blurreds = []
        if blended:
            print(f"--- Applying blur blending to masks ---")
            for i in range(len(binary_masks)):
                mask_np = np.array(binary_masks[i])
                if mask_np.ndim == 3: mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)
                mask_blurred_np = cv2.GaussianBlur(mask_np, (21, 21), 0) / 255.0
                binary_mask_np = 1.0 - (1.0 - mask_np / 255.0) * (1.0 - mask_blurred_np)
                mask_blurreds.append(Image.fromarray((binary_mask_np * 255).astype(np.uint8)))
            binary_masks = mask_blurreds # Use the blurred versions for composition

        comp_frames = []
        frames_for_composition = resized_frames_ori # Use the original resized frames
        print(f"--- Composing {len(images)} generated frames with original frames ---")
        for i in range(len(images)):
            if i >= len(frames_for_composition) or i >= len(binary_masks):
                 print(f"Warning: Skipping composition for frame {i} due to list length mismatch.")
                 continue # Skip if lists are somehow shorter than generated images

            mask_np = np.expand_dims(np.array(binary_masks[i]), axis=2) / 255.0
            img_generated_np = np.array(images[i]).astype(np.uint8)
            img_original_np = np.array(frames_for_composition[i]).astype(np.uint8)

            img_comp_np = (img_generated_np * mask_np + img_original_np * (1.0 - mask_np)).astype(np.uint8)
            comp_frames.append(Image.fromarray(img_comp_np))

        del images, frames_for_composition, binary_masks, mask_blurreds # Free memory
        if 'validation_images_input' in locals(): del validation_images_input
        if 'validation_masks_input' in locals(): del validation_masks_input
        gc.collect()

        ################ Write Video ################
        print(f"--- Writing final video to: {output_path} ---")
        if not comp_frames:
             print("Error: No frames to write to video!")
             # Clean up potentially opened resources if needed before returning
             return None # Or raise error

        default_fps = fps
        output_size = comp_frames[0].size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v") # Common codec
        writer = cv2.VideoWriter(output_path, fourcc, default_fps, output_size)

        if not writer.isOpened():
             print(f"Error: Could not open video writer for {output_path}")
             return None

        try:
            for f_idx, frame_pil in enumerate(comp_frames):
                img_np_bgr = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
                writer.write(img_np_bgr)
        finally:
             writer.release() # Ensure writer is released even if errors occur during write

        print(f"--- Video writing complete ---")

        return output_path
            

    def _prepare_noise_and_run_pre_inference(
        self, nframes, real_video_length, tar_height, tar_width, prompt_embeds_dtype,
        generator, latents, validation_masks_input, validation_images_input,
        validation_prompt, guidance_scale_final
    ):
        """
        Generates noise, performs optional pre-inference updating latents in-place.

        Args:
            nframes (int): The number of frames per processing chunk.
            real_video_length (int): The total number of frames in the video.
            tar_height (int): Target height for latents.
            tar_width (int): Target width for latents.
            prompt_embeds_dtype (torch.dtype): Data type for noise generation.
            generator (torch.Generator or None): Random number generator.
            latents (torch.Tensor): The initial latent tensor (will be modified in-place).
            validation_masks_input (list[PIL.Image]): List of mask images for pre-inference.
            validation_images_input (list[PIL.Image]): List of masked images for pre-inference.
            validation_prompt (str): Text prompt for the pipeline.
            guidance_scale_final (float): Guidance scale for the pipeline.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - noise: The full noise tensor tiled/trimmed for the video length.
                - timesteps_add_noise: Timesteps used for adding noise ([0]).
        """
        ################ Prepare Noise Shape and Base Noise ################
        shape = (
            nframes, # Base shape uses nframes
            self.pipeline.unet.config.in_channels, # Get channels from unet config
            tar_height // self.vae_scale_factor,
            tar_width // self.vae_scale_factor
        )
        print(f"--- Preparing base noise for shape {shape} with dtype {prompt_embeds_dtype} ---")
        noise_pre = randn_tensor(shape, device=self.device, dtype=prompt_embeds_dtype, generator=generator)

        ################ Repeat Noise for Full Video Length ################
        # Calculate necessary repeats based on real_video_length and nframes
        n_repeats_needed = (real_video_length + nframes - 1) // nframes
        print(f"--- Repeating noise pattern {n_repeats_needed} times for {real_video_length} frames ---")
        # Use einops repeat correctly: repeat the *temporal* dimension 't'
        noise = repeat(noise_pre, "t c h w -> (repeat t) c h w", repeat=n_repeats_needed)
        # Trim excess noise to match the exact video length
        noise = noise[:real_video_length, ...]
        print(f"--- Final noise tensor shape: {noise.shape} ---")

        ################ Timesteps for Noise Addition ################
        # Fixed timestep [0] seems intended for how schedulers like UniPC/TCD/LCM are used here.
        timesteps_add_noise = torch.tensor([0], device=self.device)
        timesteps_add_noise = timesteps_add_noise.long()

        ################ Pre-inference (Optional) ################
        latents_pre_out = None # Initialize
        sample_index = None    # Initialize

        # Condition uses the actual nframes value
        if real_video_length > nframes * 2:
            print(f"--- Running Pre-inference using nframes={nframes} on sampled frames ---")
            ## Sample indices based on nframes
            step = real_video_length / nframes # Use real_video_length for sampling step
            num_samples = min(nframes, real_video_length) # Ensure we don't sample more indices than available frames or the desired nframes
            sample_index = [int(i * step) for i in range(num_samples)]
            sample_index = [idx for idx in sample_index if idx < real_video_length] # Ensure indices are within bounds
            num_samples = len(sample_index) # Update num_samples after bounds check

            if num_samples == 0: # Handle edge case where sampling yields no indices
                print("--- Pre-inference sampling resulted in 0 frames. Skipping pre-inference. ---")
                latents_pre = None
            elif latents.shape[0] > max(sample_index): # Safely gather latents for sampled indices
                 print(f"--- Sampled {num_samples} indices for pre-inference: {sample_index[:10]}... ---") # Print first 10
                 latents_pre = torch.stack([latents[i] for i in sample_index])
                 # Gather corresponding inputs
                 validation_masks_input_pre = [validation_masks_input[i] for i in sample_index]
                 validation_images_input_pre = [validation_images_input[i] for i in sample_index]
            else:
                 print(f"ERROR: Not enough latents ({latents.shape[0]}) generated for pre-inference sampling (max index: {max(sample_index)}). Skipping pre-inference.")
                 latents_pre = None # Indicate failure/skip

            if latents_pre is not None: # Only proceed if latents were sampled correctly
                # Ensure noise_pre matches the number of frames being processed (num_samples)
                if noise_pre.shape[0] != num_samples:
                     print(f"Warning: Adjusting noise_pre shape from {noise_pre.shape[0]} to {num_samples} for pre-inference")
                     noise_pre_adjusted = noise_pre[:num_samples].clone() # Use clone to avoid modifying original noise_pre if needed elsewhere (though likely not here)
                else:
                     noise_pre_adjusted = noise_pre

                ## Add noise using the sampled latents and adjusted noise
                noisy_latents_pre = self.noise_scheduler.add_noise(latents_pre, noise_pre_adjusted, timesteps_add_noise)
                latents_pre_input = noisy_latents_pre # Use noisy latents as input to pipeline

                # Run the pipeline on the sampled subset
                print(f"--- Calling pipeline for pre-inference ({num_samples} frames) ---")
                with torch.no_grad():
                    # Pass the actual number of frames being processed
                    pipeline_output = self.pipeline(
                        num_frames=num_samples,
                        prompt=validation_prompt,
                        images=validation_images_input_pre,
                        masks=validation_masks_input_pre,
                        num_inference_steps=self.num_inference_steps,
                        generator=generator,
                        guidance_scale=guidance_scale_final,
                        latents=latents_pre_input, # Start from noisy latents
                        output_type="latent" # Get latents directly
                    )
                    # Check output format
                    if isinstance(pipeline_output, dict) and 'latents' in pipeline_output:
                         latents_pre_out = pipeline_output.latents
                    elif isinstance(pipeline_output, DiffuEraserPipelineOutput):
                         latents_pre_out = pipeline_output.latents
                    else:
                         latents_pre_out = pipeline_output # Assume direct output
                         print("Warning: Pre-inference pipeline output format unexpected, assuming latents tensor.")

                print(f"--- Pre-inference pipeline call complete ---")
                torch.cuda.empty_cache()
                del latents_pre_input, noisy_latents_pre, noise_pre_adjusted # Clean up intermediate tensors

                # Update main latents tensor with pre-inference results
                if latents_pre_out is not None and latents_pre_out.shape[0] == num_samples:
                    print(f"--- Updating main latents tensor with {num_samples} pre-inference results ---")
                    for i, index in enumerate(sample_index):
                        if i < latents_pre_out.shape[0] and index < latents.shape[0]:
                            # Update latents (move pre-inf latent back to GPU if needed, already on device from pipeline)
                            latents[index] = latents_pre_out[i] # Modify the input 'latents' tensor
                        else:
                            print(f"Warning: Index mismatch during pre-inference LATENT update (i={i}, index={index}). Skipping update for this index.")
                    del latents_pre_out # Free memory
                    torch.cuda.empty_cache()
                else: # If latents_pre_out was None or shape mismatch
                     print("--- Skipping pre-inference result application due to missing/mismatched latents_pre_out ---")
                     if latents_pre_out is not None: del latents_pre_out # Clean up if exists
                     latents_pre_out = None # Ensure it's None for logic flow

            else: # latents_pre was None (sampling failed or error)
                 print("--- Skipping pre-inference pipeline call due to sampling issues or condition not met ---")
                 latents_pre_out = None # Ensure it's None

        else: # real_video_length <= nframes * 2
            print("--- Skipping Pre-inference step (video too short or condition not met) ---")
            latents_pre_out = None # Ensure variable exists but is None
            sample_index = None

        # Final cleanup before returning
        del noise_pre # Base noise pattern is no longer needed outside this scope
        if 'latents_pre_out' in locals() and latents_pre_out is not None:
            del latents_pre_out # Should have been deleted earlier, but just in case
        if 'latents_pre' in locals() and latents_pre is not None:
            del latents_pre # Sampled latents before noise addition
        gc.collect()
        torch.cuda.empty_cache()

        # Return the full noise tensor and the timesteps for adding noise.
        # The 'latents' tensor passed as input has been modified in-place if pre-inference ran.
        return noise, timesteps_add_noise


