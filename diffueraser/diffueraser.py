
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
from propainter.utils.video_read_write import read_frames_high_fidelity_ffmpeg, write_video_with_ffmpeg

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


class DiffuEraser:
    def __init__(
            self, device, base_model_path, vae_path, diffueraser_path, 
            revision=None,
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
                save_mode='lossless_ffv1_rgb',
                max_img_size = 1280, max_video_length=10, mask_dilation_iter=4,
                nframes=22, # Keep this default or set slightly lower (e.g., 16, 18) if needed
                seed=None, revision = None, guidance_scale=None, blended=True):
        validation_prompt = "" 
        guidance_scale_final = self.guidance_scale if guidance_scale is None else guidance_scale

        if (max_img_size<256 or max_img_size>1920): # Increased upper limit slightly based on code analysis
            raise ValueError("The max_img_size must be between 256 and 1920.")

        ############### read input video ################
        print(f"--- Reading video: {validation_image} using FFmpeg ---")
        try:
            frames_pil, fps, img_size, color_info, n_total_frames = read_frames_high_fidelity_ffmpeg(
                video_path=validation_image,
                max_length=max_video_length if max_video_length > 0 else 99999.0,
            )
            if not frames_pil: raise ValueError("FFmpeg reader returned no frames.")
            frames = frames_pil # Assign to the variable name used later
        except Exception as e:
             print(f"FATAL ERROR reading video with FFmpeg: {e}")
             raise # Re-raise the error to stop execution
        video_len = len(frames) # video_len is now n_total_frames directly
        print(f"--- Video read: {video_len} frames, FPS: {fps}, Size: {img_size} ---")

        ################     read mask    ################
        print(f"--- Reading mask: {validation_mask} ---")
        mask_frames_pil = []
        # Determine if mask is a video or image
        mask_is_video = validation_mask.lower().endswith(('.mp4', '.mov', '.avi'))

        if mask_is_video:
            print("--- Mask is video, reading with FFmpeg ---")
            try:
                mask_frames_pil, mask_fps, mask_size, _, mask_frames_read = read_frames_high_fidelity_ffmpeg(
                    video_path=validation_mask,
                    max_length=max_video_length if max_video_length > 0 else 99999.0,
                )
                if not mask_frames_pil: raise ValueError("FFmpeg mask reader returned no frames.")
                # Optional: Check consistency
                if abs(mask_fps - fps) > 0.1: print(f"Warning: Mask FPS {mask_fps} differs significantly from video FPS {fps}.")
                if mask_size != img_size: print(f"Warning: Mask size {mask_size} differs from video size {img_size}. Resizing mask.")
                if mask_frames_read < video_len: print(f"Warning: Mask video has fewer frames ({mask_frames_read}) than video ({video_len}).")
                # Trim or pad mask_frames_pil if necessary to match video_len (simplest: trim)
                mask_frames_pil = mask_frames_pil[:video_len]

            except Exception as e:
                 print(f"FATAL ERROR reading mask video with FFmpeg: {e}")
                 raise
        else:
            print("--- Mask is image, reading with PIL ---")
            try:
                # Read single image mask
                mask_img = Image.open(validation_mask)
                # Repeat the single mask for all video frames
                mask_frames_pil = [mask_img] * video_len
            except Exception as e:
                 print(f"FATAL ERROR reading mask image: {e}")
                 raise
        # Process mask frames (dilation, create masked images)
        print(f"--- Processing {len(mask_frames_pil)} mask frames (dilation, creating masked images) ---")
        validation_masks_input = [] # Will store dilated PIL masks
        validation_images_input = [] # Will store masked PIL images
        for i in range(video_len): # Iterate up to the actual video length
            if i >= len(mask_frames_pil):
                print(f"Warning: Ran out of mask frames at index {i}. Using last available mask.")
                mask_pil = mask_frames_pil[-1] # Reuse last mask
            else:
                mask_pil = mask_frames_pil[i]

            # Resize mask to match video frame size (use LANCZOS for consistency if possible, else NEAREST)
            if mask_pil.size != img_size:
                # Use LANCZOS if high quality is needed, NEAREST if strict binary mask preferred
                mask_pil = mask_pil.resize(img_size, Image.Resampling.NEAREST)

            # Convert to grayscale and NumPy
            mask_np = np.array(mask_pil.convert('L'))

            # Perform dilation (using cv2 like original `read_mask` for consistency)
            m = np.array(mask_np > 0).astype(np.uint8) # Binarize
            
            # --- Dilation step ---
            if mask_dilation_iter > 0:
                m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=mask_dilation_iter)

            # Store dilated mask as PIL Image
            dilated_mask_pil = Image.fromarray(m * 255)
            validation_masks_input.append(dilated_mask_pil)

            # Create masked image (using the correctly read `frames`)
            frame_np = np.array(frames[i]) # Get the corresponding video frame
            mask_for_apply = np.expand_dims(m, axis=2) # Shape (H, W, 1), values 0 or 1
            masked_image_np = frame_np * (1 - mask_for_apply) # Apply mask (0 keeps original, 1 makes black)
            masked_image_pil = Image.fromarray(masked_image_np.astype(np.uint8))
            validation_images_input.append(masked_image_pil)

        del mask_frames_pil # Free memory
        gc.collect()
        print(f"--- Mask processing complete: {len(validation_masks_input)} masks, {len(validation_images_input)} masked images ---")

        ################    read priori   ################
        print(f"--- Reading priori: {priori} using FFmpeg ---")
        try:
            priori_frames_pil, priori_fps, priori_size, _, priori_frames_read = read_frames_high_fidelity_ffmpeg(
                video_path=priori,
                max_length=0, # Read full priori video
            )
            if not priori_frames_pil: raise ValueError("FFmpeg priori reader returned no frames.")
            prioris = priori_frames_pil # Assign to variable name used later

            # Optional: Consistency checks
            if abs(priori_fps - fps) > 0.1: print(f"Warning: Priori FPS {priori_fps} differs from video FPS {fps}.")
            if priori_size != img_size: print(f"Warning: Priori size {priori_size} differs from video size {img_size}. Priori frames will be used as is.")
            if priori_frames_read != video_len: print(f"Warning: Priori frames read ({priori_frames_read}) differs from video length ({video_len}). Using {priori_frames_read} frames.")
            # Update video_len based on shortest input
            video_len = min(video_len, priori_frames_read) # Adjust length based on actual priori frames
        except Exception as e:
            print(f"FATAL ERROR reading priori video with FFmpeg: {e}")
            raise

        os.remove(priori) # Keep removal of the intermediate file
        print(f"--- Priori read: {len(prioris)} frames ---")

        ## recheck frame counts and trim lists to the minimum common length
        # This block might need adjustment based on the new video_len update above
        n_total_frames = min(video_len, len(validation_masks_input), len(prioris)) # Use updated video_len
        if n_total_frames != video_len:
             print(f"--- Trimming all inputs to shortest length: {n_total_frames} frames ---")
             # Trim all lists to the actual shortest length determined
             validation_masks_input = validation_masks_input[:n_total_frames]
             validation_images_input = validation_images_input[:n_total_frames]
             frames = frames[:n_total_frames] # Trim original frames list too
             prioris = prioris[:n_total_frames]
        # Check nframes requirement again AFTER trimming
        if(n_total_frames < nframes):
            raise ValueError(f"The final effective video duration ({n_total_frames} frames) is shorter than nframes ({nframes}). Processing requires at least nframes.")
        real_video_length = n_total_frames # Use this consistent length
        print(f"--- Confirmed final effective length: {real_video_length} frames ---")

        print(f"--- Verifying final frame sizes against target: {img_size} ---")
        # --- Update Resizing Calls / Verification ---
        # Check if resizing is still needed after FFmpeg read (img_size should be correct now)
        if prioris[0].size != img_size:
             print(f"Warning: Resizing prioris from {prioris[0].size} to {img_size}")
             prioris = resize_frames(prioris, img_size) # Use your resize_frames function
        if validation_masks_input[0].size != img_size:
             print(f"Warning: Resizing masks from {validation_masks_input[0].size} to {img_size}")
             validation_masks_input = resize_frames(validation_masks_input, img_size)
        if validation_images_input[0].size != img_size:
             print(f"Warning: Resizing masked images from {validation_images_input[0].size} to {img_size}")
             validation_images_input = resize_frames(validation_images_input, img_size)
        if frames[0].size != img_size: # Check the original frames list
             print(f"Warning: Resizing original frames from {frames[0].size} to {img_size}")
             resized_frames = resize_frames(frames, img_size)
        else:
             resized_frames = frames # No resize needed if size matches
        # Create deep copies *after* potential resizing for final composition
        resized_frames_ori = copy.deepcopy(resized_frames)
        validation_masks_input_ori = copy.deepcopy(validation_masks_input)
        print(f"--- Final processing dimensions confirmed: {resized_frames_ori[0].size} ---")

        # Get final target dimensions (redundant if assigned above, but safe)
        tar_width, tar_height = resized_frames_ori[0].size
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
        print(f"--- Writing final video (Mode: {save_mode}) to: {output_path} using FFmpeg ---") # Log the mode
        if not comp_frames:
             print("Error: No frames to write to video!")
             return None
        # Prepare frames as list of NumPy uint8 RGB arrays
        frames_np_list = [np.array(frame_pil).astype(np.uint8) for frame_pil in comp_frames]
        try:
            write_video_with_ffmpeg(
                frames_list=frames_np_list,
                output_path=output_path,
                fps=fps,
                size=frames_np_list[0].shape[1::-1], # Get (width, height) from np array
                original_video_path=validation_image, # Pass original crop for YUV tag probing if using YUV save_mode
                save_mode=save_mode
            )
            print(f"--- Video writing complete (Mode: {save_mode}) ---")
        except Exception as e:
             print(f"ERROR during FFmpeg video saving in DiffuEraser: {e}")
             # Handle error, maybe try to delete partial output file
             if os.path.exists(output_path):
                 try: os.remove(output_path)
                 except OSError: pass
             return None # Indicate failure

        del frames_np_list# Cleanup intermediate NumPy list if large
        gc.collect()
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


