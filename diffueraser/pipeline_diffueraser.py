import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import os
import gc
import PIL.Image
from einops import rearrange, repeat
from dataclasses import dataclass
import copy
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, IPAdapterMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, ImageProjection
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
    BaseOutput
)
from diffusers.utils.torch_utils import is_compiled_module, is_torch_version, randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UniPCMultistepScheduler,
)

from libs.unet_2d_condition import UNet2DConditionModel
from libs.brushnet_CA import BrushNetModel

os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "4"


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

def get_frames_context_swap(total_frames=192, overlap=4, num_frames_per_clip=24):
    if total_frames<num_frames_per_clip:
        num_frames_per_clip = total_frames
    context_list = []
    context_list_swap = []
    for i in range(1, 2):  # i=1
        sample_interval = np.array(range(0,total_frames,i))
        n = len(sample_interval)
        if n>num_frames_per_clip:
            ## [0,num_frames_per_clip-1], [num_frames_per_clip, 2*num_frames_per_clip-1]....
            for k in range(0,n-num_frames_per_clip,num_frames_per_clip-overlap):
                context_list.append(sample_interval[k:k+num_frames_per_clip])
            if k+num_frames_per_clip < n and i==1:
                context_list.append(sample_interval[n-num_frames_per_clip:n])
            context_list_swap.append(sample_interval[0:num_frames_per_clip])
            for k in range(num_frames_per_clip//2, n-num_frames_per_clip, num_frames_per_clip-overlap):
                context_list_swap.append(sample_interval[k:k+num_frames_per_clip])
            if k+num_frames_per_clip < n and i==1:
                context_list_swap.append(sample_interval[n-num_frames_per_clip:n])
        if n==num_frames_per_clip:
            context_list.append(sample_interval[n-num_frames_per_clip:n])
            context_list_swap.append(sample_interval[n-num_frames_per_clip:n])
    return context_list, context_list_swap

@dataclass
class DiffuEraserPipelineOutput(BaseOutput):
    frames: Union[torch.Tensor, np.ndarray]
    latents: Union[torch.Tensor, np.ndarray]

class StableDiffusionDiffuEraserPipeline(
    DiffusionPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    LoraLoaderMixin,
    IPAdapterMixin,
    FromSingleFileMixin,
):
    r"""
    Pipeline for video inpainting using Video Diffusion Model with BrushNet guidance.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.LoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.LoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        brushnet ([`BrushNetModel`]`):
            Provides additional conditioning to the `unet` during the denoising process.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        brushnet: BrushNetModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            brushnet=brushnet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.register_to_config(requires_safety_checker=requires_safety_checker)
        print("--- Enabling gradient checkpointing for UNet and BrushNet ---")
        # Ensure the underlying models support this method
        if hasattr(self.unet, "enable_gradient_checkpointing"):
            self.unet.enable_gradient_checkpointing()
        else:
             print("Warning: UNet does not support enable_gradient_checkpointing()")
        if hasattr(self.brushnet, "enable_gradient_checkpointing"):
            self.brushnet.enable_gradient_checkpointing()
        else:
             print("Warning: BrushNet does not support enable_gradient_checkpointing()")

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        **kwargs,
    ):
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        prompt_embeds_tuple = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            **kwargs,
        )

        # concatenate for backwards comp
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        return prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image
    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        if output_hidden_states:
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            image_embeds = self.image_encoder(image).image_embeds
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_embeds = torch.zeros_like(image_embeds)

            return image_embeds, uncond_image_embeds
        
    def decode_latents(self, latents, weight_dtype):
        latents = 1 / self.vae.config.scaling_factor * latents
        video = []
        for t in range(latents.shape[0]):
            video.append(self.vae.decode(latents[t:t+1, ...].to(weight_dtype)).sample)
        video = torch.concat(video, dim=0)
        
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        video = video.float()
        return video

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_ip_adapter_image_embeds
    def prepare_ip_adapter_image_embeds(
        self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        if ip_adapter_image_embeds is None:
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            image_embeds = []
            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )
                single_image_embeds = torch.stack([single_image_embeds] * num_images_per_prompt, dim=0)
                single_negative_image_embeds = torch.stack(
                    [single_negative_image_embeds] * num_images_per_prompt, dim=0
                )

                if do_classifier_free_guidance:
                    single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds])
                    single_image_embeds = single_image_embeds.to(device)

                image_embeds.append(single_image_embeds)
        else:
            repeat_dims = [1]
            image_embeds = []
            for single_image_embeds in ip_adapter_image_embeds:
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    single_image_embeds = single_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_image_embeds.shape[1:]))
                    )
                    single_negative_image_embeds = single_negative_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_negative_image_embeds.shape[1:]))
                    )
                    single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds])
                else:
                    single_image_embeds = single_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_image_embeds.shape[1:]))
                    )
                image_embeds.append(single_image_embeds)

        return image_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    # Copied from diffusers.pipelines.text_to_video_synthesis/pipeline_text_to_video_synth.TextToVideoSDPipeline.decode_latents
    def decode_latents(self, latents, weight_dtype):
        latents = 1 / self.vae.config.scaling_factor * latents
        video = []
        for t in range(latents.shape[0]):
            video.append(self.vae.decode(latents[t:t+1, ...].to(weight_dtype)).sample)
        video = torch.concat(video, dim=0)
        
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        video = video.float()
        return video

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        images,
        masks,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        brushnet_conditioning_scale=1.0,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
        callback_on_step_end_tensor_inputs=None,
    ):
        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        # Check `image`
        is_compiled = hasattr(F, "scaled_dot_product_attention") and isinstance(
            self.brushnet, torch._dynamo.eval_frame.OptimizedModule
        )
        if (
            isinstance(self.brushnet, BrushNetModel)
            or is_compiled
            and isinstance(self.brushnet._orig_mod, BrushNetModel)
        ):
            self.check_image(images, masks, prompt, prompt_embeds)
        else:
            assert False

        # Check `brushnet_conditioning_scale`
        if (
            isinstance(self.brushnet, BrushNetModel)
            or is_compiled
            and isinstance(self.brushnet._orig_mod, BrushNetModel)
        ):
            if not isinstance(brushnet_conditioning_scale, float):
                raise TypeError("For single brushnet: `brushnet_conditioning_scale` must be type `float`.")
        else:
            assert False

        if not isinstance(control_guidance_start, (tuple, list)):
            control_guidance_start = [control_guidance_start]

        if not isinstance(control_guidance_end, (tuple, list)):
            control_guidance_end = [control_guidance_end]

        if len(control_guidance_start) != len(control_guidance_end):
            raise ValueError(
                f"`control_guidance_start` has {len(control_guidance_start)} elements, but `control_guidance_end` has {len(control_guidance_end)} elements. Make sure to provide the same number of elements to each list."
            )

        for start, end in zip(control_guidance_start, control_guidance_end):
            if start >= end:
                raise ValueError(
                    f"control guidance start: {start} cannot be larger or equal to control guidance end: {end}."
                )
            if start < 0.0:
                raise ValueError(f"control guidance start: {start} can't be smaller than 0.")
            if end > 1.0:
                raise ValueError(f"control guidance end: {end} can't be larger than 1.0.")

        if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
            raise ValueError(
                "Provide either `ip_adapter_image` or `ip_adapter_image_embeds`. Cannot leave both `ip_adapter_image` and `ip_adapter_image_embeds` defined."
            )

        if ip_adapter_image_embeds is not None:
            if not isinstance(ip_adapter_image_embeds, list):
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be of type `list` but is {type(ip_adapter_image_embeds)}"
                )
            elif ip_adapter_image_embeds[0].ndim not in [3, 4]:
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be a list of 3D or 4D tensors but is {ip_adapter_image_embeds[0].ndim}D"
                )

    def check_image(self, images, masks, prompt, prompt_embeds):
        for image in images:
            image_is_pil = isinstance(image, PIL.Image.Image)
            image_is_tensor = isinstance(image, torch.Tensor)
            image_is_np = isinstance(image, np.ndarray)
            image_is_pil_list = isinstance(image, list) and isinstance(image[0], PIL.Image.Image)
            image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)
            image_is_np_list = isinstance(image, list) and isinstance(image[0], np.ndarray)

            if (
                not image_is_pil
                and not image_is_tensor
                and not image_is_np
                and not image_is_pil_list
                and not image_is_tensor_list
                and not image_is_np_list
            ):
                raise TypeError(
                    f"image must be passed and be one of PIL image, numpy array, torch tensor, list of PIL images, list of numpy arrays or list of torch tensors, but is {type(image)}"
                )
        for mask in masks:
            mask_is_pil = isinstance(mask, PIL.Image.Image)
            mask_is_tensor = isinstance(mask, torch.Tensor)
            mask_is_np = isinstance(mask, np.ndarray)
            mask_is_pil_list = isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image)
            mask_is_tensor_list = isinstance(mask, list) and isinstance(mask[0], torch.Tensor)
            mask_is_np_list = isinstance(mask, list) and isinstance(mask[0], np.ndarray)

            if (
                not mask_is_pil
                and not mask_is_tensor
                and not mask_is_np
                and not mask_is_pil_list
                and not mask_is_tensor_list
                and not mask_is_np_list
            ):
                raise TypeError(
                    f"mask must be passed and be one of PIL image, numpy array, torch tensor, list of PIL images, list of numpy arrays or list of torch tensors, but is {type(mask)}"
                )

        if image_is_pil:
            image_batch_size = 1
        else:
            image_batch_size = len(image)

        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)
        elif prompt_embeds is not None:
            prompt_batch_size = prompt_embeds.shape[0]

        if image_batch_size != 1 and image_batch_size != prompt_batch_size:
            raise ValueError(
                f"If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}"
            )

    def prepare_image(
        self,
        images,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        images_new = []
        for image in images:
            image = self.image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
            image_batch_size = image.shape[0]

            if image_batch_size == 1:
                repeat_by = batch_size
            else:
                # image batch size is the same as prompt batch size
                repeat_by = num_images_per_prompt

            image = image.repeat_interleave(repeat_by, dim=0)

            image = image.to(device=device, dtype=dtype)

            # if do_classifier_free_guidance and not guess_mode:
            #     image = torch.cat([image] * 2)
            images_new.append(image)

        return images_new

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None):
        # shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        #b,c,n,h,w
        shape = (
            batch_size,
            num_channels_latents,
            num_frames,
            height // self.vae_scale_factor, 
            width // self.vae_scale_factor
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            # noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            noise = rearrange(randn_tensor(shape, generator=generator, device=device, dtype=dtype), "b c t h w -> (b t) c h w")
        else:
            noise = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = noise * self.scheduler.init_noise_sigma
        return latents, noise
    
    @staticmethod
    def temp_blend(a, b, overlap):
        factor = torch.arange(overlap).to(b.device).view(overlap, 1, 1, 1) / (overlap - 1)
        a[:overlap, ...] = (1 - factor) * a[:overlap, ...] + factor * b[:overlap, ...]
        a[overlap:, ...] = b[overlap:, ...]
        return a

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            timesteps (`torch.Tensor`):
                generate embedding vectors at these timesteps
            embedding_dim (`int`, *optional*, defaults to 512):
                dimension of the embeddings to generate
            dtype:
                data type of the generated embeddings

        Returns:
            `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @torch.no_grad()
    def __call__(
        self,
        num_frames: Optional[int] = 24,
        prompt: Union[str, List[str]] = None,
        images: PipelineImageInput = None, ##masked images
        masks: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        brushnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            images (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The BrushNet branch input condition to provide guidance to the `unet` for generation. Usually masked images.
            masks (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The BrushNet branch input condition masks to provide guidance to the `unet` for generation.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`. Shape expected (T, C, H, W).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument. Shape [B, Seq, Dim].
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument. Shape [B, Seq, Dim].
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.FloatTensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of IP-adapters.
                Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should contain the negative image embedding
                if `do_classifier_free_guidance` is set to `True`.
                If not provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array` or `pt`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] of the UNet.
            brushnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the BrushNet are multiplied by `brushnet_conditioning_scale` before they are added
                to the residual in the original `unet`.
            guess_mode (`bool`, *optional*, defaults to `False`):
                In guess mode, the BrushNet encoder tries to recognize the content of the input image even if you remove
                all prompts. A `guidance_scale` value between 3.0 and 5.0 is recommended.
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the BrushNet starts applying.
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the BrushNet stops applying.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeine class.

        Returns:
            [`~pipelines.diffueraser.pipeline_diffueraser.DiffuEraserPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.diffueraser.pipeline_diffueraser.DiffuEraserPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated frames.
        """
        # --- 0. Initial Setup ---
        # Get BrushNet reference early
        brushnet = self.brushnet._orig_mod if is_compiled_module(self.brushnet) else self.brushnet

        # Handle callbacks, validate inputs, set internal state, normalize control guidance
        callback, callback_steps, control_guidance_start, control_guidance_end = \
            self._setup_pipeline_state_and_validate(
                prompt, images, masks, negative_prompt, prompt_embeds, negative_prompt_embeds,
                ip_adapter_image, ip_adapter_image_embeds, brushnet_conditioning_scale,
                control_guidance_start, control_guidance_end, callback_on_step_end_tensor_inputs,
                guidance_scale, clip_skip, cross_attention_kwargs, **kwargs
            )

        # --- 1. Prepare Common Inputs (Prompts, IP Adapter, basic params) ---
        batch_size, device, video_length, effective_guess_mode, prompt_embeds_unet, added_cond_kwargs = \
            self._prepare_common_inputs(
                prompt, prompt_embeds, negative_prompt, negative_prompt_embeds, images,
                num_images_per_prompt, ip_adapter_image, ip_adapter_image_embeds,
                brushnet, guess_mode
            )

        # --- 2. Prepare Image-Based Inputs (VAE Encode, Masks, Conditioning Tensor) ---
        conditioning_latents, height, width, latent_height, latent_width = \
            self._prepare_image_based_inputs(
                images, masks, width, height, batch_size, num_images_per_prompt, device
            )

        # --- 3. Prepare Latents and Scheduler ---
        latents, timesteps, num_inference_steps, extra_step_kwargs = \
            self._prepare_latents_and_scheduler(
                batch_size, num_images_per_prompt, video_length, height, width,
                prompt_embeds_unet, device, generator, latents,
                num_inference_steps, timesteps, eta
            )

        # --- 4. Initialize Denoising Loop Variables ---
        (
            brushnet_keep, context_list, context_list_swap, scheduler_status, scheduler_status_swap,
            value, count, num_warmup_steps, is_unet_compiled, is_brushnet_compiled, is_torch_higher_equal_2_1,
            unet_device, cpu_device, timestep_cond
        ) = self._initialize_denoising_loop_variables(
            timesteps, control_guidance_start, control_guidance_end, brushnet,
            video_length, num_frames, latents, device, num_inference_steps
        )
        print(f"--- Device Setup: Target Computation Device = {unet_device}, CPU on {cpu_device} ---")

        print("--- Explicitly moving Text Encoder and VAE to CPU before main loop ---")
        if hasattr(self, 'text_encoder') and self.text_encoder is not None:
            self.text_encoder.to('cpu')
        if hasattr(self, 'vae') and self.vae is not None:
            # VAE is needed again for decoding AFTER the loop, so offloading is fine.
            self.vae.to('cpu')
        if hasattr(self, 'image_encoder') and self.image_encoder is not None:
             print("--- Offloading IP Adapter Image Encoder ---")
             self.image_encoder.to('cpu')
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # --- 5. Denoising loop ---
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                value.zero_()
                count.zero_()
                value = value.to(unet_device)
                count = count.to(unet_device)

                if (i % 2 == 1):
                    context_list_choose = context_list_swap
                    scheduler_status_choose = scheduler_status_swap
                else:
                    context_list_choose = context_list
                    scheduler_status_choose = scheduler_status

                # Inner loop processing context windows
                for j, context in enumerate(context_list_choose):
                    current_latents_for_context = latents[context, :, :, :]

                    value, count = self._process_context_step(
                        j=j, context=context, scheduler_state=scheduler_status_choose[j],
                        current_latents_context=current_latents_for_context,
                        value=value, count=count, t=t, i=i,
                        unet_device=unet_device, cpu_device=cpu_device,
                        original_prompt_embeds=prompt_embeds_unet, # Use UNet embeds
                        conditioning_latents=conditioning_latents,
                        timestep_cond=timestep_cond,
                        brushnet_keep=brushnet_keep,
                        brushnet_conditioning_scale=brushnet_conditioning_scale,
                        guess_mode=effective_guess_mode, # Use effective guess mode
                        is_unet_compiled=is_unet_compiled,
                        is_brushnet_compiled=is_brushnet_compiled,
                        is_torch_higher_equal_2_1=is_torch_higher_equal_2_1,
                        extra_step_kwargs=extra_step_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                    )
                # End inner context loop

                latents = self._finalize_timestep(
                    value=value, latents=latents, i=i, t=t, timesteps=timesteps,
                    num_warmup_steps=num_warmup_steps, progress_bar=progress_bar,
                    prompt_embeds_unet=prompt_embeds_unet, # Pass embeds for callback
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                    callback=callback, callback_steps=callback_steps # Pass legacy args
                )
        # --- End Denoising Loop ---

        # --- 6. Post-processing and Output ---
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            print("--- Offloading UNet and BrushNet ---")
            self.unet.to("cpu")
            self.brushnet.to("cpu")
            torch.cuda.empty_cache()

        print("--- Final garbage collection before returning from pipeline __call__ ---")
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        if output_type == "latent":
            print("--- Returning latents directly (output_type='latent') ---")
            output = DiffuEraserPipelineOutput(frames=None, latents=latents)
        else:
            print(f"--- Decoding final latents to output_type='{output_type}' ---")
            decode_dtype = self.vae.dtype
            if latents.dtype != decode_dtype and not torch.backends.mps.is_available():
                print(f"--- Casting latents from {latents.dtype} to {decode_dtype} for VAE decoding ---")
                latents_for_decode = latents.to(decode_dtype)
            else:
                latents_for_decode = latents

            video_tensor = self.decode_latents(latents_for_decode, weight_dtype=decode_dtype)

            if output_type == "pt":
                video = video_tensor
            else:
                video = self.image_processor.postprocess(video_tensor, output_type=output_type)

            output = DiffuEraserPipelineOutput(frames=video, latents=latents)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (output.frames,)
        else:
            print("--- Returning decoded frames and final latents ---")
            return output



    def _setup_pipeline_state_and_validate(
        self,
        prompt, images, masks,
        negative_prompt, prompt_embeds, negative_prompt_embeds,
        ip_adapter_image, ip_adapter_image_embeds,
        brushnet_conditioning_scale,
        control_guidance_start, control_guidance_end,
        callback_on_step_end_tensor_inputs,
        guidance_scale, clip_skip, cross_attention_kwargs,
        **kwargs,
    ) -> Tuple[Optional[Callable], Optional[int], List[float], List[float]]:
        """Handles callbacks, validates inputs, normalizes control guidance, and sets internal state."""
        # Handle callbacks deprecation
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        # Align format for control guidance start/end to be lists
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            control_guidance_start, control_guidance_end = (
                [control_guidance_start],
                [control_guidance_end],
            )
        # 1. Check inputs. Raise error if not correct
        # Note: callback_steps is passed here for validation if needed by check_inputs
        self.check_inputs(
            prompt,
            images,
            masks,
            callback_steps, # Pass the potentially None value
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            brushnet_conditioning_scale,
            control_guidance_start, # Pass the normalized list
            control_guidance_end,   # Pass the normalized list
            callback_on_step_end_tensor_inputs,
        )
        # Set internal attributes used by helper methods/properties
        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        return callback, callback_steps, control_guidance_start, control_guidance_end


    def _prepare_common_inputs(
        self,
        prompt, prompt_embeds, negative_prompt, negative_prompt_embeds,
        images, # Needed for video_length
        num_images_per_prompt,
        ip_adapter_image, ip_adapter_image_embeds,
        brushnet, # Pass BrushNet reference
        guess_mode,
    ) -> Tuple[int, torch.device, int, bool, torch.Tensor, Optional[Dict[str, Any]]]:
        """Determines call parameters, encodes prompts, and prepares IP adapter."""
        # Determine batch size
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        elif prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        else:
            batch_size = 1 # Default if no prompt info

        device = self._execution_device # Get the primary device

        # Determine guess mode
        global_pool_conditions = getattr(brushnet.config, "global_pool_conditions", False)
        effective_guess_mode = guess_mode or global_pool_conditions
        video_length = len(images)

        # Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        original_prompt_embeds, original_negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )
        # Prepare the specific prompt embeddings required by the UNet based on CFG setting
        if self.do_classifier_free_guidance:
            prompt_embeds_unet = torch.cat([original_negative_prompt_embeds, original_prompt_embeds])
        else:
            prompt_embeds_unet = original_prompt_embeds

        # Prepare IP Adapter embeddings if needed
        added_cond_kwargs = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )
            added_cond_kwargs = {"image_embeds": image_embeds}

        return batch_size, device, video_length, effective_guess_mode, prompt_embeds_unet, added_cond_kwargs


    def _prepare_image_based_inputs(
        self,
        images: PipelineImageInput,
        masks: PipelineImageInput,
        width: Optional[int],
        height: Optional[int],
        batch_size: int,
        num_images_per_prompt: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, int, int, int, int]:
        """Prepares image/mask inputs, VAE encodes, prepares conditioning tensor."""
        # Get BrushNet module reference (needed for dtype)
        brushnet = self.brushnet._orig_mod if is_compiled_module(self.brushnet) else self.brushnet
        if not isinstance(brushnet, BrushNetModel):
                raise ValueError("BrushNet model not found or is not the expected type.")

        # Preprocess the list of masked images (input `images`)
        prepared_images = self.prepare_image(
            images=images,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=self.vae.dtype, # Prepare images in VAE dtype for encoding
        )
        # Preprocess the list of masks (input `masks`)
        prepared_masks = self.prepare_image(
            images=masks,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=torch.float32, # Prepare masks as float for processing
        )
        # Get final height/width from prepared tensors if not provided
        height, width = prepared_images[0].shape[-2:]
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor

        # Convert prepared masks to binary single channel format (INVERTED: 0=masked)
        inverted_masks_new = []
        for processed_mask_tensor in prepared_masks:
                inverted_binary_mask = (processed_mask_tensor.sum(dim=1, keepdim=True) < 1e-5).to(processed_mask_tensor.dtype)
                inverted_masks_new.append(inverted_binary_mask)
        prepared_masks_processed = inverted_masks_new # List of [B, 1, H, W] tensors

        # VAE Encode prepared images (conditioning latents)
        images_cat = torch.cat(prepared_images, dim=0)
        conditioning_latents_list = []
        num_vae_batch_cond = 4 # Adjust based on VRAM
        print(f"--- Preparing conditioning latents using VAE batch size {num_vae_batch_cond} ---")
        vae_original_device_cond = self.vae.device
        cpu_device = torch.device("cpu")
        try:
            self.vae.to(device)
            with torch.no_grad():
                    for i_vae in range(0, images_cat.shape[0], num_vae_batch_cond):
                        batch_images_cond = images_cat[i_vae : i_vae + num_vae_batch_cond]
                        encoded_latent_dist = self.vae.encode(batch_images_cond).latent_dist
                        conditioning_latents_list.append(encoded_latent_dist.sample().to(cpu_device))
            conditioning_latents = torch.cat(conditioning_latents_list, dim=0).to(device)
        finally:
            self.vae.to(vae_original_device_cond)
            del conditioning_latents_list
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect()

        conditioning_latents = conditioning_latents * self.vae.config.scaling_factor

        print(f"--- Deleting intermediate 'images_cat' tensor (shape: {images_cat.shape}) ---")
        del images_cat, prepared_images
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

        # Interpolate masks and concatenate with conditioning latents
        inverted_masks_cat = torch.cat(prepared_masks_processed, dim=0)
        masks_interp = torch.nn.functional.interpolate(
            inverted_masks_cat, size=(latent_height, latent_width)
        )
        print(f"--- Deleting intermediate 'inverted_masks_cat' tensor (shape: {inverted_masks_cat.shape}) ---")
        del inverted_masks_cat, prepared_masks_processed
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

        conditioning_latents = torch.cat([
            conditioning_latents,
            masks_interp.to(conditioning_latents.device, dtype=conditioning_latents.dtype)
        ], dim=1)
        print(f"--- Final conditioning_latents shape (incl. INVERTED mask): {conditioning_latents.shape} ---")
        del masks_interp
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

        return conditioning_latents, height, width, latent_height, latent_width


    def _prepare_latents_and_scheduler(
        self,
        batch_size: int,
        num_images_per_prompt: int,
        num_frames_for_noise: int, # Use video_length here
        height: int,
        width: int,
        prompt_embeds_unet: torch.Tensor, # For dtype
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
        latents: Optional[torch.FloatTensor], # User-provided latents
        num_inference_steps: int,
        timesteps: Optional[List[int]], # User-provided timesteps
        eta: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, Dict[str, Any]]:
        """Prepares initial latents, timesteps, and scheduler kwargs."""
        # 5. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps) # Set internal state

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents_shape = (
            num_frames_for_noise, # Use video_length
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor
        )

        if latents is None:
            print(f"--- Generating initial noise for shape (T,C,H,W): {latents_shape} ---")
            noise_dtype = prompt_embeds_unet.dtype
            # NOTE: Assuming batch_size * num_images_per_prompt = 1 for video generation noise
            effective_batch_size_for_noise = 1
            # Adapt prepare_latents call if needed, or generate directly
            noise = randn_tensor(latents_shape, generator=generator, device=device, dtype=noise_dtype)
            # Scale the initial noise by the scheduler's sigma
            latents = noise * self.scheduler.init_noise_sigma
            del noise
        else:
            print(f"--- Using provided latents, shape: {latents.shape} ---")
            latents = latents.to(device=device, dtype=prompt_embeds_unet.dtype)
            if latents.shape[1:] != latents_shape[1:]: # Check channel/spatial dims match
                    raise ValueError(f"Provided latents shape {latents.shape} C/H/W dimensions do not match expected {latents_shape}")
            if latents.shape[0] != num_frames_for_noise: # Check temporal dim
                    print(f"Warning: Provided latents temporal dim {latents.shape[0]} != video length {num_frames_for_noise}.")
                    # Decide handling: truncate, error, etc. Let's assume it should match.
                    if latents.shape[0] < num_frames_for_noise:
                        raise ValueError("Provided latents are shorter than video length.")
                    latents = latents[:num_frames_for_noise] # Truncate if longer
            # Provided latents are assumed already noisy, no scaling needed.

        # 7. Prepare extra step kwargs for scheduler
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        return latents, timesteps, num_inference_steps, extra_step_kwargs


    def _initialize_denoising_loop_variables(
        self,
        timesteps: torch.Tensor,
        control_guidance_start: List[float],
        control_guidance_end: List[float],
        brushnet, # Pass BrushNet reference
        video_length: int,
        num_frames: int, # Context window size
        latents: torch.Tensor,
        device: torch.device,
        num_inference_steps: int,
    ) -> Tuple[
        List[float], List, List, List[Dict], List[Dict], torch.Tensor, torch.Tensor,
        int, bool, bool, bool, torch.device, torch.device, Optional[torch.Tensor]
    ]:
        """Initializes variables needed for the main denoising loop."""
        # Calculate brushnet_keep schedule
        brushnet_keep = []
        for i_keep in range(len(timesteps)):
            keeps = [
                1.0 - float(i_keep / len(timesteps) < s or (i_keep + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            brushnet_keep.append(keeps[0] if isinstance(brushnet, BrushNetModel) else keeps)

        # Prepare context lists and scheduler states
        overlap = num_frames // 4
        context_list, context_list_swap = get_frames_context_swap(video_length, overlap=overlap, num_frames_per_clip=num_frames)
        scheduler_status = [copy.deepcopy(self.scheduler.__dict__)] * len(context_list)
        scheduler_status_swap = [copy.deepcopy(self.scheduler.__dict__)] * len(context_list_swap)

        # Initialize accumulators
        value = torch.zeros_like(latents)
        count = torch.zeros_like(latents[:, 0:1, :, :]) # Single channel count

        # Denoising loop setup parameters
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        is_unet_compiled = is_compiled_module(self.unet)
        is_brushnet_compiled = is_compiled_module(self.brushnet)
        is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")

        # Determine target devices
        unet_device = device # Use the main pipeline device
        cpu_device = torch.device("cpu")

        # Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            # Assuming batch_size = 1 for video
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(1)
            target_dtype = latents.dtype
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim,
                dtype=target_dtype # Request float16 directly from embedding function
            ).to(device=device, dtype=latents.dtype)

        return (
            brushnet_keep, context_list, context_list_swap, scheduler_status, scheduler_status_swap,
            value, count, num_warmup_steps, is_unet_compiled, is_brushnet_compiled, is_torch_higher_equal_2_1,
            unet_device, cpu_device, timestep_cond
        )


    # Define the helper method within the class
    def _process_context_step(
        self,
        j: int,
        context: np.ndarray, # Or torch.Tensor depending on context_list type
        scheduler_state: Dict,
        current_latents_context: torch.Tensor, # Input latents SLICE for this context
        value: torch.Tensor, # Accumulated results tensor (modified by blending/assignment)
        count: torch.Tensor, # Accumulated counts tensor (incremented)
        t: Union[torch.Tensor, int], # Current timestep
        i: int, # Outer loop index (for brushnet_keep)
        unet_device: torch.device,
        cpu_device: torch.device,
        original_prompt_embeds: torch.Tensor, # Embeds for UNet (potentially cfg-concatenated)
        conditioning_latents: torch.Tensor,
        timestep_cond: Optional[torch.Tensor],
        brushnet_keep: List,
        brushnet_conditioning_scale: Union[float, List[float]],
        guess_mode: bool,
        is_unet_compiled: bool,
        is_brushnet_compiled: bool,
        is_torch_higher_equal_2_1: bool,
        extra_step_kwargs: Dict,
        added_cond_kwargs: Optional[Dict],
    ) -> Tuple[torch.Tensor, torch.Tensor]: # Returns updated value, count (modified in place)
        """
        Processes a single context window (j) within a denoising timestep (t).
        Updates the `value` and `count` tensors based on the processing of `current_latents_context`.
        (Keep full docstring as before)
        """
        # Restore scheduler state for this context window
        self.scheduler.__dict__.update(scheduler_state)

        # Ensure context latents are on the main device
        latents_j = current_latents_context.to(unet_device)

        # Mark step for CUDA graphs if applicable
        if (is_unet_compiled and is_brushnet_compiled) and is_torch_higher_equal_2_1:
            torch._inductor.cudagraph_mark_step_begin()

        # Expand latents for classifier-free guidance
        latent_model_input = torch.cat([latents_j] * 2) if self.do_classifier_free_guidance else latents_j
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        # --- Prepare BrushNet Inputs ---
        # Determine control input and context-specific prompt embeddings for BrushNet
        if guess_mode and self.do_classifier_free_guidance:
            control_model_input = latents_j # No need to repeat here
            control_model_input = self.scheduler.scale_model_input(control_model_input, t)
            # Use only positive prompts, reshaped for context length
            brushnet_prompt_embeds = original_prompt_embeds.chunk(2)[1]
            brushnet_prompt_embeds = rearrange(repeat(brushnet_prompt_embeds, "b c d -> b t c d", t=len(context)), 'b t c d -> (b t) c d')
        else:
            control_model_input = latent_model_input
            # Reshape original prompts (potentially cfg-concatenated) for context length
            if self.do_classifier_free_guidance:
                neg_brushnet_prompt_embeds, pos_brushnet_prompt_embeds = original_prompt_embeds.chunk(2)
                pos_brushnet_prompt_embeds = rearrange(repeat(pos_brushnet_prompt_embeds, "b c d -> b t c d", t=len(context)), 'b t c d -> (b t) c d')
                neg_brushnet_prompt_embeds = rearrange(repeat(neg_brushnet_prompt_embeds, "b c d -> b t c d", t=len(context)), 'b t c d -> (b t) c d')
                brushnet_prompt_embeds = torch.cat([neg_brushnet_prompt_embeds, pos_brushnet_prompt_embeds])
            else:
                brushnet_prompt_embeds = rearrange(repeat(original_prompt_embeds, "b c d -> b t c d", t=len(context)), 'b t c d -> (b t) c d')

        # Determine BrushNet conditioning scale
        if isinstance(brushnet_keep[i], list):
            cond_scale = [c * s for c, s in zip(brushnet_conditioning_scale, brushnet_keep[i])]
        else:
            brushnet_cond_scale = brushnet_conditioning_scale
            if isinstance(brushnet_cond_scale, list):
                brushnet_cond_scale = brushnet_cond_scale[0]
            cond_scale = brushnet_cond_scale * brushnet_keep[i]
        # --- End Prepare BrushNet Inputs ---

        # --- BrushNet Inference ---
        self.brushnet.to(unet_device)
        t_val = t if isinstance(t, int) else t.item() # For printing
        print(f"[t={t_val}, j={j}] Running BrushNet on {unet_device}...")

        # Ensure inputs are on GPU
        control_model_input_gpu = control_model_input.to(unet_device)
        brushnet_prompt_embeds_gpu = brushnet_prompt_embeds.to(unet_device)
        brushnet_cond_indices = context
        # Slice conditioning_latents (already on GPU from earlier prep)
        brushnet_cond_sliced = conditioning_latents[brushnet_cond_indices, :, :, :]
        # Repeat for CFG if needed
        brushnet_cond_input = torch.cat([brushnet_cond_sliced]*2) if self.do_classifier_free_guidance else brushnet_cond_sliced
        brushnet_cond_input_gpu = brushnet_cond_input.to(unet_device) # Already on device, but ensures

        # Run BrushNet - Results stay on GPU
        down_block_res_samples, mid_block_res_sample, up_block_res_samples = self.brushnet(
            control_model_input_gpu,
            t,
            encoder_hidden_states=brushnet_prompt_embeds_gpu,
            brushnet_cond=brushnet_cond_input_gpu,
            conditioning_scale=cond_scale,
            guess_mode=guess_mode,
            return_dict=False,
        )

        # Clean up BrushNet *INPUT* GPU tensors (Keep results on GPU)
        del control_model_input_gpu, brushnet_prompt_embeds_gpu, brushnet_cond_input_gpu, brushnet_cond_sliced

        # Apply guess mode logic directly on GPU tensors if needed
        if guess_mode and self.do_classifier_free_guidance:
            print(f"[t={t_val}, j={j}] Applying guess mode logic on GPU...")
            down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
            mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])
            up_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in up_block_res_samples]

        # --- UNet Inference ---
        self.unet.to(unet_device)
        print(f"[t={t_val}, j={j}] Running UNet on {unet_device}...")
        latent_model_input_gpu = latent_model_input.to(unet_device)
        # Use the ORIGINAL prompt embeddings (already prepared for CFG if needed) for the UNet
        unet_prompt_embeds_gpu = original_prompt_embeds.to(unet_device)
        timestep_cond_gpu = timestep_cond.to(unet_device) if timestep_cond is not None else None

        # Assign BrushNet results (already on GPU) directly for UNet input
        down_block_add_samples_unet = down_block_res_samples
        mid_block_add_sample_unet = mid_block_res_sample
        up_block_add_samples_unet = up_block_res_samples

        # Clear cache here (Brush net causes a large vram growth)
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # Run UNet
        noise_pred = self.unet(
            latent_model_input_gpu, # Shape (2*T, C, H, W) or (T, C, H, W)
            t,
            encoder_hidden_states=unet_prompt_embeds_gpu, # Shape (2*B, Seq, Dim) or (B, Seq, Dim)
            timestep_cond=timestep_cond_gpu,
            cross_attention_kwargs=self.cross_attention_kwargs,
            down_block_add_samples=down_block_add_samples_unet, # Directly use GPU tensors
            mid_block_add_sample=mid_block_add_sample_unet,     # Directly use GPU tensors
            up_block_add_samples=up_block_add_samples_unet,     # Directly use GPU tensors
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
            num_frames=len(context), # Pass the length of the current context
        )[0] # noise_pred shape matches latent_model_input_gpu

        # Clean up UNet inputs and BrushNet GPU results (which were inputs to UNet)
        del down_block_add_samples_unet, mid_block_add_sample_unet, up_block_add_samples_unet
        del latent_model_input_gpu, unet_prompt_embeds_gpu
        if timestep_cond_gpu is not None: del timestep_cond_gpu
        # Crucially clear cache *here* before the scheduler step if memory is tight
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        # --- End UNet Section ---

        # Perform guidance
        if self.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            del noise_pred_uncond, noise_pred_text # Delete chunks

        # Ensure inputs are on the correct device and correct type for scheduler
        noise_pred_step = noise_pred.to(unet_device)
        latents_j_step = latents_j.to(unet_device) # latents_j was already moved to unet_device

        if not torch.is_tensor(t):
            t_step = torch.tensor([t], device=unet_device, dtype=torch.long)
        else:
            # Ensure t is long and on the correct device
            t_step = t.to(device=unet_device, dtype=torch.long)

        # Compute the previous noisy sample x_t -> x_t-1
        try:
            # Squeeze t_step if it's not scalar (some schedulers might expect scalar/1D)
            latents_j_out = self.scheduler.step(
                noise_pred_step,
                t_step.squeeze(),
                latents_j_step,
                **extra_step_kwargs,
                return_dict=False
            )[0]
        except RuntimeError as e:
            print(f"ERROR during scheduler.step: {e}")
            print(f"Device Check: noise_pred={noise_pred_step.device}, t={t_step.device}, latents={latents_j_step.device}")
            raise e

        # Clean up noise prediction and step-specific tensors
        del noise_pred, noise_pred_step, latents_j_step
        # Optional: Clear cache again if scheduler step itself used significant temporary memory
        # if torch.cuda.is_available(): torch.cuda.empty_cache()

        # Result latents_j_out is on unet_device
        latents_j = latents_j_out # Reuse variable name for the output of the step
        # --- End Scheduler Step ---

        # Ensure latents_j is definitely on the target device (redundant but safe)
        latents_j_update = latents_j.to(unet_device)

        # Increment count for the processed context indices
        # count shape is [T, 1, H, W], broadcasting works with [context, ...] indexing
        count[context, ...] += 1

        # --- Blending/Assignment Logic (Modifies 'value' in-place) ---
        if j == 0:
            # First context window: direct assignment
            print(f"[t={t_val}, j={j}] Assigning (j=0): value[{list(context.shape)}] = latents_j[{list(latents_j_update.shape)}]")
            value[context, ...] = latents_j_update
        else:
            # Subsequent windows: blend overlap, assign non-overlap
            # Find indices within the *current context* that have already been processed (count > 1)
            # Need to check the count tensor *at the global indices* defined by context
            # Indexing count[:, 0, 0, 0] is valid for shape [T, 1, H, W]
            overlap_indices_local = [idx for idx, global_idx in enumerate(context) if count[global_idx, 0, 0, 0] > 1]
            overlap_len = len(overlap_indices_local)

            if overlap_len > 0:
                # Calculate blending ratios on the correct device
                ratio_next = torch.linspace(0, 1, overlap_len + 2, device=unet_device, dtype=latents_j_update.dtype)[1:-1]
                ratio_pre = 1.0 - ratio_next

                print(f"[t={t_val}, j={j}] Blending {overlap_len} frames...")
                # Blend the overlapping part
                for k, local_idx in enumerate(overlap_indices_local):
                    global_idx = context[local_idx] # Global index in 'value'/'count'
                    # Ensure parts being blended are on the correct device
                    value_part = value[global_idx, ...].to(unet_device) # Get existing value
                    latents_j_part = latents_j_update[local_idx, ...] # Get new value for this index
                    # Blend: value = value * ratio_pre + latents_j * ratio_next
                    value[global_idx, ...] = value_part * ratio_pre[k] + latents_j_part * ratio_next[k]

                # Assign the non-overlapping part (overwrite)
                assign_start_local_idx = overlap_indices_local[-1] + 1
                if assign_start_local_idx < len(context): # Check if there are non-overlapping frames
                    # Get global indices for assignment
                    assign_global_indices = context[assign_start_local_idx:]
                    # Get corresponding latents data
                    assign_latents_data = latents_j_update[assign_start_local_idx:, ...]
                    value[assign_global_indices, ...] = assign_latents_data
            else:
                # No overlap found with previous windows in this timestep
                print(f"[t={t_val}, j={j}] Assigning full chunk (j>0, no overlap found)...")
                value[context, ...] = latents_j_update # Assign the whole chunk
        # --- End Update Section ---

        # Return the modified tensors (caller will use them)
        return value, count


    def _finalize_timestep(
        self,
        value: torch.Tensor,
        latents: torch.Tensor,
        i: int,
        t: Union[int, torch.Tensor],
        timesteps: Union[List[int], torch.Tensor], # Accept List or Tensor
        num_warmup_steps: int,
        progress_bar, # Type hint can be tricky (e.g., tqdm object)
        prompt_embeds_unet: torch.Tensor, # Pass embeds needed for potential callback use
        callback_on_step_end: Optional[Callable],
        callback_on_step_end_tensor_inputs: List[str],
        callback: Optional[Callable], # Legacy callback
        callback_steps: Optional[int] # Legacy callback steps
    ) -> torch.Tensor: # Returns the potentially updated latents
        """
        Finalizes a denoising timestep by updating latents from accumulated values,
        handling callbacks, and updating the progress bar.

        Args:
            value: Accumulated latent values from context window processing for this timestep.
            latents: The latents tensor *before* this finalization step.
            i: The index of the current timestep in the main loop.
            t: The current timestep value.
            timesteps: The list or tensor of all timesteps.
            num_warmup_steps: Number of warmup steps.
            progress_bar: The progress bar object.
            prompt_embeds_unet: The prompt embeddings used by the UNet (for callbacks).
            callback_on_step_end: The main callback function.
            callback_on_step_end_tensor_inputs: Keys requested by the main callback.
            callback: Legacy callback function.
            callback_steps: Frequency for the legacy callback.
        Returns:
            The updated latents tensor for the next timestep.
        """
        # --- Update latents for the next timestep ---
        # The accumulated results are in 'value'. Clone it for the next iteration.
        # Revert to original logic: simply clone the accumulated value.
        latents = value.clone()
        # Log the update
        t_val = t if isinstance(t, int) else t.item() # For printing
        print(f"--- [t={t_val}] Updated latents tensor for next step (cloned from value), shape: {latents.shape} on {latents.device} ---")
        # ---

        # --- Handle callbacks ---
        if callback_on_step_end is not None:
            callback_kwargs = {}
            # Build the dictionary of requested tensors for the callback
            # Map known parameter names to potential callback keys
            param_map = {
                "latents": latents, # Pass the newly updated latents
                "prompt_embeds": prompt_embeds_unet,
                "timestep": t,
                "step_index": i,
                # Add other variables passed as parameters if they might be requested
            }
            for k in callback_on_step_end_tensor_inputs:
                if k in param_map:
                    callback_kwargs[k] = param_map[k]
                elif hasattr(self, k): # Check instance attributes if not in direct params
                     callback_kwargs[k] = getattr(self, k)
                # else: print(f"Warning: Callback requested key '{k}' not found.")

            # Call the callback function
            callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

            # Update pipeline state based on callback output (if provided)
            # Primarily update latents if the callback returned a modified version
            latents = callback_outputs.pop("latents", latents)
            # Note: Modifying other state like prompt_embeds via callback is possible but complex

        # --- Update progress bar ---
        # Update based on scheduler order (updates less frequently for some schedulers)
        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
            progress_bar.update()
            # --- Handle legacy callback ---
            if callback is not None and callback_steps is not None and i % callback_steps == 0:
                 # Calculate step index compatible with legacy expectations
                 step_idx = i // getattr(self.scheduler, "order", 1)
                 # Call legacy callback with the potentially modified latents
                 callback(step_idx, t, latents)

        # Return the potentially modified latents to be used in the next iteration
        return latents