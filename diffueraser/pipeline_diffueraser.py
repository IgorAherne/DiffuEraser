import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
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

    # based on BrushNet: https://github.com/TencentARC/BrushNet/blob/main/src/diffusers/pipelines/brushnet/pipeline_brushnet.py
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
                The BrushNet branch input condition to provide guidance to the `unet` for generation. 
            masks (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The BrushNet branch input condition to provide guidance to the `unet` for generation. 
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
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.FloatTensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of IP-adapters.
                Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should contain the negative image embedding
                if `do_classifier_free_guidance` is set to `True`.
                If not provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            brushnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the BrushNet are multiplied by `brushnet_conditioning_scale` before they are added
                to the residual in the original `unet`. If multiple BrushNets are specified in `init`, you can set
                the corresponding scale as a list.
            guess_mode (`bool`, *optional*, defaults to `False`):
                The BrushNet encoder tries to recognize the content of the input image even if you remove all
                prompts. A `guidance_scale` value between 3.0 and 5.0 is recommended.
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

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

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

        brushnet = self.brushnet._orig_mod if is_compiled_module(self.brushnet) else self.brushnet

        # align format for control guidance
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
        self.check_inputs(
            prompt,
            images,
            masks,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            brushnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        global_pool_conditions = (
            brushnet.config.global_pool_conditions
            if isinstance(brushnet, BrushNetModel)
            else brushnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions
        video_length = len(images)

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
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
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare image
        if isinstance(brushnet, BrushNetModel):
            images = self.prepare_image(
                images=images,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=brushnet.dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
            original_masks_prepared = self.prepare_image( # Use a distinct variable name initially
                images=masks, # Pass the original masks list here
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=torch.float32, # Prepare masks in float32 for consistent thresholding
                do_classifier_free_guidance=self.do_classifier_free_guidance, # Should likely be False for masks
                guess_mode=guess_mode, # Should likely be False for masks
            )
            original_masks_processed = []
            for mask_tensor in original_masks_prepared:
                # Assuming prepare_image outputs range [0, 1] or similar non-negative
                # Convert to grayscale and threshold to get binary mask [1, 1, H, W]
                # Take the mean across channels (simple grayscale conversion)
                mask_gray = mask_tensor.mean(dim=1, keepdim=True)
                # Threshold: mask is typically white (1) where inpainting should occur
                mask_binary = (mask_gray > 0.5).to(brushnet.dtype) # Threshold at 0.5 and convert to brushnet's dtype
                original_masks_processed.append(mask_binary)
            # Rename for consistency with the rest of the code downstream
            original_masks = original_masks_processed # This is now a list of [1, 1, H, W] tensors
            
            height, width = images[0].shape[-2:]
        else:
            assert False# Should not happen if brushnet is correctly passe

        # 5. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents, noise = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        height, width = images[0].shape[-2:] # Get dimensions before potential deletion

        # 6.1 Prepare condition latents (Optimized VAE Encoding & Tensor Release)
        print("--- Preparing conditioning latents (Optimized) ---")
        # Keep full-res images list on CPU if possible, or ensure it's deleted after use.
        # Let's assume 'images' is the list of GPU tensors from prepare_image
        images_gpu = torch.cat(images, dim=0) # Create GPU tensor for VAE input
        del images # Delete the list to free references if it holds tensors
        images_gpu = images_gpu.to(dtype=prompt_embeds.dtype) # Ensure correct dtype

        conditioning_latents_batches_cpu = [] # Store batches on CPU
        num=4
        vae_original_device = self.vae.device # Remember VAE device
        vae_dtype = self.vae.dtype

        try:
            self.vae.to(device) # Move VAE to GPU for encoding
            print(f"--- Moved VAE to {device} for encoding loop ---")
            with torch.no_grad():
                for i in range(0, images_gpu.shape[0], num):
                    image_batch_gpu = images_gpu[i : i + num]
                    # Encode directly on GPU
                    latent_batch_gpu = self.vae.encode(image_batch_gpu).latent_dist.sample()
                    conditioning_latents_batches_cpu.append(latent_batch_gpu.cpu()) # Move result to CPU
                    # Optional: print progress less often
                    # if i % (num * 10) == 0: print(f"--- VAE Encoded batch up to frame {i+num} ---")

            print("--- VAE encoding loop finished ---")

        finally:
            self.vae.to(vae_original_device) # Move VAE back
            print(f"--- Returned VAE to {vae_original_device} ---")
            # Crucially, delete the full-res GPU image tensor NOW
            del images_gpu
            print("--- Deleted full-resolution input images tensor ---")
            gc.collect()
            torch.cuda.empty_cache()

        # Concatenate latents on CPU, then move to GPU
        conditioning_latents_4ch = torch.cat(conditioning_latents_batches_cpu, dim=0)
        del conditioning_latents_batches_cpu # Free CPU list
        conditioning_latents_4ch = conditioning_latents_4ch.to(device=device, dtype=prompt_embeds.dtype)
        conditioning_latents_4ch = conditioning_latents_4ch * self.vae.config.scaling_factor
        print(f"--- Created 4ch conditioning latents on GPU: {conditioning_latents_4ch.shape} ---")


        # Prepare and downscale masks (Optimized Release)
        original_masks_gpu = torch.cat(original_masks, dim=0) # Create GPU tensor
        del original_masks # Delete list if it holds tensors
        original_masks_gpu = original_masks_gpu.to(dtype=prompt_embeds.dtype) # Ensure correct dtype

        masks = torch.nn.functional.interpolate(
            original_masks_gpu,
            size=(latents.shape[-2], latents.shape[-1])
        ).to(device=device, dtype=prompt_embeds.dtype) # Ensure on device and correct dtype

        # Delete the original full-res mask tensor NOW
        del original_masks_gpu
        print("--- Deleted full-resolution original masks tensor ---")
        print(f"--- Created downscaled masks on GPU: {masks.shape} ---")
        gc.collect()
        torch.cuda.empty_cache()

        # *** DO NOT concatenate conditioning_latents and masks here yet ***
        # We will do it inside the loop for the specific context chunk
        # conditioning_latents=torch.concat([conditioning_latents,masks],1)

        # 6.5 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if ip_adapter_image is not None or ip_adapter_image_embeds is not None
            else None
        )

        # 7.2 Create tensor stating which brushnets to keep
        brushnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            brushnet_keep.append(keeps[0] if isinstance(brushnet, BrushNetModel) else keeps)


        overlap = num_frames//4
        context_list, context_list_swap = get_frames_context_swap(video_length, overlap=overlap, num_frames_per_clip=num_frames)
        scheduler_status = [copy.deepcopy(self.scheduler.__dict__)] * len(context_list)
        scheduler_status_swap = [copy.deepcopy(self.scheduler.__dict__)] * len(context_list_swap)
        count = torch.zeros_like(latents)
        value = torch.zeros_like(latents)

        # *** Create count and value tensors on CPU ***
        print("--- Creating count and value tensors on CPU ---")
        # latents (initial noise) is already on GPU from prepare_latents
        count = torch.zeros_like(latents, device='cpu')
        value = torch.zeros_like(latents, device='cpu')
        

        # 8. Denoising loop
        print("--- Starting Denoising Loop ---")
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        is_unet_compiled = is_compiled_module(self.unet)
        is_brushnet_compiled = is_compiled_module(self.brushnet)
        is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")

        # Ensure latents (initial noise) is on the GPU before the loop starts
        latents = latents.to(device)
        # Make sure 4ch condition latents and masks are also on GPU
        conditioning_latents_4ch = conditioning_latents_4ch.to(device)
        masks = masks.to(device)
        # Embeddings should also be on the device
        prompt_embeds = prompt_embeds.to(device)
        if self.do_classifier_free_guidance:
             # prompt_embeds is already concatenated if CFG is on
             pass # No need to move negative_prompt_embeds separately if already concatenated
        elif negative_prompt_embeds is not None: # Handle case where CFG is off but neg prompts provided
             negative_prompt_embeds = negative_prompt_embeds.to(device)

        # Make sure timestep_cond is on device if used
        if timestep_cond is not None:
            timestep_cond = timestep_cond.to(device)

        # Use value tensor on CPU as planned for overlap handling
        # (count is already on CPU from previous optimization)
        value = torch.zeros_like(latents, device='cpu') # Ensure value starts on CPU

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                print(f"\n--- Starting Timestep {i+1}/{len(timesteps)} (t={t.item()}) ---") # Progress indicator

                # Reset CPU count tensor for the new timestep iteration
                count.zero_()

                # Determine context list and scheduler status for this timestep
                if (i % 2 == 1):
                    context_list_choose = context_list_swap
                    scheduler_status_choose = scheduler_status_swap
                else:
                    context_list_choose = context_list
                    scheduler_status_choose = scheduler_status

                # Inner loop iterating through temporal contexts (chunks)
                for j, context in enumerate(context_list_choose):
                    print(f"  Processing context chunk {j+1}/{len(context_list_choose)} with indices: {context[:3]}...{context[-3:]}") # Fine-grained progress

                    # Restore scheduler state for this chunk
                    self.scheduler.__dict__.update(scheduler_status_choose[j])

                    # --- Prepare Inputs for this Chunk ---

                    # Get the latents slice for this context (GPU)
                    # Note: 'latents' comes from the *previous* timestep's blended 'value' tensor, moved to GPU
                    latents_j = latents[context, ...]

                    # Expand latents for classifier-free guidance if needed (GPU)
                    latent_model_input = torch.cat([latents_j] * 2) if self.do_classifier_free_guidance else latents_j
                    # Scale model input according to scheduler (GPU)
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # Determine BrushNet input (usually same as UNet input unless guess_mode CFG)
                    if guess_mode and self.do_classifier_free_guidance:
                        control_model_input = self.scheduler.scale_model_input(latents_j, t) # Use uncond latents
                    else:
                        control_model_input = latent_model_input

                    # Prepare BrushNet prompt embeddings for this chunk (GPU)
                    # Needs careful handling for CFG duplication and repeating for num_frames
                    if guess_mode and self.do_classifier_free_guidance:
                        # Use only conditional prompt embeds, repeat for num_frames
                        brushnet_prompt_embeds = prompt_embeds.chunk(2)[1] # Get conditional part
                        brushnet_prompt_embeds = rearrange(repeat(brushnet_prompt_embeds, "b c d -> b t c d", t=num_frames), 'b t c d -> (b t) c d')
                    else:
                        # Handle CFG duplication for brushnet prompts
                        if self.do_classifier_free_guidance:
                            # We need embeds for uncond and cond, repeated for num_frames
                            neg_brushnet_p, pos_brushnet_p = prompt_embeds.chunk(2)
                            neg_brushnet_p_rep = rearrange(repeat(neg_brushnet_p, "b c d -> b t c d", t=num_frames), 'b t c d -> (b t) c d')
                            pos_brushnet_p_rep = rearrange(repeat(pos_brushnet_p, "b c d -> b t c d", t=num_frames), 'b t c d -> (b t) c d')
                            brushnet_prompt_embeds = torch.cat([neg_brushnet_p_rep, pos_brushnet_p_rep])
                            del neg_brushnet_p_rep, pos_brushnet_p_rep, neg_brushnet_p, pos_brushnet_p # Clean up intermediate tensors
                        else:
                            # Just repeat the single prompt embed for num_frames
                            brushnet_prompt_embeds = rearrange(repeat(prompt_embeds, "b c d -> b t c d", t=num_frames), 'b t c d -> (b t) c d')

                    # Prepare BrushNet conditioning input (latent + mask) for this chunk (GPU)
                    with torch.no_grad(): # No grads needed for slicing/concatenating
                        cond_latents_context = conditioning_latents_4ch[context, ...] # Slice 4ch latents
                        masks_context = masks[context, ...] # Slice 1ch masks
                        # Concatenate ONLY the slices needed
                        brushnet_cond_context = torch.cat([cond_latents_context, masks_context], dim=1)
                        # Handle classifier-free guidance duplication
                        if self.do_classifier_free_guidance:
                             brushnet_cond_input = torch.cat([brushnet_cond_context] * 2)
                        else:
                             brushnet_cond_input = brushnet_cond_context
                        # Delete slices after concatenation
                        del cond_latents_context, masks_context, brushnet_cond_context

                    # Determine BrushNet conditioning scale for this step
                    if isinstance(brushnet_keep[i], list):
                        # This case seems unlikely if brushnet_conditioning_scale is float, but handle it
                        cond_scale = [c * s for c, s in zip(brushnet_conditioning_scale, brushnet_keep[i])]
                    else: # brushnet_keep[i] is float (likely 1.0 or 0.0)
                        current_brushnet_cond_scale = brushnet_conditioning_scale
                        # Handle if conditioning_scale itself is a list (e.g., for multiple brushnets, though we only have one)
                        if isinstance(current_brushnet_cond_scale, list):
                            current_brushnet_cond_scale = current_brushnet_cond_scale[0]
                        cond_scale = current_brushnet_cond_scale * brushnet_keep[i] # Apply step-wise guidance weight


                    # --- BrushNet Forward Pass ---
                    print(f"    Running BrushNet for chunk {j+1}...") # Debug
                    # Use autocast for potential mixed-precision benefits
                    with torch.cuda.amp.autocast(enabled=(self.brushnet.dtype == torch.float16)):
                         down_block_res_samples, mid_block_res_sample, up_block_res_samples = self.brushnet(
                             control_model_input,
                             t,
                             encoder_hidden_states=brushnet_prompt_embeds,
                             brushnet_cond=brushnet_cond_input,
                             conditioning_scale=cond_scale,
                             guess_mode=guess_mode,
                             return_dict=False,
                         )
                    print(f"    BrushNet Done.") # Debug

                    # --- Clean up BrushNet Inputs ---
                    del control_model_input
                    del brushnet_cond_input
                    del brushnet_prompt_embeds # Delete potentially large repeated embeds


                    # Handle 'guess_mode' output manipulation if enabled
                    if guess_mode and self.do_classifier_free_guidance:
                        # Move zeros_like to the correct device
                        down_block_res_samples = [torch.cat([torch.zeros_like(d, device=d.device), d]) for d in down_block_res_samples]
                        mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample, device=mid_block_res_sample.device), mid_block_res_sample])
                        up_block_res_samples = [torch.cat([torch.zeros_like(d, device=d.device), d]) for d in up_block_res_samples]

                    # --- UNet Forward Pass ---
                    print(f"    Running UNet for chunk {j+1}...") # Debug
                    # Use autocast for potential mixed-precision benefits
                    with torch.cuda.amp.autocast(enabled=(self.unet.dtype == torch.float16)):
                         # Note: UNet gets the original (non-repeated) prompt embeddings
                         unet_prompt_embeds = prompt_embeds # This should be (batch*2, seq, dim) if CFG, or (batch, seq, dim) otherwise
                         noise_pred = self.unet(
                             latent_model_input, # Input latents (potentially duplicated for CFG)
                             t,
                             encoder_hidden_states=unet_prompt_embeds,
                             timestep_cond=timestep_cond,
                             cross_attention_kwargs=cross_attention_kwargs,
                             down_block_add_samples=down_block_res_samples, # Residuals from BrushNet
                             mid_block_add_sample=mid_block_res_sample,      # Residuals from BrushNet
                             up_block_add_samples=up_block_res_samples,      # Residuals from BrushNet
                             added_cond_kwargs=added_cond_kwargs,          # For SDXL-like features (None here)
                             return_dict=False,
                             num_frames=num_frames,                         # Pass num_frames for Motion Module
                         )[0] # Get the primary output (sample)
                    print(f"    UNet Done.") # Debug

                    # --- Clean up UNet Inputs & BrushNet Residuals ---
                    del latent_model_input
                    del down_block_res_samples, mid_block_res_sample, up_block_res_samples
                    if 'unet_prompt_embeds' in locals(): del unet_prompt_embeds # Clean up reference


                    # --- Perform Guidance ---
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        guidance_factor = self.guidance_scale # Use the pipeline's guidance scale
                        noise_pred = noise_pred_uncond + guidance_factor * (noise_pred_text - noise_pred_uncond)
                        del noise_pred_uncond, noise_pred_text # Clean up chunks


                    # --- Scheduler Step ---
                    # Compute the previous noisy sample x_t -> x_t-1
                    latents_j_new = self.scheduler.step(noise_pred, t, latents_j, **extra_step_kwargs, return_dict=False)[0]
                    print(f"    Scheduler step done for chunk {j+1}.") # Debug

                    # --- Clean up Prediction and Old Latents ---
                    del noise_pred
                    del latents_j # Delete the input latents for this chunk


                    # --- Overlap Blending (Using CPU Tensors) ---
                    print(f"    Performing overlap blend for chunk {j+1}...") # Debug
                    # Move the computation result to CPU
                    latents_j_cpu = latents_j_new.cpu()
                    del latents_j_new # Delete the GPU version

                    # Access CPU tensor slices for count and value
                    # These are direct references, so updates modify the main tensors
                    count_context_cpu = count[context, ...]
                    value_context_cpu = value[context, ...]

                    # Increment count on the CPU slice
                    count_context_cpu += 1

                    # Perform the blend logic using CPU tensors
                    if j == 0: # First chunk, just add
                        value_context_cpu += latents_j_cpu
                    else: # Subsequent chunks, blend overlaps
                        # Find overlapping indices within the current context based on CPU counts
                        overlap_indices_in_context = torch.where(count_context_cpu[:, 0, 0, 0] > 1)[0]

                        if overlap_indices_in_context.numel() > 0: # If overlaps exist
                            overlap_cur = overlap_indices_in_context.numel()
                            # Ratios on CPU
                            ratio_dtype = value.dtype # Match dtype of value tensor (likely float32 or float16 on CPU)
                            ratio_next = torch.linspace(0, 1, overlap_cur + 2, device='cpu', dtype=ratio_dtype)[1:-1]
                            ratio_pre = 1.0 - ratio_next
                            # Make ratios broadcastable
                            ratio_next_b = ratio_next.view(-1, 1, 1, 1)
                            ratio_pre_b = ratio_pre.view(-1, 1, 1, 1)

                            # Get overlapping slices from CPU value tensor and the current chunk's CPU result
                            value_overlap_cpu = value_context_cpu[overlap_indices_in_context, ...]
                            latents_j_overlap_cpu = latents_j_cpu[overlap_indices_in_context, ...]

                            # Blend on CPU
                            blended_overlap_cpu = torch.lerp(value_overlap_cpu, latents_j_overlap_cpu, ratio_next_b)
                            # Or: blended_overlap_cpu = value_overlap_cpu * ratio_pre_b + latents_j_overlap_cpu * ratio_next_b

                            # Write blended result back to the CPU value slice
                            value_context_cpu[overlap_indices_in_context, ...] = blended_overlap_cpu

                            # Determine where the non-overlapping part starts in this chunk's latents_j_cpu
                            non_overlap_start_idx_in_context = overlap_indices_in_context[-1] + 1
                        else: # No overlaps in this chunk
                            non_overlap_start_idx_in_context = 0

                        # Assign the non-overlapping tail part of latents_j_cpu to the value slice (CPU to CPU)
                        if non_overlap_start_idx_in_context < num_frames:
                            value_context_cpu[non_overlap_start_idx_in_context:, ...] = latents_j_cpu[non_overlap_start_idx_in_context:, ...]

                    # Clean up the CPU tensor for this chunk's result
                    del latents_j_cpu
                    print(f"    Overlap blend done.") # Debug

                # --- End of Inner Loop (j loop over contexts) ---

                # After processing all contexts for this timestep, update the main 'latents' tensor on GPU
                # The 'value' tensor on CPU now holds the fully blended result for this timestep
                print(f"--- Updating latents for next timestep from CPU value tensor ---")
                latents = value.to(device) # Move updated results from CPU to GPU

                # Optional: Callback handling
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        # Handle potential deletion of variables if k is not present
                        if k in locals():
                             callback_kwargs[k] = locals()[k]
                        else:
                             callback_kwargs[k] = None # Or skip if not critical
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    # Update variables based on callback output if necessary
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    # Handle negative prompts if separate and modified by callback
                    if "negative_prompt_embeds" in callback_outputs:
                         negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # Legacy callback handling
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                # --- Optional Cleanup at end of Timestep Iteration ---
                print(f"--- End of Timestep {i+1}. Cleaning up... ---")
                gc.collect()
                torch.cuda.empty_cache()

        # --- End of Outer Loop (i loop over timesteps) ---
        # Final 'latents' tensor is on GPU


        # If we do sequential model offloading, let's offload unet and brushnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.brushnet.to("cpu")
            torch.cuda.empty_cache()

        if output_type == "latent":
            # No need to decode VAE, return the computed latents directly
            # Assign the final latents tensor to the 'latents' field of the output object.
            # Set 'frames' to None as we are not decoding.
            print("--- Returning latents directly (output_type='latent') ---") # Optional: Add log
            return DiffuEraserPipelineOutput(frames=None, latents=latents) # <<< CORRECTED RETURN

        # --- Code for output_type != "latent" ---
        print(f"--- Decoding final latents to output_type='{output_type}' ---") # Optional: Add log
        video_tensor = self.decode_latents(latents, weight_dtype=prompt_embeds.dtype) # VAE Decode happens here

        if output_type == "pt":
            video = video_tensor
        else:
            # Ensure postprocess handles pt input correctly
            video = self.image_processor.postprocess(video_tensor, output_type=output_type)

        # Offload all models if needed (maybe_free_model_hooks likely handles offloaded models)
        self.maybe_free_model_hooks()

        if not return_dict:
            # Original code had 'has_nsfw_concept' here, but safety checker is likely off.
            # Return only the video frames/tensor in a tuple.
            return (video,)

        # Return the standard output object for non-latent types
        print("--- Returning decoded frames and final latents ---") # Optional: Add log
        return DiffuEraserPipelineOutput(frames=video, latents=latents)
