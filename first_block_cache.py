import contextlib
import dataclasses
import unittest
import sys
from collections import defaultdict
from typing import DefaultDict, Dict, Optional

import torch
import torch.nn.functional as F


# ==============================================================================
# CACHE CONTEXT - CORRECTED VERSION
# ==============================================================================

@dataclasses.dataclass
class CacheContext:
    buffers: Dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)
    step_counter: int = 0  # Diffusion step counter
    last_threshold: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    
    @torch.compiler.disable()
    def get_buffer(self, name: str) -> Optional[torch.Tensor]:
        return self.buffers.get(name)
    
    @torch.compiler.disable()
    def set_buffer(self, name: str, buffer: torch.Tensor):
        # Ensures buffer is contiguous and detached
        if buffer is not None:
            self.buffers[name] = buffer.detach().contiguous()
    
    def clear_buffers(self):
        self.buffers.clear()
        self.step_counter = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    def increment_step(self):
        self.step_counter += 1


# ==============================================================================
# GLOBAL CACHE FUNCTIONS
# ==============================================================================

_current_cache_context: Optional[CacheContext] = None


def create_cache_context() -> CacheContext:
    return CacheContext()


def get_current_cache_context() -> Optional[CacheContext]:
    return _current_cache_context


def set_current_cache_context(cache_context: Optional[CacheContext] = None):
    global _current_cache_context
    _current_cache_context = cache_context


@contextlib.contextmanager
def cache_context(cache_context: CacheContext):
    global _current_cache_context
    old_cache_context = _current_cache_context
    _current_cache_context = cache_context
    try:
        yield cache_context
    finally:
        _current_cache_context = old_cache_context


@torch.compiler.disable()
def get_buffer(name: str) -> Optional[torch.Tensor]:
    ctx = get_current_cache_context()
    if ctx is None:
        return None
    return ctx.get_buffer(name)


@torch.compiler.disable()
def set_buffer(name: str, buffer: torch.Tensor):
    ctx = get_current_cache_context()
    if ctx is not None:
        ctx.set_buffer(name, buffer)


# ==============================================================================
# EXECUTION PATCH - COMPATIBLE WITH 0.8.2
# ==============================================================================

def patch_get_output_data():
    execution = None
    
    # Tries to import from multiple locations
    for module_name in ['execution', 'comfy.execution', 'comfy.execution_manager']:
        try:
            if module_name in sys.modules:
                execution = sys.modules[module_name]
                break
            else:
                exec_module = __import__(module_name, fromlist=[''])
                execution = exec_module
                break
        except (ImportError, AttributeError):
            continue
    
    if execution is None:
        return
    
    get_output_data = getattr(execution, "get_output_data", None)
    if get_output_data is None or getattr(get_output_data, "_wavespeed_patched", False):
        return
    
    # Wrapper that works with 2 or 3 return values
    def new_get_output_data(*args, **kwargs):
        result = get_output_data(*args, **kwargs)
        
        ctx = get_current_cache_context()
        if ctx is not None:
            ctx.clear_buffers()
            set_current_cache_context(None)
        
        return result
    
    new_get_output_data._wavespeed_patched = True
    execution.get_output_data = new_get_output_data


# ==============================================================================
# SIMILARITY AND CACHE FUNCTIONS - CORRECTED VERSION
# ==============================================================================

@torch.compiler.disable()
def compute_tensor_similarity(t1: torch.Tensor, t2: torch.Tensor, threshold: float) -> bool:
    """
    Computes similarity using multiple metrics for greater robustness.
    """
    if t1.shape != t2.shape:
        return False
    
    # Convert to float32 to avoid numerical issues
    t1_f = t1.float()
    t2_f = t2.float()
    
    # Metric 1: Relative MAE (Mean Absolute Error)
    diff_abs = (t1_f - t2_f).abs().mean()
    base_abs = t1_f.abs().mean().clamp(min=1e-8)
    mae_relative = (diff_abs / base_abs).item()
    
    # Metric 2: Cosine similarity (vector direction)
    t1_flat = t1_f.flatten()
    t2_flat = t2_f.flatten()
    cos_sim = F.cosine_similarity(t1_flat.unsqueeze(0), t2_flat.unsqueeze(0)).item()
    
    # Metric 3: Relative MSE to capture outliers
    mse = ((t1_f - t2_f) ** 2).mean()
    base_var = (t1_f ** 2).mean().clamp(min=1e-8)
    mse_relative = (mse / base_var).item()
    
    # Combined decision: uses MAE as primary, but validates with cosine
    is_similar = (mae_relative < threshold) and (cos_sim > 0.95)
    
    # Uncomment for debugging:
    # print(f"[WS] MAE: {mae_relative:.4f} | Cos: {cos_sim:.4f} | Threshold: {threshold} | {'HIT' if is_similar else 'MISS'}")
    
    return is_similar


@torch.compiler.disable()
def get_can_use_cache(current_first_residual: torch.Tensor, threshold: float) -> bool:
    """
    Determines whether to use cache based on first residual similarity.
    """
    prev_residual = get_buffer("first_hidden_states_residual")
    
    if prev_residual is None:
        return False
    
    return compute_tensor_similarity(prev_residual, current_first_residual, threshold)


@torch.compiler.disable()
def apply_cached_residual(
    hidden_states: torch.Tensor,
    current_first_residual: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None
):
    """
    Applies cached residual with adaptive scale correction.
    """
    cached_residual = get_buffer("hidden_states_residual")
    prev_first_residual = get_buffer("first_hidden_states_residual")
    
    if cached_residual is None:
        if encoder_hidden_states is None:
            return hidden_states
        return hidden_states, encoder_hidden_states
    
    # Computes scale factor based on first residual change
    scale_factor = 1.0
    if prev_first_residual is not None:
        # Uses RMS norm (Root Mean Square) which is more stable
        prev_norm = prev_first_residual.square().mean().sqrt().clamp(min=1e-8)
        curr_norm = current_first_residual.square().mean().sqrt().clamp(min=1e-8)
        
        # Computes ratio, but limits to avoid explosion/collapse
        scale_factor = (curr_norm / prev_norm).item()
        scale_factor = max(0.85, min(1.15, scale_factor))
    
    # Applies scaled residual
    hidden_states = hidden_states + (cached_residual * scale_factor)
    hidden_states = hidden_states.contiguous()
    
    if encoder_hidden_states is None:
        return hidden_states
    
    # Processes encoder_hidden_states if it exists
    cached_encoder_residual = get_buffer("encoder_hidden_states_residual")
    if cached_encoder_residual is not None:
        encoder_hidden_states = encoder_hidden_states + (cached_encoder_residual * scale_factor)
        encoder_hidden_states = encoder_hidden_states.contiguous()
    
    return hidden_states, encoder_hidden_states


# ==============================================================================
# FLUX PATCH - COMPLETELY REWRITTEN VERSION
# ==============================================================================

def create_patch_flux_forward(
    model,
    *,
    residual_diff_threshold: float,
    validate_can_use_cache_function=None
):
    """
    Creates patch for FLUX model with corrected cache logic.
    """
    from torch import Tensor
    
    # Imports timestep_embedding from correct location
    try:
        from comfy.ldm.flux.model import timestep_embedding
    except ImportError:
        try:
            from comfy.ldm.flux.layers import timestep_embedding
        except ImportError:
            # Fallback: uses own implementation
            def timestep_embedding(timesteps, dim, max_period=10000):
                half = dim // 2
                freqs = torch.exp(
                    -torch.log(torch.tensor(max_period)) * 
                    torch.arange(start=0, end=half, dtype=torch.float32) / half
                ).to(device=timesteps.device)
                args = timesteps[:, None].float() * freqs[None]
                embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
                if dim % 2:
                    embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
                return embedding
    
    def call_remaining_blocks(
        self, blocks_replace, control, img, txt, vec, pe,
        attn_mask, ca_idx, timesteps, transformer_options
    ):
        """
        Executes blocks 1-N and returns total residual.
        """
        # Saves initial state to compute total residual
        img_initial = img.clone()
        
        extra_kwargs = {}
        if attn_mask is not None:
            extra_kwargs["attn_mask"] = attn_mask
        
        # Double blocks (starting from index 1)
        for i, block in enumerate(self.double_blocks):
            if i < 1:
                continue
            
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(
                        img=args["img"], txt=args["txt"], 
                        vec=args["vec"], pe=args["pe"], 
                        **extra_kwargs
                    )
                    return out
                
                out = blocks_replace[("double_block", i)](
                    {"img": img, "txt": txt, "vec": vec, "pe": pe, **extra_kwargs},
                    {"original_block": block_wrap, "transformer_options": transformer_options}
                )
                txt = out["txt"]
                img = out["img"]
            else:
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe, **extra_kwargs)
            
            # Controlnet
            if control is not None:
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img = img + add
            
            # PuLID
            if getattr(self, "pulid_data", {}):
                if i % self.pulid_double_interval == 0:
                    for _, node_data in self.pulid_data.items():
                        if torch.any((node_data['sigma_start'] >= timesteps) & 
                                   (timesteps >= node_data['sigma_end'])):
                            img = img + node_data['weight'] * self.pulid_ca[ca_idx](
                                node_data['embedding'], img
                            )
                    ca_idx += 1
        
        # Concatenates for single blocks
        img = torch.cat((txt, img), 1)
        
        # Single blocks
        for i, block in enumerate(self.single_blocks):
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], vec=args["vec"], pe=args["pe"], **extra_kwargs)
                    return out
                
                out = blocks_replace[("single_block", i)](
                    {"img": img, "vec": vec, "pe": pe, **extra_kwargs},
                    {"original_block": block_wrap, "transformer_options": transformer_options}
                )
                img = out["img"]
            else:
                img = block(img, vec=vec, pe=pe, **extra_kwargs)
            
            # Controlnet
            if control is not None:
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:, txt.shape[1]:, ...] = img[:, txt.shape[1]:, ...] + add
            
            # PuLID
            if getattr(self, "pulid_data", {}):
                real_img, txt_part = img[:, txt.shape[1]:, ...], img[:, :txt.shape[1], ...]
                if i % self.pulid_single_interval == 0:
                    for _, node_data in self.pulid_data.items():
                        if torch.any((node_data['sigma_start'] >= timesteps) & 
                                   (timesteps >= node_data['sigma_end'])):
                            real_img = real_img + node_data['weight'] * self.pulid_ca[ca_idx](
                                node_data['embedding'], real_img
                            )
                    ca_idx += 1
                img = torch.cat((txt_part, real_img), 1)
        
        # Extracts only image part
        img = img[:, txt.shape[1]:, ...]
        img = img.contiguous()
        
        # Computes total residual: (final output) - (input after block 0)
        residual = img - img_initial
        residual = residual.contiguous()
        
        return img, residual
    
    def forward(
        self,
        img: Tensor,
        timesteps: Tensor,
        **kwargs
    ) -> Tensor:
        """
        Modified forward with intelligent caching.
        """
        # Extracts arguments
        txt = kwargs.get("txt") or kwargs.get("context")
        y = kwargs.get("y")
        guidance = kwargs.get("guidance")
        control = kwargs.get("control")
        transformer_options = kwargs.get("transformer_options", {})
        attn_mask = kwargs.get("attn_mask")
        img_ids = kwargs.get("img_ids")
        txt_ids = kwargs.get("txt_ids")
        
        if txt is None or y is None:
            raise ValueError("[WaveSpeed] Missing required arguments: txt and y")
        
        patches_replace = transformer_options.get("patches_replace", {})
        
        # Packing logic for 4D tensors
        is_packed = False
        unpack_params = None
        
        if img.ndim == 4 and img_ids is None:
            is_packed = True
            b, c, h, w = img.shape
            img = F.pixel_unshuffle(img, 2)
            b, c_packed, h_packed, w_packed = img.shape
            
            unpack_params = (b, c, h, w, c_packed, h_packed, w_packed)
            
            # Creates IDs
            img_ids = torch.zeros(h_packed, w_packed, 3, device=img.device, dtype=img.dtype)
            img_ids[..., 1] = img_ids[..., 1] + torch.arange(h_packed, device=img.device, dtype=img.dtype)[:, None]
            img_ids[..., 2] = img_ids[..., 2] + torch.arange(w_packed, device=img.device, dtype=img.dtype)[None, :]
            img_ids = img_ids.reshape(1, h_packed * w_packed, 3).repeat(b, 1, 1)
            
            if txt_ids is None:
                b_txt, l, c_txt = txt.shape
                txt_ids = torch.zeros(b_txt, l, 3, device=txt.device, dtype=txt.dtype)
            
            img = img.permute(0, 2, 3, 1).reshape(b, h_packed * w_packed, c_packed)
        
        # Embeddings
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
        
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("[WaveSpeed] Guidance required for guidance_embed model")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))
        
        vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])
        txt = self.txt_in(txt)
        
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
        
        ca_idx = 0
        extra_kwargs = {}
        if attn_mask is not None:
            extra_kwargs["attn_mask"] = attn_mask
        
        blocks_replace = patches_replace.get("dit", {})
        
        # EXECUTES ONLY BLOCK 0
        img_before_block0 = img.clone()
        
        for i, block in enumerate(self.double_blocks):
            if i >= 1:
                break
            
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(
                        img=args["img"], txt=args["txt"],
                        vec=args["vec"], pe=args["pe"],
                        **extra_kwargs
                    )
                    return out
                
                out = blocks_replace[("double_block", i)](
                    {"img": img, "txt": txt, "vec": vec, "pe": pe, **extra_kwargs},
                    {"original_block": block_wrap, "transformer_options": transformer_options}
                )
                txt = out["txt"]
                img = out["img"]
            else:
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe, **extra_kwargs)
            
            # Controlnet in block 0
            if control is not None:
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img = img + add
            
            # PuLID in block 0
            if getattr(self, "pulid_data", {}):
                if i % self.pulid_double_interval == 0:
                    for _, node_data in self.pulid_data.items():
                        if torch.any((node_data['sigma_start'] >= timesteps) & 
                                   (timesteps >= node_data['sigma_end'])):
                            img = img + node_data['weight'] * self.pulid_ca[ca_idx](
                                node_data['embedding'], img
                            )
                    ca_idx += 1
        
        # Computes block 0 residual
        first_residual = img - img_before_block0
        first_residual = first_residual.contiguous()
        
        # Decides whether to use cache
        can_use_cache = get_can_use_cache(first_residual, residual_diff_threshold)
        
        if validate_can_use_cache_function:
            can_use_cache = validate_can_use_cache_function(can_use_cache)
        
        # Graph break for PyTorch
        torch._dynamo.graph_break()
        
        if can_use_cache:
            # CACHE HIT: Applies cached residual with scale correction
            img = apply_cached_residual(img, first_residual)
            
            ctx = get_current_cache_context()
            if ctx:
                ctx.cache_hits += 1
        else:
            # CACHE MISS: Executes remaining blocks and saves to cache
            img, total_residual = call_remaining_blocks(
                self, blocks_replace, control, img, txt, vec, pe,
                attn_mask, ca_idx, timesteps, transformer_options
            )
            
            set_buffer("first_hidden_states_residual", first_residual)
            set_buffer("hidden_states_residual", total_residual)
            
            ctx = get_current_cache_context()
            if ctx:
                ctx.cache_misses += 1
        
        torch._dynamo.graph_break()
        
        # Final layer
        img = self.final_layer(img, vec)
        
        # Unpacking if necessary
        if is_packed and unpack_params:
            b, c, h, w, c_packed, h_packed, w_packed = unpack_params
            img = img.reshape(b, h_packed, w_packed, c_packed)
            img = img.permute(0, 3, 1, 2)
            img = F.pixel_shuffle(img, 2)
        
        return img
    
    new_forward = forward.__get__(model)
    
    @contextlib.contextmanager
    def patch_forward_context():
        with unittest.mock.patch.object(model, "forward", new_forward):
            yield
    
    return patch_forward_context


# ==============================================================================
# UNET PATCH (For SD1.5/SDXL)
# ==============================================================================

def create_patch_unet_model__forward(
    model,
    *,
    residual_diff_threshold: float,
    validate_can_use_cache_function=None
):
    """
    Patch for UNet models (SD 1.5, SDXL, etc).
    """
    from comfy.ldm.modules.diffusionmodules.openaimodel import (
        timestep_embedding, forward_timestep_embed, apply_control
    )
    
    def call_remaining_blocks(
        self, transformer_options, control, transformer_patches,
        hs, h, *args, **kwargs
    ):
        h_initial = h.clone()
        
        for id, module in enumerate(self.input_blocks):
            if id < 2:
                continue
            
            transformer_options["block"] = ("input", id)
            h = forward_timestep_embed(module, h, *args, **kwargs)
            h = apply_control(h, control, 'input')
            
            if "input_block_patch" in transformer_patches:
                patch = transformer_patches["input_block_patch"]
                for p in patch:
                    h = p(h, transformer_options)
            
            hs.append(h)
            
            if "input_block_patch_after_skip" in transformer_patches:
                patch = transformer_patches["input_block_patch_after_skip"]
                for p in patch:
                    h = p(h, transformer_options)
        
        transformer_options["block"] = ("middle", 0)
        if self.middle_block is not None:
            h = forward_timestep_embed(self.middle_block, h, *args, **kwargs)
        h = apply_control(h, control, 'middle')
        
        for id, module in enumerate(self.output_blocks):
            transformer_options["block"] = ("output", id)
            hsp = hs.pop()
            hsp = apply_control(hsp, control, 'output')
            
            if "output_block_patch" in transformer_patches:
                patch = transformer_patches["output_block_patch"]
                for p in patch:
                    h, hsp = p(h, hsp, transformer_options)
            
            h = torch.cat([h, hsp], dim=1)
            del hsp
            
            output_shape = hs[-1].shape if len(hs) > 0 else None
            h = forward_timestep_embed(module, h, *args, output_shape, **kwargs)
        
        residual = h - h_initial
        return h, residual.contiguous()
    
    def unet_model__forward(
        self, x, timesteps=None, context=None, y=None,
        control=None, transformer_options={}, **kwargs
    ):
        transformer_options["original_shape"] = list(x.shape)
        transformer_options["transformer_index"] = 0
        transformer_patches = transformer_options.get("patches", {})
        
        num_video_frames = kwargs.get("num_video_frames", self.default_num_video_frames)
        image_only_indicator = kwargs.get("image_only_indicator", None)
        time_context = kwargs.get("time_context", None)
        
        assert (y is not None) == (self.num_classes is not None), \
            "must specify y if and only if the model is class-conditional"
        
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(x.dtype)
        emb = self.time_embed(t_emb)
        
        if "emb_patch" in transformer_patches:
            patch = transformer_patches["emb_patch"]
            for p in patch:
                emb = p(emb, self.model_channels, transformer_options)
        
        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)
        
        can_use_cache = False
        h = x
        
        for id, module in enumerate(self.input_blocks):
            if id >= 2:
                break
            
            transformer_options["block"] = ("input", id)
            
            if id == 1:
                h_before = h.clone()
            
            h = forward_timestep_embed(
                module, h, emb, context, transformer_options,
                time_context=time_context, num_video_frames=num_video_frames,
                image_only_indicator=image_only_indicator
            )
            h = apply_control(h, control, 'input')
            
            if "input_block_patch" in transformer_patches:
                patch = transformer_patches["input_block_patch"]
                for p in patch:
                    h = p(h, transformer_options)
            
            hs.append(h)
            
            if "input_block_patch_after_skip" in transformer_patches:
                patch = transformer_patches["input_block_patch_after_skip"]
                for p in patch:
                    h = p(h, transformer_options)
            
            if id == 1:
                first_residual = (h - h_before).contiguous()
                can_use_cache = get_can_use_cache(first_residual, residual_diff_threshold)
                
                if validate_can_use_cache_function:
                    can_use_cache = validate_can_use_cache_function(can_use_cache)
                
                if not can_use_cache:
                    set_buffer("first_hidden_states_residual", first_residual)
                
                del first_residual, h_before
        
        torch._dynamo.graph_break()
        
        if can_use_cache:
            # Note: UNet does not use encoder_hidden_states in cache
            h = apply_cached_residual(h, get_buffer("first_hidden_states_residual"))
        else:
            h, residual = call_remaining_blocks(
                self, transformer_options, control, transformer_patches,
                hs, h, emb, context, transformer_options,
                time_context=time_context, num_video_frames=num_video_frames,
                image_only_indicator=image_only_indicator
            )
            set_buffer("hidden_states_residual", residual)
        
        torch._dynamo.graph_break()
        
        h = h.type(x.dtype)
        
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)
    
    new__forward = unet_model__forward.__get__(model)
    
    @contextlib.contextmanager
    def patch__forward():
        with unittest.mock.patch.object(model, "_forward", new__forward):
            yield
    
    return patch__forward


# ==============================================================================
# CACHEDTRANSFORMERBLOCKS CLASS (For backward compatibility with old code)
# ==============================================================================

class CachedTransformerBlocks(torch.nn.Module):
    """
    Legacy class for backward compatibility with old wavespeed code.
    Use create_patch_flux_forward or create_patch_unet_model__forward for new projects.
    """
    def __init__(
        self,
        transformer_blocks,
        single_transformer_blocks=None,
        *,
        residual_diff_threshold,
        validate_can_use_cache_function=None,
        return_hidden_states_first=True,
        accept_hidden_states_first=True,
        cat_hidden_states_first=False,
        return_hidden_states_only=False,
        clone_original_hidden_states=False,
    ):
        super().__init__()
        self.transformer_blocks = transformer_blocks
        self.single_transformer_blocks = single_transformer_blocks
        self.residual_diff_threshold = residual_diff_threshold
        self.validate_can_use_cache_function = validate_can_use_cache_function
        self.return_hidden_states_first = return_hidden_states_first
        self.accept_hidden_states_first = accept_hidden_states_first
        self.cat_hidden_states_first = cat_hidden_states_first
        self.return_hidden_states_only = return_hidden_states_only
        self.clone_original_hidden_states = clone_original_hidden_states

    def forward(self, *args, **kwargs):
        # Simplified implementation - use patch functions above for production
        img_arg_name = None
        if "img" in kwargs: img_arg_name = "img"
        elif "hidden_states" in kwargs: img_arg_name = "hidden_states"
        
        txt_arg_name = None
        if "txt" in kwargs: txt_arg_name = "txt"
        elif "context" in kwargs: txt_arg_name = "context"
        elif "encoder_hidden_states" in kwargs: txt_arg_name = "encoder_hidden_states"

        if self.accept_hidden_states_first:
            if args: img = args[0]; args = args[1:]
            else: img = kwargs.pop(img_arg_name)
            if args: txt = args[0]; args = args[1:]
            else: txt = kwargs.pop(txt_arg_name)
        else:
            if args: txt = args[0]; args = args[1:]
            else: txt = kwargs.pop(txt_arg_name)
            if args: img = args[0]; args = args[1:]
            else: img = kwargs.pop(img_arg_name)

        hidden_states = img
        encoder_hidden_states = txt
        
        # Execution without cache if threshold <= 0
        if self.residual_diff_threshold <= 0.0:
            for block in self.transformer_blocks:
                if txt_arg_name == "encoder_hidden_states":
                    hidden_states = block(hidden_states, *args, encoder_hidden_states=encoder_hidden_states, **kwargs)
                else:
                    if self.accept_hidden_states_first: 
                        hidden_states = block(hidden_states, encoder_hidden_states, *args, **kwargs)
                    else: 
                        hidden_states = block(encoder_hidden_states, hidden_states, *args, **kwargs)
                
                if not self.return_hidden_states_only:
                    hidden_states, encoder_hidden_states = hidden_states
                    if not self.return_hidden_states_first: 
                        hidden_states, encoder_hidden_states = encoder_hidden_states, hidden_states
            
            if self.single_transformer_blocks is not None:
                hidden_states = torch.cat(
                    [hidden_states, encoder_hidden_states] if self.cat_hidden_states_first 
                    else [encoder_hidden_states, hidden_states], 
                    dim=1
                )
                for block in self.single_transformer_blocks: 
                    hidden_states = block(hidden_states, *args, **kwargs)
                hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:]

            if self.return_hidden_states_only: 
                return hidden_states
            return (
                (hidden_states, encoder_hidden_states) if self.return_hidden_states_first 
                else (encoder_hidden_states, hidden_states)
            )

        # Execution with cache
        original_hidden_states = hidden_states.clone() if self.clone_original_hidden_states else hidden_states
        
        first_block = self.transformer_blocks[0]
        if txt_arg_name == "encoder_hidden_states":
            hidden_states = first_block(hidden_states, *args, encoder_hidden_states=encoder_hidden_states, **kwargs)
        else:
            if self.accept_hidden_states_first: 
                hidden_states = first_block(hidden_states, encoder_hidden_states, *args, **kwargs)
            else: 
                hidden_states = first_block(encoder_hidden_states, hidden_states, *args, **kwargs)
        
        if not self.return_hidden_states_only:
            hidden_states, encoder_hidden_states = hidden_states
            if not self.return_hidden_states_first: 
                hidden_states, encoder_hidden_states = encoder_hidden_states, hidden_states
        
        first_residual = (hidden_states - original_hidden_states).contiguous()
        
        can_use = get_can_use_cache(first_residual, self.residual_diff_threshold)
        if self.validate_can_use_cache_function: 
            can_use = self.validate_can_use_cache_function(can_use)
        
        torch._dynamo.graph_break()
        
        if can_use:
            result = apply_cached_residual(hidden_states, first_residual, encoder_hidden_states)
            if isinstance(result, tuple):
                hidden_states, encoder_hidden_states = result
            else:
                hidden_states = result
        else:
            set_buffer("first_hidden_states_residual", first_residual)
            hidden_states, encoder_hidden_states, hr, er = self.call_remaining_transformer_blocks(
                hidden_states, encoder_hidden_states, *args, txt_arg_name=txt_arg_name, **kwargs
            )
            set_buffer("hidden_states_residual", hr)
            if er is not None: 
                set_buffer("encoder_hidden_states_residual", er)
        
        torch._dynamo.graph_break()
        
        if self.return_hidden_states_only: 
            return hidden_states
        return (
            (hidden_states, encoder_hidden_states) if self.return_hidden_states_first 
            else (encoder_hidden_states, hidden_states)
        )

    def call_remaining_transformer_blocks(self, hidden_states, encoder_hidden_states, *args, txt_arg_name=None, **kwargs):
        original_hidden_states = hidden_states.clone() if self.clone_original_hidden_states else hidden_states
        original_encoder_hidden_states = (
            encoder_hidden_states.clone() if self.clone_original_hidden_states and encoder_hidden_states is not None 
            else encoder_hidden_states
        )

        for block in self.transformer_blocks[1:]:
            if txt_arg_name == "encoder_hidden_states":
                hidden_states = block(hidden_states, *args, encoder_hidden_states=encoder_hidden_states, **kwargs)
            else:
                if self.accept_hidden_states_first: 
                    hidden_states = block(hidden_states, encoder_hidden_states, *args, **kwargs)
                else: 
                    hidden_states = block(encoder_hidden_states, hidden_states, *args, **kwargs)
            
            if not self.return_hidden_states_only:
                hidden_states, encoder_hidden_states = hidden_states
                if not self.return_hidden_states_first: 
                    hidden_states, encoder_hidden_states = encoder_hidden_states, hidden_states
        
        if self.single_transformer_blocks is not None:
            hidden_states = torch.cat(
                [hidden_states, encoder_hidden_states] if self.cat_hidden_states_first 
                else [encoder_hidden_states, hidden_states], 
                dim=1
            )
            for block in self.single_transformer_blocks: 
                hidden_states = block(hidden_states, *args, **kwargs)
            
            if self.cat_hidden_states_first:
                hidden_states, encoder_hidden_states = hidden_states.split(
                    [hidden_states.shape[1] - encoder_hidden_states.shape[1], encoder_hidden_states.shape[1]], 
                    dim=1
                )
            else:
                encoder_hidden_states, hidden_states = hidden_states.split(
                    [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], 
                    dim=1
                )

        hidden_states = hidden_states.flatten().contiguous().reshape(hidden_states.shape)
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.flatten().contiguous().reshape(encoder_hidden_states.shape)

        res_h = (hidden_states - original_hidden_states).contiguous()
        res_e = (encoder_hidden_states - original_encoder_hidden_states).contiguous() if encoder_hidden_states is not None else None
        
        return hidden_states, encoder_hidden_states, res_h, res_e


# Exports main functions
__all__ = [
    'CacheContext',
    'create_cache_context',
    'get_current_cache_context',
    'set_current_cache_context',
    'cache_context',
    'patch_get_output_data',
    'create_patch_flux_forward',
    'create_patch_unet_model__forward',
    'CachedTransformerBlocks',
]
