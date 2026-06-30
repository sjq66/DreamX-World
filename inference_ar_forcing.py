"""
AR-Forcing inference script for chunk-wise causal video generation with camera control.

Input JSON format:
[
    {
        "image_path": "/path/to/image.png",
        "caption": "description text",
        "action_seq": ["wi", "s"],
        "action_speed_list": [6, 4]
    },
    ...
]

Usage:
    CUDA_VISIBLE_DEVICES=0 python inference_ar_forcing.py \
        --config_path configs/ar_forcing/causal_camera_forcing_5b.yaml \
        --base_checkpoint_path /path/to/baseline.pt \
        --data_path configs/dreamx/eval.json \
        --output_folder outputs_ar/ \
        --num_output_frames 21
"""

import argparse
import csv
import json
import os

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

from pipeline.pipeline_causal_camera import CausalCameraInferencePipeline
from utils.misc import set_seed
from utils.trajectory_processor import generate_trajectory_from_json, Camera
from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller
from utils.postprocess import postprocess_video_frames


# ─────────────────────────────────────────────────────────────────────────────
# Camera: cam_params (numpy) → PRoPE dict {viewmats, K}
# ─────────────────────────────────────────────────────────────────────────────

def _invert_SE3(mats):
    """Invert batch of 4x4 SE(3) matrices."""
    R_inv = mats[..., :3, :3].transpose(-1, -2)
    result = torch.zeros_like(mats)
    result[..., :3, :3] = R_inv
    result[..., :3, 3] = -torch.einsum("...ij,...j->...i", R_inv, mats[..., :3, 3])
    result[..., 3, 3] = 1.0
    return result


def get_relative_pose(cam_params):
    """Compute relative c2w poses (first frame as origin)."""
    abs_w2cs = [cp.w2c_mat for cp in cam_params]
    abs_c2ws = [cp.c2w_mat for cp in cam_params]
    target_cam_c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w] + [abs2rel @ c2w for c2w in abs_c2ws[1:]]
    return np.array(ret_poses, dtype=np.float32)


def cam_params_to_prope_dict(cam_params, device, dtype=torch.bfloat16, chunk_relative=False):
    """
    Convert Camera objects → PRoPE conditioning dict.

    Steps:
      1. Subsample to latent-aligned frames (1+4k pattern)
      2. Compute relative c2w poses (global or chunk-relative)
      3. Invert to w2c (viewmats)
      4. Expand each frame to 880 spatial tokens (22x40 patches)
      5. Build normalized intrinsic K matrices

    Args:
        chunk_relative: If True, compute relative poses per chunk (chunk_size=3).

    Returns:
        {'viewmats': [1, T_latent*880, 4, 4], 'K': [1, T_latent*880, 3, 3]}
    """
    num_frames = len(cam_params)
    latent_frame_count = 1 + (num_frames - 1) // 4
    aligned_indices = [0] + [1 + 4 * i for i in range(latent_frame_count - 1)]
    cam_params_sub = [cam_params[i] for i in aligned_indices]

    if chunk_relative:
        chunk_size = 3
        all_relative_poses = []
        for chunk_start in range(0, len(cam_params_sub), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(cam_params_sub))
            if chunk_start == 0:
                chunk_cams = cam_params_sub[chunk_start:chunk_end]
            else:
                reference_cam = cam_params_sub[chunk_start - 1]
                chunk_cams = [reference_cam] + cam_params_sub[chunk_start:chunk_end]
            chunk_poses = get_relative_pose(chunk_cams)
            if chunk_start == 0:
                all_relative_poses.append(chunk_poses)
            else:
                all_relative_poses.append(chunk_poses[1:])
        c2w_poses = np.concatenate(all_relative_poses, axis=0)
    else:
        c2w_poses = get_relative_pose(cam_params_sub)
    c2ws = torch.as_tensor(c2w_poses, dtype=dtype, device=device)

    viewmats = _invert_SE3(c2ws)  # [T_latent, 4, 4]
    # Expand to per-token: 880 = 22*40 spatial patches per frame
    viewmats = viewmats.unsqueeze(1).expand(-1, 880, -1, -1).reshape(1, -1, 4, 4)

    # Normalized intrinsics (fixed, matching training config)
    fx_norm = 969.6969696969696 / (960.0 * 2)  # ≈ 0.505
    fy_norm = 969.6969696969696 / (540.0 * 2)  # ≈ 0.898

    K = torch.zeros((1, 3, 3), dtype=dtype, device=device)
    K[:, 0, 0] = fx_norm
    K[:, 1, 1] = fy_norm
    K[:, 0, 2] = 0.5
    K[:, 1, 2] = 0.5
    K[:, 2, 2] = 1.0
    K = K.unsqueeze(1).expand(-1, viewmats.shape[1], -1, -1).reshape(1, -1, 3, 3)

    return {'viewmats': viewmats, 'K': K}


def parse_args():
    parser = argparse.ArgumentParser(description="AR-Forcing causal video generation")
    # Model and config paths
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to AR-forcing YAML config file")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Path to the folder containing Wan2.2 base model weights (text encoder, tokenizer, VAE)")
    parser.add_argument("--transformer_path", type=str, default=None,
                        help="Path to the folder containing AR-forcing transformer config.json")
    parser.add_argument("--vae_path", type=str, default=None,
                        help="Path to VAE checkpoint file (overrides model_name/Wan2.2_VAE.pth)")
    parser.add_argument("--base_checkpoint_path", type=str, default=None,
                        help="Path to base .pt checkpoint (generator_ema key)")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to additional checkpoint (.pt or .safetensors)")

    # Input/Output
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)

    # Generation parameters
    parser.add_argument("--num_output_frames", type=int, default=21)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--local_attn_size", type=int, default=None,
                        help="Override AR self-attention local window size in latent frames")
    parser.add_argument("--sink_size", type=int, default=None,
                        help="Override number of sink latent frames kept at the cache front")

    # Post-processing
    parser.add_argument("--color_correction_strength", type=float, default=0.3)

    # Camera
    parser.add_argument("--chunk_relative", action="store_true",
                        help="Compute relative camera poses per chunk (chunk_size=3) instead of globally")

    # KV cache eviction
    parser.add_argument("--kv_evict_policy", type=str, default="fifo",
                        choices=["fifo", "similarity", "stride", "pyramid"],
                        help="KV cache eviction policy for the AR self-attention cache")
    parser.add_argument("--kv_similarity_threshold", type=float, default=0.95,
                        help="Cosine threshold for similarity-based KV chunk eviction")
    parser.add_argument("--kv_similarity_recent_keep_chunks", type=int, default=1,
                        help="Number of most recent non-sink chunks protected from similarity eviction")
    parser.add_argument("--kv_similarity_source", type=str, default="k",
                        choices=["k", "v", "kv"],
                        help="Cache tensor used for average-pooled chunk similarity")
    parser.add_argument("--kv_stride_anchor_chunks", type=int, default=1,
                        help="For stride eviction, preserve this many oldest non-sink chunks before evicting a middle chunk")
    parser.add_argument("--kv_pyramid_recent_keep_chunks", type=int, default=3,
                        help="For pyramid eviction, keep this many most recent non-sink chunks dense")
    parser.add_argument("--kv_pyramid_long_keep_chunks", type=int, default=4,
                        help="For pyramid eviction, keep this many older non-sink chunks before thinning")
    parser.add_argument("--kv_similarity_debug", action="store_true",
                        help="Print similarity-eviction decisions during AR inference")
    parser.add_argument("--kv_eviction_log_path", type=str, default=None,
                        help="Optional CSV path for per-roll KV eviction decisions")

    # LoRA
    parser.add_argument("--lora_ckpt", type=str, default=None,
                        help="Path to LoRA checkpoint (requires adapter section in config)")
    return parser.parse_args()


def maybe_enable_eprope_from_checkpoint(args, config):
    """Match PRoPE attention compression to DreamX safetensors checkpoints."""
    checkpoint_path = args.checkpoint_path
    if not checkpoint_path or not checkpoint_path.endswith(".safetensors"):
        return

    try:
        from safetensors import safe_open
        with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            q_proj_key = None
            for candidate in (
                "blocks.0.cam_self_attn.q_proj.weight",
                "model.blocks.0.cam_self_attn.q_proj.weight",
            ):
                if candidate in keys:
                    q_proj_key = candidate
                    break
            if q_proj_key is None:
                return
            q_proj_shape = tuple(f.get_tensor(q_proj_key).shape)
    except Exception as exc:
        print(f"Warning: failed to inspect checkpoint shape for eprope auto-detect: {exc}")
        return

    if len(q_proj_shape) != 2:
        return

    out_dim, in_dim = q_proj_shape
    if out_dim < in_dim and in_dim % out_dim == 0:
        compress = in_dim // out_dim
        if compress > 1:
            config.model_kwargs.eprope = True
            print(
                "Detected compressed PRoPE checkpoint "
                f"({q_proj_key} shape={q_proj_shape}); enabling eprope "
                f"(attn_compress={compress})."
            )


def save_video(output_path, frames, fps):
    """Write RGB video frames with imageio/ffmpeg.

    torchvision.io.write_video is brittle with newer PyAV versions because it
    sets VideoFrame.pict_type to a string. imageio uses ffmpeg directly and is
    stable in the current uv environment.
    """
    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().clamp(0, 255).to(torch.uint8).numpy()
    else:
        frames = np.asarray(frames).clip(0, 255).astype(np.uint8)

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected RGB frames with shape [T,H,W,3], got {frames.shape}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimwrite(
        output_path,
        frames,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )


def load_pipeline(args, config, device):
    """Load the CausalCameraInferencePipeline with checkpoints."""
    maybe_enable_eprope_from_checkpoint(args, config)

    if args.local_attn_size is not None:
        config.model_kwargs.local_attn_size = args.local_attn_size
    if args.sink_size is not None:
        config.model_kwargs.sink_size = args.sink_size

    # Build explicit paths from --model_name and --transformer_path if provided
    text_encoder_path = None
    tokenizer_path = None
    vae_path = args.vae_path
    model_config_path = None

    if args.model_name:
        text_encoder_path = os.path.join(args.model_name, "models_t5_umt5-xxl-enc-bf16.pth")
        tokenizer_path = os.path.join(args.model_name, "google/umt5-xxl/")
        if vae_path is None:
            vae_path = os.path.join(args.model_name, "Wan2.2_VAE.pth")

    if args.transformer_path:
        model_config_path = os.path.join(args.transformer_path, "config.json")

    pipeline = CausalCameraInferencePipeline(
        config, device=device, num_output_frames=args.num_output_frames,
        model_config_path=model_config_path,
        text_encoder_path=text_encoder_path,
        tokenizer_path=tokenizer_path,
        vae_path=vae_path,
    )

    if args.base_checkpoint_path:
        state_dict = torch.load(args.base_checkpoint_path, map_location="cpu")
        checkpoint_key = "generator_ema" if "generator_ema" in state_dict else "generator"
        gen_sd = state_dict.get(checkpoint_key, state_dict)
        try:
            missing, unexpected = pipeline.generator.load_state_dict(gen_sd, strict=False)
        except RuntimeError:
            fixed = {k.replace("model._fsdp_wrapped_module.", "model.", 1): v for k, v in gen_sd.items()}
            missing, unexpected = pipeline.generator.load_state_dict(fixed, strict=False)
        print(f"Base checkpoint loaded — missing: {len(missing)}, unexpected: {len(unexpected)}")

    if args.checkpoint_path:
        if args.checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd = load_file(args.checkpoint_path)
            sd = {"model." + k: v for k, v in sd.items()}
        elif args.checkpoint_path.endswith(".pt"):
            raw = torch.load(args.checkpoint_path, map_location="cpu")
            sd = raw.get("generator_ema", raw.get("generator", raw))
        else:
            import glob
            from safetensors.torch import load_file
            sd = {}
            for f in glob.glob(args.checkpoint_path + "/*.safetensors"):
                for k, v in load_file(f).items():
                    if args.base_checkpoint_path is None or 'cam_self_attn' in k:
                        sd['model.' + k] = v
        missing, unexpected = pipeline.generator.load_state_dict(sd, strict=False)
        print(f"Checkpoint loaded — missing: {len(missing)}, unexpected: {len(unexpected)}")

    # LoRA support
    lora_ckpt_path = args.lora_ckpt
    if getattr(config, "adapter", None) and lora_ckpt_path:
        import peft
        from utils.lora_peft import configure_lora_for_model
        print(f"Applying LoRA with config: {config.adapter}")
        pipeline.generator.model = configure_lora_for_model(
            pipeline.generator.model,
            model_name="generator",
            lora_config=config.adapter,
        )
        print(f"Loading LoRA checkpoint from {lora_ckpt_path}")
        lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
        if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])
        else:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)
        print("LoRA weights loaded for generator")

    return pipeline


def write_kv_eviction_rows(log_path, rows, fieldnames):
    if not log_path or not rows:
        return
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def estimate_kv_cache_gb(pipeline, dtype_bytes=2):
    local_attn_size = pipeline.generator.model.local_attn_size
    if local_attn_size == -1:
        return float("nan")
    num_blocks = getattr(pipeline, "num_transformer_blocks", 30)
    frame_seq_length = getattr(pipeline, "frame_seq_length", 880)
    num_heads = 24
    head_dim = 128
    tensors_per_cache = 2  # K and V
    total_bytes = (
        num_blocks * local_attn_size * frame_seq_length *
        num_heads * head_dim * tensors_per_cache * dtype_bytes
    )
    return total_bytes / (1024 ** 3)


def main():
    args = parse_args()
    device = torch.device("cuda")
    set_seed(args.seed)
    torch.set_grad_enabled(False)

    print(f"Free VRAM: {get_cuda_free_memory_gb(gpu):.1f} GB")
    low_memory = get_cuda_free_memory_gb(gpu) < 40

    # Load config
    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load(
        os.path.join(os.path.dirname(args.config_path), "default_config.yaml"))
    config = OmegaConf.merge(default_config, config)

    # Load pipeline
    pipeline = load_pipeline(args, config, device)
    pipeline = pipeline.to(dtype=torch.bfloat16)
    print(
        "KV cache config: "
        f"local_attn_size={pipeline.generator.model.local_attn_size}, "
        f"sink_size={pipeline.generator.model.sink_size}, "
        f"estimated_kv_cache={estimate_kv_cache_gb(pipeline):.2f} GiB"
    )
    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
    else:
        pipeline.text_encoder.to(device=gpu)
    pipeline.generator.to(device=gpu)
    pipeline.vae.to(device=gpu)

    # Load JSON data
    with open(args.data_path, 'r') as f:
        items = json.load(f)
    print(f"Loaded {len(items)} items from {args.data_path}")

    # Image transform (fixed 704x1280)
    transform = transforms.Compose([
        transforms.Resize((704, 1280)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    num_pixel_frames = (args.num_output_frames - 1) * 4 + 1
    os.makedirs(args.output_folder, exist_ok=True)
    kv_cache_policy = {
        "policy": args.kv_evict_policy,
        "similarity_threshold": args.kv_similarity_threshold,
        "recent_keep_chunks": args.kv_similarity_recent_keep_chunks,
        "similarity_source": args.kv_similarity_source,
        "stride_anchor_chunks": args.kv_stride_anchor_chunks,
        "pyramid_recent_keep_chunks": args.kv_pyramid_recent_keep_chunks,
        "pyramid_long_keep_chunks": args.kv_pyramid_long_keep_chunks,
        "debug": args.kv_similarity_debug,
        "stats": [],
    }
    kv_eviction_fields = [
        "task_id", "output_name", "event_index", "layer", "policy", "reason",
        "best_similarity", "threshold", "evict_start_index", "fifo_start_index",
        "chosen_is_fifo", "num_candidates", "num_new_tokens", "num_evicted_tokens",
        "old_local_end", "global_end_index", "candidate_starts",
        "candidate_similarities",
    ]
    if args.kv_evict_policy != "fifo":
        print(
            "KV cache policy: "
            f"{args.kv_evict_policy}, threshold={args.kv_similarity_threshold}, "
            f"recent_keep_chunks={args.kv_similarity_recent_keep_chunks}, "
            f"source={args.kv_similarity_source}, "
            f"stride_anchor_chunks={args.kv_stride_anchor_chunks}, "
            f"pyramid_recent_keep_chunks={args.kv_pyramid_recent_keep_chunks}, "
            f"pyramid_long_keep_chunks={args.kv_pyramid_long_keep_chunks}"
        )
    if args.local_attn_size is not None or args.sink_size is not None:
        print(
            "KV cache window override: "
            f"local_attn_size={args.local_attn_size}, sink_size={args.sink_size}"
        )

    # ─── Inference loop ───
    for idx, item in enumerate(items):
        image_path = item['image_path']
        caption = item.get('caption', item.get('prompt', ''))
        action_seq = item['action_seq']
        action_speed_list = item['action_speed_list']
        task_id = item.get('task_id', str(idx))

        img_parent = os.path.basename(os.path.dirname(image_path))
        img_basename = os.path.splitext(os.path.basename(image_path))[0]
        output_name = f"{img_parent}_{img_basename}" if img_parent else img_basename
        output_path = os.path.join(args.output_folder, f'{task_id}_{output_name}.mp4')
        if os.path.exists(output_path):
            print(f"[{idx}] Skip (exists): {output_path}")
            continue

        print(f"[{idx}] Generating: {output_path}")
        stats_start = len(kv_cache_policy["stats"])
        torch.cuda.reset_peak_memory_stats(device)

        # 1) Encode input image
        pil_image = Image.open(image_path).convert('RGB')
        image_tensor = transform(pil_image).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)
        initial_latent = pipeline.vae.encode_to_latent(image_tensor).to(device=device, dtype=torch.bfloat16)

        # 2) Build noise (first frame = encoded image)
        sampled_noise = torch.randn([1, args.num_output_frames, 48, 44, 80], device=device, dtype=torch.bfloat16)
        sampled_noise[:, 0] = initial_latent

        # 3) Build camera trajectory → PRoPE dict
        action_seq_lower = [a.lower() for a in action_seq]
        trajectory_spec = list(zip(action_seq_lower, action_speed_list))
        _, cam_params_np, _ = generate_trajectory_from_json(
            trajectory_spec=trajectory_spec,
            num_frames=num_pixel_frames,
            return_cam_params=True,
        )
        cam_objects = [Camera(cam_params_np[i].tolist()) for i in range(cam_params_np.shape[0])]
        control_camera = cam_params_to_prope_dict(cam_objects, device=device, chunk_relative=args.chunk_relative)

        # 4) Run inference
        video, latents = pipeline.inference(
            noise=sampled_noise,
            text_prompts=[caption],
            y=None,
            y_camera=control_camera,
            return_latents=True,
            kv_cache_policy=kv_cache_policy,
        )

        # 5) Post-process and save video
        video = rearrange(video, 'b t c h w -> b t h w c').cpu()
        video = 255.0 * video
        pipeline.vae.model.clear_cache()

        reference_frame = video[0, 0] if video.shape[1] > 0 else None
        video = postprocess_video_frames(
            video,
            reference_frame=reference_frame,
            color_correction_strength=args.color_correction_strength,
        )

        save_video(output_path, video[0], fps=args.fps)
        print(f"    Saved: {output_path}")
        print(f"    Peak reserved VRAM: {torch.cuda.max_memory_reserved(device) / (1024 ** 3):.2f} GiB")

        new_stats = kv_cache_policy["stats"][stats_start:]
        rows = []
        for event_offset, stat in enumerate(new_stats):
            row = {field: "" for field in kv_eviction_fields}
            row.update(stat)
            row["task_id"] = task_id
            row["output_name"] = output_name
            row["event_index"] = stats_start + event_offset
            if isinstance(row.get("candidate_starts"), list):
                row["candidate_starts"] = ";".join(str(v) for v in row["candidate_starts"])
            if isinstance(row.get("candidate_similarities"), list):
                row["candidate_similarities"] = ";".join(f"{v:.6f}" for v in row["candidate_similarities"])
            rows.append(row)
        write_kv_eviction_rows(args.kv_eviction_log_path, rows, kv_eviction_fields)

    print("Done.")


if __name__ == "__main__":
    main()
