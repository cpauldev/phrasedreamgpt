from __future__ import annotations

import importlib.util
import random
import sys
import time
from collections.abc import Callable, Collection
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    MODEL_RESIDUAL_MODE_ATTNRES,
    MODEL_RESIDUAL_MODE_ATTNRES_BLOCK,
    Dataset,
    GenerationConfig,
    ModelConfig,
    PrecisionSettings,
    RuntimeSettings,
    TrainingConfig,
    TrainingResult,
    fail,
    print_section,
)
from .source_filter import DEFAULT_SOURCE_FILTER_MAX_RETRIES, BloomSourceFilter

if TYPE_CHECKING:
    from .artifacts import ArtifactBundle


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def has_mps() -> bool:
    return (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            fail(
                "--device cuda was requested but CUDA is not available.",
                (
                    "Install a CUDA-enabled PyTorch build, verify "
                    "`torch.cuda.is_available()`, or rerun with --device cpu."
                ),
            )
        return torch.device("cuda")

    if device_arg == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_built():
            fail(
                "--device mps was requested but this PyTorch build does not include MPS support.",
                "Install a PyTorch build with MPS support or rerun with --device cpu.",
            )
        if not torch.backends.mps.is_available():
            fail(
                "--device mps was requested but MPS is unavailable on this machine.",
                "Use an Apple Silicon machine with MPS enabled, or rerun with --device cpu.",
            )
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if has_mps():
        return torch.device("mps")
    return torch.device("cpu")


def load_dataset(path: str, *, shuffle: bool = True) -> Dataset:
    input_path = Path(path)
    if not input_path.exists():
        fail(
            f"Dataset file not found: {input_path}",
            "Pass a valid path with --dataset PATH.",
        )

    try:
        with input_path.open("r", encoding="utf-8") as handle:
            docs = [line.strip() for line in handle if line.strip()]
    except UnicodeDecodeError as exc:
        fail(
            f"Failed to decode input file as UTF-8: {input_path}",
            f"Save the dataset as UTF-8 text and try again. Original error: {exc}",
        )
    except OSError as exc:
        fail(
            f"Failed to read input file: {input_path}",
            f"Check that the file exists and is readable. Original error: {exc}",
        )

    if not docs:
        fail(
            f"Input file has no non-empty lines: {input_path}",
            "Add at least one non-empty line to the dataset file.",
        )

    if shuffle:
        random.shuffle(docs)

    id_to_char = sorted(set("".join(docs)))
    char_to_id = {ch: i for i, ch in enumerate(id_to_char)}
    bos_id = len(id_to_char)
    vocab_size = bos_id + 1

    stream = [bos_id]
    for doc in docs:
        stream.extend(char_to_id[ch] for ch in doc)
        stream.append(bos_id)

    return Dataset(
        data=torch.tensor(stream, dtype=torch.long),
        id_to_char=id_to_char,
        bos_id=bos_id,
        vocab_size=vocab_size,
    )


def dataset_document_count(dataset: Dataset) -> int:
    bos_count = int((dataset.data == dataset.bos_id).sum().item())
    return max(0, bos_count - 1)


def print_dataset_summary(path: str, dataset: Dataset) -> None:
    print_section("dataset")
    print(f"file    {Path(path).name}")
    print(f"docs    {dataset_document_count(dataset):,}")
    print(f"vocab   {dataset.vocab_size:,} chars")
    print(f"tokens  {dataset.data.numel():,}")


def ensure_dataset_supports_block_size(dataset: Dataset, block_size: int) -> None:
    minimum_tokens = block_size + 2
    if dataset.data.numel() < minimum_tokens:
        fail(
            f"The dataset is too small for --block-size {block_size}.",
            (
                "Reduce --block-size or provide more input data. This setup needs "
                f"at least {minimum_tokens} tokens after tokenization."
            ),
        )


def unwrap_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def dtype_name(dtype: torch.dtype | None) -> str | None:
    if dtype is None:
        return None
    return str(dtype).replace("torch.", "")


def is_compiled_model(model: nn.Module) -> bool:
    return hasattr(model, "_orig_mod")


def capture_rng_state() -> dict:
    state = {
        "python_random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    if (
        hasattr(torch, "mps")
        and hasattr(torch.mps, "get_rng_state")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        state["mps_rng_state"] = torch.mps.get_rng_state()
    return state


def as_cpu_bytetensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu", dtype=torch.uint8).contiguous()


def restore_rng_state(rng_state: dict) -> None:
    if "python_random_state" in rng_state:
        random.setstate(rng_state["python_random_state"])
    if "torch_rng_state" in rng_state:
        torch.set_rng_state(as_cpu_bytetensor(rng_state["torch_rng_state"]))

    cuda_rng_state_all = rng_state.get("cuda_rng_state_all")
    if cuda_rng_state_all is not None:
        if not torch.cuda.is_available():
            fail(
                "This checkpoint requires CUDA RNG state restoration, but CUDA is not available.",
                (
                    "Resume on a CUDA-enabled machine or load the model from "
                    "the interactive models menu for generation only."
                ),
            )
        torch.cuda.set_rng_state_all([as_cpu_bytetensor(state) for state in cuda_rng_state_all])

    mps_rng_state = rng_state.get("mps_rng_state")
    if mps_rng_state is not None:
        if not (
            hasattr(torch, "mps")
            and hasattr(torch.mps, "set_rng_state")
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            fail(
                "This checkpoint requires MPS RNG state restoration, but MPS is not available.",
                (
                    "Resume on a machine with MPS support or load the model "
                    "from the interactive models menu for generation only."
                ),
            )
        torch.mps.set_rng_state(as_cpu_bytetensor(mps_rng_state))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class AttentionResidual(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.proj = nn.Linear(dim, 1, bias=False)

    def forward(self, history: list[torch.Tensor]) -> torch.Tensor:
        # Full AttnRes from Eq. 2-4: one learned pseudo-query attends over
        # the embedding and all previously produced layer outputs.
        stacked = torch.stack(history, dim=0)
        logits = self.proj(self.norm(stacked)).squeeze(-1)
        weights = torch.softmax(logits, dim=0)
        return (weights.unsqueeze(-1) * stacked).sum(dim=0)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(channels, dim=2)

        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.c_proj(y)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, n_embd: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(n_embd, hidden_dim, bias=False)
        self.up_proj = nn.Linear(n_embd, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.residual_mode = config.residual_mode
        self.norm1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config.n_embd, config.n_head)
        self.norm2 = RMSNorm(config.n_embd)
        self.feed_forward = SwiGLUFeedForward(config.n_embd, config.mlp_hidden_dim)
        if self.residual_mode == MODEL_RESIDUAL_MODE_ATTNRES:
            self.attn_residual = AttentionResidual(config.n_embd)
            self.feed_forward_residual = AttentionResidual(config.n_embd)
        if self.residual_mode == MODEL_RESIDUAL_MODE_ATTNRES_BLOCK:
            self.attn_block_residual = AttentionResidual(config.n_embd)
            self.feed_forward_block_residual = AttentionResidual(config.n_embd)

    def apply_attention(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(self.norm1(x))

    def apply_feed_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feed_forward(self.norm2(x))

    @staticmethod
    def _block_attnres_sources(
        completed_blocks: list[torch.Tensor],
        partial_block: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        if partial_block is None:
            return list(completed_blocks)
        return [*completed_blocks, partial_block]

    @staticmethod
    def _merge_partial_block(
        partial_block: torch.Tensor | None,
        sublayer_out: torch.Tensor,
    ) -> torch.Tensor:
        return sublayer_out if partial_block is None else partial_block + sublayer_out

    def _forward_block_attnres_sublayer(
        self,
        *,
        residual: AttentionResidual,
        transform: Callable[[torch.Tensor], torch.Tensor],
        completed_blocks: list[torch.Tensor],
        partial_block: torch.Tensor | None,
    ) -> torch.Tensor:
        sublayer_input = residual(self._block_attnres_sources(completed_blocks, partial_block))
        sublayer_out = transform(sublayer_input)
        return self._merge_partial_block(partial_block, sublayer_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual_mode in {MODEL_RESIDUAL_MODE_ATTNRES, MODEL_RESIDUAL_MODE_ATTNRES_BLOCK}:
            raise TypeError(
                "Attention Residual blocks require a residual-mode-specific forward path."
            )
        x = x + self.apply_attention(x)
        x = x + self.apply_feed_forward(x)
        return x

    def forward_attnres(self, history: list[torch.Tensor]) -> list[torch.Tensor]:
        attn_input = self.attn_residual(history)
        attn_out = self.apply_attention(attn_input)
        history_with_attn = history + [attn_out]

        ff_input = self.feed_forward_residual(history_with_attn)
        ff_out = self.apply_feed_forward(ff_input)
        return history_with_attn + [ff_out]

    def forward_block_attnres_attention(
        self,
        completed_blocks: list[torch.Tensor],
        partial_block: torch.Tensor | None,
    ) -> torch.Tensor:
        # Block AttnRes from Eq. 5-6 / Fig. 2: first layer in a block
        # attends over completed block summaries only; later layers also see
        # the current intra-block partial sum.
        return self._forward_block_attnres_sublayer(
            residual=self.attn_block_residual,
            transform=self.apply_attention,
            completed_blocks=completed_blocks,
            partial_block=partial_block,
        )

    def forward_block_attnres_feed_forward(
        self,
        completed_blocks: list[torch.Tensor],
        partial_block: torch.Tensor | None,
    ) -> torch.Tensor:
        return self._forward_block_attnres_sublayer(
            residual=self.feed_forward_block_residual,
            transform=self.apply_feed_forward,
            completed_blocks=completed_blocks,
            partial_block=partial_block,
        )


def residual_site_count(transformer_blocks: int) -> int:
    return transformer_blocks * 2


def resolve_block_attnres_layout(
    total_residual_sites: int,
    requested_block_count: int,
) -> tuple[int, tuple[int, ...]]:
    block_count = min(requested_block_count, total_residual_sites)
    base_block_layers = total_residual_sites // block_count
    remainder = total_residual_sites % block_count
    block_end_indices: list[int] = []
    next_end = -1

    for block_index in range(block_count):
        block_layers = base_block_layers
        if block_index == block_count - 1:
            block_layers += remainder
        next_end += block_layers
        block_end_indices.append(next_end)

    return block_count, tuple(block_end_indices)


class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        self.residual_mode = config.residual_mode
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        if self.residual_mode in {MODEL_RESIDUAL_MODE_ATTNRES, MODEL_RESIDUAL_MODE_ATTNRES_BLOCK}:
            self.output_residual = AttentionResidual(config.n_embd)
        if self.residual_mode == MODEL_RESIDUAL_MODE_ATTNRES_BLOCK:
            self.total_residual_sites = residual_site_count(config.n_layer)
            (
                self.effective_residual_block_count,
                self.block_end_indices,
            ) = resolve_block_attnres_layout(
                self.total_residual_sites,
                config.residual_block_count,
            )
        self.norm_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        self._init_attention_residual_queries()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.08)

    def _init_attention_residual_queries(self) -> None:
        for module in self.modules():
            if isinstance(module, AttentionResidual):
                # The paper initializes every pseudo-query to zero so AttnRes
                # starts as uniform averaging over its available sources.
                nn.init.zeros_(module.proj.weight)

    @staticmethod
    def _maybe_close_block(
        *,
        depth_index: int,
        block_end_indices: set[int],
        completed_blocks: list[torch.Tensor],
        partial_block: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if depth_index not in block_end_indices:
            return partial_block
        if partial_block is None:
            raise RuntimeError("Block AttnRes cannot close a block without a partial sum.")
        completed_blocks.append(partial_block)
        return None

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, seq_len = idx.shape
        if seq_len > self.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block_size {self.block_size}")

        positions = torch.arange(0, seq_len, device=idx.device)
        x = self.wte(idx) + self.wpe(positions)[None, :, :]
        if self.residual_mode == MODEL_RESIDUAL_MODE_ATTNRES:
            history = [x]
            for block in self.blocks:
                history = block.forward_attnres(history)
            x = self.output_residual(history)
        elif self.residual_mode == MODEL_RESIDUAL_MODE_ATTNRES_BLOCK:
            completed_blocks = [x]
            partial_block: torch.Tensor | None = None
            block_end_indices = set(self.block_end_indices)
            depth_index = 0
            for block in self.blocks:
                partial_block = block.forward_block_attnres_attention(
                    completed_blocks,
                    partial_block,
                )
                partial_block = self._maybe_close_block(
                    depth_index=depth_index,
                    block_end_indices=block_end_indices,
                    completed_blocks=completed_blocks,
                    partial_block=partial_block,
                )
                depth_index += 1

                partial_block = block.forward_block_attnres_feed_forward(
                    completed_blocks,
                    partial_block,
                )
                partial_block = self._maybe_close_block(
                    depth_index=depth_index,
                    block_end_indices=block_end_indices,
                    completed_blocks=completed_blocks,
                    partial_block=partial_block,
                )
                depth_index += 1
            sources = (
                completed_blocks if partial_block is None else completed_blocks + [partial_block]
            )
            x = self.output_residual(sources)
        else:
            for block in self.blocks:
                x = block(x)
        x = self.norm_f(x)
        return self.lm_head(x)


def build_model(model_config: ModelConfig, state_dict: dict | None, device: torch.device) -> GPT:
    model = GPT(model_config).to(device)
    if state_dict is None:
        return model

    try:
        model.load_state_dict(state_dict)
    except Exception as exc:
        fail(
            "Checkpoint weights could not be loaded into the model.",
            f"The checkpoint may be incompatible or corrupted. Original error: {exc}",
        )
    return model


class BatchProvider:
    def __init__(self, data: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
        self.data = data.to(device)
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.offsets = torch.arange(block_size, device=device)

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = self.data.numel() - self.block_size - 1
        indices = torch.randint(0, max_start + 1, (self.batch_size,), device=self.device)
        positions = indices[:, None] + self.offsets[None, :]
        x = self.data[positions]
        y = self.data[positions + 1]
        return x, y


@dataclass(frozen=True)
class TrainingTracePoint:
    """A checkpoint captured during one `train_with_trace` call."""

    run_step: int
    completed_steps: int
    total_tokens: int
    elapsed: float
    final_loss: float


def resolve_amp_settings(
    amp_pref: bool | None, dtype_arg: str, device: torch.device
) -> PrecisionSettings:
    if dtype_arg != "auto" and device.type != "cuda":
        fail(
            (
                f"--dtype {dtype_arg} was requested but non-auto dtype selection "
                "requires --device cuda."
            ),
            "Use --device cuda for fp16/bf16 selection, or switch back to --dtype auto.",
        )

    if amp_pref is True and device.type != "cuda":
        fail(
            "--amp was requested but AMP is only supported with --device cuda in this script.",
            "Use --device cuda or rerun with --no-amp.",
        )

    if device.type != "cuda":
        return PrecisionSettings(amp_dtype=None, use_amp=False)

    if dtype_arg == "fp32":
        if amp_pref is True:
            fail(
                "--amp with --dtype fp32 is invalid.",
                "Use --dtype fp16, --dtype bf16, or --dtype auto when AMP is enabled.",
            )
        return PrecisionSettings(amp_dtype=None, use_amp=False)

    if dtype_arg == "bf16":
        if not torch.cuda.is_bf16_supported():
            fail(
                "--dtype bf16 was requested but this GPU/PyTorch build does not support bfloat16.",
                "Use --dtype fp16, --dtype auto, or rerun with --no-amp.",
            )
        amp_dtype = torch.bfloat16
    elif dtype_arg == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if amp_pref is False:
        return PrecisionSettings(amp_dtype=amp_dtype, use_amp=False)

    return PrecisionSettings(amp_dtype=amp_dtype, use_amp=True)


def make_optimizer(
    model: nn.Module, training_config: TrainingConfig, device: torch.device
) -> torch.optim.Optimizer:
    kwargs = {
        "lr": training_config.learning_rate,
        "betas": (training_config.beta1, training_config.beta2),
        "eps": training_config.eps,
        "weight_decay": training_config.weight_decay,
    }
    if device.type == "cuda":
        try:
            return torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
        except TypeError:
            pass
    return torch.optim.AdamW(model.parameters(), **kwargs)


def has_triton() -> bool:
    return importlib.util.find_spec("triton") is not None


def triton_install_hint() -> str:
    if sys.platform == "win32":
        return (
            "Install triton-windows (it provides the `triton` module on Windows), "
            "or rerun with --no-compile."
        )
    return "Install triton, or rerun with --no-compile."


def maybe_compile(model: nn.Module, compile_pref: bool | None, device: torch.device) -> nn.Module:
    if compile_pref is False:
        return model

    if compile_pref is True:
        if device.type != "cuda":
            fail(
                (
                    "--compile was requested but this script only supports compile "
                    "mode on --device cuda."
                ),
                "Use --device cuda or rerun with --no-compile.",
            )
        if not has_triton():
            fail(
                "--compile was requested but Triton is not installed.",
                triton_install_hint(),
            )
        if not hasattr(torch, "compile"):
            fail(
                "--compile was requested but this PyTorch build does not expose torch.compile.",
                "Upgrade PyTorch or rerun with --no-compile.",
            )
        try:
            return torch.compile(model)
        except Exception as exc:  # pragma: no cover - backend-specific failure path
            fail(
                "torch.compile failed to initialize.",
                f"Rerun with --no-compile. Original error: {exc}",
            )

    if device.type != "cuda":
        return model
    if not has_triton():
        return model
    if not hasattr(torch, "compile"):
        return model

    try:
        return torch.compile(model)
    except Exception:
        return model


def configure_matmul(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def create_grad_scaler(settings: PrecisionSettings, device: torch.device):
    if not (settings.use_amp and settings.amp_dtype == torch.float16 and device.type == "cuda"):
        return None
    return torch.amp.GradScaler("cuda", enabled=True)


def autocast_context(device: torch.device, settings: PrecisionSettings):
    if not settings.use_amp:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=settings.amp_dtype, enabled=True)


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def move_value_to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_value_to_device(inner, device) for key, inner in value.items()}
    if isinstance(value, list):
        return [move_value_to_device(inner, device) for inner in value]
    if isinstance(value, tuple):
        return tuple(move_value_to_device(inner, device) for inner in value)
    return value


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            state[key] = move_value_to_device(value, device)


def verify_resume_runtime(saved_training_config: dict, runtime: RuntimeSettings) -> None:
    expected_device = saved_training_config["resolved_device"]
    expected_amp_enabled = saved_training_config["amp_enabled"]
    expected_amp_dtype = saved_training_config["amp_dtype"]
    expected_compile_enabled = saved_training_config["compile_enabled"]

    if runtime.resolved_device != expected_device:
        fail(
            (
                "Exact resume requires the same execution backend that was used "
                "when the checkpoint was created."
            ),
            (
                f"Resume on {expected_device} or load the artifact from the "
                "models menu for generation "
                f"only. Current device resolved to {runtime.resolved_device}."
            ),
        )

    if runtime.amp_enabled != expected_amp_enabled:
        fail(
            (
                "Exact resume requires the same AMP setting that was active when "
                "the checkpoint was created."
            ),
            (
                "Resume with the original environment/settings, or start a new "
                "training run instead of exact resume."
            ),
        )

    if runtime.amp_dtype != expected_amp_dtype:
        fail(
            (
                "Exact resume requires the same AMP dtype that was active when "
                "the checkpoint was created."
            ),
            (
                "Resume with the original environment/settings. Expected "
                f"{expected_amp_dtype}, got {runtime.amp_dtype}."
            ),
        )

    if runtime.compile_enabled != expected_compile_enabled:
        fail(
            (
                "Exact resume requires the same compile setting that was active "
                "when the checkpoint was created."
            ),
            (
                "Resume with the original environment/settings, or start a new "
                "training run instead of exact resume."
            ),
        )


def _normalize_trace_steps(
    requested_steps: Collection[int] | None,
    total_steps: int,
) -> tuple[int, ...]:
    if requested_steps is None:
        requested_steps = ()

    normalized_steps = {
        step
        for step in requested_steps
        if isinstance(step, int) and step > 0 and step <= total_steps
    }
    normalized_steps.add(total_steps)
    return tuple(sorted(normalized_steps))


def _record_trace_point(
    trace: list[TrainingTracePoint],
    *,
    run_step: int,
    completed_steps_before: int,
    total_tokens: int,
    started_at: float,
    final_loss: float,
) -> None:
    elapsed = max(time.perf_counter() - started_at, 1e-9)
    trace.append(
        TrainingTracePoint(
            run_step=run_step,
            completed_steps=completed_steps_before + run_step,
            total_tokens=total_tokens,
            elapsed=elapsed,
            final_loss=final_loss,
        )
    )


def train_with_trace(
    training_config: TrainingConfig,
    dataset: Dataset,
    device: torch.device,
    *,
    resume_bundle: ArtifactBundle | None = None,
    trace_steps: Collection[int] | None = None,
    report_progress: bool = True,
) -> tuple[TrainingResult, list[TrainingTracePoint]]:
    """Train once and optionally capture intermediate checkpoints.

    `trace_steps` are run-local step numbers such as `(1000, 2000, 3000)`.
    The final step is always recorded in the returned trace, even if it is not
    listed explicitly, so downstream benchmark code can rely on a terminal
    checkpoint being present.
    """
    configure_matmul(device)

    if resume_bundle is None:
        model = GPT(training_config.model).to(device)
    else:
        model = build_model(training_config.model, resume_bundle.state_dict, device)
    model = maybe_compile(model, training_config.compile_requested, device)

    optimizer = make_optimizer(model, training_config, device)
    batcher = BatchProvider(
        dataset.data,
        training_config.model.block_size,
        training_config.batch_size,
        device,
    )
    precision = resolve_amp_settings(
        training_config.amp_requested,
        training_config.requested_dtype,
        device,
    )
    scaler = create_grad_scaler(precision, device)
    runtime = RuntimeSettings(
        requested_device=training_config.requested_device,
        resolved_device=str(device),
        requested_dtype=training_config.requested_dtype,
        amp_requested=training_config.amp_requested,
        amp_enabled=precision.use_amp,
        amp_dtype=dtype_name(precision.amp_dtype) if precision.use_amp else None,
        compile_requested=training_config.compile_requested,
        compile_enabled=is_compiled_model(model),
    )

    completed_steps_before = 0
    total_tokens_before = 0
    if resume_bundle is not None:
        verify_resume_runtime(resume_bundle.training_metadata, runtime)
        optimizer.load_state_dict(resume_bundle.optimizer_state)
        move_optimizer_state_to_device(optimizer, device)
        if scaler is not None and resume_bundle.scaler_state:
            scaler.load_state_dict(resume_bundle.scaler_state)

        completed_steps_before = int(resume_bundle.resume_state.get("completed_steps", 0))
        total_tokens_before = int(resume_bundle.resume_state.get("total_tokens", 0))
        restore_rng_state(resume_bundle.rng_state)

    total_tokens = total_tokens_before
    final_loss = float("nan")
    target_total_steps = completed_steps_before + training_config.steps

    raw_model = unwrap_model(model)
    param_count = sum(p.numel() for p in raw_model.parameters())
    cfg = raw_model.config

    device_label = str(device)
    if device.type == "cuda":
        device_label += f"  ({torch.cuda.get_device_name(0)})"
    amp_label = runtime.amp_dtype if runtime.amp_enabled else "off"

    if report_progress:
        print_section("model")
        print(f"params   {param_count:,}")
        print(f"layers   {cfg.n_layer}")
        print(f"heads    {cfg.n_head}")
        print(f"embd     {cfg.n_embd}")
        print(f"block    {cfg.block_size}")
        print(f"resid    {cfg.residual_mode}")
        if cfg.residual_mode == MODEL_RESIDUAL_MODE_ATTNRES_BLOCK:
            print(f"rblocks  {unwrap_model(model).effective_residual_block_count}")

        print_section("training")
        print(f"device   {device_label}")
        print(f"amp      {amp_label}")
        print(f"compile  {'on' if runtime.compile_enabled else 'off'}")
        print(f"steps    {target_total_steps:,}")
        print(f"batch    {training_config.batch_size}")
        print(f"lr       {training_config.learning_rate:.2e}")
        if resume_bundle is not None:
            print(f"from     step {completed_steps_before:,}")
        print()

    step_w = len(str(target_total_steps))
    started_at = time.perf_counter()
    model.train()
    normalized_trace_steps = _normalize_trace_steps(trace_steps, training_config.steps)
    trace_step_set = set(normalized_trace_steps)
    trace: list[TrainingTracePoint] = []

    for step in range(training_config.steps):
        global_step = completed_steps_before + step
        lr_t = training_config.learning_rate * (1.0 - global_step / max(target_total_steps, 1))
        optimizer.param_groups[0]["lr"] = lr_t

        xb, yb = batcher.get()

        with autocast_context(device, precision):
            logits = model(xb)
            loss = F.cross_entropy(
                logits.view(-1, dataset.vocab_size),
                yb.reshape(-1),
            )

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        final_loss = float(loss.item())
        total_tokens += xb.numel()
        run_step = step + 1

        if run_step in trace_step_set:
            _record_trace_point(
                trace,
                run_step=run_step,
                completed_steps_before=completed_steps_before,
                total_tokens=total_tokens,
                started_at=started_at,
                final_loss=final_loss,
            )

        if report_progress and (
            (run_step % training_config.print_every == 0) or run_step == training_config.steps
        ):
            elapsed = max(time.perf_counter() - started_at, 1e-9)
            run_tokens = total_tokens - total_tokens_before
            tok_s = run_tokens / elapsed
            steps_s = run_step / elapsed
            print(
                f"step {global_step + 1:{step_w}d}/{target_total_steps}"
                f"  loss {final_loss:.4f}"
                f"  tok/s {tok_s:,.0f}"
                f"  step/s {steps_s:.2f}",
                flush=True,
            )

    synchronize_device(device)

    elapsed = max(time.perf_counter() - started_at, 1e-9)
    run_tokens = total_tokens - total_tokens_before
    result = TrainingResult(
        model=model,
        elapsed=elapsed,
        total_tokens=total_tokens,
        tok_s=run_tokens / elapsed,
        final_loss=final_loss,
        completed_steps=target_total_steps,
        optimizer_state=optimizer.state_dict(),
        scaler_state=scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        runtime=runtime,
    )
    return result, trace


def train_once(
    training_config: TrainingConfig,
    dataset: Dataset,
    device: torch.device,
    *,
    resume_bundle: ArtifactBundle | None = None,
) -> TrainingResult:
    result, _ = train_with_trace(
        training_config,
        dataset,
        device,
        resume_bundle=resume_bundle,
        report_progress=True,
    )
    return result


@torch.no_grad()
def generate_sample_once(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    generation_config: GenerationConfig,
) -> str:
    idx = torch.tensor([[dataset.bos_id]], dtype=torch.long, device=device)
    chars: list[str] = []

    for _ in range(generation_config.requested_block_size):
        logits = model(idx[:, -generation_config.requested_block_size :])
        next_logits = logits[:, -1, :] / generation_config.temperature
        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        token = int(next_id.item())
        if token == dataset.bos_id:
            break
        chars.append(dataset.id_to_char[token])
        idx = torch.cat((idx, next_id), dim=1)

    return "".join(chars)


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    generation_config: GenerationConfig,
    *,
    source_filter: BloomSourceFilter | None = None,
) -> list[str]:
    model.eval()
    samples: list[str] = []

    for _ in range(generation_config.num_samples):
        for _ in range(DEFAULT_SOURCE_FILTER_MAX_RETRIES):
            sample = generate_sample_once(model, dataset, device, generation_config)
            if source_filter is None or not source_filter.matches(sample):
                samples.append(sample)
                break
        else:
            fail(
                "Failed to sample a non-source line within the retry limit.",
                (
                    f"Increase --temperature, train for fewer steps, "
                    "or resave the model. The fixed retry limit is "
                    f"{DEFAULT_SOURCE_FILTER_MAX_RETRIES} attempts."
                ),
            )

    return samples


def detect_compare_accelerator() -> tuple[torch.device, str] | None:
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if has_mps():
        return torch.device("mps"), "mps"
    return None


def compare_training(
    training_config: TrainingConfig,
    dataset: Dataset,
    compare_steps: int,
    *,
    accel_device: torch.device | None = None,
    accel_label: str | None = None,
) -> tuple[TrainingResult, TrainingResult, str]:
    if accel_device is None or accel_label is None:
        detected = detect_compare_accelerator()
        if detected is None:
            fail(
                "benchmark requires an accelerator but neither CUDA nor MPS is available.",
                "Install a CUDA-enabled PyTorch build or run on Apple Silicon with MPS support.",
            )
        accel_device, accel_label = detected

    cpu_config = TrainingConfig(
        dataset_path=training_config.dataset_path,
        seed=training_config.seed,
        steps=compare_steps,
        batch_size=training_config.batch_size,
        model=training_config.model,
        learning_rate=training_config.learning_rate,
        beta1=training_config.beta1,
        beta2=training_config.beta2,
        eps=training_config.eps,
        weight_decay=training_config.weight_decay,
        requested_device="cpu",
        requested_dtype=training_config.requested_dtype,
        amp_requested=training_config.amp_requested,
        compile_requested=training_config.compile_requested,
        print_every=training_config.print_every,
    )
    accel_config = TrainingConfig(
        dataset_path=training_config.dataset_path,
        seed=training_config.seed,
        steps=compare_steps,
        batch_size=training_config.batch_size,
        model=training_config.model,
        learning_rate=training_config.learning_rate,
        beta1=training_config.beta1,
        beta2=training_config.beta2,
        eps=training_config.eps,
        weight_decay=training_config.weight_decay,
        requested_device=training_config.requested_device,
        requested_dtype=training_config.requested_dtype,
        amp_requested=training_config.amp_requested,
        compile_requested=training_config.compile_requested,
        print_every=training_config.print_every,
    )

    print_section("benchmark")
    label_w = len("accelerator")
    print(f"{'accelerator':{label_w}}  {accel_label}")
    print(f"{'comparing':{label_w}}  cpu vs {accel_label}")

    print(f"\nrunning cpu for {compare_steps} steps...")
    cpu_result = train_once(cpu_config, dataset, torch.device("cpu"))

    print(f"\nrunning {accel_label} for {compare_steps} steps...")
    accel_result = train_once(accel_config, dataset, accel_device)

    return cpu_result, accel_result, accel_label


def print_training_summary(result: TrainingResult) -> None:
    print(
        f"\ndone  elapsed {result.elapsed:.2f}s"
        f"  loss {result.final_loss:.4f}"
        f"  tok/s {result.tok_s:,.0f}"
    )


def resolve_generation_block_size(
    model: nn.Module, generation_config: GenerationConfig
) -> tuple[int, str | None]:
    model_block_size = unwrap_model(model).block_size
    generation_block_size = min(generation_config.requested_block_size, model_block_size)
    if generation_config.requested_block_size > model_block_size:
        message = (
            "requested block size "
            f"{generation_config.requested_block_size} exceeds checkpoint/model block size "
            f"{model_block_size}; using {generation_block_size} for generation"
        )
        return generation_block_size, message
    return generation_block_size, None
