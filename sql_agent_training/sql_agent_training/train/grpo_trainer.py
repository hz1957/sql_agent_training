"""Hugging Face Trainer-backed online GRPO for SQL-agent rollouts."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import yaml

from sql_agent_training.agent.model_client import HuggingFaceInMemoryModelClient, ModelClient, VllmOpenAIModelClient
from sql_agent_training.data.spider_dataset import SpiderExample
from sql_agent_training.train.distributed import (
    DistributedContext,
    all_ranks_true,
    barrier,
    broadcast_object,
    init_distributed,
    rank_suffix_path,
    unwrap_distributed_model,
)
from sql_agent_training.train.grpo_batch import GrpoBatch, summarize_grpo_batch
from sql_agent_training.train.grpo_rollouts import (
    RolloutJsonlWriter,
    RolloutSource,
    build_rollout_batch_from_config,
    load_rollout_source_from_config,
)
from sql_agent_training.train.grpo_train import (
    GrpoLossConfig,
    GrpoTrainMetrics,
    GrpoTrainingBatch,
    _apply_peft_adapter_if_configured,
    _loss_config_from_config,
    _new_checkpoint_dir,
    _sample_step_examples,
    _save_policy,
    _sequence_logprobs,
    _sequence_logprobs_microbatched,
    _shard_examples_for_rank,
    _torch_dtype_from_config,
    build_training_tensors,
    compute_group_advantages,
    create_tiny_causal_lm,
)


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency check
        raise RuntimeError("Install the train extra to run GRPO training: uv sync --extra train") from exc
    return torch


def _deepspeed_config(training: dict[str, Any]) -> str | dict[str, Any] | None:
    value = training.get("deepspeed")
    if value is None or value is False:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    if isinstance(value, dict):
        return value
    raise ValueError("training.deepspeed must be a config path, inline config dict, false, or null")


def _save_strategy(training: dict[str, Any]) -> str:
    value = str(training.get("save_strategy", "steps" if int(training.get("save_every_steps", 0) or 0) else "no"))
    normalized = value.strip().lower()
    if normalized not in {"no", "steps", "epoch", "best"}:
        raise ValueError("training.save_strategy must be one of 'no', 'steps', 'epoch', or 'best'")
    return normalized


def _update_epochs(config: dict[str, Any]) -> int:
    epochs = int(config.get("training", {}).get("update_epochs", 1))
    if epochs <= 0:
        raise ValueError("training.update_epochs must be positive")
    return epochs


def _logprob_micro_batch_size(config: dict[str, Any]) -> int:
    size = int(config.get("training", {}).get("logprob_micro_batch_size", 1))
    if size <= 0:
        raise ValueError("training.logprob_micro_batch_size must be positive")
    return size


def _task_batch_size(config: dict[str, Any]) -> int | None:
    training = config.get("training", {})
    rollout = config.get("rollout", {})
    value = training.get("task_batch_size", rollout.get("task_batch_size"))
    if value is None:
        return None
    size = int(value)
    return size if size > 0 else None


def _optimizer_steps(config: dict[str, Any]) -> int:
    training = config.get("training", {})
    rollout_steps = int(training.get("max_steps", 1))
    if rollout_steps <= 0:
        raise ValueError("training.max_steps must be positive")
    return rollout_steps * _update_epochs(config)


def _save_steps(config: dict[str, Any]) -> int:
    training = config.get("training", {})
    if training.get("save_steps") is not None:
        return int(training["save_steps"])
    save_every_rollout_steps = int(training.get("save_every_steps", 0) or 0)
    return max(1, save_every_rollout_steps * _update_epochs(config)) if save_every_rollout_steps > 0 else 500


def _rollout_backend(config: dict[str, Any]) -> str:
    backend = str(config.get("rollout", {}).get("backend", "hf")).strip().lower()
    if backend not in {"hf", "vllm"}:
        raise ValueError("rollout.backend must be one of 'hf' or 'vllm'")
    return backend


def _vllm_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("rollout", {}).get("vllm", {}))


def _should_sync_lora_adapter(*, rollout_step: int, sync_every: int, adapter_loaded: bool) -> bool:
    if sync_every <= 0:
        raise ValueError("rollout.vllm.sync_every_rollout_steps must be positive")
    return not adapter_loaded or rollout_step % sync_every == 0


def _resolve_latest_run(root: str | Path) -> Path:
    candidates = [path for path in Path(root).iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint run directories found under {root}")
    return max(candidates, key=lambda path: path.name)


def _resolve_adapter_path(model_config: dict[str, Any]) -> str | None:
    if model_config.get("adapter_path"):
        return str(model_config["adapter_path"])
    adapter_root = model_config.get("adapter_root")
    if not adapter_root:
        return None
    run_dir = _resolve_latest_run(adapter_root)
    checkpoint_name = model_config.get("adapter_checkpoint")
    if checkpoint_name:
        return str(run_dir / str(checkpoint_name))
    return str(run_dir)


def _save_lora_adapter_for_rollout(model: Any, output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    if not hasattr(model, "peft_config"):
        raise RuntimeError("vLLM rollout sync requires a PEFT LoRA model with save_pretrained support")
    if not hasattr(model, "save_pretrained"):
        raise RuntimeError("vLLM rollout sync requires model.save_pretrained")
    model.save_pretrained(path)
    adapter_config = path / "adapter_config.json"
    if not adapter_config.exists():
        raise RuntimeError(f"LoRA adapter export did not write {adapter_config}")
    return path


def _cleanup_old_lora_exports(root: Path, *, keep_last: int) -> None:
    if keep_last <= 0 or not root.exists():
        return
    candidates = sorted([path for path in root.iterdir() if path.is_dir() and path.name.startswith("step_")])
    for stale_path in candidates[:-keep_last]:
        shutil.rmtree(stale_path)


def _build_training_args(config: dict[str, Any], output_dir: Path) -> Any:
    try:
        from transformers import TrainingArguments
    except ImportError as exc:  # pragma: no cover - optional dependency check
        raise RuntimeError("Install the train extra to use Hugging Face Trainer.") from exc

    training = config.get("training", {})
    gradient_accumulation_steps = int(training.get("gradient_accumulation_steps", 1))
    if gradient_accumulation_steps != 1:
        raise ValueError("HF Trainer GRPO currently requires training.gradient_accumulation_steps: 1")
    if float(training.get("kl_beta", 0.0)) != 0.0:
        raise ValueError("HF Trainer GRPO currently requires training.kl_beta: 0.0")

    torch_dtype = str(config.get("model", {}).get("torch_dtype", "")).lower()
    bf16 = bool(training.get("bf16", torch_dtype in {"bf16", "bfloat16"}))
    fp16 = bool(training.get("fp16", torch_dtype in {"fp16", "float16", "half"}))
    save_strategy = _save_strategy(training)
    return TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        max_steps=_optimizer_steps(config),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=float(training.get("learning_rate", 1e-4)),
        warmup_ratio=float(training.get("warmup_ratio", 0.0)),
        weight_decay=float(training.get("weight_decay", 0.0)),
        max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        logging_strategy="steps",
        logging_steps=int(training.get("logging_steps", 1)),
        save_strategy=save_strategy,
        save_steps=_save_steps(config),
        save_total_limit=int(training["save_total_limit"]) if training.get("save_total_limit") is not None else None,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=bool(training.get("gradient_checkpointing", False)),
        remove_unused_columns=False,
        dataloader_num_workers=int(training.get("dataloader_num_workers", 0)),
        report_to=training.get("report_to", "none"),
        seed=int(training.get("seed", 0)),
        deepspeed=_deepspeed_config(training),
    )


def _build_policy_model_and_tokenizer(config: dict[str, Any]) -> tuple[Any, Any | None, int]:
    model_config = config.get("model", {})
    backend = str(model_config.get("backend", "hf"))
    if backend == "tiny":
        model = create_tiny_causal_lm(
            vocab_size=int(model_config.get("vocab_size", 1024)),
            hidden_size=int(model_config.get("hidden_size", 32)),
        )
        pad_token_id = int(config.get("training", {}).get("pad_token_id", 0))
        return model, None, pad_token_id

    if backend != "hf":
        raise ValueError(f"Unknown GRPO model backend: {backend}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency check
        raise RuntimeError("Install the train extra to use Hugging Face GRPO training.") from exc

    model_path = str(model_config["path"])
    tokenizer_config = config.get("tokenizer", {})
    tokenizer_path = str(model_config.get("tokenizer_path") or tokenizer_config.get("path") or model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_device_hint = "cpu" if str(config.get("training", {}).get("device", "auto")) == "cpu" else "cuda"
    policy_base = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=_torch_dtype_from_config(config, dtype_device_hint),
    )
    adapter_path = _resolve_adapter_path(model_config)
    policy = _apply_peft_adapter_if_configured(policy_base, adapter_path, is_trainable=True)
    if bool(config.get("training", {}).get("gradient_checkpointing", False)) and hasattr(policy.config, "use_cache"):
        policy.config.use_cache = False
    if bool(config.get("training", {}).get("gradient_checkpointing", False)) and hasattr(
        policy, "enable_input_require_grads"
    ):
        policy.enable_input_require_grads()
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    return policy, tokenizer, int(pad_token_id)


class RolloutStepDataset:
    """Sized iterable of placeholder optimizer steps."""

    def __init__(self, length: int) -> None:
        torch = _require_torch()

        class _Dataset(torch.utils.data.IterableDataset):
            def __iter__(self_nonlocal):
                for index in range(length):
                    yield {"step_index": index}

            def __len__(self_nonlocal) -> int:
                return length

        self.dataset = _Dataset()


class AgentGRPOTrainer:
    """Factory wrapper that creates a Transformers Trainer subclass with GRPO state."""

    @staticmethod
    def build(
        *,
        config: dict[str, Any],
        model: Any,
        args: Any,
        tokenizer: Any | None,
        pad_token_id: int,
        source: RolloutSource,
        context: DistributedContext,
        rollout_writer: RolloutJsonlWriter,
        metrics_jsonl: Path,
    ) -> Any:
        try:
            from torch.utils.data import DataLoader
            from transformers import Trainer
        except ImportError as exc:  # pragma: no cover - optional dependency check
            raise RuntimeError("Install the train extra to use Hugging Face Trainer.") from exc

        class _Trainer(Trainer):
            def __init__(self, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                self.grpo_config = config
                self.source = source
                self.context = context
                self.rollout_writer = rollout_writer
                self.metrics_jsonl = metrics_jsonl
                self.hf_tokenizer = tokenizer
                self.pad_token_id = pad_token_id
                self.loss_config: GrpoLossConfig = _loss_config_from_config(config)
                self.logprob_micro_batch_size = _logprob_micro_batch_size(config)
                self.update_epochs = _update_epochs(config)
                self.task_batch_size = _task_batch_size(config)
                self.rollout_backend = _rollout_backend(config)
                self.vllm_config = _vllm_config(config)
                self.vllm_lora_name = str(self.vllm_config.get("lora_name", "current_policy"))
                self.vllm_current_model_name = str(self.vllm_config.get("model", self.vllm_lora_name))
                self.vllm_adapter_loaded = False
                self.rollout_step = 0
                self.cached_prepared: GrpoTrainingBatch | None = None
                self.cached_batch_stats: dict[str, Any] = {}
                self.cached_global_task_batch_size = 0
                self.cached_local_task_batch_size = 0
                self.cached_update_epoch = 0
                self.total_trajectories = 0
                self.rows_written = 0
                self.metrics_history: list[dict[str, Any]] = []
                if self.rollout_backend == "vllm" and self.hf_tokenizer is None:
                    raise ValueError("rollout.backend: vllm requires a Hugging Face tokenizer")

            def get_train_dataloader(self) -> Any:
                return DataLoader(self.train_dataset, batch_size=None)

            def _unwrap_for_generation(self, model_for_step: Any) -> Any:
                if hasattr(self, "accelerator") and self.accelerator is not None:
                    return self.accelerator.unwrap_model(model_for_step)
                return unwrap_distributed_model(model_for_step)

            def _rollout_model_client(self, model_for_step: Any) -> ModelClient | None:
                if self.hf_tokenizer is None:
                    return None
                if self.grpo_config.get("rollout", {}).get("scripted_responses"):
                    return None
                rollout = self.grpo_config.get("rollout", {})
                if self.rollout_backend == "vllm":
                    return VllmOpenAIModelClient(
                        base_url=str(self.vllm_config.get("base_url", "http://127.0.0.1:8000/v1")),
                        model_name=self.vllm_current_model_name,
                        tokenizer=self.hf_tokenizer,
                        api_key=str(self.vllm_config.get("api_key") or os.environ.get("VLLM_API_KEY") or "") or None,
                        timeout_seconds=float(self.vllm_config.get("timeout_seconds", 300.0)),
                        max_new_tokens=int(rollout.get("max_response_length", 256)),
                        temperature=float(rollout.get("temperature", 0.0)),
                        top_p=float(rollout["top_p"]) if rollout.get("top_p") is not None else None,
                        top_k=int(rollout["top_k"]) if rollout.get("top_k") is not None else None,
                    )
                generation_model = self._unwrap_for_generation(model_for_step)
                return HuggingFaceInMemoryModelClient(
                    generation_model,
                    self.hf_tokenizer,
                    device=str(self.args.device),
                    max_new_tokens=int(rollout.get("max_response_length", 256)),
                    temperature=float(rollout.get("temperature", 0.0)),
                    top_p=float(rollout["top_p"]) if rollout.get("top_p") is not None else None,
                    top_k=int(rollout["top_k"]) if rollout.get("top_k") is not None else None,
                )

            def _sync_lora_to_vllm_if_needed(self, model_for_step: Any) -> None:
                if self.rollout_backend != "vllm":
                    return
                if not bool(self.vllm_config.get("sync_lora", True)):
                    self.vllm_current_model_name = str(self.vllm_config.get("model", self.vllm_lora_name))
                    return

                sync_every = int(self.vllm_config.get("sync_every_rollout_steps", 1))
                if not _should_sync_lora_adapter(
                    rollout_step=self.rollout_step,
                    sync_every=sync_every,
                    adapter_loaded=self.vllm_adapter_loaded,
                ):
                    return

                exported_path_value = None
                if self.context.is_main_process:
                    export_root = Path(
                        self.vllm_config.get(
                            "lora_tmp_dir",
                            f"/dev/shm/{os.environ.get('USER', 'user')}/sql_agent_training/vllm_lora",
                        )
                    )
                    export_path = export_root / f"step_{self.rollout_step:06d}"
                    model_to_export = self._unwrap_for_generation(model_for_step)
                    _save_lora_adapter_for_rollout(model_to_export, export_path)
                    client = VllmOpenAIModelClient(
                        base_url=str(self.vllm_config.get("base_url", "http://127.0.0.1:8000/v1")),
                        model_name=self.vllm_lora_name,
                        tokenizer=self.hf_tokenizer,
                        api_key=str(self.vllm_config.get("api_key") or os.environ.get("VLLM_API_KEY") or "") or None,
                        timeout_seconds=float(self.vllm_config.get("timeout_seconds", 300.0)),
                    )
                    client.load_lora_adapter(
                        lora_name=self.vllm_lora_name,
                        lora_path=export_path,
                        load_inplace=bool(self.vllm_config.get("load_inplace", True)),
                    )
                    _cleanup_old_lora_exports(
                        export_root,
                        keep_last=int(self.vllm_config.get("keep_last_adapters", 3)),
                    )
                    exported_path_value = str(export_path)

                broadcast_object(exported_path_value, self.context)
                barrier(self.context)
                self.vllm_current_model_name = self.vllm_lora_name
                self.vllm_adapter_loaded = True

            def _sample_examples_for_rollout_step(self) -> tuple[list[SpiderExample], list[SpiderExample]]:
                seed = int(self.grpo_config.get("training", {}).get("seed", 0))
                rng = random.Random(seed + self.rollout_step)
                global_examples = _sample_step_examples(
                    self.source.examples,
                    task_batch_size=self.task_batch_size,
                    rng=rng,
                )
                local_examples = _shard_examples_for_rank(global_examples, self.context)
                if not local_examples:
                    raise ValueError("Each distributed rank must receive at least one task")
                return global_examples, local_examples

            def _prepare_next_rollout_batch(self, model_for_step: Any) -> None:
                self.rollout_step += 1
                self._sync_lora_to_vllm_if_needed(model_for_step)
                global_examples, local_examples = self._sample_examples_for_rollout_step()
                batch = build_rollout_batch_from_config(
                    self.grpo_config,
                    rollout_writer=self.rollout_writer,
                    model_client=self._rollout_model_client(model_for_step),
                    hf_tokenizer=self.hf_tokenizer,
                    examples=local_examples,
                    source=self.source,
                )
                if not all_ranks_true(bool(batch.groups), self.context):
                    raise ValueError("GRPO rollout produced an empty batch on at least one rank")
                self.cached_batch_stats = summarize_grpo_batch(batch)
                self.cached_prepared = self._prepare_training_batch(model_for_step, batch)
                self.cached_global_task_batch_size = len(global_examples)
                self.cached_local_task_batch_size = len(local_examples)
                self.cached_update_epoch = 0
                self.total_trajectories += batch.num_trajectories
                self.rows_written = self.rollout_writer.count

            def _prepare_training_batch(self, model_for_step: Any, batch: GrpoBatch) -> GrpoTrainingBatch:
                torch = _require_torch()
                advantages = compute_group_advantages(
                    batch,
                    normalize=self.loss_config.normalize_advantages,
                    epsilon=self.loss_config.advantage_epsilon,
                )
                tensors = build_training_tensors(
                    batch,
                    advantages=advantages,
                    pad_token_id=self.pad_token_id,
                    device=str(self.args.device),
                )
                model_for_step.eval()
                with torch.no_grad():
                    old_logprobs = _sequence_logprobs_microbatched(
                        model_for_step,
                        tensors["input_ids"],
                        tensors["attention_mask"],
                        micro_batch_size=self.logprob_micro_batch_size,
                    ).detach()

                return GrpoTrainingBatch(
                    input_ids=tensors["input_ids"],
                    attention_mask=tensors["attention_mask"],
                    response_mask=tensors["response_mask"],
                    advantages=tensors["advantages"],
                    rewards=tensors["rewards"],
                    old_logprobs=old_logprobs,
                    reference_logprobs=None,
                    rollout_ids=tensors["rollout_ids"],
                )

            def _accelerator_backward(self, loss: Any) -> None:
                kwargs: dict[str, Any] = {}
                distributed_type = getattr(getattr(self, "accelerator", None), "distributed_type", None)
                if getattr(distributed_type, "name", "") == "DEEPSPEED":
                    kwargs["scale_wrt_gas"] = False
                self.accelerator.backward(loss, **kwargs)

            def _backward_prepared_batch(
                self,
                model_for_step: Any,
                batch: GrpoTrainingBatch,
            ) -> tuple[Any, GrpoTrainMetrics]:
                torch = _require_torch()
                model_for_step.train()
                total_tokens = batch.response_mask.sum().clamp_min(1.0)
                batch_size = int(batch.input_ids.shape[0])
                micro_batch_size = self.logprob_micro_batch_size
                starts = list(range(0, batch_size, micro_batch_size if micro_batch_size > 0 else batch_size))
                metric_sums = {
                    "policy_loss": 0.0,
                    "kl": 0.0,
                    "clip_fraction": 0.0,
                    "ratio": 0.0,
                    "policy_approx_kl": 0.0,
                }
                ratio_min: float | None = None
                ratio_max: float | None = None
                trainable_tokens = int(batch.response_mask.sum().detach().cpu().item())
                detached_loss = torch.zeros((), dtype=torch.float32, device=self.args.device)

                for micro_index, start in enumerate(starts):
                    stop = start + (micro_batch_size if micro_batch_size > 0 else batch_size)
                    sync_context = (
                        nullcontext()
                        if micro_index == len(starts) - 1 or not hasattr(model_for_step, "no_sync")
                        else model_for_step.no_sync()
                    )
                    with sync_context:
                        with self.compute_loss_context_manager():
                            new_logprobs = _sequence_logprobs(
                                model_for_step,
                                batch.input_ids[start:stop],
                                batch.attention_mask[start:stop],
                            )
                            old_logprobs = batch.old_logprobs[start:stop]
                            response_mask = batch.response_mask[start:stop]
                            token_advantages = batch.advantages[start:stop].unsqueeze(-1)
                            log_ratio = new_logprobs - old_logprobs
                            ratio = torch.exp(log_ratio)
                            unclipped = ratio * token_advantages
                            clipped_ratio = torch.clamp(
                                ratio,
                                1.0 - self.loss_config.clip_epsilon,
                                1.0 + self.loss_config.clip_epsilon,
                            )
                            clipped = clipped_ratio * token_advantages
                            policy_loss_tokens = -torch.minimum(unclipped, clipped)
                            policy_kl_tokens = (ratio - 1.0) - log_ratio
                            kl_tokens = torch.zeros_like(policy_loss_tokens)
                            micro_policy_loss = (policy_loss_tokens * response_mask).sum() / total_tokens
                            micro_kl_loss = self.loss_config.kl_beta * (kl_tokens * response_mask).sum() / total_tokens
                            micro_loss = micro_policy_loss + micro_kl_loss

                        self._accelerator_backward(micro_loss)
                        detached_loss = detached_loss + micro_loss.detach()

                        with torch.no_grad():
                            response_ratios = ratio[response_mask.bool()]
                            metric_sums["policy_loss"] += float(
                                (policy_loss_tokens * response_mask).sum().detach().cpu()
                            )
                            metric_sums["kl"] += float((kl_tokens * response_mask).sum().detach().cpu())
                            clip_mask = ((ratio - 1.0).abs() > self.loss_config.clip_epsilon).float()
                            metric_sums["clip_fraction"] += float((clip_mask * response_mask).sum().detach().cpu())
                            metric_sums["policy_approx_kl"] += float(
                                (policy_kl_tokens * response_mask).sum().detach().cpu()
                            )
                            if response_ratios.numel() > 0:
                                metric_sums["ratio"] += float(response_ratios.sum().detach().cpu())
                                current_min = float(response_ratios.min().detach().cpu())
                                current_max = float(response_ratios.max().detach().cpu())
                                ratio_min = current_min if ratio_min is None else min(ratio_min, current_min)
                                ratio_max = current_max if ratio_max is None else max(ratio_max, current_max)

                denominator = float(max(trainable_tokens, 1))
                policy_loss_value = metric_sums["policy_loss"] / denominator
                approx_kl_value = metric_sums["kl"] / denominator
                kl_loss_value = self.loss_config.kl_beta * approx_kl_value
                metrics = GrpoTrainMetrics(
                    loss=policy_loss_value + kl_loss_value,
                    policy_loss=policy_loss_value,
                    kl_loss=kl_loss_value,
                    approx_kl=approx_kl_value,
                    clip_fraction=metric_sums["clip_fraction"] / denominator,
                    mean_reward=float(batch.rewards.mean().detach().cpu()),
                    mean_advantage=float(batch.advantages.mean().detach().cpu()),
                    ratio_mean=metric_sums["ratio"] / denominator if trainable_tokens else 0.0,
                    ratio_min=ratio_min if ratio_min is not None else 0.0,
                    ratio_max=ratio_max if ratio_max is not None else 0.0,
                    policy_approx_kl=metric_sums["policy_approx_kl"] / denominator,
                    trainable_tokens=trainable_tokens,
                )
                return detached_loss, metrics

            def training_step(self, model: Any, inputs: dict[str, Any], num_items_in_batch: Any = None) -> Any:
                del inputs, num_items_in_batch
                if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                    self.optimizer.train()
                if self.cached_prepared is None or self.cached_update_epoch >= self.update_epochs:
                    self._prepare_next_rollout_batch(model)
                assert self.cached_prepared is not None
                self.cached_update_epoch += 1
                loss, metrics = self._backward_prepared_batch(model, self.cached_prepared)
                metric_row = {
                    "step": self.rollout_step,
                    "update_epoch": self.cached_update_epoch,
                    "optimizer_step": self.state.global_step + 1,
                    "rank": self.context.rank,
                    "world_size": self.context.world_size,
                    "global_task_batch_size": self.cached_global_task_batch_size,
                    "local_task_batch_size": self.cached_local_task_batch_size,
                    **self.cached_batch_stats,
                    **metrics.__dict__,
                }
                self.metrics_history.append(metric_row)
                if self.is_world_process_zero():
                    self.log(
                        {
                            "grpo_loss": metrics.loss,
                            "grpo_policy_loss": metrics.policy_loss,
                            "grpo_mean_reward": metrics.mean_reward,
                            "grpo_policy_approx_kl": metrics.policy_approx_kl,
                        }
                    )
                _write_jsonl_row(self.metrics_jsonl, metric_row)
                if self.cached_update_epoch >= self.update_epochs:
                    self.cached_prepared = None
                    self.cached_batch_stats = {}
                return loss.detach()

            def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
                del _internal_call
                target_dir = output_dir or self.args.output_dir
                if self.is_world_process_zero():
                    model_to_save = self.accelerator.unwrap_model(self.model) if self.accelerator else self.model
                    _save_policy(model_to_save, self.hf_tokenizer, self.grpo_config, target_dir)
                if self.args.should_save:
                    self.state.save_to_json(os.path.join(target_dir, "trainer_state.json"))

        dataset = RolloutStepDataset(_optimizer_steps(config)).dataset
        return _Trainer(model=model, args=args, train_dataset=dataset)


def _write_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _write_run_config(config: dict[str, Any], checkpoint_dir: Path, *, is_main_process: bool) -> Path:
    path = checkpoint_dir / "run_config.yaml"
    if is_main_process:
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def train_grpo_with_hf_trainer_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Run online SQL-agent GRPO through Hugging Face Trainer."""

    torch = _require_torch()
    seed = int(config.get("training", {}).get("seed", 0))
    torch.manual_seed(seed)
    context = init_distributed(str(config.get("training", {}).get("device", "auto")))
    checkpoint_dir_value = (
        str(_new_checkpoint_dir(config.get("output", {}).get("checkpoint_dir", "artifacts/checkpoints/grpo_trainer")))
        if context.is_main_process
        else None
    )
    checkpoint_dir = Path(broadcast_object(checkpoint_dir_value, context))
    if context.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
    barrier(context)
    run_config_path = _write_run_config(config, checkpoint_dir, is_main_process=context.is_main_process)
    output = config.get("output", {})
    metrics_jsonl = rank_suffix_path(output.get("metrics_jsonl", checkpoint_dir / "metrics.jsonl"), context)
    rollouts_jsonl = rank_suffix_path(output.get("rollouts_jsonl", checkpoint_dir / "rollouts.jsonl"), context)
    for path in (metrics_jsonl, rollouts_jsonl):
        if path.exists():
            path.unlink()

    source = load_rollout_source_from_config(config)
    try:
        args = _build_training_args(config, checkpoint_dir)
        model, tokenizer, pad_token_id = _build_policy_model_and_tokenizer(config)
        with RolloutJsonlWriter(rollouts_jsonl, include_text=bool(output.get("include_text", True))) as rollout_writer:
            trainer = AgentGRPOTrainer.build(
                config=config,
                model=model,
                args=args,
                tokenizer=tokenizer,
                pad_token_id=pad_token_id,
                source=source,
                context=context,
                rollout_writer=rollout_writer,
                metrics_jsonl=metrics_jsonl,
            )
            trainer.train()
            trainer.save_model(str(checkpoint_dir))
            metrics_history = list(trainer.metrics_history)
            rows_written = trainer.rows_written
            total_trajectories = trainer.total_trajectories
    finally:
        source.close()

    metrics_path = rank_suffix_path(output.get("metrics_json", checkpoint_dir / "metrics.json"), context)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_history, indent=2), encoding="utf-8")
    final_metrics = metrics_history[-1] if metrics_history else {}
    return {
        **final_metrics,
        "steps": int(config.get("training", {}).get("max_steps", 1)),
        "update_epochs": _update_epochs(config),
        "optimizer_steps": _optimizer_steps(config),
        "rank": context.rank,
        "world_size": context.world_size,
        "rows_written": rows_written,
        "trajectories": total_trajectories,
        "checkpoint_dir": str(checkpoint_dir),
        "rollouts_jsonl": str(rollouts_jsonl),
        "metrics_json": str(metrics_path),
        "metrics_jsonl": str(metrics_jsonl),
        "run_config": str(run_config_path),
    }


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return loaded or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HF Trainer-backed online SQL-agent GRPO.")
    parser.add_argument("--config", default="configs/grpo.local_dryrun.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Force built-in demo rollouts.")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    config = _load_config(args.config)
    if args.dry_run:
        config["dry_run"] = True
    summary = train_grpo_with_hf_trainer_from_config(config)
    if int(summary.get("rank", 0)) == 0:
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
