"""Ray-based asynchronous GRPO pipeline with separately placed worker modules.

Architecture
------------
- `RolloutActor`: vLLM inference plus SQLite-backed SQL-agent rollout/reward.
- `ReferenceLogprobWorker`: frozen reference model forward pass.
- `LearnerWorker`: policy HF model training only.
  Receives the batch and pre-computed reference log-probs, computes old
  log-probabilities, runs GRPO gradient updates, and returns updated LoRA
  weights plus step metrics.
- `train_grpo_ray` (CPU driver): coordinates the three workers in strict
  on-policy order; each rollout uses the latest weights before training begins.

The three modules are independent in code. Deployment decides whether rollout
and reference share one physical GPU (2-GPU runs) or use separate GPUs
(3-GPU runs).

Per-step timeline (on-policy, rollout is the bottleneck)::

    RolloutActor:      [vLLM rollout] -> GrpoBatch -----------------------> reload LoRA
    ReferenceWorker:                  [reference logprobs] ---------------->
    LearnerWorker:                                        [GRPO update] -> LoRA weights

Usage::

    python -m sql_agent_training.train.grpo_train \\
        --config configs/grpo.ray_14b_lora.yaml --ray
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import time
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses shared between workers
# ---------------------------------------------------------------------------


@dataclass
class LearnerStepResult:
    """Return value from one `LearnerWorker.train_step` call."""

    metrics: dict[str, Any]
    """Scalar metrics from `GrpoTrainMetrics.__dict__`."""

    lora_state_dict: dict[str, Any]
    """LoRA-only CPU state dict (small, suitable for Ray Object Store transfer)."""

    optimizer_step: int
    """Cumulative number of optimizer steps taken so far."""

    batch_stats: dict[str, Any] = field(default_factory=dict)
    """Diagnostic statistics from `summarize_grpo_batch`."""


# ---------------------------------------------------------------------------
# Lazy imports helpers
# ---------------------------------------------------------------------------


def _require_ray() -> Any:
    try:
        import ray
    except ImportError as exc:
        raise RuntimeError("Install ray to use the Ray GRPO pipeline: uv sync --extra train") from exc
    return ray


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install torch to run GRPO training.") from exc
    return torch


def _normalize_cuda_device(requested_device: str) -> str:
    """Map legacy physical cuda:N config to cuda:0 inside single-GPU Ray actors."""

    if not requested_device.startswith("cuda:"):
        return requested_device
    visible_devices = [item for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item]
    if len(visible_devices) == 1:
        return "cuda:0"
    return requested_device


# ---------------------------------------------------------------------------
# RolloutActor
# ---------------------------------------------------------------------------


def _role_num_gpus(config: dict[str, Any], role: str) -> float:
    """Return Ray GPU resource demand for one worker role."""

    ray_cfg = config.get("ray", {})
    colocate_reference = bool(ray_cfg.get("reference_colocate_with_rollout", True))
    defaults = {
        "rollout": 0.5 if colocate_reference else 1.0,
        "reference": 0.5 if colocate_reference else 1.0,
        "learner": 1.0,
    }
    if role not in defaults:
        raise ValueError(f"unknown Ray GRPO role: {role}")
    value = ray_cfg.get(f"{role}_num_gpus", defaults[role])
    num_gpus = float(value)
    if num_gpus < 0:
        raise ValueError(f"ray.{role}_num_gpus must be non-negative")
    return num_gpus


class RolloutActor:
    """Runs SQL-agent rollouts through vLLM.

    Initialised once per training run by the coordinator.  Internally it:

    1. Starts a vLLM server subprocess on the assigned GPU.
    2. Builds scored SQL-agent trajectories.
    3. Exposes `rollout` and `reload_lora` for the coordinator.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        import torch
        from transformers import AutoTokenizer

        self._config = config
        model_cfg = config.get("model", {})
        ray_cfg = config.get("ray", {})
        self._device = _normalize_cuda_device(str(ray_cfg.get("rollout_device", ray_cfg.get("actor_device", "cuda:0"))))
        self._vllm_port = int(ray_cfg.get("vllm_port", 8100))
        self._vllm_model_name = str(model_cfg.get("path", ""))
        self._adapter_path: str | None = str(model_cfg["adapter_path"]) if model_cfg.get("adapter_path") else None

        torch_dtype_str = str(model_cfg.get("torch_dtype", "bf16")).lower()
        if torch_dtype_str in {"bf16", "bfloat16"}:
            self._torch_dtype: Any = torch.bfloat16
        elif torch_dtype_str in {"fp16", "float16", "half"}:
            self._torch_dtype = torch.float16
        else:
            self._torch_dtype = None

        rollout_cfg = config.get("rollout", {})
        self._max_new_tokens = int(rollout_cfg.get("max_response_length", 2048))
        self._max_model_len = int(
            rollout_cfg.get("max_model_len", int(rollout_cfg.get("max_prompt_length", 4096)) + self._max_new_tokens)
        )
        self._temperature = float(rollout_cfg.get("temperature", 0.8))
        self._top_p: float | None = float(rollout_cfg["top_p"]) if rollout_cfg.get("top_p") is not None else None
        colocate_reference = bool(ray_cfg.get("reference_colocate_with_rollout", True))
        self._vllm_gpu_memory_utilization = float(
            ray_cfg.get("vllm_gpu_memory_utilization", 0.45 if colocate_reference else 0.85)
        )

        tokenizer_path = (
            model_cfg.get("tokenizer_path") or config.get("tokenizer", {}).get("path") or self._vllm_model_name
        )
        logger.info("RolloutActor: loading tokenizer from %s", tokenizer_path)
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Start vLLM subprocess
        self._vllm_proc: subprocess.Popen[str] | None = None
        self._start_vllm()

        # Build rollout source
        from sql_agent_training.train.grpo_rollouts import load_rollout_source_from_config

        self._source = load_rollout_source_from_config(config)
        logger.info("RolloutActor: rollout source loaded (%d examples)", len(self._source.examples))

    # ------------------------------------------------------------------
    # vLLM lifecycle
    # ------------------------------------------------------------------

    def _vllm_base_url(self) -> str:
        return f"http://127.0.0.1:{self._vllm_port}/v1"

    def _start_vllm(self) -> None:
        """Launch vLLM OpenAI-compatible server as a subprocess."""
        import torch

        cmd = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self._vllm_model_name,
            "--port",
            str(self._vllm_port),
            "--dtype",
            "bfloat16" if self._torch_dtype == torch.bfloat16 else "float16",
            "--gpu-memory-utilization",
            str(self._vllm_gpu_memory_utilization),
            "--max-model-len",
            str(self._max_model_len),
            "--trust-remote-code",
            "--enable-lora",
        ]
        if self._adapter_path:
            cmd += ["--lora-modules", f"policy={self._adapter_path}"]
        logger.info("RolloutActor: starting vLLM: %s", " ".join(cmd))
        self._vllm_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._wait_for_vllm_ready()

    def _wait_for_vllm_ready(self, timeout: float = 300.0, poll_interval: float = 3.0) -> None:
        """Block until vLLM /health endpoint responds or timeout elapses."""
        import urllib.error
        import urllib.request

        health_url = f"http://127.0.0.1:{self._vllm_port}/health"
        deadline = time.time() + timeout
        logger.info("RolloutActor: waiting for vLLM to be ready at %s", health_url)
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(health_url, timeout=2.0):
                    logger.info("RolloutActor: vLLM is ready")
                    return
            except (urllib.error.URLError, OSError):
                time.sleep(poll_interval)
        raise RuntimeError(f"vLLM did not become ready within {timeout}s")

    def _build_vllm_client(self, lora_name: str | None = None) -> Any:
        from sql_agent_training.agent.model_client import VllmOpenAIModelClient

        model_name = f"{lora_name}" if lora_name else self._vllm_model_name
        return VllmOpenAIModelClient(
            base_url=self._vllm_base_url(),
            model_name=model_name,
            tokenizer=self._tokenizer,
            max_new_tokens=self._max_new_tokens,
            temperature=self._temperature,
            top_p=self._top_p,
        )

    # ------------------------------------------------------------------
    # Public remote methods
    # ------------------------------------------------------------------

    def rollout(
        self,
        step_examples: list[Any],
        *,
        rollout_jsonl_path: str | None = None,
        include_text: bool = True,
    ) -> Any:
        """Run vLLM rollouts and return a scored GRPO batch.

        Args:
            step_examples: `SpiderExample` list for this training step.
            rollout_jsonl_path: Optional path to append rollout JSONL rows.
            include_text: Whether to include prompt/response text in JSONL.

        Returns:
            `GrpoBatch` containing scored tokenized trajectories.
        """
        from sql_agent_training.train.grpo_rollouts import (
            RolloutJsonlWriter,
            _build_batch_from_examples,
        )
        from sql_agent_training.agent.tokenization import ExistingHuggingFaceTokenizer

        text_tokenizer = ExistingHuggingFaceTokenizer(self._tokenizer)
        lora_name = "policy" if self._adapter_path else None
        vllm_client = self._build_vllm_client(lora_name=lora_name)

        writer: RolloutJsonlWriter | None = None
        if rollout_jsonl_path:
            writer = RolloutJsonlWriter(rollout_jsonl_path, include_text=include_text)
            writer.__enter__()

        try:
            batch = _build_batch_from_examples(
                examples=step_examples,
                schema_prompts=self._source.schema_prompts,
                sqlite_paths=self._source.sqlite_paths,
                config=self._config,
                rollout_writer=writer,
                model_client=vllm_client,
                tokenizer=text_tokenizer,
            )
        finally:
            if writer is not None:
                writer.__exit__(None, None, None)

        return batch

    def reload_lora(self, lora_state_dict: dict[str, Any], *, lora_name: str = "policy") -> None:
        """Hot-reload updated LoRA weights into the vLLM server.

        Args:
            lora_state_dict: LoRA adapter state dict from `LearnerWorker`.
            lora_name: The named LoRA slot registered in vLLM.
        """
        from sql_agent_training.agent.model_client import VllmOpenAIModelClient
        import torch

        # Write state dict to a temporary directory vLLM can read from
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter_dir = Path(tmp_dir) / lora_name
            adapter_dir.mkdir()
            # Copy adapter config from original path so vLLM can parse it
            if self._adapter_path:
                import shutil

                for fname in ("adapter_config.json",):
                    src = Path(self._adapter_path) / fname
                    if src.exists():
                        shutil.copy2(src, adapter_dir / fname)
            # Save state dict
            torch.save(lora_state_dict, adapter_dir / "adapter_model.bin")

            client = VllmOpenAIModelClient(
                base_url=self._vllm_base_url(),
                model_name=self._vllm_model_name,
            )
            client.load_lora_adapter(lora_name=lora_name, lora_path=str(adapter_dir))
            logger.info("RolloutActor: vLLM LoRA weights reloaded from tmp dir")

    def close(self) -> None:
        """Terminate the vLLM subprocess and release resources."""
        self._source.close()
        if self._vllm_proc is not None:
            logger.info("RolloutActor: terminating vLLM subprocess")
            self._vllm_proc.terminate()
            try:
                self._vllm_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._vllm_proc.kill()
            self._vllm_proc = None


# ---------------------------------------------------------------------------
# ReferenceLogprobWorker
# ---------------------------------------------------------------------------


class ReferenceLogprobWorker:
    """Runs frozen reference-model log-probability computation."""

    def __init__(self, config: dict[str, Any]) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        self._config = config
        model_cfg = config.get("model", {})
        ray_cfg = config.get("ray", {})
        self._device = _normalize_cuda_device(
            str(ray_cfg.get("reference_device", ray_cfg.get("actor_device", "cuda:0")))
        )

        torch_dtype_str = str(model_cfg.get("torch_dtype", "bf16")).lower()
        if torch_dtype_str in {"bf16", "bfloat16"}:
            torch_dtype: Any = torch.bfloat16
        elif torch_dtype_str in {"fp16", "float16", "half"}:
            torch_dtype = torch.float16
        else:
            torch_dtype = None

        reference_8bit = bool(model_cfg.get("reference_load_in_8bit", False))
        reference_4bit = bool(model_cfg.get("reference_load_in_4bit", False))
        model_path = str(model_cfg["path"])
        ref_path = str(model_cfg.get("reference_path") or model_path)
        adapter_path = str(model_cfg["adapter_path"]) if model_cfg.get("adapter_path") else ""
        ref_adapter = str(model_cfg.get("reference_adapter_path") or adapter_path)
        logger.info("ReferenceLogprobWorker: loading reference model from %s", ref_path)

        load_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if reference_8bit:
            load_kwargs["load_in_8bit"] = True
        elif reference_4bit:
            load_kwargs["load_in_4bit"] = True
        elif torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype

        ref_base = AutoModelForCausalLM.from_pretrained(ref_path, **load_kwargs)
        if ref_adapter:
            try:
                from peft import PeftModel

                ref_base = PeftModel.from_pretrained(ref_base, ref_adapter, is_trainable=False)
            except ImportError as exc:
                raise RuntimeError("Install peft to load LoRA adapter on reference model.") from exc
        if not reference_8bit and not reference_4bit:
            ref_base = ref_base.to(self._device)
        self._reference_model = ref_base
        self._reference_model.eval()
        for param in self._reference_model.parameters():
            param.requires_grad_(False)
        logger.info("ReferenceLogprobWorker: initialised on device %s", self._device)

    def compute_ref_logprobs(self, batch: Any) -> dict[str, list[float]]:
        """Compute per-token reference log-probabilities for all trajectories."""
        import torch

        ref_logprobs: dict[str, list[float]] = {}
        self._reference_model.eval()

        with torch.no_grad():
            for trajectory in batch.trajectories:
                sequence = trajectory.prompt_ids + trajectory.response_ids
                input_ids = torch.tensor([sequence], dtype=torch.long, device=self._device)
                attention_mask = torch.ones_like(input_ids)
                outputs = self._reference_model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, :-1, :]  # (1, seq-1, vocab)
                labels = input_ids[:, 1:]  # (1, seq-1)
                target_logits = logits.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
                log_probs = target_logits - torch.logsumexp(logits, dim=-1)
                ref_logprobs[trajectory.rollout_id] = log_probs[0].cpu().tolist()

        return ref_logprobs

    def close(self) -> None:
        """Release GPU cache held by the reference worker."""
        try:
            import torch

            del self._reference_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.debug("ReferenceLogprobWorker: close cleanup skipped", exc_info=True)


# ---------------------------------------------------------------------------
# LearnerWorker
# ---------------------------------------------------------------------------


class LearnerWorker:
    """Runs GRPO gradient updates on the learner GPU.

    Receives `GrpoBatch` from `RolloutActor` plus `ref_logprobs` from
    `ReferenceLogprobWorker`, computes old log-probs from the current policy,
    performs clipped GRPO updates, and returns the updated LoRA adapter state
    dict plus step metrics.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._config = config
        model_cfg = config.get("model", {})
        ray_cfg = config.get("ray", {})
        self._device = _normalize_cuda_device(str(ray_cfg.get("learner_device", "cuda:0")))
        self._update_epochs = int(config.get("training", {}).get("update_epochs", 1))

        torch_dtype_str = str(model_cfg.get("torch_dtype", "bf16")).lower()
        if torch_dtype_str in {"bf16", "bfloat16"}:
            torch_dtype: Any = torch.bfloat16
        elif torch_dtype_str in {"fp16", "float16", "half"}:
            torch_dtype = torch.float16
        else:
            torch_dtype = None

        model_path = str(model_cfg["path"])
        adapter_path: str | None = str(model_cfg["adapter_path"]) if model_cfg.get("adapter_path") else None
        tokenizer_path = model_cfg.get("tokenizer_path") or config.get("tokenizer", {}).get("path") or model_path

        logger.info("LearnerWorker: loading tokenizer from %s", tokenizer_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        self._pad_token_id = int(tokenizer.pad_token_id or tokenizer.eos_token_id or 0)

        load_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        logger.info("LearnerWorker: loading policy base model from %s", model_path)
        policy_base = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        if adapter_path:
            try:
                from peft import PeftModel

                policy_base = PeftModel.from_pretrained(policy_base, adapter_path, is_trainable=True)
                logger.info("LearnerWorker: LoRA adapter attached from %s", adapter_path)
            except ImportError as exc:
                raise RuntimeError("Install peft for LoRA GRPO training.") from exc
        self._policy_model = policy_base.to(self._device)

        training_cfg = config.get("training", {})
        from sql_agent_training.train.grpo_train import GrpoLossConfig, GrpoTrainer

        loss_config = GrpoLossConfig(
            clip_epsilon=float(training_cfg.get("clip_epsilon", 0.2)),
            kl_beta=float(training_cfg.get("kl_beta", 0.02)),
            advantage_epsilon=float(training_cfg.get("advantage_epsilon", 1e-6)),
            normalize_advantages=bool(training_cfg.get("normalize_advantages", True)),
            max_grad_norm=(
                float(training_cfg["max_grad_norm"]) if training_cfg.get("max_grad_norm") is not None else None
            ),
        )
        logprob_micro_batch_size = int(training_cfg.get("logprob_micro_batch_size", 1))
        optimizer = torch.optim.AdamW(
            self._policy_model.parameters(),
            lr=float(training_cfg.get("learning_rate", 1e-4)),
        )
        # Pass a dummy reference (unused: we receive ref_logprobs externally)
        self._trainer = GrpoTrainer(
            self._policy_model,
            self._policy_model,  # placeholder; ref logprobs come from ReferenceLogprobWorker
            optimizer,
            pad_token_id=self._pad_token_id,
            loss_config=loss_config,
            device=self._device,
            logprob_micro_batch_size=logprob_micro_batch_size,
        )
        self._optimizer_steps = 0
        logger.info("LearnerWorker: initialised on device %s", self._device)

    def train_step(
        self,
        grpo_batch: Any,
        ref_logprobs: dict[str, list[float]],
    ) -> LearnerStepResult:
        """Run one GRPO training step.

        Args:
            grpo_batch: `GrpoBatch` from `RolloutActor.rollout`.
            ref_logprobs: Per-token reference log-probs keyed by `rollout_id`.

        Returns:
            `LearnerStepResult` with metrics, LoRA state dict, and batch stats.
        """
        import torch
        from sql_agent_training.train.grpo_train import (
            compute_group_advantages,
            build_training_tensors,
            GrpoTrainingBatch,
            _sequence_logprobs_microbatched,
        )
        from sql_agent_training.train.grpo_batch import summarize_grpo_batch

        batch_stats = summarize_grpo_batch(grpo_batch)
        advantages = compute_group_advantages(
            grpo_batch,
            normalize=self._trainer.loss_config.normalize_advantages,
            epsilon=self._trainer.loss_config.advantage_epsilon,
        )
        tensors = build_training_tensors(
            grpo_batch,
            advantages=advantages,
            pad_token_id=self._pad_token_id,
            device=self._device,
        )

        # Build reference logprobs tensor aligned to shifted labels
        reference_logprobs_tensor = self._build_ref_logprobs_tensor(
            grpo_batch,
            ref_logprobs,
            tensors["input_ids"],
            tensors["response_mask"],
        )

        # Compute old log-probs from the current (pre-update) policy
        self._policy_model.eval()
        with torch.no_grad():
            old_logprobs = _sequence_logprobs_microbatched(
                self._policy_model,
                tensors["input_ids"],
                tensors["attention_mask"],
                micro_batch_size=self._trainer.logprob_micro_batch_size,
            ).detach()

        prepared = GrpoTrainingBatch(
            input_ids=tensors["input_ids"],
            attention_mask=tensors["attention_mask"],
            response_mask=tensors["response_mask"],
            advantages=tensors["advantages"],
            rewards=tensors["rewards"],
            old_logprobs=old_logprobs,
            reference_logprobs=reference_logprobs_tensor,
            rollout_ids=tensors["rollout_ids"],
        )

        last_metrics = None
        for _ in range(self._update_epochs):
            self._optimizer_steps += 1
            last_metrics = self._trainer.train_prepared_batch(prepared)

        assert last_metrics is not None
        lora_state_dict = self._extract_lora_state_dict()
        logger.info(
            "LearnerWorker: step complete: loss=%.4f reward=%.4f optimizer_step=%d",
            last_metrics.loss,
            last_metrics.mean_reward,
            self._optimizer_steps,
        )
        return LearnerStepResult(
            metrics=last_metrics.__dict__,
            lora_state_dict=lora_state_dict,
            optimizer_step=self._optimizer_steps,
            batch_stats=batch_stats,
        )

    def _build_ref_logprobs_tensor(
        self,
        batch: Any,
        ref_logprobs: dict[str, list[float]],
        input_ids: Any,
        response_mask: Any,
    ) -> Any:
        """Reconstruct reference log-prob tensor aligned to shifted label positions.

        The ReferenceLogprobWorker returns per-token logprobs over the full sequence
        (shifted by one).  Here we slice out the positions corresponding to
        each trajectory's response tokens and pad to match `input_ids`.

        Args:
            batch: `GrpoBatch` with trajectory lengths.
            ref_logprobs: Raw per-token logprobs from ReferenceLogprobWorker.
            input_ids: Padded input tensor `(B, L)`.
            response_mask: Shifted response mask `(B, L-1)`.

        Returns:
            Reference logprobs tensor `(B, L-1)` on `self._device`.
        """
        import torch

        trajectories = batch.trajectories
        seq_len = int(input_ids.shape[1])
        rows: list[list[float]] = []
        for trajectory in trajectories:
            full_logprobs = ref_logprobs.get(trajectory.rollout_id, [])
            # full_logprobs covers positions [0..len(prompt)+len(response)-2]
            # We pad/truncate to seq_len - 1 (shifted positions)
            target_len = seq_len - 1
            if len(full_logprobs) >= target_len:
                row = full_logprobs[:target_len]
            else:
                row = full_logprobs + [0.0] * (target_len - len(full_logprobs))
            rows.append(row)
        return torch.tensor(rows, dtype=torch.float32, device=self._device)

    def _extract_lora_state_dict(self) -> dict[str, Any]:
        """Extract LoRA adapter parameters as a CPU state dict.

        Returns:
            Dict mapping parameter names to CPU tensors.  Falls back to the
            full model state dict when PEFT is not in use.
        """
        import torch

        # Try PEFT get_adapter_state_dict (PEFT >= 0.7)
        try:
            from peft import get_peft_model_state_dict

            state = get_peft_model_state_dict(self._policy_model)
            return {k: v.detach().cpu() for k, v in state.items()}
        except (ImportError, AttributeError):
            pass

        # Fallback: filter LoRA keys by name convention
        lora_keys = {k for k in self._policy_model.state_dict() if "lora_" in k}
        if lora_keys:
            full = self._policy_model.state_dict()
            return {k: v.detach().cpu() for k, v in full.items() if k in lora_keys}

        # Last resort: full state dict (may be large for non-LoRA models)
        logger.warning("LearnerWorker: could not isolate LoRA params; returning full state dict")
        return {k: v.detach().cpu() for k, v in self._policy_model.state_dict().items()}


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


def _new_checkpoint_dir(base_dir: str | Path) -> Path:
    root = Path(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root / timestamp
    counter = 1
    while candidate.exists():
        candidate = root / f"{timestamp}_{counter:02d}"
        counter += 1
    return candidate


def _write_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _sample_step_examples(
    examples: list[Any],
    *,
    task_batch_size: int | None,
    rng: random.Random,
) -> list[Any]:
    if not examples:
        raise ValueError("training examples must be non-empty")
    if task_batch_size is None or task_batch_size >= len(examples):
        return list(examples)
    return rng.sample(examples, task_batch_size)


def _save_lora_checkpoint(
    lora_state_dict: dict[str, Any],
    adapter_path: str | None,
    output_dir: Path,
) -> None:
    """Save LoRA adapter weights to ``output_dir``.

    If an original ``adapter_path`` is given, the ``adapter_config.json`` is
    copied alongside the new weights so the checkpoint is self-contained.
    """
    import shutil
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state_dict, output_dir / "adapter_model.bin")
    if adapter_path:
        src_cfg = Path(adapter_path) / "adapter_config.json"
        if src_cfg.exists():
            shutil.copy2(src_cfg, output_dir / "adapter_config.json")
    logger.info("Checkpoint saved to %s", output_dir)


def train_grpo_ray(config: dict[str, Any]) -> dict[str, Any]:
    """Launch the Ray GRPO pipeline and return summary metrics.

    This is the entry point called by `grpo_train.main()` when ``--ray`` is
    passed. It creates rollout, reference-logprob, and learner workers, then
    runs the training loop in strict on-policy order.

    Args:
        config: Loaded YAML config dict (same schema as `train_grpo_from_config`
            plus an optional ``ray`` section for GPU assignments and vLLM port).

    Returns:
        Summary dict with final metrics, checkpoint path, and rollout counts.
    """
    ray = _require_ray()
    if not ray.is_initialized():
        ray.init()
    logger.info("Ray initialised: %s", ray.cluster_resources())

    seed = int(config.get("training", {}).get("seed", 0))
    rng = random.Random(seed)
    training = config.get("training", {})
    output = config.get("output", {})
    max_steps = int(training.get("max_steps", 1))
    if max_steps <= 0:
        raise ValueError("training.max_steps must be positive")
    task_batch_size_cfg = training.get("task_batch_size")
    task_batch_size: int | None = int(task_batch_size_cfg) if task_batch_size_cfg is not None else None
    save_every_steps = int(training.get("save_every_steps", 0) or 0)
    include_text = bool(output.get("include_text", True))

    checkpoint_dir = _new_checkpoint_dir(output.get("checkpoint_dir", "artifacts/checkpoints/grpo_ray"))
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    run_config_path = checkpoint_dir / "run_config.yaml"
    run_config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    rollouts_jsonl = Path(output.get("rollouts_jsonl", checkpoint_dir / "rollouts.jsonl"))
    metrics_jsonl = Path(output.get("metrics_jsonl", checkpoint_dir / "metrics.jsonl"))
    if metrics_jsonl.exists():
        metrics_jsonl.unlink()

    # Build remote classes at runtime to avoid module-level ray.remote decoration
    RemoteRolloutActor = ray.remote(num_gpus=_role_num_gpus(config, "rollout"))(RolloutActor)
    RemoteReferenceLogprobWorker = ray.remote(num_gpus=_role_num_gpus(config, "reference"))(ReferenceLogprobWorker)
    RemoteLearnerWorker = ray.remote(num_gpus=_role_num_gpus(config, "learner"))(LearnerWorker)

    rollout_actor: Any = RemoteRolloutActor.remote(config)
    reference_worker: Any = RemoteReferenceLogprobWorker.remote(config)
    learner: Any = RemoteLearnerWorker.remote(config)
    logger.info(
        "Worker resources: rollout=%s GPU, reference=%s GPU, learner=%s GPU",
        _role_num_gpus(config, "rollout"),
        _role_num_gpus(config, "reference"),
        _role_num_gpus(config, "learner"),
    )

    # Load example list from the source (driver side, no GPU needed)
    from sql_agent_training.train.grpo_rollouts import load_rollout_source_from_config

    source = load_rollout_source_from_config(config)
    try:
        adapter_path: str | None = (
            str(config["model"]["adapter_path"]) if config.get("model", {}).get("adapter_path") else None
        )
        metrics_history: list[dict[str, Any]] = []
        total_trajectories = 0
        final_batch_stats: dict[str, Any] = {}

        for step_index in range(max_steps):
            step = step_index + 1
            step_examples = _sample_step_examples(
                source.examples,
                task_batch_size=task_batch_size,
                rng=rng,
            )
            logger.info("Step %d/%d: running rollout on %d examples", step, max_steps, len(step_examples))

            # Step A: RolloutActor generates trajectories.
            rollout_ref = rollout_actor.rollout.remote(
                step_examples,
                rollout_jsonl_path=str(rollouts_jsonl),
                include_text=include_text,
            )
            grpo_batch = ray.get(rollout_ref)

            if not grpo_batch.groups:
                logger.warning("Step %d: empty batch; skipping update", step)
                metric_row: dict[str, Any] = {
                    "step": step,
                    "skipped_update": True,
                    "skip_reason": "empty_batch",
                }
                _write_jsonl_row(metrics_jsonl, metric_row)
                metrics_history.append(metric_row)
                continue

            # Step B: ReferenceLogprobWorker computes frozen reference log-probs.
            logger.info("Step %d: computing reference logprobs", step)
            ref_logprobs = ray.get(reference_worker.compute_ref_logprobs.remote(grpo_batch))

            # Step C: Learner old_logprobs + GRPO update.
            logger.info("Step %d: starting Learner train_step", step)
            train_ref = learner.train_step.remote(grpo_batch, ref_logprobs)
            result: LearnerStepResult = ray.get(train_ref)

            metric_row = {
                "step": step,
                "optimizer_step": result.optimizer_step,
                "trajectories": grpo_batch.num_trajectories,
                "groups": len(grpo_batch.groups),
                "task_batch_size": len(step_examples),
                **result.batch_stats,
                **result.metrics,
            }
            _write_jsonl_row(metrics_jsonl, metric_row)
            metrics_history.append(metric_row)
            total_trajectories += grpo_batch.num_trajectories
            final_batch_stats = result.batch_stats

            # Step D: push updated LoRA weights back to RolloutActor / vLLM.
            logger.info("Step %d: reloading LoRA weights in vLLM", step)
            ray.get(rollout_actor.reload_lora.remote(result.lora_state_dict))

            # Step E: optional checkpoint.
            if save_every_steps > 0 and step % save_every_steps == 0 and step != max_steps:
                _save_lora_checkpoint(
                    result.lora_state_dict,
                    adapter_path,
                    checkpoint_dir / f"step_{step:06d}",
                )

    finally:
        source.close()
        ray.get([rollout_actor.close.remote(), reference_worker.close.remote()])

    # Save final checkpoint
    final_lora = result.lora_state_dict if metrics_history else {}  # type: ignore[possibly-undefined]
    if final_lora:
        _save_lora_checkpoint(final_lora, adapter_path, checkpoint_dir)

    metrics_path = Path(output.get("metrics_json", checkpoint_dir / "metrics.json"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_history, indent=2), encoding="utf-8")

    final_metrics = metrics_history[-1] if metrics_history else {}
    logger.info("Ray GRPO training complete: %d steps, %d trajectories", max_steps, total_trajectories)
    return {
        **final_batch_stats,
        **final_metrics,
        "steps": max_steps,
        "trajectories": total_trajectories,
        "checkpoint_dir": str(checkpoint_dir),
        "metrics_json": str(metrics_path),
        "metrics_jsonl": str(metrics_jsonl),
        "rollouts_jsonl": str(rollouts_jsonl),
        "run_config": str(run_config_path),
    }
