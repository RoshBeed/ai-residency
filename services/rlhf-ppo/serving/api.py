"""FastAPI service for this repo's generation demo.

The HTTP equivalent of `ui/app.py`, built for the portfolio hub that runs all six
services as backend containers behind one frontend. It reuses the same model-loading and
generation path the UI uses rather than reimplementing it.

Every response carries a `note` field saying plainly that this is not a PPO-tuned
completion, so a caller cannot mistake it for one. `make train` does now write a
policy adapter, but it is the product of a single PPO step, so this service
deliberately does not load it: serving it would imply an improvement that one
gradient step cannot honestly claim.

`/ppo_step` is the other half of this service: it runs `PPOTrainer.ppo_step` for
real, all ten stages, and returns everything each one produced -- not a simulation
of PPO, the same class `make train` uses. It loads three copies of the base model
(policy, frozen reference, value) plus the OpenAssistant DeBERTa reward model, which
is why this service's pod needs far more memory than `/generate` alone would.
"""

import queue
import threading
from collections.abc import AsyncIterator, Callable, Generator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ai_residency_common import (
    add_health_routes,
    format_sse,
    get_device,
    get_logger,
    retrying,
    run_app,
)
from base_setup import BaseModelSetup
from config import PPOConfig
from models import PolicyModel
from ppo_trainer import PPOStepResult, PPOTrainer
from training.run_ppo import EXAMPLE_PROMPTS

__all__ = ["app"]

logger = get_logger(__name__)

NO_CHECKPOINT_NOTE = (
    "Completion from LoRA-initialized SmolLM2-135M-Instruct, not a PPO-tuned checkpoint. "
    "No policy improvement has been applied. See the PPO section below for the trained pipeline."
)

PPO_NOTE = (
    "This runs PPOTrainer.ppo_step for real -- all twelve stages, on this repo's "
    "actual pipeline -- with the OpenAssistant DeBERTa-v3 reward model "
    "(OpenAssistant/reward-model-deberta-v3-large-v2) providing real reward scores "
    "trained on human preference data including Reddit TL;DR. The reward signal is "
    "genuine; the headroom for improvement is limited by the single-step PPO update "
    "this endpoint performs each call. Repeated calls compound on the same policy/"
    "value weights cached in this pod's memory until it restarts, the same way a "
    "real PPO run would."
)

_models_ready = False
"""Whether the startup preload finished. Reported by `/ready`."""


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load the PPOTrainer at boot so the first request is warm.

    Three SmolLM2-135M copies + the OpenAssistant DeBERTa reward model are built
    once and cached for the pod's lifetime. The checkpoint directory is also
    created here so the first save does not race with the volume mount.

    A failure here leaves the app running and `/ready` answering 503, rather than
    killing the process: the pod stays inspectable, and the Service simply does not
    route to it.
    """
    ensure_loaded()
    yield


def ensure_loaded() -> bool:
    """Build the PPO models if they are not built, and report whether they are."""
    global _models_ready
    if _models_ready:
        return True
    try:
        PPOConfig().checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Pre-loading PPOTrainer at startup")
        _load_ppo_trainer()
        _models_ready = True
    except Exception as error:
        logger.error("Could not pre-load the PPO models, /ready will report 503: %s", error)
        return False
    return _models_ready


app = FastAPI(
    title="RLHF / PPO Generation Service",
    description=(
        "Generation endpoint backed by this repo's PolicyModel (LoRA-wrapped "
        "SmolLM2-135M-Instruct). See /generate's response 'note' field for the honest "
        "no-trained-checkpoint disclosure."
    ),
    lifespan=lifespan,
)


class GenerateRequest(BaseModel):
    """Input for `/generate`."""

    prompt: str
    max_new_tokens: int = Field(default=50, gt=0, le=512)


class TokenCandidate(BaseModel):
    """One candidate token the sampler could have picked at a given step."""

    token: str
    probability: float


class GenerationStep(BaseModel):
    """One step of the autoregressive sampling loop."""

    step: int
    token: str
    probability: float
    top_candidates: list[TokenCandidate]


class GenerateResponse(BaseModel):
    """Output of `/generate`."""

    prompt: str
    completion: str
    note: str
    prompt_tokens: list[str] = []
    generation_trace: list[GenerationStep] = []


@lru_cache(maxsize=1)
def _load_policy_model() -> tuple[PolicyModel, Any, torch.device, PPOConfig]:
    """Load the policy model once per process, the same way the UI does."""
    config = PPOConfig()
    device = get_device()
    logger.info("Loading %s on %s", config.model_name, device)

    base_setup = BaseModelSetup(config)
    policy_model = PolicyModel(base_setup).to(device)
    policy_model.eval()
    return policy_model, base_setup.tokenizer, device, config


_ppo_trainer: PPOTrainer | None = None
"""The PPOTrainer loaded on first call to `/ppo_step`, cached for the pod's lifetime."""


def _load_ppo_trainer(
    on_progress: Callable[[str], None] | None = None,
) -> PPOTrainer:
    """Return the cached PPOTrainer, building it with progress events on first call.

    Three SmolLM2-135M base-model copies plus the OpenAssistant reward model are
    built on the first invocation. That
    takes a while, and an SSE client has no idea anything is happening unless
    progress events arrive. `on_progress(detail)` is called throughout init so the
    streaming endpoint can relay each step to the client.
    """
    global _ppo_trainer
    if _ppo_trainer is not None:
        if on_progress is not None:
            on_progress("PPO trainer already loaded in this pod.")
        return _ppo_trainer

    logger.info("Loading PPOTrainer (4 model copies) for /ppo_step")
    _ppo_trainer = PPOTrainer(PPOConfig(), on_progress=on_progress)
    return _ppo_trainer


class SAPairOut(BaseModel):
    """One (state, action) pair, fully populated -- every stage's output on it."""

    state_text: str
    action_token: str
    action_token_id: int
    prompt_idx: int
    step: int
    logprob: float
    ref_logprob: float
    kl_div: float
    adjusted_reward: float
    value: float
    td_error: float
    advantage: float
    reward_to_go: float
    clipped_ratio: float
    policy_ratio: float


class PPOStepRequest(BaseModel):
    """Input for `/ppo_step`."""

    prompts: list[str] | None = None
    """Prompts to run one PPO step over. Defaults to this repo's own example prompts."""

    ppo_steps: int = 5
    """Number of PPO steps to run in this request. Each step is a full 12-stage
    pipeline (generate, score, credit-assign, update, save)."""


class PPOStepResponse(BaseModel):
    """Output of `/ppo_step`: every stage of one real PPO step."""

    prompts: list[str]
    generated_texts: list[str]
    rewards: list[float]
    sa_pairs: list[SAPairOut]
    value_loss: float
    policy_loss: float
    step_count: int
    note: str


@dataclass
class GenerationResult:
    """Everything `_generate` produces: the completion plus the sampling trace."""

    completion: str
    prompt_tokens: list[str]
    generation_trace: list[GenerationStep]


def _generate(
    policy_model: PolicyModel,
    tokenizer: Any,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
) -> GenerationResult:
    """Sample a completion for one prompt, tracing the sampling loop that produced it.

    Args:
        policy_model: The loaded policy model.
        tokenizer: Shared tokenizer.
        device: Device to run on.
        prompt: Text to complete.
        max_new_tokens: How many tokens to generate.

    Returns:
        The generated continuation, the tokenized prompt, and a per-step trace of
        what the sampler was choosing between and which token it actually drew.
    """
    inputs = tokenizer([prompt], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = policy_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

    prompt_length = inputs["input_ids"].shape[1]
    sequence = outputs.sequences[0]
    generated_ids = sequence[prompt_length:]
    completion: str = tokenizer.decode(generated_ids, skip_special_tokens=True)

    prompt_tokens = [tokenizer.decode([token_id]) for token_id in inputs["input_ids"][0].tolist()]

    # `outputs.scores` is one logit tensor (batch, vocab) per generated step, already
    # warped by the top_k/top_p/temperature settings above -- these are the real
    # probabilities the sampling step drew from, not the raw model logits.
    generation_trace: list[GenerationStep] = []
    for step, step_scores in enumerate(outputs.scores):
        probs = torch.softmax(step_scores[0], dim=-1)
        top_probs, top_ids = torch.topk(probs, 5)
        top_candidates = [
            TokenCandidate(token=tokenizer.decode([int(token_id)]), probability=float(prob))
            for token_id, prob in zip(top_ids.tolist(), top_probs.tolist(), strict=True)
        ]
        sampled_id = int(sequence[prompt_length + step])
        generation_trace.append(
            GenerationStep(
                step=step,
                token=tokenizer.decode([sampled_id]),
                probability=float(probs[sampled_id]),
                top_candidates=top_candidates,
            )
        )

    return GenerationResult(
        completion=completion,
        prompt_tokens=prompt_tokens,
        generation_trace=generation_trace,
    )


# There is no PPO-tuned checkpoint to load -- the policy starts from LoRA-initialized
# SmolLM2 and improves within the pod's lifetime -- so `checkpoint_loaded` reports
# whether the models this service actually needs are resident, which is the thing that
# determines whether it can answer.
add_health_routes(
    app,
    # Retried rather than decided once. A resolve that failed at startup -- a DNS blip
    # during a mass restart, say -- would otherwise leave this pod not-ready for the
    # rest of its life, with liveness unable to help because the process itself is
    # perfectly healthy. `retrying` re-attempts, at most once a minute, from /ready.
    ready=retrying(ensure_loaded),
    detail=lambda: {
        "device": str(get_device()),
        "models_loaded": _models_ready,
        "checkpoint_loaded": _models_ready,
    },
)


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    """Complete a prompt with the LoRA-wrapped policy model.

    Args:
        request: The prompt and how many tokens to generate.

    Returns:
        The completion, plus the no-trained-checkpoint disclosure.

    Raises:
        HTTPException: 400 if the prompt is empty.
    """
    if not request.prompt.strip():
        raise HTTPException(status_code=422, detail="prompt must not be empty")

    policy_model, tokenizer, device, _ = _load_policy_model()
    result = _generate(policy_model, tokenizer, device, request.prompt, request.max_new_tokens)
    return GenerateResponse(
        prompt=request.prompt,
        completion=result.completion,
        note=NO_CHECKPOINT_NOTE,
        prompt_tokens=result.prompt_tokens,
        generation_trace=result.generation_trace,
    )


@app.post("/ppo_step", response_model=PPOStepResponse)
def ppo_step(request: PPOStepRequest) -> PPOStepResponse:
    """Run one real PPO step and return what every stage produced.

    Args:
        request: Optional prompts to run the step over.

    Returns:
        Every (s, a) pair with all ten stages' fields, the rewards, both losses, and
        the honest disclosure of what's real here and what isn't.
    """
    prompts = request.prompts if request.prompts else EXAMPLE_PROMPTS
    if not prompts or any(not prompt.strip() for prompt in prompts):
        raise HTTPException(status_code=422, detail="prompts must be non-empty")

    trainer = _load_ppo_trainer()
    result: PPOStepResult | None = None
    for _step in range(request.ppo_steps):
        result = trainer.ppo_step(prompts)
    if result is None:
        raise HTTPException(status_code=500, detail="ppo_step produced no result")

    return PPOStepResponse(
        prompts=prompts,
        generated_texts=result["generated_texts"],
        rewards=result["rewards"],
        sa_pairs=[SAPairOut(**pair) for pair in result["sa_pairs"]],
        value_loss=result["value_loss"],
        policy_loss=result["policy_loss"],
        step_count=trainer.step_count,
        note=PPO_NOTE,
    )


@app.post("/ppo_step/stream")
def ppo_step_stream(request: PPOStepRequest) -> StreamingResponse:
    """Same pipeline as `/ppo_step`, streamed as Server-Sent Events per stage.

    ``PPOTrainer.ppo_step``'s own ``on_stage`` callback already fires once per real
    stage boundary; this just relays each call through a queue from a background
    thread, so a caller sees the pipeline unfold live instead of waiting on one
    request that can take a while (four model copies, ten stages, two backward
    passes).

    An initial ``{"stage": "loading", ...}`` event is pushed immediately, followed
    by progress events as models load (the expensive first-call case), heartbeats
    every 30s so no proxy times out, and finally the per-stage events the trainer
    itself emits.

    Args:
        request: Optional prompts to run the step over.

    Returns:
        A ``text/event-stream`` response. Each event is ``{"stage": ..., ...}``,
        matching ``PPOTrainer.ppo_step``'s own stage names; the final event is
        ``{"stage": "done", ...}`` (same shape as ``/ppo_step``'s body), or
        ``{"stage": "error", "detail": ...}`` if the step raised.
    """
    prompts = request.prompts if request.prompts else EXAMPLE_PROMPTS
    if not prompts or any(not prompt.strip() for prompt in prompts):
        raise HTTPException(status_code=422, detail="prompts must be non-empty")

    events: queue.Queue[dict[str, Any] | None] = queue.Queue()

    def worker() -> None:
        try:
            trainer = _load_ppo_trainer()

            def on_stage(stage: str, payload: dict[str, Any]) -> None:
                events.put({"stage": stage, **payload})

            final_result: PPOStepResult | None = None
            for step in range(1, request.ppo_steps + 1):
                events.put({"stage": "step_boundary", "step": step, "total": request.ppo_steps})
                final_result = trainer.ppo_step(prompts, on_stage=on_stage)
            if final_result is None:
                raise RuntimeError("ppo_step produced no result")

            events.put(
                {
                    "stage": "done",
                    "prompts": prompts,
                    "generated_texts": final_result["generated_texts"],
                    "rewards": final_result["rewards"],
                    "sa_pairs": final_result["sa_pairs"],
                    "value_loss": final_result["value_loss"],
                    "policy_loss": final_result["policy_loss"],
                    "step_count": trainer.step_count,
                    "note": PPO_NOTE,
                }
            )
        except Exception as error:  # relay any failure to the client rather than hiding it
            events.put({"stage": "error", "detail": str(error)})
        finally:
            events.put(None)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    def event_generator() -> Generator[str, None, None]:
        heartbeat = 0
        while True:
            try:
                event = events.get(timeout=5)
            except queue.Empty:
                heartbeat += 1
                yield format_sse({"stage": "heartbeat", "elapsed": heartbeat * 5})
                continue
            if event is None:
                break
            yield format_sse(event)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def main() -> None:
    """Entry point for `make serve`."""
    run_app(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
