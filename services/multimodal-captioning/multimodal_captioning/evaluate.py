"""Score captions by whether they describe the image, not by held-out loss.

Loss is what training optimises and it is a poor read on this service. The adapter can
lower it by learning the *register* of Flickr30k captions -- plausible sentences about
people and streets -- while describing the wrong picture, which is exactly what the
single-token checkpoints did: fluent, image-independent, and better than an untrained
adapter on loss.

So this measures agreement instead. CLIP embeds the image and the generated caption into
the same space, and their cosine similarity says whether the caption is *about* that
image. Two reference points make the number mean something, both scored the same way on
the same clips:

* an untrained adapter, which is the floor -- whatever score a random projection earns
* the captions a person wrote, which is the ceiling -- CLIP's own view of a good caption

Reporting agreement without both is how a 0.20 reads as competent. The published run
scored 0.1882 untrained and 0.3280 for human captions, so 0.20 is a tenth of the way
across a gap that is itself only 0.14 wide.

Using CLIP to score captions conditioned on CLIP features flatters the model: the
generator and the judge share an encoder, so a caption can agree with the image
representation without agreeing with the image. It is still the cheapest signal that
separates "describes this picture" from "reads like a caption", which loss does not.
"""

import argparse
import statistics

import torch
from PIL.Image import Image as PILImage

from ai_residency_common import get_logger, push_checkpoint
from config import Config
from multimodal_captioning.captioning import caption_image
from multimodal_captioning.data import build_datasets
from multimodal_captioning.models import (
    Backbones,
    build_adapter,
    load_adapter,
    load_backbones,
)

__all__ = ["agreement", "evaluate"]

logger = get_logger(__name__)


def agreement(images: list[PILImage], texts: list[str], backbones: Backbones) -> float:
    """Mean CLIP cosine similarity between each image and its text.

    Both sides go through CLIP's own projection heads, so this is the space CLIP was
    trained to align -- the same quantity its contrastive objective maximised.
    """
    device = backbones.device
    scores: list[float] = []
    with torch.no_grad():
        for image, text in zip(images, texts, strict=True):
            pixels = backbones.image_processor(images=image.convert("RGB"), return_tensors="pt")
            pixels = {k: v.to(device) for k, v in pixels.items()}
            # Explicit projections rather than `get_image_features` /
            # `get_text_features`: in transformers 5.x those return a
            # `BaseModelOutputWithPooling`, not a tensor. Going through the towers and
            # the two projection heads is also exactly the path `captioning.py` takes,
            # so the judge sees what the generator saw.
            image_embed = backbones.clip_model.visual_projection(
                backbones.vision_encoder(**pixels).pooler_output
            )
            # 77 tokens is CLIP's text limit; a long caption is truncated rather than
            # rejected, which matches how the published numbers were produced.
            tokens = backbones.clip_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=77, padding=True
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}
            text_embed = backbones.clip_model.text_projection(
                backbones.text_encoder(**tokens).pooler_output
            )
            scores.append(
                float(torch.nn.functional.cosine_similarity(image_embed, text_embed).mean())
            )
    return statistics.fmean(scores)


def evaluate(config: Config, *, limit: int = 200, push: bool = False) -> dict[str, float]:
    """Score the shipped adapter against the untrained floor and the human ceiling.

    Args:
        config: Supplies the checkpoint, the dataset and the seeded split.
        limit: Held-out clips to score. The published numbers used a few hundred;
            agreement is a mean over independent pairs, so this converges quickly.
        push: Upload the scored checkpoint to the Hub.

    Returns:
        The three means, keyed `untrained`, `trained` and `human`.
    """
    backbones = load_backbones(config)
    device = backbones.device
    _, held_out = build_datasets(config)
    # The split is a seeded `random_split` over the first `num_samples` rows, so this
    # only scores genuinely held-out images when `num_samples` matches the run that
    # produced the checkpoint. Scored at the default 100 against a checkpoint trained
    # on 5,000, the "held-out" 20 are drawn from a different pool and some of them were
    # in training.
    logger.info(
        "Split over the first %d rows: %d held out. Set AI_RESIDENCY_NUM_SAMPLES to the "
        "value the checkpoint was trained with.",
        config.num_samples,
        len(held_out),
    )

    examples = [held_out[i] for i in range(min(limit, len(held_out)))]
    images = [example[0] for example in examples]
    human = [example[1] for example in examples]
    logger.info("Scoring %d held-out pairs", len(images))

    trained = load_adapter(config, config.checkpoint_name, backbones).to(device)
    untrained = build_adapter(
        backbones.clip_dim, backbones.qwen_dim, final_layer_norm=config.final_layer_norm
    ).to(device)

    results: dict[str, float] = {}
    for name, adapter in (("untrained", untrained), ("trained", trained)):
        captions = [caption_image(image, adapter, backbones, config).caption for image in images]
        results[name] = agreement(images, captions, backbones)
        logger.info("%s adapter: CLIP agreement %.4f", name, results[name])
        if name == "trained":
            for image_caption in captions[:3]:
                logger.info("  sample: %s", image_caption[:90])

    results["human"] = agreement(images, human, backbones)
    logger.info("human captions: CLIP agreement %.4f", results["human"])

    gap = results["human"] - results["untrained"]
    closed = (results["trained"] - results["untrained"]) / gap if gap else 0.0
    logger.info(
        "Closed %.1f%% of the gap from an untrained adapter (%.4f) to human captions (%.4f)",
        100 * closed,
        results["untrained"],
        results["human"],
    )

    if push and config.hf_repo_id is not None:
        push_checkpoint(config.hf_repo_id, config.checkpoint_path)

    return results


def main() -> None:
    """Entry point for `make evaluate`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=200, help="held-out pairs to score")
    parser.add_argument("--push", action="store_true", help="upload the scored checkpoint")
    args = parser.parse_args()
    evaluate(Config(), limit=args.limit, push=args.push)


if __name__ == "__main__":
    main()
