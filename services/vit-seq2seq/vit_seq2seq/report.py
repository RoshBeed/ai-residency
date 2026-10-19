"""The two figures in this service's write-up, drawn from the shipped checkpoint.

Both assets used to be slide exports, and both drifted away from the model without
anything noticing: they showed a four-digit composite laid out two-by-two decoding to
`0 2 3 1`, 16 patch tokens, `nn.Linear(11, 13)` and a `<pad>` id. The shipped model
reads a 112x112 row of one to three digits as 49 patch tokens at `d_model` 256, projects
to a twelve-token vocabulary, and deliberately has no pad id -- shorter targets are
padded with repeated EOS and the loss is masked after the first one, which is what keeps
the output head twelve wide.

A figure that can only be redrawn by whoever still has the slide deck is a figure that
goes stale. Every shape here is read off `Seq2SeqArch` and a real forward pass, and
every decoded sequence is a real `generate` call on held-out composites, so `make
figures` reproduces both from whatever checkpoint is current.

Figure 1 is shared with `vit-classifier`, which shows the single-label half of the same
picture. The target paths are arguments rather than hardcoded, so one run writes both
copies and they cannot disagree.
"""

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.transforms import Transform
from torchvision import datasets, transforms

from ai_residency_common import get_device, get_logger, resolve_required, set_seed
from ai_residency_common.figures import (
    AQUA,
    BLUE,
    INK,
    INK_SOFT,
    MUTED,
    ORANGE,
    RULE,
    SURFACE,
    VIOLET,
    apply_style,
    save_figure,
)
from config import Config
from datasets.multi_digit_dataset import MultiDigitDataset, MultiDigitMNISTDataset
from vit_seq2seq.evaluate import digits_of
from vit_seq2seq.seq2seq import Seq2SeqVisionTransformer

__all__ = ["architecture_figure", "main", "task_figure"]

logger = get_logger(__name__)

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


class Draw:
    """Boxes, arrows and labels in figure coordinates.

    These are schematics rather than plots -- there is no data to put on an axis, so the
    whole figure is drawn in its own coordinate space. Wrapping that in one object keeps
    the layout readable as positions rather than as transform boilerplate.
    """

    def __init__(self, figure: Figure) -> None:
        self.figure = figure
        self.transform: Transform = figure.transFigure

    def box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        colour: str,
        label: str = "",
        *,
        sub: str = "",
        dashed: bool = False,
        size: float = 12,
        fill: str = SURFACE,
    ) -> None:
        """A labelled block, optionally with a smaller line of shape underneath."""
        self.figure.patches.append(
            mpatches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.003,rounding_size=0.006",
                transform=self.transform,
                facecolor=fill,
                edgecolor=colour,
                linewidth=1.6,
                linestyle="--" if dashed else "-",
                zorder=2,
            )
        )
        centre = y + h / 2 + (0.012 if sub else 0)
        if label:
            self.figure.text(
                x + w / 2,
                centre,
                label,
                ha="center",
                va="center",
                color=colour,
                fontsize=size,
                family="monospace",
                zorder=3,
            )
        if sub:
            self.figure.text(
                x + w / 2,
                centre - 0.030,
                sub,
                ha="center",
                va="center",
                color=MUTED,
                fontsize=9.5,
                family="monospace",
                zorder=3,
            )

    def arrow(
        self, x0: float, y0: float, x1: float, y1: float, colour: str = MUTED, bend: float = 0.0
    ) -> None:
        """A single arrow, straight unless `bend` curves it."""
        self.figure.patches.append(
            mpatches.FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                transform=self.transform,
                connectionstyle=f"arc3,rad={bend}",
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.3,
                color=colour,
                shrinkA=0,
                shrinkB=0,
                zorder=1 if bend else 2,
            )
        )

    def rule(self, x: float, y0: float = 0.08, y1: float = 0.92) -> None:
        """A vertical divider between panels."""
        self.figure.lines.append(
            Line2D([x, x], [y0, y1], transform=self.transform, color=RULE, linewidth=1.0)
        )

    def note(self, x: float, y: float, text: str, colour: str = MUTED, size: float = 9.5) -> None:
        """A small monospace annotation."""
        self.figure.text(
            x, y, text, ha="center", va="center", color=colour, fontsize=size, family="monospace"
        )

    def image(
        self, box: tuple[float, float, float, float], pixels: np.ndarray, edge: str = RULE
    ) -> Axes:
        """A greyscale panel, drawn dark-on-light so it reads on the page."""
        axes = self.figure.add_axes(box)
        axes.imshow(1.0 - pixels, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes.set_xticks([])
        axes.set_yticks([])
        axes.grid(False)
        for spine in axes.spines.values():
            spine.set_edgecolor(edge)
        return axes


def _held_out_composites(config: Config) -> tuple[list[np.ndarray], list[str], list[str]]:
    """One held-out composite per digit count, with what the shipped model reads off it.

    Returns:
        The images as pixels in [0, 1], the decoded digits, and the targets.
    """
    set_seed(config.seed)
    device = get_device()
    arch = config.seq2seq_arch

    base = MultiDigitMNISTDataset(
        data_dir=config.data_dir,
        image_size=arch.img_size,
        max_digits=config.max_digits,
        min_digits=config.min_digits,
        digit_size=config.digit_size,
        augment_probability=config.augment_probability,
        rotation_degrees=config.augment_rotation_degrees,
        shear_degrees=config.augment_shear_degrees,
        scale_range=config.augment_scale_range,
        samples_per_class=1,
        batch_size=1,
        num_workers=0,
        seed=config.seed,
    )
    by_length: dict[int, int] = {}
    for number in base.test_numbers:
        by_length.setdefault(len(str(number)), number)
    numbers = [by_length[length] for length in sorted(by_length)]

    split = MultiDigitDataset(
        digit_images=base.digit_images,
        numbers=numbers,
        image_size=arch.img_size,
        max_digits=config.max_digits,
        digit_size=config.digit_size,
        augment_probability=config.augment_probability,
        rotation_degrees=config.augment_rotation_degrees,
        shear_degrees=config.augment_shear_degrees,
        scale_range=config.augment_scale_range,
    )

    model = Seq2SeqVisionTransformer(arch).to(device)
    path, source = resolve_required(
        config.seq2seq_checkpoint_path,
        config.hf_repo_id,
        rebuild_hint="Train it first with `make train`.",
    )
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()
    logger.info("Drawing figures from %s (%s)", config.seq2seq_checkpoint_path.name, source)

    images: list[np.ndarray] = []
    reads: list[str] = []
    for index in range(len(split.numbers)):
        tensor, _ = split[index]
        with torch.no_grad():
            generated = model.generate(
                tensor.unsqueeze(0).to(device),
                start_token=arch.start_token,
                max_length=config.decoder_seq_length,
            )
        digits = digits_of(generated[0].tolist(), arch.start_token, arch.eos_token)
        reads.append("".join(str(digit) for digit in digits))
        images.append(np.clip(tensor.squeeze(0).numpy() * MNIST_STD + MNIST_MEAN, 0.0, 1.0))
    return images, reads, [str(number) for number in split.numbers]


def _mnist_panel(config: Config) -> tuple[np.ndarray, np.ndarray, int]:
    """Ten examples of each digit for the dataset panel, plus one digit on its own."""
    raw = datasets.MNIST(
        str(config.data_dir), train=False, download=True, transform=transforms.ToTensor()
    )
    grid: dict[int, list[np.ndarray]] = {digit: [] for digit in range(10)}
    single: np.ndarray | None = None
    label = 4
    for image, target in raw:
        digit = int(target)
        if len(grid[digit]) < 10:
            grid[digit].append(image.squeeze(0).numpy())
        if single is None and digit == label:
            single = image.squeeze(0).numpy()
        if single is not None and all(len(v) == 10 for v in grid.values()):
            break
    assert single is not None
    tiles = np.vstack([np.hstack([grid[d][row] for d in range(10)]) for row in range(10)])
    return tiles, single, label


def task_figure(config: Config, outputs: list[Path]) -> None:
    """Figure 1: one label on the left, a self-terminating sequence on the right.

    Args:
        config: Supplies the architecture, the data directory and the checkpoint.
        outputs: Every path to write. Both ViT services show this figure, so one run
            writes both copies rather than leaving them to drift apart.
    """
    images, reads, _ = _held_out_composites(config)
    tiles, single, label = _mnist_panel(config)

    apply_style()
    figure = plt.figure(figsize=(16, 7))
    figure.patch.set_facecolor(SURFACE)
    draw = Draw(figure)

    axes = draw.image((0.035, 0.12, 0.26, 0.72), tiles)
    axes.set_xticks([28 * digit + 14 for digit in range(10)])
    axes.set_xticklabels([str(digit) for digit in range(10)], fontsize=9, color=MUTED)
    axes.xaxis.tick_top()
    for spine in axes.spines.values():
        spine.set_visible(True)
    figure.text(
        0.165,
        0.055,
        "MNIST, the handwriting both services read",
        ha="center",
        color=INK_SOFT,
        fontsize=11,
    )
    draw.rule(0.345)

    draw.image((0.405, 0.13, 0.09, 0.19), single)
    draw.arrow(0.45, 0.335, 0.45, 0.40)
    draw.box(0.395, 0.40, 0.11, 0.30, AQUA, "Encoder")
    draw.arrow(0.45, 0.705, 0.45, 0.775)
    figure.text(
        0.45,
        0.80,
        str(label),
        ha="center",
        va="bottom",
        color=INK,
        fontsize=22,
        family="monospace",
    )
    figure.text(0.45, 0.055, "vit-classifier: one label", ha="center", color=INK_SOFT, fontsize=11)
    draw.rule(0.555)

    for image, read, top in zip(images, reads, (0.63, 0.42, 0.21), strict=True):
        draw.image((0.578, top, 0.075, 0.155), image)
        draw.arrow(0.656, top + 0.078, 0.678, top + 0.078)
        draw.arrow(0.850, top + 0.078, 0.872, top + 0.078, colour=ORANGE)
        figure.text(
            0.878,
            top + 0.078,
            " ".join(read) + "  <eos>",
            ha="left",
            va="center",
            color=INK,
            fontsize=14,
            family="monospace",
        )
    draw.box(0.680, 0.20, 0.070, 0.60, AQUA, "Encoder")
    draw.arrow(0.752, 0.50, 0.775, 0.50)
    draw.box(0.777, 0.20, 0.070, 0.60, ORANGE, "Decoder")
    draw.arrow(0.812, 0.152, 0.812, 0.198)
    figure.text(
        0.812,
        0.142,
        "<start>",
        ha="center",
        va="top",
        color=MUTED,
        fontsize=10,
        family="monospace",
    )
    figure.text(
        0.78,
        0.055,
        "vit-seq2seq: one to three digits, ending at <eos>",
        ha="center",
        color=INK_SOFT,
        fontsize=11,
    )

    for output in outputs:
        save_figure(figure, output, tight=False)
        logger.info("Wrote %s", output)
    plt.close(figure)


def architecture_figure(config: Config, outputs: list[Path]) -> None:
    """Figure 6: the encoder-decoder, with the shapes the model actually has.

    Args:
        config: Supplies the architecture, the data directory and the checkpoint.
        outputs: Every path to write.
    """
    images, reads, _ = _held_out_composites(config)
    composite, read = images[-1], reads[-1]
    arch = config.seq2seq_arch
    grid_side = arch.img_size // arch.patch_size
    tokens = grid_side * grid_side
    patch_dim = arch.patch_size * arch.patch_size * arch.in_channels
    target_length = config.decoder_seq_length - 1

    apply_style()
    figure = plt.figure(figsize=(16, 9))
    figure.patch.set_facecolor(SURFACE)
    draw = Draw(figure)

    left_x, left_w = 0.055, 0.30
    draw.image((left_x + 0.005, 0.055, 0.085, 0.145), composite)
    draw.note(left_x + 0.0475, 0.030, f"{arch.in_channels} x {arch.img_size} x {arch.img_size}")

    patches = draw.image((left_x + 0.125, 0.055, 0.17, 0.145), composite, edge=VIOLET)
    for edge in range(1, grid_side):
        patches.axhline(arch.patch_size * edge - 0.5, color=VIOLET, linewidth=0.9)
        patches.axvline(arch.patch_size * edge - 0.5, color=VIOLET, linewidth=0.9)
    draw.note(
        left_x + 0.21,
        0.030,
        f"{grid_side} x {grid_side} patches of {arch.patch_size} x {arch.patch_size}",
    )
    draw.arrow(left_x + 0.095, 0.128, left_x + 0.122, 0.128)

    draw.arrow(left_x + left_w / 2, 0.205, left_x + left_w / 2, 0.245)
    draw.box(
        left_x,
        0.245,
        left_w,
        0.065,
        VIOLET,
        "Linear projection of flattened patches",
        size=11.5,
        sub=f"{patch_dim} -> {arch.d_model}",
    )
    draw.arrow(left_x + left_w / 2, 0.315, left_x + left_w / 2, 0.355)
    draw.note(left_x + left_w / 2 + 0.085, 0.335, f"{tokens} x {arch.d_model}   + positions")

    draw.box(left_x, 0.355, left_w, 0.265, ORANGE, dashed=True)
    for label, y in (("Encoder Block", 0.565), ("Encoder Block", 0.495), ("...", 0.425)):
        draw.box(left_x + 0.022, y, left_w - 0.044, 0.045, ORANGE, label, size=11.5)
    draw.box(left_x + 0.022, 0.370, left_w - 0.044, 0.045, ORANGE, "Encoder Block", size=11.5)
    draw.note(
        left_x + left_w / 2,
        0.638,
        f"x {arch.num_encoder_layers}, {arch.heads} heads of {arch.head_dim}, "
        f"feed-forward {arch.ff_hidden_dim}",
    )
    draw.arrow(left_x + left_w / 2, 0.650, left_x + left_w / 2, 0.690)
    draw.box(
        left_x + 0.055,
        0.690,
        left_w - 0.11,
        0.055,
        ORANGE,
        "encoder output",
        size=11.5,
        sub=f"{tokens} x {arch.d_model}",
    )

    right_x, right_w = 0.515, 0.30
    draw.box(right_x, 0.055, right_w, 0.055, INK_SOFT, "<start>  " + "  ".join(read), size=12.5)
    draw.arrow(right_x + right_w / 2, 0.115, right_x + right_w / 2, 0.155)
    draw.box(right_x, 0.155, right_w, 0.055, BLUE, "Token embedding + positions", size=11.5)
    draw.arrow(right_x + right_w / 2, 0.215, right_x + right_w / 2, 0.255)
    draw.note(right_x + right_w / 2 + 0.080, 0.235, f"{target_length} x {arch.d_model}")

    draw.box(right_x, 0.255, right_w, 0.420, BLUE, dashed=True)
    draw.note(
        right_x + right_w / 2,
        0.690,
        f"x {arch.num_decoder_layers}, causal self-attention then cross-attention",
    )
    draw.box(right_x + 0.022, 0.600, right_w - 0.044, 0.055, BLUE, "Decoder Block", size=11.5)
    draw.box(right_x + 0.022, 0.530, right_w - 0.044, 0.055, BLUE, "...", size=11.5)

    draw.box(right_x + 0.022, 0.275, right_w - 0.044, 0.235, AQUA, fill="#f2faf6")
    draw.note(right_x + right_w / 2, 0.487, "inside one block", colour=AQUA, size=10)
    rows = (
        ("Q  from the decoder", f"{target_length} x {arch.d_model}"),
        ("K, V  from the encoder output", f"{tokens} x {arch.d_model}"),
        (f"softmax(Q K' / sqrt({arch.head_dim}))", f"{arch.heads} x {target_length} x {tokens}"),
        ("attention x V, heads joined", f"{target_length} x {arch.d_model}"),
    )
    for index, (what, dims) in enumerate(rows):
        y = 0.447 - index * 0.042
        figure.text(
            right_x + 0.038,
            y,
            what,
            ha="left",
            va="center",
            color=INK_SOFT,
            fontsize=10,
            family="monospace",
        )
        figure.text(
            right_x + right_w - 0.038,
            y,
            dims,
            ha="right",
            va="center",
            color=MUTED,
            fontsize=10,
            family="monospace",
        )

    draw.arrow(left_x + left_w - 0.055, 0.7175, right_x + 0.022, 0.393, colour=ORANGE, bend=-0.18)
    figure.text(
        0.455,
        0.60,
        "cross-attention",
        ha="center",
        color=ORANGE,
        fontsize=10.5,
        family="monospace",
        rotation=-52,
    )

    draw.arrow(right_x + right_w / 2, 0.680, right_x + right_w / 2, 0.720)
    draw.box(
        right_x,
        0.720,
        right_w,
        0.055,
        VIOLET,
        f"nn.Linear({arch.d_model}, {arch.vocab_size})",
        size=12,
    )
    draw.arrow(right_x + right_w / 2, 0.780, right_x + right_w / 2, 0.820)
    draw.note(right_x + right_w / 2 + 0.075, 0.800, f"{target_length} x {arch.vocab_size} logits")
    draw.box(right_x, 0.820, right_w, 0.060, INK, "  ".join(read) + "  <eos>", size=13.5)

    vocab_x = 0.845
    draw.box(vocab_x, 0.255, 0.115, 0.42, RULE, fill="#fbfbf7")
    draw.note(vocab_x + 0.0575, 0.645, "vocabulary", colour=INK_SOFT, size=10.5)
    figure.text(
        vocab_x + 0.018,
        0.608,
        "id",
        ha="left",
        va="center",
        color=MUTED,
        fontsize=9.5,
        family="monospace",
    )
    figure.text(
        vocab_x + 0.098,
        0.608,
        "means",
        ha="right",
        va="center",
        color=MUTED,
        fontsize=9.5,
        family="monospace",
    )
    figure.lines.append(
        Line2D(
            [vocab_x + 0.014, vocab_x + 0.101],
            [0.592, 0.592],
            transform=draw.transform,
            color=RULE,
            linewidth=0.8,
        )
    )
    entries = [(str(digit), str(digit)) for digit in range(10)]
    entries.append((str(arch.start_token), "<start>"))
    entries.append((str(arch.eos_token), "<eos>"))
    for index, (token, meaning) in enumerate(entries):
        y = 0.566 - index * 0.0252
        figure.text(
            vocab_x + 0.018,
            y,
            token,
            ha="left",
            va="center",
            color=MUTED,
            fontsize=9.5,
            family="monospace",
        )
        figure.text(
            vocab_x + 0.098,
            y,
            meaning,
            ha="right",
            va="center",
            color=INK_SOFT,
            fontsize=9.5,
            family="monospace",
        )
    figure.text(
        vocab_x + 0.0575,
        0.225,
        "no pad id: shorter targets are\n"
        "padded with repeated <eos>\n"
        "and the loss is masked after\n"
        "the first one",
        ha="center",
        va="top",
        color=MUTED,
        fontsize=9.5,
        family="monospace",
    )

    for output in outputs:
        save_figure(figure, output, tight=False)
        logger.info("Wrote %s", output)
    plt.close(figure)


def main() -> None:
    """Entry point for `make figures`."""
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Redraw this service's write-up figures")
    parser.add_argument(
        "--task",
        type=Path,
        nargs="+",
        default=[
            here / "assets/task_overview.png",
            here.parent / "vit-classifier/assets/task_overview.png",
        ],
        help="where to write Figure 1; both ViT services show it",
    )
    parser.add_argument(
        "--architecture",
        type=Path,
        nargs="+",
        default=[here / "assets/seq2seq_slide_architecture.png"],
        help="where to write Figure 6",
    )
    args = parser.parse_args()
    config = Config()
    task_figure(config, list(args.task))
    architecture_figure(config, list(args.architecture))


if __name__ == "__main__":
    main()
