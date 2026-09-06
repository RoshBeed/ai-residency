"""Which services exist, where they are, and what each one is.

Pure data, no Streamlit. Extracted from `app.py` so it can be imported and asserted
against without a Streamlit runtime, back when `app.py` ran its whole page flow at
import and the frontend was the one service with no tests at all. `st.navigation`
routes to `views/` now and every page is an ordinary function, but keeping the metadata
here is still right: it is the one thing both the pages and the deployment agree on.

`API_URLS` keys are the canonical service names, and they match the directory under
`services/`, the Deployment name in the Helm chart, and the compose service name. The
environment variable is that name uppercased with hyphens as underscores, which is what
lets the chart generate the frontend's environment from `.Values.backends` without a
per-service special case.

The defaults are localhost ports, not `http://<service>:8000`, and that is deliberate
even though service-name DNS resolves identically under compose and in the cluster.
Both deployments inject the URLs anyway -- the chart generates them, compose lists them
(and `deploy/test_workspace.py` fails if that list and the compose services disagree) --
so the default is only ever used by `make run`, a frontend started on the host against
port-forwarded backends. A container-shaped default would break exactly that case and
help nothing that already sets the variable.
"""

import os
from typing import Any

__all__ = ["ALL_PAGES", "API_URLS", "PAGES", "PROJECTS", "TAG_URLS", "tag_link"]

API_URLS = {
    "word-embeddings": os.getenv("WORD_EMBEDDINGS_API_URL", "http://localhost:8001"),
    "content-scoring": os.getenv("CONTENT_SCORING_API_URL", "http://localhost:8007"),
    "vector-search-retrieval": os.getenv(
        "VECTOR_SEARCH_RETRIEVAL_API_URL", "http://localhost:8002"
    ),
    "vit-classifier": os.getenv("VIT_CLASSIFIER_API_URL", "http://localhost:8003"),
    "vit-seq2seq": os.getenv("VIT_SEQ2SEQ_API_URL", "http://localhost:8008"),
    "multimodal-captioning": os.getenv("MULTIMODAL_CAPTIONING_API_URL", "http://localhost:8004"),
    "whisper-fine-tuning": os.getenv("WHISPER_FINE_TUNING_API_URL", "http://localhost:8005"),
    "speech-emotion-recognition": os.getenv(
        "SPEECH_EMOTION_RECOGNITION_API_URL", "http://localhost:8009"
    ),
    "rlhf-ppo": os.getenv("RLHF_PPO_API_URL", "http://localhost:8006"),
}

ALL_PAGES = {
    "word-embeddings": "Word Embeddings",
    "content-scoring": "Content Scoring",
    "vector-search-retrieval": "Vector Search & Retrieval",
    "vit-classifier": "ViT Classifier",
    "vit-seq2seq": "ViT Seq2Seq",
    "multimodal-captioning": "Multimodal Captioning",
    "whisper-fine-tuning": "Whisper Fine-Tuning",
    "speech-emotion-recognition": "Speech Emotion Recognition",
    "rlhf-ppo": "RLHF / PPO",
}


def _selected_pages() -> dict[str, str]:
    """The pages this instance should show, from `AI_RESIDENCY_PAGES`.

    Unset means all of them, which is the portfolio hub and the default. Setting it
    to one or more service names makes the same image the UI for just those, so
    running a single system does not mean looking at a sidebar of eight pages whose
    backends were never started -- and does not mean a second UI living in the
    backend image either. The frontend stays the only place Streamlit runs.

    Names are comma- or space-separated and must match `ALL_PAGES`. An unknown name
    raises rather than being skipped: a typo that silently shows the wrong set is
    worse to debug than one that refuses to start and says which names are valid.
    """
    raw = os.getenv("AI_RESIDENCY_PAGES", "").replace(",", " ").split()
    if not raw:
        return dict(ALL_PAGES)

    unknown = [name for name in raw if name not in ALL_PAGES]
    if unknown:
        raise ValueError(
            f"AI_RESIDENCY_PAGES names {unknown!r}, which are not services. "
            f"Valid names: {', '.join(ALL_PAGES)}"
        )
    # Dedupe while keeping ALL_PAGES' order, so the sidebar reads the same however
    # the variable was written.
    return {name: label for name, label in ALL_PAGES.items() if name in set(raw)}


PAGES = _selected_pages()

TAG_URLS: dict[str, str] = {
    "NLP": "https://en.wikipedia.org/wiki/Natural_language_processing",
    "Embeddings": "https://en.wikipedia.org/wiki/Word_embedding",
    "Representation Learning": "https://en.wikipedia.org/wiki/Representation_learning",
    "CBOW": "https://en.wikipedia.org/wiki/Bag-of-words_model",
    "Regression": "https://en.wikipedia.org/wiki/Regression_analysis",
    "RAG": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
    "Semantic Search": "https://en.wikipedia.org/wiki/Semantic_search",
    "Vector Search": "https://en.wikipedia.org/wiki/Nearest_neighbor_search",
    "Dual Encoder": "https://en.wikipedia.org/wiki/Embedding_(machine_learning)",
    "HNSW": "https://en.wikipedia.org/wiki/Hierarchical_navigable_small_world",
    "Computer Vision": "https://en.wikipedia.org/wiki/Computer_vision",
    "Vision Transformer": "https://en.wikipedia.org/wiki/Vision_transformer",
    "Self-Attention": "https://en.wikipedia.org/wiki/Attention_(machine_learning)",
    "Seq2Seq": "https://en.wikipedia.org/wiki/Seq2seq",
    "Multimodal": "https://en.wikipedia.org/wiki/Multimodal_learning",
    "Vision-Language": "https://en.wikipedia.org/wiki/Vision-language_model",
    "CLIP": "https://en.wikipedia.org/wiki/Contrastive_Language%E2%80%93Image_Pre-training",
    "Adapter Fine-tuning": "https://en.wikipedia.org/wiki/LoRA_(machine_learning)",
    "ASR": "https://en.wikipedia.org/wiki/Speech_recognition",
    "Speech Recognition": "https://en.wikipedia.org/wiki/Speech_recognition",
    "Fine-tuning": "https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning)",
    "Transfer Learning": "https://en.wikipedia.org/wiki/Transfer_learning",
    "RLHF": "https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback",
    "LLM Alignment": "https://en.wikipedia.org/wiki/AI_alignment",
    "PPO": "https://en.wikipedia.org/wiki/Proximal_policy_optimization",
    "GAE": "https://en.wikipedia.org/wiki/Temporal_difference_learning",
    "Reward Modelling": "https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback#Reward_model",
}


def tag_link(tag: str) -> str:
    """Render one technique tag, linked to a reference when there is one for it."""
    url = TAG_URLS.get(tag)
    if url:
        return f'<a href="{url}" target="_blank" class="tag-chip tag-link">{tag}</a>'
    return f'<span class="tag-chip">{tag}</span>'


# ---------------------------------------------------------------------------
# Project metadata used for the home-page card grid and the tech-depth sidebar
# on each project's demo page.  Real numbers pulled from actual training logs,
# W&B runs, and each project's own README -- nothing invented.
# ---------------------------------------------------------------------------
PROJECTS: dict[str, dict[str, Any]] = {
    "word-embeddings": {
        "icon": "🧬",
        "title": "Word Embeddings",
        "subtitle": "CBOW embeddings trained from scratch, negative sampling",
        "description": (
            "Trains 300-dimensional word vectors from raw text with skip-gram and "
            "CBOW architectures and negative sampling — no pretrained embeddings, "
            "no HuggingFace model."
        ),
        "techniques": ["NLP", "Embeddings", "Representation Learning", "CBOW", "Skip-gram"],
        "papers": [
            (
                "Mikolov et al.",
                "Efficient Estimation of Word Representations in Vector Space",
                "https://arxiv.org/abs/1301.3781",
            ),
        ],
        "datasets": [
            ("text8", "https://huggingface.co/datasets/roshbeed/ai-residency-text8"),
            (
                "Eval benchmarks",
                "https://huggingface.co/datasets/roshbeed/ai-residency-word-embeddings-eval",
            ),
        ],
        "model_links": [
            (
                "Word vectors",
                "https://huggingface.co/roshbeed/ai-residency-word-embeddings",
            ),
        ],
        "stats": [
            ("Vocabulary", "71,290 words"),
            ("Embedding dim", "300"),
            ("WordSim-353", "0.73 rho"),
            ("Google analogies", "46.4%"),
        ],
        "intro": (
            "Trains word2vec from scratch on text8 — 17M tokens of cleaned Wikipedia, the "
            "standard benchmark corpus for this task. Both architectures are trained: CBOW "
            "averages the context to predict the centre word, skip-gram predicts each context "
            "word from the centre. Both use negative sampling over a 71,290-word "
            "vocabulary.\n\n"
            "**Approach.** The negative-sampling loss, the unigram^0.75 sampler, frequent-word "
            "subsampling and the batching are all implemented directly — no gensim, no "
            "pretrained vectors. A 48-run Bayesian sweep through W&B picked the published "
            "configuration, ranked on benchmark score rather than loss.\n\n"
            "**Results.** Spearman 0.73 on WordSim-353, 0.30 on SimLex-999 and 46.4% on the "
            "Google analogy set — at or above published word2vec results for this corpus. "
            "Skip-gram is served; CBOW is published alongside it.\n\n"
            "**This page** calls the nearest-neighbour and analogy API: look up a word's "
            "closest neighbours in the learned space, or complete the canonical "
            "`king - man + woman` analogy."
        ),
        "docs_path": "services/word-embeddings/README.md",
        "diagrams": [
            (
                "assets/word-embeddings/fig9-cbow-forward-pass.png",
                "Annotated CBOW forward pass — context word embeddings are averaged, then "
                "scored against every word in the vocabulary to produce a probability distribution.",
            ),
            (
                "assets/word-embeddings/fig3-late-early-fusion.png",
                "Late fusion vs early fusion — late fusion averages word embeddings first, then "
                "feeds them to a downstream head. This portfolio uses late fusion.",
            ),
        ],
        "model_specs": [
            "**Skip-gram** (served) — 71,290 × 300 embedding matrices (in + out), context "
            "window 8, 10 negative samples. 42.8M parameters trained from scratch on text8.",
            "**CBOW** — the same two matrices, context window 10, 15 negative samples. Wins "
            "WordSim-353 (0.730 against 0.726) and loses the analogy set by 5.6 points.",
        ],
    },
    "content-scoring": {
        "icon": "📝",
        "title": "Content Scoring",
        "subtitle": "Early vs late fusion of title text and post metadata, on 872k HN posts",
        "description": (
            "Predicts a Hacker News post's score from its title and metadata, four ways: "
            "text alone, metadata alone, early fusion and late fusion — which score the "
            "same, because the task turns out to be additive."
        ),
        "techniques": ["NLP", "Embeddings", "Regression", "Transfer Learning", "Feature Fusion"],
        "papers": [
            (
                "Mikolov et al.",
                "Efficient Estimation of Word Representations in Vector Space",
                "https://arxiv.org/abs/1301.3781",
            ),
        ],
        "datasets": [
            ("HN Posts (872k)", "https://huggingface.co/datasets/roshbeed/ai-residency-hn-posts"),
        ],
        "model_links": [
            (
                "Predictors + extended vectors",
                "https://huggingface.co/roshbeed/ai-residency-content-scoring",
            ),
        ],
        "stats": [
            ("Posts", "872,554"),
            ("Split", "time-based, 711k/93k/69k"),
            ("Vocabulary", "71,290 → 86,438"),
            ("Title OOV", "11.3% → 3.2%"),
        ],
        "intro": (
            "Predicts a Hacker News post's score from its title and the things you know "
            "when you submit it — when, where it links, how the title is written, and how "
            "the author's earlier posts did. Built on the word vectors trained from "
            "scratch in the sibling word-embeddings project, with the vocabulary extended "
            "onto 710k Hacker News titles: text8 has no `gpt-4`, no `kubernetes`, no "
            "`postgres`, and adding them takes out-of-vocabulary tokens from 11.3% of a "
            "title to 3.2%.\n\n"
            "**The question is where the title and the metadata should meet.** Early "
            "fusion concatenates everything and hands one MLP the lot, free to find any "
            "interaction but obliged to discover the group structure itself. Late fusion "
            "gives each group its own tower and joins them at the head. Both are compared "
            "against the two ablations that make the comparison mean anything — text "
            "alone, and metadata alone — and against TF-IDF + Ridge, which is the "
            "baseline that decides whether any of it earned its keep.\n\n"
            "**Split by time, not at random.** Scores drift and same-day posts share a "
            "front page, so a random split lets a model see the future of its own test "
            "set. Author history is computed on a strictly backward-looking window for "
            "the same reason.\n\n"
            "**This page** calls `/compare`, which runs all four architectures over the "
            "post you type, and `/predict/stream`, which relays one architecture's "
            "pipeline a stage at a time."
        ),
        "docs_path": "services/content-scoring/README.md",
        "diagrams": [
            (
                "assets/content-scoring/fusion-architectures.png",
                "The first layer is the only place early and late fusion differ: early wires "
                "every input group to every hidden unit, late keeps the five blocks on the "
                "diagonal. The shared layer underneath is identical in both.",
            ),
            (
                "assets/word-embeddings/fig3-late-early-fusion.png",
                "Early fusion versus late fusion — this service builds both and compares them "
                "head to head, alongside the text-only and features-only ablations.",
            ),
        ],
        "model_specs": [
            "**Early fusion** (served) — 150,593 parameters. A dense `329 → 256` first "
            "layer, then a shared `256 → 256` and the head. The 329 is the 300-d pooled "
            "title plus 21 numeric features plus the 8-d domain embedding.",
            "**Late fusion** — 136,467 parameters. The same layer with the cross-group "
            "weights removed: `300→233 | 6→5 | 9→7 | 11→9 | 3→2`, so each group reaches "
            "only its own hidden units. Identical shared layer and head, so the two differ "
            "in connectivity alone.",
            "**Text only** — 143,169 parameters, `300 → 256` then the shared stack.",
            "**Features only** — 73,793 parameters, `29 → 256` then the shared stack.",
            "**Domain embedding** — `1003 × 8`, shared by all four. Domains seen fewer than "
            "50 times in training collapse to one `<rare>` row.",
            "**Word vectors** — `86,438 × 300` skip-gram, text8 continued on the Hacker News "
            "title corpus. Frozen, and kept out of the checkpoints: all four models share "
            "one 104MB matrix.",
        ],
    },
    "vector-search-retrieval": {
        "icon": "🔍",
        "title": "Vector Search & Retrieval",
        "subtitle": "Dual-encoder GRU architecture, triplet loss, Redis HNSW indexing",
        "description": (
            "Retrieves relevant documents through a dual-encoder architecture "
            "that maps queries and documents into a shared embedding space, "
            "indexed for fast approximate nearest-neighbour lookup."
        ),
        "techniques": ["RAG", "Semantic Search", "Vector Search", "Dual Encoder", "HNSW"],
        "papers": [
            (
                "Bromley et al.",
                "Signature Verification using a Siamese Time Delay Neural Network",
                "https://proceedings.neurips.cc/paper/1993/hash/288cc0ff022877bd3df94bc9360b9c5d-Abstract.html",
            ),
        ],
        "datasets": [("MS MARCO", "https://huggingface.co/datasets/microsoft/ms_marco")],
        "model_links": [],
        "stats": [
            ("Corpus", "MS MARCO (8.8M passages)"),
            ("Vocabulary", "7,692 words"),
            ("Embedding dim", "128"),
            ("Ranking accuracy", "52.0% pairwise"),
        ],
        "intro": (
            "Retrieves relevant documents from MS MARCO's 8.8M passages through a "
            "dual-encoder architecture: a GRU query tower and a GRU document tower that "
            "map queries and documents into a shared embedding space, indexed in Redis "
            "via HNSW for approximate nearest-neighbour lookup.\n\n"
            "**Architecture.** CBOW word embeddings (trained on MS MARCO's vocabulary) "
            "feed both towers. The towers are trained with triplet loss on (query, "
            "relevant passage, irrelevant passage) triples — the loss pulls the query "
            "and relevant document embeddings together while pushing the irrelevant one "
            "away. Once trained, the document tower embeds every passage in the corpus "
            "once, offline, into a Redis HNSW index. At query time, only the query "
            "tower runs — one forward pass, then a cosine-distance ANN search over the "
            "index. Redis is a real runtime dependency throughout, not optional.\n\n"
            "**Approach.** Late fusion — separate query and document towers, combined "
            "only at the cosine-distance comparison — means the document tower runs once "
            "offline regardless of how many queries come in. This is the defining "
            "architectural decision behind dual-encoder retrieval: a query completes in "
            "milliseconds where a cross-encoder would need to re-encode every candidate "
            "pair. The towers are designed to support MPS acceleration where it helps "
            "(CBOW embeddings) and intentionally stay on CPU where dispatch overhead on "
            "tiny tensors makes the GPU slower (GRU training on single triples)."
        ),
        "docs_path": "services/vector-search-retrieval/README.md",
        "diagrams": [
            (
                "assets/dual-encoder-retrieval/fig6-offline-online-stages.png",
                "Offline pipeline (train towers, embed corpus, index in Redis) and online query "
                "flow (encode query → ANN search → ranked results).",
            ),
            (
                "assets/dual-encoder-retrieval/fig5-triplet-loss.png",
                "Triplet loss — pulls the query and relevant document embeddings together while "
                "pushing the irrelevant one apart, with a margin to prevent trivial solutions.",
            ),
        ],
        "model_specs": [
            "**GRU towers** — single-layer GRU, hidden size 128, embedding dim 128. "
            "~99K parameters per tower, sharing frozen CBOW embeddings from a 7,692-word vocabulary.",
        ],
    },
    "vit-classifier": {
        "icon": "🔢",
        "title": "ViT Classifier",
        "subtitle": "Encoder-only Vision Transformer for a single MNIST digit, no torchvision",
        "description": (
            "Patch embeddings and multi-head self-attention, written from scratch, "
            "classifying one handwritten digit."
        ),
        "techniques": ["Computer Vision", "Vision Transformer", "Self-Attention"],
        "papers": [
            (
                "Dosovitskiy et al.",
                "An Image is Worth 16x16 Words",
                "https://arxiv.org/abs/2010.11929",
            ),
            ("Vaswani et al.", "Attention Is All You Need", "https://arxiv.org/abs/1706.03762"),
        ],
        "datasets": [("MNIST", "https://huggingface.co/datasets/mnist")],
        "model_links": [],
        "stats": [
            ("Test accuracy", "91.78%"),
            ("Majority baseline", "11.35%"),
            ("Trained for", "1 epoch"),
        ],
        "intro": (
            "An image is not a sequence, so the first thing a Vision Transformer has to do "
            "is invent one: cut the image into fixed-size patches, project each into a "
            "vector, and hand the transformer a sequence of patch tokens the way a language "
            "model gets word tokens. Everything after that is the transformer architecture "
            "applied unchanged to a new modality.\n\n"
            "**Architecture.** Images are split into 4x4 patches, linearly projected, and fed "
            "through six multi-head self-attention blocks with post-norm, matching the "
            "original Transformer paper rather than ViT's pre-norm. A [CLS] token and MLP "
            "head produce the ten-way prediction.\n\n"
            "**Shared blocks.** The attention, encoder and patch-embedding modules live in "
            "`libs/vision-transformer`, shared with the seq2seq model rather than duplicated "
            "— the two models are the same blocks wired differently."
        ),
        "docs_path": "services/vit-classifier/README.md",
        "diagrams": [
            (
                "assets/vision-transformer/task_overview.png",
                "Recognizing one digit versus recognizing several — the two tasks these "
                "models split between them.",
            ),
        ],
        "model_specs": [
            "6 encoder layers, d_model=128, 8 heads, FF hidden=512, 4x4 patches. ~1.7M parameters.",
            "Built from scratch — no `torch.nn.Transformer` shortcuts.",
        ],
    },
    "vit-seq2seq": {
        "icon": "🔠",
        "title": "ViT Seq2Seq",
        "subtitle": "Encoder-decoder Vision Transformer reading multi-digit numbers",
        "description": (
            "An encoder reads an image of three digits; a causal decoder generates the "
            "number one digit at a time, until it decides it is finished."
        ),
        "techniques": ["Computer Vision", "Vision Transformer", "Seq2Seq", "Cross-Attention"],
        "papers": [
            (
                "Dosovitskiy et al.",
                "An Image is Worth 16x16 Words",
                "https://arxiv.org/abs/2010.11929",
            ),
            ("Vaswani et al.", "Attention Is All You Need", "https://arxiv.org/abs/1706.03762"),
        ],
        "datasets": [("MNIST", "https://huggingface.co/datasets/mnist")],
        "model_links": [],
        "stats": [
            ("Exact-sequence match", "68.0%"),
            ("Random baseline", "0.10%"),
            ("Held-out sequences", "500"),
        ],
        "intro": (
            "Classifying one digit needs a single label. Reading a number needs an ordered "
            "sequence, of a length the model decides on, which is a different problem: the "
            "model has to know when it has finished.\n\n"
            "**Architecture.** The encoder is the same stack the classifier uses, over 16x16 "
            "patches of an 84x84 composite. The decoder adds a start token, a causal mask so "
            "each position only sees what came before it, cross-attention onto the encoded "
            "image, and an EOS token it emits when the number is done.\n\n"
            "**Per-sequence termination.** Generation tracks completion per sequence within a "
            "batch. Stopping the whole batch at the first EOS silently truncates every other "
            "sequence still decoding — including inside the trainer's own evaluation loop, "
            "which makes the reported accuracy wrong rather than merely noisy."
        ),
        "docs_path": "services/vit-seq2seq/README.md",
        "diagrams": [
            (
                "assets/vision-transformer/seq2seq_slide_architecture.png",
                "Encoder-decoder with cross-attention — the decoder attends to encoder output "
                "at each generation step, producing digits token by token until EOS.",
            ),
        ],
        "model_specs": [
            "6 encoder + 6 decoder layers, d_model=256, 8 heads, FF hidden=1024, 16x16 patches. "
            "~7.2M parameters. Vocab of 12 tokens (0-9 + START + EOS), max sequence length 5.",
            "Built from scratch — no `torch.nn.Transformer` shortcuts.",
        ],
    },
    "multimodal-captioning": {
        "icon": "🖼️",
        "title": "Multimodal Captioning",
        "subtitle": "CLIP ViT‑B/32 encoder, trained adapter, Qwen3 decoder pipeline",
        "description": (
            "Projects images into a language model's input space through a "
            "trained adapter, with both vision and language backbones kept "
            "frozen while the adapter learns to bridge the two modalities."
        ),
        "techniques": ["Multimodal", "Vision-Language", "CLIP", "Adapter Fine-tuning"],
        "papers": [
            (
                "Radford et al.",
                "Learning Transferable Visual Models From Natural Language Supervision",
                "https://arxiv.org/abs/2103.00020",
            ),
        ],
        "datasets": [("Flickr8k", "https://huggingface.co/datasets/nlphuji/flickr8k")],
        "model_links": [
            ("CLIP ViT‑B/32", "https://huggingface.co/openai/clip-vit-base-patch32"),
            ("Qwen3‑0.6B", "https://huggingface.co/Qwen/Qwen3-0.6B"),
        ],
        "stats": [
            ("Vision encoder", "CLIP ViT-B/32"),
            ("Decoder", "Qwen3-0.6B"),
            ("Adapter params", "<1M (learned)"),
            ("Training", "CPU-only"),
        ],
        "intro": (
            "Image captioning by bridging a frozen CLIP ViT-B/32 vision encoder to a "
            "frozen Qwen3-0.6B language model through a small trainable adapter. The "
            "adapter projects CLIP's image embedding into Qwen3's token space as a "
            "single soft token, prepended to a text prompt before autoregressive "
            "decoding — the same shape LLaVA and PaLiGemma use.\n\n"
            "**Architecture.** The adapter is a compact projection layer that maps "
            "CLIP's 512-dim vision embeddings to Qwen3's 896-dim hidden size. During "
            "training, only the adapter's weights are updated — both the vision encoder "
            "and the 596M-parameter language model stay frozen. The loss is standard "
            "cross-entropy on caption tokens. The intended training pipeline uses "
            "batched data loading, gradient accumulation, and checkpoint persistence "
            "with planned expansion to synthetically generated captions from a larger "
            "vision-language model. The repo also includes a standalone Part 1 for CLIP "
            "zero-shot classification over sample images.\n\n"
            "**Approach.** This is transfer learning at its most efficient — two "
            "pre-trained backbones, neither modified, communicating through a learned "
            "bridge of under a million parameters. The pattern (frozen encoder → trained "
            "projection → frozen decoder) is the same one behind most production "
            "vision-language systems. The design supports training on Flickr30k with "
            "planned expansion to VQAv2 and synthetic data."
        ),
        "docs_path": "services/multimodal-captioning/README.md",
        "diagrams": [
            (
                "assets/image-captioning/fig3-paligemma-architecture.png",
                "The PaliGemma architecture — SigLIP vision encoder → linear projection → "
                "Gemma decoder. The canonical reference pattern: a frozen vision backbone "
                "feeds a frozen language model through a learned projection layer.",
            ),
            (
                "assets/image-captioning/fig1-clip-contrastive-pretraining.png",
                "CLIP contrastive pretraining — image and text encoders are trained jointly to "
                "maximise cosine similarity between matched pairs while minimising it for "
                "unmatched ones.",
            ),
        ],
        "model_specs": [
            "**CLIP ViT-B/32** (frozen) — 512-dim image embeddings, 12 layers, 12 heads.",
            "**Adapter** — Linear(512→1024) → LayerNorm → GELU → Linear(1024→1024). "
            "~1.6M trainable parameters.",
            "**Qwen3-0.6B** (frozen) — 28 layers, 16 attn heads (8 KV, GQA), FFN hidden 3072, "
            "vocab 151,936. ~598M total parameters, only the adapter is trained.",
        ],
    },
    "whisper-fine-tuning": {
        "icon": "🎙️",
        "title": "Whisper Fine-Tuning",
        "subtitle": "Whisper‑tiny, real gradient updates per request, transcript convergence",
        "description": (
            "Fine-tunes Whisper on a single clip, live, per request. Five real "
            "gradient steps run while you watch, each one streamed with its loss and "
            "the transcript the updated weights now produce."
        ),
        "techniques": ["ASR", "Speech Recognition", "Fine-tuning", "Transfer Learning"],
        "papers": [
            (
                "Radford et al.",
                "Robust Speech Recognition via Large-Scale Weak Supervision",
                "https://arxiv.org/abs/2212.04356",
            ),
        ],
        "datasets": [
            ("EmoDB", "http://emodb.bilderbar.info/"),
            ("CREMA-D", "https://github.com/CheyneyComputerScience/CREMA-D"),
        ],
        "model_links": [
            ("Whisper‑tiny", "https://huggingface.co/openai/whisper-tiny"),
        ],
        "stats": [
            ("Base model", "Whisper-tiny (39M params)"),
            ("Datasets", "EmoDB + CREMA-D"),
            ("Fine-tune steps", "5 per request"),
            ("Verified result", '"My name is Roosh" → "My name is Rosh"'),
        ],
        "intro": (
            "Extends OpenAI's Whisper-tiny with emotion tokens while preserving its "
            "transcription ability. Fine-tunes the model in real time per request — "
            "genuine gradient updates on live weights, with loss tracking and per-step "
            "transcript diffs showing convergence step by step.\n\n"
            "**Architecture.** The system has two parts. Part 1 (the live demo) loads a "
            "fresh Whisper-tiny, transcribes input audio for a baseline, then runs "
            "gradient steps fine-tuning toward a user-provided target text with loss and "
            "transcript streamed back via SSE. Part 2 (the data pipeline) prepares "
            "speech-emotion datasets (CREMA-D, EmoDB) with per-file emotion labels. The "
            "intended goal is to add special emotion tokens to Whisper's tokenizer and "
            "fine-tune on emotion-labelled speech, with the transcription task preserved "
            "through the loss function to avoid catastrophic forgetting.\n\n"
            "**Approach.** Audio flows through the full speech pipeline: raw waveform → "
            "Fourier transform → mel spectrogram → Whisper encoder. The real-time "
            "fine-tuning demonstrates that even a few gradient steps can meaningfully "
            "shift a pre-trained model's output for a specific target. The emotion-token "
            "extension explores the practical reality of modifying a pre-trained "
            "tokenizer and the tradeoff between task specialization and preserving "
            "original capabilities. "
            "(Whisper shells out to it for audio decoding)."
        ),
        "docs_path": "services/whisper-fine-tuning/README.md",
        "diagrams": [
            (
                "assets/whisper-finetuning/fig2-whisper-architecture.png",
                "Whisper encoder-decoder architecture — log-mel spectrogram input, encoder "
                "blocks, cross-attention decoder producing text tokens.",
            ),
            (
                "assets/whisper-finetuning/fig3-multitask-format.png",
                "Whisper's multitask training format — a single model handles transcription, "
                "translation, language identification, and voice activity detection through "
                "special tokens prepended to the decoder input.",
            ),
        ],
        "model_specs": [
            "**Whisper-tiny** — 4 encoder + 4 decoder layers, d_model=384, 6 attention heads, "
            "80 mel bins. ~39M parameters.",
            "**Custom tokens** — 6 emotion tokens injected into the decoder vocabulary: "
            "`⟨emotion_happy⟩`, `⟨emotion_sad⟩`, `⟨emotion_fearful⟩`, `⟨emotion_neutral⟩`, "
            "`⟨emotion_angry⟩`, `⟨emotion_disgust⟩`.",
        ],
    },
    "speech-emotion-recognition": {
        "icon": "😐",
        "title": "Speech Emotion Recognition",
        "subtitle": "Whisper fine-tuned to emit an emotion token, not a separate classifier head",
        "description": (
            "Predicts the emotion in a clip through the same mechanism Whisper already "
            "uses for language and task: a special token its decoder emits."
        ),
        "techniques": ["ASR", "Speech Emotion Recognition", "Fine-tuning", "Tokenization"],
        "papers": [
            (
                "Radford et al.",
                "Robust Speech Recognition via Large-Scale Weak Supervision",
                "https://arxiv.org/abs/2212.04356",
            ),
        ],
        "datasets": [
            ("EmoDB", "http://emodb.bilderbar.info/"),
            ("CREMA-D", "https://github.com/CheyneyComputerScience/CREMA-D"),
        ],
        "model_links": [
            ("Whisper-small", "https://huggingface.co/openai/whisper-small"),
        ],
        "stats": [
            ("Emotions", "6"),
            ("Corpora", "EmoDB + CREMA-D"),
            ("Baseline", "majority emotion"),
        ],
        "intro": (
            "Emotion could be a classifier head bolted onto the encoder. It is not, here. "
            "Whisper's decoder already emits special tokens for language and task, so "
            "adding one per emotion to the same vocabulary lets the model say which "
            "emotion it heard through the mechanism it already has.\n\n"
            "**The pipeline.** Two corpora, each encoding the emotion inside the audio "
            "filename in a different format at a different position. That is resolved "
            "once, in configuration, into manifests of `audio_path, label` — the whole "
            "interface between the data pipeline and training.\n\n"
            "**What the target is.** EmoDB and CREMA-D are fixed-sentence corpora: every "
            "speaker reads the same handful of sentences and what varies is delivery. So "
            "the manifests carry no transcript, and the training target is the emotion "
            "token alone rather than a transcript with an emotion attached.\n\n"
            "**Baseline.** Six emotions are not evenly represented, so always answering "
            "the most common one is a genuinely competitive score on a corpus this size. "
            "That is what the model is reported against."
        ),
        "docs_path": "services/speech-emotion-recognition/README.md",
        "diagrams": [],
        "model_specs": [
            "Whisper-small, with the tokenizer extended by one special token per emotion and "
            "the embedding table resized to match.",
            "The emotion is read off the decoder's first step, which gives every emotion's "
            "probability rather than only the winner.",
        ],
    },
    "rlhf-ppo": {
        "icon": "🎯",
        "title": "RLHF / PPO",
        "subtitle": "From-scratch PPO, GAE, clip objective, trained reward model",
        "description": (
            "Aligns a language model through PPO with a trained reward model "
            "to improve text summarisation. Generation, GAE advantage estimation, "
            "and the clipped policy update are all traced per token in real time."
        ),
        "techniques": ["RLHF", "LLM Alignment", "PPO", "GAE", "Reward Modelling"],
        "papers": [
            ("Schulman et al.", "Proximal Policy Optimization", "https://arxiv.org/abs/1707.06347"),
        ],
        "datasets": [("TL;DR", "https://huggingface.co/datasets/openai/summarize_from_feedback")],
        "model_links": [
            ("SmolLM2‑135M", "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct"),
            ("DeBERTa‑v3", "https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2"),
        ],
        "stats": [
            ("Base model", "SmolLM2-135M + LoRA"),
            ("Reward model", "OpenAssistant DeBERTa-v3"),
            ("Pipeline", "12 stages per step"),
            ("Checkpoints", "Policy + value persist across restarts"),
        ],
        "intro": (
            "A modular, from-scratch implementation of Proximal Policy Optimization for "
            "aligning a language model through human feedback — the full RLHF pipeline "
            "from generation through advantage estimation to the clipped policy gradient "
            "update.\n\n"
            "**Architecture.** The intended full pipeline has three stages: (1) "
            "supervised fine-tuning of a base model on a summarisation dataset, (2) "
            "training a reward model on human preference data to score completions, "
            "(3) the PPO loop optimizing the policy against the reward model's signal. "
            "The implementation focuses on stage 3: a `PPOTrainer.ppo_step()` "
            "orchestrating 12 independently testable modules — policy sampling "
            "(LoRA-wrapped SmolLM2-135M), reference-model KL baseline, reward scoring "
            "(DeBERTa-v3), KL-penalized reward, value function estimation, temporal "
            "difference error, GAE advantage estimation with λ-return, rewards-to-go, "
            "value function MSE loss, and the PPO-clip objective on the policy. Four "
            "models coexist per step: policy, reference, reward, and value.\n\n"
            "**Approach.** Each of the 12 stages is an independently testable module, "
            "making the full PPO algorithm inspectable step by step. The pipeline traces "
            "every intermediate quantity per token — log-probabilities, KL divergence, "
            "advantages, the policy ratio before and after clipping — so the "
            "optimization dynamics are visible rather than black-box. The PPO-clip "
            "mechanism, KL penalty for preventing reward hacking, and GAE's bias-"
            "variance tradeoff are all implemented directly from the original papers. "
            "The SFT and reward-model training stages are designed as follow-on work on "
            "preference datasets."
        ),
        "docs_path": "services/rlhf-ppo/README.md",
        "diagrams": [
            (
                "assets/rlhf-ppo/fig1-rlhf-three-stage.png",
                "The RLHF pipeline — human feedback → reward model training → PPO policy "
                "optimisation with four models active per step (policy, reference, reward, value).",
            ),
            (
                "assets/rlhf-ppo/ppo-workflow.png",
                "PPO workflow — sampling from the environment → GAE advantage estimation → "
                "loss computation → policy gradient update. The iterative loop that trains the "
                "policy model through exploration and exploitation.",
            ),
        ],
        "model_specs": [
            "**SmolLM2-135M** (LoRA-wrapped) — 30 layers, d_model=576, 9 heads, FFN hidden 2304. ~135M base parameters.",
            "**LoRA** — rank 16, alpha 32, applied to all 4 attention projections (q, k, v, o). ~1.6M trainable per model copy.",
            "**Four models** — policy, reference, reward, value — share the LoRA-wrapped base. "
            "Reward and value add a Linear(576→1) scalar head. Total ~6.4M LoRA parameters across all copies.",
        ],
    },
}
