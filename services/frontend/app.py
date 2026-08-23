"""
ML systems portfolio -- a single Streamlit frontend that calls every
backend APIs directly over HTTP. Each page below is a thin client around one
project's real, already-verified endpoint contract -- no inference logic lives
here, only request/response wiring and display.

Every page also renders a real trace of what the backend actually did to
produce its answer (which words were recognised, per-token probabilities,
a live fine-tuning loss curve, the PPO policy's sampling candidates at each
generation step, ...) rather than only the final result. That data comes
straight from each project's own API response -- nothing here recomputes or
simulates it.
"""

import base64
import difflib
import html
import io
import json
import math
import os
from collections.abc import Generator
from itertools import pairwise
from typing import Any

import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageDraw
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="ML Systems Portfolio", page_icon="🧩", layout="wide")

LOGO_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 96 96">
  <rect width="96" height="96" rx="18" fill="#2a78d6"/>
  <text x="48" y="38" text-anchor="middle" fill="white" font-size="22" font-weight="600" font-family="system-ui,-apple-system,sans-serif">ML</text>
  <text x="48" y="70" text-anchor="middle" fill="#b7d3f6" font-size="14" font-family="system-ui,-apple-system,sans-serif">folio</text>
</svg>"""
LOGO_URI = f"data:image/svg+xml;base64,{base64.b64encode(LOGO_SVG.encode()).decode()}"
st.logo(LOGO_URI)

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

PAGES = {
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


def tag_link(t: str) -> str:
    url = TAG_URLS.get(t)
    if url:
        return f'<a href="{url}" target="_blank" class="tag-chip tag-link">{t}</a>'
    return f'<span class="tag-chip">{t}</span>'


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
            "Trains 128-dimensional word vectors from raw text with a "
            "continuous bag-of-words architecture and negative sampling — no "
            "pretrained embeddings, no HuggingFace model."
        ),
        "techniques": ["NLP", "Embeddings", "Representation Learning", "CBOW"],
        "papers": [
            (
                "Mikolov et al.",
                "Efficient Estimation of Word Representations in Vector Space",
                "https://arxiv.org/abs/1301.3781",
            ),
        ],
        "datasets": [("text8", "https://huggingface.co/datasets/ardMLX/text8"), ("HN Posts", "#")],
        "model_links": [],
        "stats": [
            ("Vocabulary", "258,196 words"),
            ("Embedding dim", "128"),
            ("Corpus", "text8 (100M chars)"),
            ("Window", "4"),
        ],
        "intro": (
            "Trains word2vec CBOW embeddings from scratch. Stage 1 builds a vocabulary "
            "from the text8 corpus (100M characters of cleaned Wikipedia text) and merges "
            "it with Hacker News post titles. Stage 2 pretrains CBOW embeddings — "
            "128-dimensional vectors trained with negative sampling over the merged corpus.\n\n"
            "**Approach.** The embedding stage is a genuine from-scratch training run. The "
            "negative-sampling loss, the vocabulary merger handling token-lemma mapping, and "
            "the bag-of-words averaging are all implemented directly. Hyperparameter sweeps "
            "over embedding dimension, context window, and learning rate are driven through "
            "W&B. The vectors are published to the Hugging Face Hub, where the content-scoring "
            "project consumes them frozen — this is the foundation for understanding how text "
            "becomes vectors.\n\n"
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
            "**CBOW** — vocab × 128 embedding matrices (in + out), context window 4, "
            "15 negative samples. ~65M parameters trained from scratch on text8.",
        ],
    },
    "content-scoring": {
        "icon": "📝",
        "title": "Content Scoring",
        "subtitle": "Regression head over frozen embeddings, predict an engagement score",
        "description": (
            "Predicts an engagement score from a title, using word vectors frozen "
            "from the word-embeddings project and a small regression head."
        ),
        "techniques": ["NLP", "Embeddings", "Regression", "Transfer Learning"],
        "papers": [
            (
                "Mikolov et al.",
                "Efficient Estimation of Word Representations in Vector Space",
                "https://arxiv.org/abs/1301.3781",
            ),
        ],
        "datasets": [("HN Posts", "#")],
        "model_links": [],
        "stats": [
            ("Labeled examples", "782 HN posts"),
            ("Embedding dim", "128"),
            ("Head", "128 → 64 → 1"),
            ("Baseline", "predict-the-mean MSE"),
        ],
        "intro": (
            "Predicts an engagement score from a post's title, on top of word embeddings "
            "trained from scratch in the sibling word-embeddings project. The vectors are "
            "frozen — pulled from that project's Hugging Face repo — and only the regression "
            "head is trained.\n\n"
            "**Honest result.** With 782 labelled posts, the trained head does not clearly "
            "beat the predict-the-mean baseline (test MSE 3,641 vs 3,596). The write-up "
            "says so directly rather than dressing it up. What the pipeline demonstrates is "
            "the transfer: frozen embeddings → averaged title vector → scalar head, the "
            "same shape as a reward model, trained here on engagement counts rather than "
            "human preferences.\n\n"
            "**This page** calls the scoring API, which embeds each query word-by-word, "
            "averages the vectors, and predicts a score — streamed live so each stage of "
            "the inference pipeline is visible in real time."
        ),
        "docs_path": "services/content-scoring/README.md",
        "diagrams": [
            (
                "assets/word-embeddings/fig3-late-early-fusion.png",
                "Late fusion vs early fusion — this project uses late fusion: average the word "
                "vectors, then feed one vector to the regression head.",
            ),
        ],
        "model_specs": [
            "**Scorer** — 128 → 128 → 64 → 1 MLP with ReLU and dropout 0.2. "
            "~17K parameters trained on 782 labelled HN posts.",
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
            "Extends a speech model with emotion tokens while preserving its "
            "transcription ability. Real gradient updates run per request with "
            "live loss tracking and transcript convergence."
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
            "original capabilities. The system `ffmpeg` binary is a hard prerequisite "
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

# ---------------------------------------------------------------------------
# Palette + shared component CSS, injected once. Sequential magnitude (word/
# token/digit confidence, patch-grid lines) stays one hue (blue) light->dark;
# known/unknown is a status pair (green/red), always paired with an icon so
# it's never color-alone. Both light and dark surfaces are declared, keyed
# off prefers-color-scheme, so this reads correctly in either OS theme.
# ---------------------------------------------------------------------------
PALETTE_CSS = """
<style>
/* Reduce default top padding so cards sit above the fold */
.main .block-container { padding-top: 1rem !important; }
h1 { text-align: center; }

/* Equal-height cards: force Streamlit columns to stretch */
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
    display: flex;
    flex-direction: column;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] > div:first-child {
    flex: 1;
}

:root {
  --surface-1: #fcfcfb;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --text-muted: #898781;
  --gridline: #e1e0d9;
  --series-1: #2a78d6;
  --series-1-muted: #b7d3f6;
  --good: #0ca30c;
  --critical: #d03b3b;
  --chip-bg: #f9f9f7;
  --border: rgba(11,11,11,0.10);
  --tag-technique-bg: #d4ddf7;
  --tag-technique-fg: #2a4cc7;
  --tag-model-bg: #f5e6d3;
  --tag-model-fg: #b86e1c;
  --tag-paper-bg: #e8e0f0;
  --tag-paper-fg: #6b4c8a;
  --tag-data-bg: #d4ece2;
  --tag-data-fg: #2d7a5f;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface-1: #1a1a19;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted: #898781;
    --gridline: #2c2c2a;
    --series-1: #3987e5;
    --series-1-muted: #184f95;
    --good: #0ca30c;
    --critical: #e66767;
    --chip-bg: #202020;
    --border: rgba(255,255,255,0.10);
    --tag-technique-bg: #1f387a;
    --tag-technique-fg: #8dbcf7;
    --tag-model-bg: #4d3514;
    --tag-model-fg: #f0b84d;
    --tag-paper-bg: #3a284d;
    --tag-paper-fg: #c4a3de;
    --tag-data-bg: #1f4733;
    --tag-data-fg: #6fd69a;
  }
}

/* ---- Home page: hero, cards, tags ---- */

.hero-section { text-align: center; padding: 2.5rem 0 1.5rem; }
.hero-title { font-size: 2.6rem; font-weight: 700; color: var(--text-primary); margin-bottom: 0.4rem; letter-spacing: -0.02em; }
.hero-tagline { font-size: 0.95rem; color: var(--text-secondary); max-width: 720px; line-height: 1.55; margin: 0 auto 1.5rem; }

.card { border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px 10px; background: var(--chip-bg); width: 100%; min-width: 0; box-sizing: border-box; }
.card-header { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
.card-icon { font-size: 1.3rem; flex-shrink: 0; }
.card-title { font-size: 0.95rem; font-weight: 600; color: var(--text-primary); }
.card-desc { font-size: 0.8rem; color: var(--text-secondary); line-height: 1.45; margin-bottom: 10px; }
.tag-row { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 6px; }
.tag-chip { font-size: 11px; padding: 3px 10px; border-radius: 12px; border: 1px solid transparent; background: var(--tag-technique-bg); color: var(--tag-technique-fg); font-weight: 500; white-space: nowrap; }
.tag-link { text-decoration: none; }
.tag-link:hover { filter: brightness(0.9); }
.tag-model { font-size: 11px; padding: 3px 10px; border-radius: 12px; border: 1px solid transparent; background: var(--tag-model-bg); color: var(--tag-model-fg); font-weight: 500; white-space: nowrap; text-decoration: none; }
.tag-model:hover { filter: brightness(0.9); }

.tag-paper { font-size: 11px; padding: 3px 10px; border-radius: 12px; border: 1px solid transparent; background: var(--tag-paper-bg); color: var(--tag-paper-fg); font-weight: 500; white-space: nowrap; text-decoration: none; }
.tag-paper:hover { filter: brightness(0.9); }
.tag-data { font-size: 11px; padding: 3px 10px; border-radius: 12px; border: 1px solid transparent; background: var(--tag-data-bg); color: var(--tag-data-fg); font-weight: 500; white-space: nowrap; text-decoration: none; }
.tag-data:hover { filter: brightness(0.9); }
.card-actions { text-align: right; margin-top: 2px; }
.card-link { font-size: 11px; color: var(--text-muted); text-decoration: none; font-weight: 500; }
.card-link:hover { color: var(--series-1); }

/* ---- Demo-page shared components ---- */

.trace-caption { font-size: 13px; color: var(--text-muted); margin: 4px 0 10px; }
.trace-row, .token-row { display:flex; flex-wrap:wrap; gap:6px; margin:6px 0 14px; }
.trace-chip {
  font-size:13px; padding:4px 10px; border-radius:6px; background:var(--chip-bg);
  border:1px solid var(--border); color:var(--text-secondary);
}
.trace-chip.known { border-left:3px solid var(--good); }
.trace-chip.unknown { border-left:3px solid var(--critical); color:var(--text-muted); text-decoration:line-through; }
.token-chip {
  font-size:14px; padding:4px 8px; border-radius:5px; font-variant-numeric:tabular-nums;
  cursor:default; border:1px solid var(--border);
}
.digit-chip {
  font-size:22px; font-weight:600; padding:10px 16px; border-radius:6px;
  cursor:default; border:1px solid var(--border);
}
.prob-chart { margin:6px 0 14px; }
.prob-row { display:flex; align-items:center; gap:8px; margin:3px 0; }
.prob-label {
  min-width:60px; max-width:140px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
  font-size:13px; color:var(--text-secondary); font-variant-numeric:tabular-nums;
}
.prob-track { flex:1; height:14px; background:var(--gridline); border-radius:4px; overflow:hidden; }
.prob-fill { height:100%; background:var(--series-1-muted); }
.prob-fill.pred { background:var(--series-1); }
.prob-value { width:56px; text-align:right; font-size:12px; color:var(--text-muted); font-variant-numeric:tabular-nums; }
.loss-svg { display:block; margin:6px 0 4px; }
.stat-row { display:flex; flex-wrap:wrap; gap:10px; margin:6px 0 14px; }
.stat-tile { padding:10px 16px; border:1px solid var(--border); border-radius:8px; background:var(--chip-bg); }
.stat-tile .label { font-size:11px; color:var(--text-muted); text-transform:uppercase; letter-spacing:.03em; }
.stat-tile .value { font-size:20px; color:var(--text-primary); font-variant-numeric:tabular-nums; }
.diff-box {
  padding:10px 14px; border:1px solid var(--border); border-radius:8px;
  background:var(--chip-bg); margin:4px 0 14px; line-height:1.7; font-size:14px;
  color:var(--text-secondary);
}
.diff-ins { color:var(--good); text-decoration:underline; font-weight:600; }
.diff-del { color:var(--critical); text-decoration:line-through; opacity:0.7; }

/* ---- Live SSE progress tracker + score bars ---- */

.tracker { font-family: ui-monospace,SFMono-Regular,monospace; font-size: 13px; line-height: 1.85; margin: 4px 0 14px; color: var(--text-secondary); }
.tracker .done { color: var(--good); }
.tracker .active { color: var(--series-1); font-weight: 500; }
.tracker .pending { color: var(--text-muted); }

.score-bars { margin: 6px 0 14px; }
.score-row { display: flex; align-items: center; gap: 8px; margin: 4px 0; }
.score-rank { min-width: 36px; font-size: 12px; color: var(--text-muted); text-align: right; font-variant-numeric: tabular-nums; }
.score-fill { height: 14px; border-radius: 3px; min-width: 4px; transition: width 0.4s ease; }
.score-fill.best { background: #0ca30c; }
.score-fill.mid { background: var(--series-1-muted); }
.score-fill.worst { background: var(--gridline); }
.score-doc { font-size: 13px; color: var(--text-secondary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
.score-val { font-size: 11px; color: var(--text-muted); font-variant-numeric: tabular-nums; width: 50px; text-align: right; }
.score-detail { font-size: 12px; color: var(--text-muted); line-height: 1.5; margin: 2px 0 8px 44px; }

.sim-matrix { border-collapse: collapse; margin: 10px 0 16px; font-size: 12px; font-variant-numeric: tabular-nums; }
.sim-matrix th { font-weight: 500; color: var(--text-muted); padding: 4px 8px; text-align: center; }
.sim-matrix td { padding: 6px 10px; text-align: center; border-radius: 2px; cursor: default; }

.stage-block { margin: 6px 0 14px; }
.stage-label { font-weight: 600; font-size: 14px; }
.stage-detail { font-size: 12px; color: var(--text-muted); }
.stage-math { font-size: 12px; color: var(--text-muted); }
.stage-completion { font-size: 13px; color: var(--text-secondary); margin-left: 4px; font-style: italic; }
.stage-scalar { font-size: 15px; color: var(--series-1); font-variant-numeric: tabular-nums; margin-left: 4px; }
.status-line { font-size: 14px; color: var(--text-muted); margin: 4px 0; }

.step-header {
  background: var(--series-1-muted); color: var(--series-1);
  padding: 8px 12px; border-radius: 4px; margin: 18px 0 8px;
  font-weight: 600; font-size: 14px;
}
.step-summary { font-size: 12px; color: var(--text-secondary); margin: 4px 0 14px 12px; }
.step-summary td { padding: 1px 8px; font-variant-numeric: tabular-nums; }
.step-summary .trend-up { color: var(--good); }
.step-summary .trend-down { color: var(--critical); }
.step-summary .trend-flat { color: var(--text-muted); }

.arch-table { border-collapse: collapse; margin: 4px 0 10px; font-size: 12px; }
.arch-table td, .arch-table th { padding: 4px 10px; text-align: left; border-bottom: 1px solid var(--border); }
.arch-table th { color: var(--text-muted); font-weight: 500; }

.summary-card {
  border: 1px solid var(--border); border-radius: 6px; padding: 14px 18px;
  margin: 14px 0; background: var(--bg-light, #fafaf9);
}
.summary-card h4 { margin: 0 0 8px; font-size: 14px; }
.summary-table { border-collapse: collapse; font-size: 13px; width: 100%; }
.summary-table td { padding: 3px 10px; border-bottom: 1px solid var(--border); font-variant-numeric: tabular-nums; }
.summary-table td:first-child { color: var(--text-muted); }
.summary-table .summary-good { color: var(--good); }
.summary-table .summary-bad { color: var(--critical); }

/* Sidebar tertiary buttons: strip chrome so they read as plain text links */
section[data-testid="stSidebar"] .stButton button[kind="tertiary"] {
    background: none; border: none; padding: 3px 0; margin: 0;
    font-size: 0.95rem; text-align: left; line-height: 1.5;
    color: var(--text-secondary); min-height: unset;
}
section[data-testid="stSidebar"] .stButton button[kind="tertiary"]:hover {
    color: var(--series-1); text-decoration: underline; background: none;
}
</style>
"""
st.markdown(PALETTE_CSS, unsafe_allow_html=True)

# Sequential blue ramp, steps 100->650 from the validated default palette
# (references/palette.md). Index 0 = near-zero, last = near-certain.
_SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]
_SEQ_BLUE_DARK_STEPS = {"#3987e5", "#2a78d6", "#1c5cab", "#104281"}
_SEQ_BLUE_RGB = (42, 120, 214)  # #2a78d6, for the PIL patch-grid overlay


def esc(value: object) -> str:
    return html.escape(str(value))


def seq_step(probability: float) -> str:
    """Map a probability in [0, 1] to one step of the sequential blue ramp."""
    index = min(max(int(probability * len(_SEQ_BLUE)), 0), len(_SEQ_BLUE) - 1)
    return _SEQ_BLUE[index]


def text_color_for(hex_bg: str) -> str:
    """White text on the ramp's dark steps, dark text on its light steps."""
    return "#ffffff" if hex_bg in _SEQ_BLUE_DARK_STEPS else "#0b0b0b"


def render_stats(items: list[tuple[str, str]]) -> str:
    tiles = "".join(
        f'<div class="stat-tile"><div class="label">{esc(label)}</div>'
        f'<div class="value">{esc(value)}</div></div>'
        for label, value in items
    )
    return f'<div class="stat-row">{tiles}</div>'


def render_word_trace(word_trace: list[dict[str, Any]]) -> str:
    """Known/unknown vocabulary chips -- a status pair, always icon + label."""
    chips = []
    for w in word_trace:
        known = bool(w["known"])
        cls = "known" if known else "unknown"
        icon = "✓" if known else "✗"
        chips.append(f'<span class="trace-chip {cls}">{icon} {esc(w["word"])}</span>')
    return f'<div class="trace-row">{"".join(chips)}</div>'


def render_probability_bars(probs: dict[str, float], predicted: str | None) -> str:
    """Single-series magnitude bars, one hue, the predicted/chosen label emphasized."""
    rows = []
    for label, p in probs.items():
        pred_cls = " pred" if label == predicted else ""
        rows.append(
            f'<div class="prob-row"><span class="prob-label" title="{esc(label)}">{esc(label)}</span>'
            f'<div class="prob-track"><div class="prob-fill{pred_cls}" '
            f'style="width:{max(p, 0.0) * 100:.1f}%"></div></div>'
            f'<span class="prob-value">{p * 100:.1f}%</span></div>'
        )
    return f'<div class="prob-chart">{"".join(rows)}</div>'


def render_token_heatmap(tokens: list[dict[str, Any]]) -> str:
    """Each token shaded by its own sampling/greedy probability -- one hue, magnitude only."""
    spans = []
    for t in tokens:
        prob = float(t["probability"])
        bg = seq_step(prob)
        fg = text_color_for(bg)
        label = esc(t["token"]) or "&nbsp;"
        spans.append(
            f'<span class="token-chip" style="background:{bg};color:{fg}" '
            f'title="{prob * 100:.1f}% confidence">{label}</span>'
        )
    return f'<div class="token-row">{"".join(spans)}</div>'


def render_digit_trace(digit_trace: list[dict[str, Any]]) -> str:
    chips = []
    for d in digit_trace:
        conf = float(d["confidence"])
        bg = seq_step(conf)
        fg = text_color_for(bg)
        chips.append(
            f'<span class="digit-chip" style="background:{bg};color:{fg}" '
            f'title="{conf * 100:.1f}% confidence">{esc(d["digit"])}</span>'
        )
    return f'<div class="token-row">{"".join(chips)}</div>'


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return (int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16))


def _lerp_hex(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> str:
    r = round(c1[0] + (c2[0] - c1[0]) * t)
    g = round(c1[1] + (c2[1] - c1[1]) * t)
    b = round(c1[2] + (c2[2] - c1[2]) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


# Diverging pair (blue <-> red, neutral gray midpoint) for signed per-token
# quantities -- KL divergence, advantages, TD errors, etc. can be negative, so the
# sequential single-hue ramp (magnitude only) doesn't fit; sign matters here too.
_DIVERGING_NEUTRAL = _hex_to_rgb("#f0efec")
_DIVERGING_BLUE = _hex_to_rgb("#2a78d6")
_DIVERGING_RED = _hex_to_rgb("#e34948")


def render_diverging_token_heatmap(tokens: list[str], values: list[float]) -> str:
    """Shade each token by a signed value: red for positive, blue for negative,
    gray at zero, scaled by the largest magnitude in this sequence."""
    max_abs = max((abs(v) for v in values), default=0.0)
    spans = []
    for token, value in zip(tokens, values, strict=True):
        t = 0.0 if max_abs <= 0 else max(-1.0, min(1.0, value / max_abs))
        bg = _lerp_hex(_DIVERGING_NEUTRAL, _DIVERGING_RED if t >= 0 else _DIVERGING_BLUE, abs(t))
        fg = "#ffffff" if abs(t) > 0.55 else "#0b0b0b"
        label = esc(token) or "&nbsp;"
        spans.append(
            f'<span class="token-chip" style="background:{bg};color:{fg}" '
            f'title="{value:+.4f}">{label}</span>'
        )
    return f'<div class="token-row">{"".join(spans)}</div>'


def render_diff(before: str, after: str) -> str:
    """Word-level diff between two transcripts: insertions green+underlined, deletions
    red+struck-through -- both color and a second visual cue, never color alone."""
    before_words = before.split()
    after_words = after.split()
    matcher = difflib.SequenceMatcher(a=before_words, b=after_words)
    spans = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            spans.append(esc(" ".join(before_words[i1:i2])))
        elif tag == "delete":
            spans.append(f'<span class="diff-del">{esc(" ".join(before_words[i1:i2]))}</span>')
        elif tag == "insert":
            spans.append(f'<span class="diff-ins">{esc(" ".join(after_words[j1:j2]))}</span>')
        elif tag == "replace":
            spans.append(f'<span class="diff-del">{esc(" ".join(before_words[i1:i2]))}</span>')
            spans.append(f'<span class="diff-ins">{esc(" ".join(after_words[j1:j2]))}</span>')
    return f'<div class="diff-box">{" ".join(spans) or "&nbsp;"}</div>'


def render_loss_curve(points: list[tuple[int, float]]) -> str:
    """A single-series inline SVG line -- one axis, one hue, hover titles as the tooltip layer."""
    width, height, pad = 460, 170, 32
    values = [v for _, v in points]
    max_v = max(values) if values else 1.0
    max_v = max_v if max_v > 0 else 1.0
    n = len(points)

    def x(i: int) -> float:
        return pad + (width - 2 * pad) * (i / (n - 1) if n > 1 else 0.0)

    def y(v: float) -> float:
        return height - pad - (height - 2 * pad) * (v / max_v)

    coords = [(x(i), y(v)) for i, (_, v) in enumerate(points)]
    poly = " ".join(f"{cx:.1f},{cy:.1f}" for cx, cy in coords)
    circles = "".join(
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" fill="var(--series-1)">'
        f"<title>step {step}: loss {v:.4f}</title></circle>"
        for (step, v), (cx, cy) in zip(points, coords, strict=True)
    )
    labels = "".join(
        f'<text x="{cx:.1f}" y="{height - pad + 16}" font-size="11" fill="var(--text-muted)" '
        f'text-anchor="middle">{step}</text>'
        for (step, _v), (cx, _cy) in zip(points, coords, strict=True)
    )
    return (
        f'<svg width="{width}" height="{height}" class="loss-svg" role="img" '
        f'aria-label="Loss per fine-tuning step">'
        f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" '
        f'stroke="var(--gridline)" stroke-width="1"/>'
        f'<polyline points="{poly}" fill="none" stroke="var(--series-1)" stroke-width="2"/>'
        f"{circles}{labels}</svg>"
    )


def draw_patch_grid(
    image_bytes: bytes, img_size: int, patch_size: int, upscale: int = 6
) -> Image.Image:
    """Resize to the model's real input resolution, then overlay its real patch grid."""
    base = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((img_size, img_size))
    overlay = base.resize((img_size * upscale, img_size * upscale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(overlay, "RGBA")
    step = patch_size * upscale
    w, h = overlay.size
    line_color = (*_SEQ_BLUE_RGB, 200)
    for gx in range(0, w + 1, step):
        draw.line([(gx, 0), (gx, h)], fill=line_color, width=1)
    for gy in range(0, h + 1, step):
        draw.line([(0, gy), (w, gy)], fill=line_color, width=1)
    return overlay


def stream_sse(
    url: str, json_body: dict[str, Any], timeout: int
) -> Generator[dict[str, Any], None, None]:
    """POST to an SSE endpoint and yield each event as a parsed dict.

    A thin wrapper over `requests`' own streaming support -- no extra dependency.
    Each backend event is one `data: {...}` line followed by a blank line; this
    reads line by line and parses only the `data:` ones, which is sufficient for
    the single-line JSON payloads these backends emit.
    """
    with requests.post(url, json=json_body, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        # Decode explicitly rather than via decode_unicode=True: the flag does
        # yield str at run time, but requests' stubs type iter_lines as bytes
        # either way, so the str comparison below has no basis under strict mypy.
        for raw in r.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            if line.startswith("data: "):
                yield json.loads(line[len("data: ") :])


# ---------------------------------------------------------------------------
# Backend utilities
# ---------------------------------------------------------------------------


def health(week: str) -> bool:
    try:
        r = requests.get(f"{API_URLS[week]}/health", timeout=3)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def bytes_to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def image_to_b64(uploaded_file: UploadedFile) -> str:
    return bytes_to_b64(uploaded_file.getvalue())


def image_file_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def canvas_to_png_bytes(image_data: np.ndarray) -> bytes:
    """Convert the drawable canvas's RGBA pixel array into grayscale PNG bytes."""
    image = Image.fromarray(image_data.astype("uint8"), "RGBA").convert("L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_week_brief(week: str) -> None:
    """Brief intro paragraph above the interactive demo."""
    project = PROJECTS[week]
    if "intro" in project:
        st.markdown(project["intro"])
    st.divider()


def render_week_details(week: str) -> None:
    """Diagrams, stats, tags, and docs link below the demo."""
    project = PROJECTS[week]

    if project.get("diagrams"):
        diagrams = project["diagrams"]
        if len(diagrams) == 2:
            c1, c2 = st.columns(2)
            with c1:
                b64 = image_file_to_b64(diagrams[0][0])
                st.markdown(
                    f'<div style="height:400px;display:flex;align-items:center;justify-content:center;padding:16px;background:#fff;border-radius:4px;">'
                    f'<img src="data:image/png;base64,{b64}" '
                    f'style="max-width:100%;max-height:400px;object-fit:contain;">'
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.caption(diagrams[0][1])
            with c2:
                b64 = image_file_to_b64(diagrams[1][0])
                st.markdown(
                    f'<div style="height:400px;display:flex;align-items:center;justify-content:center;padding:16px;background:#fff;border-radius:4px;">'
                    f'<img src="data:image/png;base64,{b64}" '
                    f'style="max-width:100%;max-height:400px;object-fit:contain;">'
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.caption(diagrams[1][1])
        else:
            for path, caption in diagrams:
                st.image(path, width=650)
                st.caption(caption)

    if project.get("model_specs"):
        st.markdown("**Model details**")
        for line in project["model_specs"]:
            st.markdown(f"- {line}")

    if project.get("stats"):
        cols = st.columns(len(project["stats"]))
        for col, (label, value) in zip(cols, project["stats"], strict=True):
            with col:
                st.metric(label, value)

    tags_html = "".join(tag_link(t) for t in project["techniques"])
    model_html = "".join(
        f'<a href="{url}" target="_blank" class="tag-model">{label}</a>'
        for label, url in project.get("model_links", [])
    )
    data_html = "".join(
        f'<a href="{url}" target="_blank" class="tag-data">{label}</a>'
        for label, url in project.get("datasets", [])
    )
    paper_html = "".join(
        f'<a href="{url}" target="_blank" class="tag-paper">{label}</a>'
        for label, _title, url in project.get("papers", [])
    )
    st.markdown(
        f'<div class="tag-row" style="margin:6px 0 12px;">{tags_html}{model_html}{data_html}{paper_html}</div>',
        unsafe_allow_html=True,
    )

    docs_link = f"https://github.com/RoshBeed/ai-residency/blob/main/{project['docs_path']}"
    st.caption(
        f"[Full technical write-up →]({docs_link}) "
        "(architecture diagrams, method walkthrough, experimental results, references)"
    )


# ---------------------------------------------------------------------------
# Home page
# ---------------------------------------------------------------------------


def render_home_page() -> None:
    st.markdown(
        "<h2 style='text-align:center;margin:0 0 2px;font-size:1.2rem;'>ML Systems Portfolio</h2>"
        "<hr style='margin:0 0 6px;border:none;border-top:1px solid var(--border);'>",
        unsafe_allow_html=True,
    )

    # Project cards — the main focus, above the fold.
    week_keys = list(PROJECTS.keys())
    for row_start in range(0, len(week_keys), 3):
        row = week_keys[row_start : row_start + 3]
        cols = st.columns(len(row), gap="medium")
        for i, wk in enumerate(row):
            p = PROJECTS[wk]
            with cols[i]:
                tags_html = "".join(tag_link(t) for t in p["techniques"])
                data_html = "".join(
                    f'<a href="{url}" target="_blank" class="tag-data">{label}</a>'
                    for label, url in p.get("datasets", [])
                )
                model_html = "".join(
                    f'<a href="{url}" target="_blank" class="tag-model">{label}</a>'
                    for label, url in p.get("model_links", [])
                )
                paper_html = "".join(
                    f'<a href="{url}" target="_blank" class="tag-paper" title="{label}: {title}">{title[:16] + "…" if len(title) > 16 else title}</a>'
                    for label, title, url in p.get("papers", [])
                )
                st.markdown(
                    f'<div class="card">'
                    f'<div class="card-header">'
                    f'<span class="card-icon">{p["icon"]}</span>'
                    f'<span class="card-title">{p["title"]}</span>'
                    f"</div>"
                    f'<div class="card-desc">{p["description"]}</div>'
                    f'<div class="tag-row">{tags_html}{model_html}{data_html}{paper_html}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if st.button("🔬 Open", key=f"try_{wk}", use_container_width=True):
                    navigate(wk)
                    st.rerun()

    st.markdown(
        "<p style='text-align:center;font-size:0.85rem;color:var(--text-muted);margin-top:8px;'>"
        "End-to-end ML pipelines spanning NLP, computer vision, speech, and reinforcement learning."
        "</p>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Session state + routing setup
# ---------------------------------------------------------------------------

if "page" in st.query_params and st.query_params["page"] != st.session_state.get("page"):
    st.session_state.page = st.query_params["page"]
if "page" not in st.session_state:
    st.session_state.page = "home"


def navigate(page: str) -> None:
    st.session_state.page = page
    st.query_params["page"] = page


_PAGE = st.session_state.page


# ---------------------------------------------------------------------------
# Sidebar: navigation + tech-depth panel + backend status
# ---------------------------------------------------------------------------

with st.sidebar:
    if st.button("← Home", key="sidebar_home", type="tertiary"):
        navigate("home")
        st.rerun()

    if _PAGE != "home":
        st.divider()
        for wk, label in PAGES.items():
            if wk == _PAGE:
                st.markdown(f"{PROJECTS[wk]['icon']} **{label}**")
            else:
                if st.button(f"{PROJECTS[wk]['icon']} {label}", key=f"nav_{wk}", type="tertiary"):
                    navigate(wk)
                    st.rerun()

    st.divider()
    st.caption("[GitHub →](https://github.com/RoshBeed/ai-residency/tree/main/services/frontend)")
    st.caption("[LinkedIn →](https://linkedin.com/in/RoshBeed)")


# ---------------------------------------------------------------------------
# Main content routing
# ---------------------------------------------------------------------------

if _PAGE == "home":
    render_home_page()
    st.stop()

# --- All week-pages share the title + health check ---

st.markdown(f"## {PAGES[_PAGE]}")

if not health(_PAGE):
    st.error(
        f"This project's backend isn't reachable at `{API_URLS[_PAGE]}`. "
        f"Start it with `docker compose up {_PAGE}` (or the equivalent local "
        "`uv run` command in that project's own repo) before using this page."
    )
    st.stop()

render_week_brief(_PAGE)
st.subheader("Try it out")

# ---------------------------------------------------------------------------
# Word Embeddings (nearest-neighbour + analogy)
# ---------------------------------------------------------------------------
if _PAGE == "word-embeddings":
    st.markdown("**Nearest neighbours** — the words closest to a query word in the learned space.")
    word = st.text_input("Word", value="python")
    if st.button("Find nearest", type="primary") and word:
        try:
            r = requests.post(f"{API_URLS[_PAGE]}/nearest", json={"word": word}, timeout=15)
            r.raise_for_status()
            data = r.json()
            st.markdown(
                '<div class="trace-caption">'
                f'Loaded from checkpoint source "{data["checkpoint_source"]}".</div>',
                unsafe_allow_html=True,
            )
            for rank, (neighbour, similarity) in enumerate(data["nearest"], start=1):
                st.markdown(
                    f'**{rank}.** `{neighbour}` <span class="dim">(cosine {similarity:.3f})</span>',
                    unsafe_allow_html=True,
                )
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")

    st.divider()
    st.markdown("**Analogy** — find the word closest to `a − b + c`.")
    c1, c2, c3 = st.columns(3)
    with c1:
        a = st.text_input("a", value="king")
    with c2:
        b = st.text_input("b", value="man")
    with c3:
        c = st.text_input("c", value="woman")
    if st.button("Solve analogy", type="primary") and a and b and c:
        try:
            r = requests.post(
                f"{API_URLS[_PAGE]}/analogy", json={"a": a, "b": b, "c": c}, timeout=15
            )
            r.raise_for_status()
            data = r.json()
            for neighbour, similarity in data["nearest"]:
                st.markdown(
                    f"`{a} − {b} + {c}` ≈ **{neighbour}** "
                    f'<span class="dim">(cosine {similarity:.3f})</span>',
                    unsafe_allow_html=True,
                )
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")

# ---------------------------------------------------------------------------
# Content Scoring (regression head over frozen embeddings)
# ---------------------------------------------------------------------------
elif _PAGE == "content-scoring":
    title = st.text_input(
        "Hacker News post title",
        value="Show HN: I built a CLI tool that writes your commit messages",
    )
    if st.button("Predict score", type="primary") and title:
        st.caption("Live -- each stage below arrives the moment it completes:")
        tracker = st.empty()
        live_words = st.empty()
        done_lines: list[str] = []
        try:
            for event in stream_sse(
                f"{API_URLS[_PAGE]}/predict/stream",
                {"title": title},
                timeout=15,
            ):
                stage = event["stage"]
                if stage == "word_lookup":
                    word_trace = event["word_trace"]
                    known_c = sum(1 for w in word_trace if w["known"])
                    total_c = len(word_trace)
                    done_lines.append(
                        f'<span class="done">✓</span> Word lookup: {known_c}/{total_c} recognised'
                    )
                    done_lines.append('<span class="active">⟳</span> Building embedding vector…')
                    tracker.markdown(
                        f'<div class="tracker">{"<br>".join(done_lines)}</div>',
                        unsafe_allow_html=True,
                    )
                    live_words.markdown(render_word_trace(word_trace), unsafe_allow_html=True)
                elif stage == "embedding":
                    norm = event["title_embedding_norm"]
                    done_lines[-1] = (
                        f'<span class="done">✓</span> Embedding built (norm = {norm:.3f})'
                    )
                    done_lines.append('<span class="active">⟳</span> Predicting score…')
                    tracker.markdown(
                        f'<div class="tracker">{"<br>".join(done_lines)}</div>',
                        unsafe_allow_html=True,
                    )
                elif stage == "done":
                    data = event
                    done_lines[-1] = (
                        f'<span class="done">✓</span> Predicted {data["predicted_score"]:.1f} upvotes'
                    )
                    tracker.markdown(
                        f'<div class="tracker">{"<br>".join(done_lines)}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        '<div class="trace-caption">'
                        "Words looked up in CBOW vocabulary (258,196 words). "
                        "Unknown words are skipped.</div>",
                        unsafe_allow_html=True,
                    )
                    live_words.empty()
                    st.markdown(render_word_trace(data["word_trace"]), unsafe_allow_html=True)
                    known = sum(1 for w in data["word_trace"] if w["known"])
                    st.markdown(
                        render_stats(
                            [
                                ("Words recognised", f"{known} / {len(data['word_trace'])}"),
                                ("Embedding dim", str(data["embedding_dim"])),
                                ("Title vector norm", f"{data['title_embedding_norm']:.3f}"),
                            ]
                        ),
                        unsafe_allow_html=True,
                    )
                    st.metric("Predicted score", f"{data['predicted_score']:.1f} upvotes")
                    with st.expander("Raw response"):
                        st.json(data)
                elif stage == "error":
                    tracker.empty()
                    st.error(f"Prediction failed: {event['detail']}")
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")

# ---------------------------------------------------------------------------
# RAG & Semantic Search
# ---------------------------------------------------------------------------
elif _PAGE == "vector-search-retrieval":
    query = st.text_input("Search query", value="was ronald reagan a democrat")
    top_k = st.slider("Top K", 1, 10, 5)
    if st.button("Search", type="primary") and query:
        st.caption("Live -- each stage below arrives the moment it completes:")
        tracker = st.empty()
        live_words = st.empty()
        retrieved_lines: list[str] = []
        try:
            for event in stream_sse(
                f"{API_URLS[_PAGE]}/query/stream",
                {"text": query, "top_k": top_k},
                timeout=15,
            ):
                stage = event["stage"]
                if stage == "vocabulary":
                    wt = event["word_trace"]
                    known_c = sum(1 for w in wt if w["known"])
                    retrieved_lines.append(
                        f'<span class="done">✓</span> Vocabulary check: '
                        f"{known_c}/{len(wt)} words recognised"
                    )
                    retrieved_lines.append(
                        '<span class="active">⟳</span> Encoding query with GRU tower…'
                    )
                    tracker.markdown(
                        f'<div class="tracker">{"<br>".join(retrieved_lines)}</div>',
                        unsafe_allow_html=True,
                    )
                    live_words.markdown(render_word_trace(wt), unsafe_allow_html=True)
                elif stage == "encoded":
                    norm = event["embedding_norm"]
                    retrieved_lines[-1] = (
                        f'<span class="done">✓</span> Query encoded (norm = {norm:.3f})'
                    )
                    retrieved_lines.append(
                        '<span class="active">⟳</span> Searching Redis via HNSW…'
                    )
                    tracker.markdown(
                        f'<div class="tracker">{"<br>".join(retrieved_lines)}</div>',
                        unsafe_allow_html=True,
                    )
                elif stage == "done":
                    data = event
                    n_res = len(data["results"])
                    retrieved_lines[-1] = f'<span class="done">✓</span> {n_res} results retrieved'
                    tracker.markdown(
                        f'<div class="tracker">{"<br>".join(retrieved_lines)}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        '<div class="trace-caption">'
                        "Words encoded by trained GRU query tower, then compared against "
                        "indexed documents via cosine distance (lower = closer).</div>",
                        unsafe_allow_html=True,
                    )
                    live_words.empty()
                    st.markdown(render_word_trace(data["word_trace"]), unsafe_allow_html=True)
                    st.markdown(
                        render_stats([("Query vector norm", f"{data['embedding_norm']:.3f}")]),
                        unsafe_allow_html=True,
                    )
                    if not data["results"]:
                        st.info("No results — the index may be empty on this deployment.")
                    else:
                        chart_data = {f"#{r['rank']}": r["score"] for r in data["results"]}
                        st.bar_chart(
                            chart_data,
                            horizontal=True,
                            x_label="cosine distance",
                            y_label="result rank",
                        )
                    for res in data["results"]:
                        st.markdown(
                            f'<div class="score-detail"><strong>#{res["rank"]}</strong> '
                            f"(cos-dist {res['score']:.4f})</div>"
                            f'<div class="score-detail">{esc(res["text"])}</div>',
                            unsafe_allow_html=True,
                        )
                    with st.expander("Raw response"):
                        st.json(data)
                elif stage == "error":
                    tracker.empty()
                    st.error(f"Retrieval failed: {event['detail']}")
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")

# ---------------------------------------------------------------------------
# Vision Transformer & Seq2Seq
# ---------------------------------------------------------------------------
elif _PAGE == "vit-classifier":
    col_draw, col_upload = st.columns(2)
    with col_draw:
        st.caption("Draw a digit")
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=18,
            stroke_color="#ffffff",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="vit_classifier_canvas",
        )
    with col_upload:
        st.caption("Or upload one")
        uploaded = st.file_uploader(
            "Upload a digit image (PNG/JPG)",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )
        if uploaded is not None:
            st.image(uploaded, width=150)

    image_bytes: bytes | None = None
    if canvas_result.image_data is not None:
        image_bytes = canvas_to_png_bytes(canvas_result.image_data)
    elif uploaded is not None:
        image_bytes = uploaded.getvalue()

    if image_bytes is not None and st.button("Classify", type="primary"):
        b64_image = bytes_to_b64(image_bytes)
        vit_data: dict[str, Any] | None = None
        try:
            r = requests.post(
                f"{API_URLS[_PAGE]}/predict",
                json={"image_data": b64_image},
                timeout=15,
            )
            r.raise_for_status()
            vit_data = r.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Classifier request failed: {e}")

        if vit_data:
            st.caption(
                f"Input: {vit_data['img_size']}x{vit_data['img_size']}px, "
                f"{vit_data['patch_size']}x{vit_data['patch_size']} patches."
            )
            st.image(
                draw_patch_grid(image_bytes, vit_data["img_size"], vit_data["patch_size"]),
                width=260,
            )
            st.markdown(f"**Predicted: {vit_data['pred']}**")
            if vit_data.get("class_probabilities"):
                st.markdown(
                    render_probability_bars(vit_data["class_probabilities"], vit_data["pred"]),
                    unsafe_allow_html=True,
                )

# ---------------------------------------------------------------------------
# ViT Seq2Seq -- multi-digit, decoded one token at a time
# ---------------------------------------------------------------------------
elif _PAGE == "vit-seq2seq":
    col_draw, col_upload = st.columns(2)
    with col_draw:
        st.caption("Draw a multi-digit number")
        seq_canvas = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=14,
            stroke_color="#ffffff",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="vit_seq2seq_canvas",
        )
    with col_upload:
        st.caption("Or upload one")
        seq_uploaded = st.file_uploader(
            "Upload a multi-digit image (PNG/JPG)",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )
        if seq_uploaded is not None:
            st.image(seq_uploaded, width=150)

    seq_bytes: bytes | None = None
    if seq_canvas.image_data is not None:
        seq_bytes = canvas_to_png_bytes(seq_canvas.image_data)
    elif seq_uploaded is not None:
        seq_bytes = seq_uploaded.getvalue()

    if seq_bytes is not None and st.button("Read the number", type="primary"):
        b64_seq = bytes_to_b64(seq_bytes)
        st.caption("Input: 84x84px, 16x16 patches. Decoded autoregressively.")
        st.image(draw_patch_grid(seq_bytes, 84, 16), width=260)

        st.caption("Live -- each digit below appears the moment the model emits it:")
        digit_display = st.empty()
        seq2seq_data = None
        digit_trace_sofar: list[dict[str, Any]] = []
        try:
            for event in stream_sse(
                f"{API_URLS[_PAGE]}/predict/stream",
                {"image_data": b64_seq},
                timeout=30,
            ):
                stage = event["stage"]
                if stage == "digit" and event["digit"]:
                    digit_trace_sofar.append(
                        {"digit": event["digit"], "confidence": event["confidence"]}
                    )
                    digit_display.markdown(
                        render_digit_trace(digit_trace_sofar),
                        unsafe_allow_html=True,
                    )
                elif stage == "done":
                    seq2seq_data = event
                    digit_display.empty()
                elif stage == "error":
                    digit_display.empty()
                    st.error(f"Seq2seq generation failed: {event['detail']}")
        except requests.exceptions.RequestException as e:
            st.error(f"SSE request failed: {e}")

        if seq2seq_data:
            st.markdown(f"**Predicted: {seq2seq_data['pred']}**")
            if seq2seq_data.get("digit_trace"):
                st.caption("Darker = higher confidence at each step:")
                st.markdown(render_digit_trace(seq2seq_data["digit_trace"]), unsafe_allow_html=True)
            with st.expander("Raw SSE response"):
                st.json(seq2seq_data)

# ---------------------------------------------------------------------------
# Multimodal Transfer Learning
# ---------------------------------------------------------------------------
elif _PAGE == "multimodal-captioning":
    col_upload, col_camera = st.columns(2)
    with col_upload:
        uploaded = st.file_uploader(
            "Upload an image", type=["png", "jpg", "jpeg"], label_visibility="collapsed"
        )
        if uploaded is not None:
            st.image(uploaded, width=250)
    with col_camera:
        camera = st.camera_input("Take a photo", label_visibility="collapsed")
        if camera is not None:
            st.image(camera, width=250)

    image_source = uploaded or camera
    if image_source is not None and st.button("Generate caption", type="primary"):
        st.caption("Live -- each token below appears the moment the model emits it:")
        tracker = st.empty()
        live_heatmap = st.empty()
        tokens: list[dict[str, Any]] = []
        data = None
        try:
            for event in stream_sse(
                f"{API_URLS[_PAGE]}/caption/stream",
                {"image_data": image_to_b64(image_source), "checkpoint_name": "adapter.pth"},
                timeout=90,
            ):
                stage = event["stage"]
                if stage == "model_loaded":
                    tracker.markdown(
                        '<div class="tracker">'
                        '<span class="done">✓</span> CLIP + Qwen3 loaded<br>'
                        '<span class="active">⟳</span> Generating caption…'
                        "</div>",
                        unsafe_allow_html=True,
                    )
                elif stage == "token":
                    tokens.append({"token": event["token"], "probability": event["probability"]})
                    live_heatmap.markdown(render_token_heatmap(tokens), unsafe_allow_html=True)
                    tracker.markdown(
                        f'<div class="tracker">'
                        f'<span class="done">✓</span> Models loaded<br>'
                        f'<span class="active">⟳</span> Token {event["step"] + 1}: '
                        f'"<strong>{esc(event["token"])}</strong>" '
                        f"({event['probability'] * 100:.1f}% confidence)"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                elif stage == "done":
                    data = event
                    tracker.markdown(
                        f'<div class="tracker">'
                        f'<span class="done">✓</span> Models loaded<br>'
                        f'<span class="done">✓</span> {len(data["token_trace"])} tokens generated'
                        "</div>",
                        unsafe_allow_html=True,
                    )
                elif stage == "error":
                    tracker.empty()
                    st.error(f"Captioning failed: {event['detail']}")
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")

        if data:
            st.caption(
                "CLIP image embedding → prepended to prompt → Qwen3 autoregressive decode. "
                "Darker = higher sampling probability."
            )
            live_heatmap.empty()
            st.markdown(render_token_heatmap(data["token_trace"]), unsafe_allow_html=True)
            st.success(data["caption"])
            st.caption(f"Checkpoint: {data['checkpoint_used']}")
            with st.expander("Raw SSE response"):
                st.json(data)

# ---------------------------------------------------------------------------
# Emotion-Aware Speech Recognition
# ---------------------------------------------------------------------------
elif _PAGE == "whisper-fine-tuning":
    col_record, col_upload = st.columns(2)
    with col_record:
        audio = st.audio_input("Record audio")
        if audio is not None:
            st.audio(audio)
    with col_upload:
        audio_file = st.file_uploader(
            "Upload audio", type=["mp3", "wav", "m4a", "aiff", "flac"], label_visibility="collapsed"
        )
        if audio_file is not None:
            st.audio(audio_file)

    audio_source = audio or audio_file
    audio_bytes: bytes | None = audio_source.getvalue() if audio_source else None

    target_text = st.text_input(
        "Fine-tune target text",
        value="My name is Rosh",
        help="What the model is fine-tuned to transcribe the clip as. The default is "
        "the demo's own ground truth for the bundled 'My name is Rosh' clip -- if you "
        "record something else, try changing this to match it, or leave it as-is to "
        "see what happens when the target doesn't match what was actually said.",
    )
    if audio_bytes is not None and st.button("Transcribe", type="primary"):
        st.caption("Live -- each event below arrives the moment that stage finishes:")
        tracker = st.empty()
        live_curve = st.empty()
        live_table = st.empty()
        baseline_loss = None
        base_output = None
        transcript_lines: list[str] = []
        steps: list[dict[str, Any]] = []
        try:
            for event in stream_sse(
                f"{API_URLS[_PAGE]}/transcribe/stream",
                {"audio_data": bytes_to_b64(audio_bytes), "target_text": target_text},
                timeout=120,
            ):
                stage = event["stage"]
                if stage == "base_transcript":
                    base_output = event["text"]
                    transcript_lines.append(
                        f'<span class="done">✓</span> Base whisper: "<em>{esc(base_output)}</em>"'
                    )
                    transcript_lines.append(
                        '<span class="active">⟳</span> Computing baseline loss…'
                    )
                    tracker.markdown(
                        f'<div class="tracker">{"<br>".join(transcript_lines)}</div>',
                        unsafe_allow_html=True,
                    )
                elif stage == "baseline_loss":
                    baseline_loss = event["loss"]
                    transcript_lines[-1] = (
                        f'<span class="done">✓</span> Baseline loss: {baseline_loss:.4f}'
                    )
                    transcript_lines.append('<span class="active">⟳</span> Step 1/5…')
                    tracker.markdown(
                        f'<div class="tracker">{"<br>".join(transcript_lines)}</div>',
                        unsafe_allow_html=True,
                    )
                elif stage == "step":
                    steps.append(event)
                    transcript_lines[-1] = (
                        f'<span class="done">✓</span> Step {event["step"]}/5: '
                        f"loss {event['loss']:.4f}"
                    )
                    if event["step"] < 5:
                        transcript_lines.append(
                            f'<span class="active">⟳</span> Step {event["step"] + 1}/5…'
                        )
                    tracker.markdown(
                        f'<div class="tracker">{"<br>".join(transcript_lines)}</div>',
                        unsafe_allow_html=True,
                    )
                    points = [(0, baseline_loss)] + [(s["step"], s["loss"]) for s in steps]
                    live_curve.markdown(render_loss_curve(points), unsafe_allow_html=True)
                    live_table.dataframe(
                        [{"step": 0, "loss": baseline_loss, "text": base_output}, *steps],
                        use_container_width=True,
                        hide_index=True,
                    )
                elif stage == "done":
                    st.session_state["audio-fine-tuning_data"] = event
                    tracker.empty()
                    live_curve.empty()
                    live_table.empty()
                elif stage == "error":
                    tracker.empty()
                    st.error(f"Fine-tuning failed: {event['detail']}")
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")

    data = st.session_state.get("audio-fine-tuning_data")
    if data:
        st.markdown(
            '<div class="trace-caption">'
            "Not a cached demo. Fine-tunes Whisper-tiny from scratch each request — "
            f'5 gradient steps toward target "{esc(target_text)}".</div>',
            unsafe_allow_html=True,
        )
        points = [(0, data["baseline_loss"])] + [
            (s["step"], s["loss"]) for s in data["fine_tune_trace"]
        ]
        st.markdown(render_loss_curve(points), unsafe_allow_html=True)
        st.markdown(
            render_stats(
                [
                    ("Baseline loss (step 0)", f"{data['baseline_loss']:.4f}"),
                    (
                        "Final loss",
                        f"{data['fine_tune_trace'][-1]['loss']:.4f}"
                        if data["fine_tune_trace"]
                        else "n/a",
                    ),
                ]
            ),
            unsafe_allow_html=True,
        )

        step_rows = [{"step": 0, "loss": data["baseline_loss"], "text": data["base_output"]}] + [
            {"step": s["step"], "loss": s["loss"], "text": s["text"]}
            for s in data["fine_tune_trace"]
        ]
        st.caption("Per-step transcript — green = added, red = dropped:")
        for prev_row, curr_row in pairwise(step_rows):
            loss = curr_row["loss"]
            st.markdown(f"**Step {curr_row['step']}** (loss = {loss:.4f})")
            st.markdown(render_diff(prev_row["text"], curr_row["text"]), unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Base Whisper**")
            st.text(data["base_output"])
        with col2:
            st.markdown("**Fine-tuned (5 steps)**")
            st.text(data["ft_output"])
        with st.expander("Raw response"):
            st.json(data)


# ---------------------------------------------------------------------------
# Speech Emotion Recognition -- the emotion is a token the decoder emits
# ---------------------------------------------------------------------------
elif _PAGE == "speech-emotion-recognition":
    st.caption(
        "Upload a clip. The model predicts which of six emotions it hears, by scoring the "
        "emotion tokens its decoder can emit."
    )
    emotion_upload = st.file_uploader(
        "Upload audio (WAV/MP3)", type=["wav", "mp3", "m4a"], label_visibility="collapsed"
    )
    if emotion_upload is not None:
        st.audio(emotion_upload)

    if emotion_upload is not None and st.button("Predict emotion", type="primary"):
        try:
            response = requests.post(
                f"{API_URLS[_PAGE]}/predict",
                json={"audio_data": bytes_to_b64(emotion_upload.getvalue())},
                timeout=120,
            )
        except requests.exceptions.RequestException as error:
            st.error(f"Request failed: {error}")
        else:
            if response.status_code == 503:
                st.info(
                    "No trained checkpoint is loaded. The pipeline, trainer and API are "
                    "wired up; train one with `make train SERVICE=speech-emotion-recognition`."
                )
            elif response.status_code != 200:
                st.error(f"Backend returned {response.status_code}: {response.text[:200]}")
            else:
                result = response.json()
                st.markdown(f"**Predicted: {result['emotion']}**")
                st.caption(f"Checkpoint source: `{result['checkpoint_source']}`")
                st.markdown(
                    render_probability_bars(result["scores"], result["emotion"]),
                    unsafe_allow_html=True,
                )
                with st.expander("Raw response"):
                    st.json(result)

# ---------------------------------------------------------------------------
# RLHF for Text Summarization
# ---------------------------------------------------------------------------
elif _PAGE == "rlhf-ppo":
    ppo_steps = st.number_input("PPO steps", min_value=1, max_value=20, value=5, step=1)
    _PPO_STAGE_LABELS = {
        "generate": "1. Sampling",
        "kl_divergence": "2. Token-Level KL Penalty",
        "rewards": "3. Reward Scoring",
        "adjusted_reward": "4. KL-Penalized Reward",
        "value": "5. Value Function",
        "td_error": "6. Temporal Difference Error",
        "advantage": "7. Generalized Advantage Estimation",
        "reward_to_go": "8. Rewards-to-Go",
        "value_loss": "9. Value Function Loss",
        "clip": "10. Clip",
        "policy_loss": "11. PPO-Clip Loss",
        "policy_ratio": "12. Policy Ratio",
    }
    _PPO_STAGE_FIELD = {
        "kl_divergence": "kl_div",
        "adjusted_reward": "adjusted_reward",
        "value": "value",
        "td_error": "td_error",
        "advantage": "advantage",
        "reward_to_go": "reward_to_go",
        "clip": "clipped_ratio",
        "policy_ratio": "policy_ratio",
    }
    if st.button("Run PPO steps", type="primary"):
        st.caption("Live — 12 stages per step. Token rows persist across stages.")
        _rendering = {
            "kl_divergence": ("2. Token-Level KL Penalty", "kl_div"),
            "adjusted_reward": ("4. KL-Penalized Reward", "adjusted_reward"),
            "value": ("5. Value Function", "value"),
            "td_error": ("6. Temporal Difference Error", "td_error"),
            "advantage": ("7. Generalized Advantage Estimation", "advantage"),
            "reward_to_go": ("8. Rewards-to-Go", "reward_to_go"),
            "clip": ("10. Clip", "clipped_ratio"),
            "policy_ratio": ("12. Policy Ratio", "policy_ratio"),
        }

        _MATH = {
            "generate": "a_t  ∼  π_θ(·∣s_t)",
            "kl_divergence": "log π_θ(a_t∣s_t) / π_ref(a_t∣s_t)",
            "rewards": "r_ψ(x, y)",
            "adjusted_reward": "r − η·KL_t",
            "value": "V_ϕ(s_t)",
            "td_error": "δ_t  =  r_t + γV_ϕ(s_{t+1}) − V_ϕ(s_t)",
            "advantage": "Â_t  =  Σ_{l≥0} (γλ)^l δ_{t+l}",
            "reward_to_go": "R̂_t  =  Â_t + V_ϕ(s_t)",
            "value_loss": "L_V  =  E[(V_ϕ − R̂_t)²]    ···  ∂V",
            "clip": "clamp(r_t(θ), 1±ε)",
            "policy_loss": "L_CLIP  =  E[min(r_tÂ_t, clip(r_t, 1±ε)Â_t)]    ···  ∂π",
            "policy_ratio": "r_t(θ)  =  π_θ / π_old    ——  readout",
        }

        try:
            status_line = st.empty()
            unified = st.empty()
            blocks: list[str] = []
            last_load: str = ""
            sampling_chips: list[str] = []

            # Per-step metrics for the post-run summary card.
            step_records: list[dict[str, float | str]] = []
            cur: dict[str, float | str] = {}

            def _render_heatmap(
                label: str, stage: str, field: str, event: dict[str, Any], math: dict[str, Any]
            ) -> str:
                html = (
                    f'<div class="stage-block">'
                    f'<span class="stage-label">{label}</span>'
                    f'<span class="stage-math">{math[stage]}</span>'
                    f"</div>"
                )
                pairs_by_prompt: dict[int, list[dict[str, Any]]] = {}
                for pair in event["sa_pairs"]:
                    pairs_by_prompt.setdefault(pair["prompt_idx"], []).append(pair)
                for pidx in sorted(pairs_by_prompt):
                    ordered = sorted(pairs_by_prompt[pidx], key=lambda p: p["step"])
                    html += render_diverging_token_heatmap(
                        [p["action_token"] for p in ordered],
                        [p[field] for p in ordered],
                    )
                return html

            def flush() -> None:
                unified.markdown("".join(blocks), unsafe_allow_html=True)

            for event in stream_sse(
                f"{API_URLS[_PAGE]}/ppo_step/stream",
                {
                    "prompts": [
                        "SUBREDDIT: r/relationships\nTITLE: Should I admit to snooping?\nPOST: ...\nTL;DR:"
                    ],
                    "ppo_steps": ppo_steps,
                },
                timeout=600,
            ):
                stage = event["stage"]
                if stage == "loading":
                    last_load = event["detail"]
                    status_line.markdown(f"⏳ {last_load}")
                elif stage == "step_boundary":
                    status_line.empty()
                    if cur:
                        step_records.append(cur)
                        cur = {}
                    step_num = event["step"]
                    step_total = event["total"]
                    blocks.append(
                        f'<div class="step-header">PPO Step {step_num} / {step_total}</div>'
                    )
                    flush()
                elif stage == "generating":
                    prob = math.exp(event["logprob"])
                    bg = seq_step(prob)
                    fg = text_color_for(bg)
                    token = esc(event["token"]) or "&nbsp;"
                    sampling_chips.append(
                        f'<span class="token-chip" style="background:{bg};color:{fg}" '
                        f'title="{prob * 100:.1f}% confidence">{token}</span>'
                    )
                    sampling_block = (
                        f'<div class="stage-block">'
                        f'<span class="stage-label">1. Sampling</span>'
                        f'<span class="stage-math">{_MATH["generate"]}</span>'
                        f"</div>"
                        f'<div class="token-row">{"".join(sampling_chips)}</div>'
                    )
                    if len(blocks) == 0 or "1. Sampling" in blocks[0]:
                        blocks[:1] = [sampling_block]
                    else:
                        blocks.insert(0, sampling_block)
                    status_line.empty()
                    flush()
                elif stage == "generate" and "sa_pairs" in event:
                    status_line.empty()
                    pairs = sorted(event["sa_pairs"], key=lambda p: p["step"])
                    chips: list[str] = []
                    for pair in pairs:
                        prob = math.exp(pair["logprob"])
                        bg = seq_step(prob)
                        fg = text_color_for(bg)
                        token = esc(pair["action_token"]) or "&nbsp;"
                        chips.append(
                            f'<span class="token-chip" style="background:{bg};color:{fg}" '
                            f'title="{prob * 100:.1f}% confidence">{token}</span>'
                        )
                    completion = esc(event["generated_texts"][0])
                    cur["completion"] = completion
                    sampling_block = (
                        f'<div class="stage-block">'
                        f'<span class="stage-label">1. Sampling</span>'
                        f'<span class="stage-math">{_MATH["generate"]}</span>'
                        f'<span class="stage-completion">{completion}</span>'
                        f"</div>"
                        f'<div class="token-row">{"".join(chips)}</div>'
                    )
                    if len(blocks) > 0 and "1. Sampling" in blocks[0]:
                        blocks[0] = sampling_block
                    else:
                        blocks.append(sampling_block)
                    sampling_chips = []
                    flush()
                elif stage == "heartbeat":
                    if last_load:
                        status_line.markdown(f"⏳ {last_load}  —  {event['elapsed']}s")
                elif stage == "done":
                    if cur:
                        step_records.append(cur)
                    st.session_state["rlhf-ppo_ppo_data"] = event
                    st.session_state["rlhf-ppo_step_records"] = step_records
                    status_line.empty()
                elif stage == "error":
                    status_line.empty()
                    st.error(f"PPO step failed: {event['detail']}")
                elif stage == "rewards" and "rewards" in event:
                    cur["reward"] = event["rewards"][0]
                    vals = ", ".join(f"{r:.3f}" for r in event["rewards"])
                    blocks.append(
                        f'<div class="stage-block">'
                        f'<span class="stage-label">3. Reward Scoring</span>'
                        f'<span class="stage-math">{_MATH["rewards"]}</span>'
                        f"</div>"
                        f'<div class="stage-scalar">{vals}</div>'
                    )
                    flush()
                elif stage in {"value_loss", "policy_loss"}:
                    label = (
                        "9. Value Function Loss" if stage == "value_loss" else "11. PPO-Clip Loss"
                    )
                    math_key = "value_loss" if stage == "value_loss" else "policy_loss"
                    value = (
                        event.get("value_loss")
                        if stage == "value_loss"
                        else event.get("policy_loss")
                    )
                    if value is not None:
                        if stage == "value_loss":
                            cur["value_loss"] = value
                        else:
                            cur["policy_loss"] = value
                        blocks.append(
                            f'<div class="stage-block">'
                            f'<span class="stage-label">{label}</span>'
                            f'<span class="stage-math">{_MATH[math_key]}</span>'
                            f"</div>"
                            f'<div class="stage-scalar">{value:.4f}</div>'
                        )
                        flush()
                elif stage == "kl_divergence" and "sa_pairs" in event:
                    kls = [p["kl_div"] for p in event["sa_pairs"]]
                    cur["avg_kl"] = sum(kls) / len(kls) if kls else 0.0
                    label, field = _rendering[stage]
                    block_html = _render_heatmap(label, stage, field, event, _MATH)
                    blocks.append(block_html)
                    flush()
                elif stage == "policy_ratio" and "sa_pairs" in event:
                    ratios = [p["policy_ratio"] for p in event["sa_pairs"]]
                    cur["avg_ratio"] = sum(ratios) / len(ratios) if ratios else 1.0
                    label, field = _rendering[stage]
                    block_html = _render_heatmap(label, stage, field, event, _MATH)
                    blocks.append(block_html)
                    # Per-step summary line.
                    step_reward = cur.get("reward", float("nan"))
                    vl = cur.get("value_loss", float("nan"))
                    pl = cur.get("policy_loss", float("nan"))
                    summary = '<div class="step-summary">'
                    summary += "<table><tr>"
                    summary += f"<td>reward</td><td>{step_reward:.2f}</td>"
                    summary += f"<td>value loss</td><td>{vl:.1f}</td>"
                    summary += f"<td>policy loss</td><td>{pl:+.3f}</td>"
                    summary += f"<td>avg KL</td><td>{cur['avg_kl']:.2f}</td>"
                    summary += f"<td>avg ratio</td><td>{cur['avg_ratio']:.3f}</td>"
                    summary += "</tr></table></div>"
                    blocks.append(summary)
                    flush()
                elif stage in _rendering and "sa_pairs" in event:
                    label, field = _rendering[stage]
                    block_html = _render_heatmap(label, stage, field, event, _MATH)
                    blocks.append(block_html)
                    flush()
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")

    ppo_data = st.session_state.get("rlhf-ppo_ppo_data")
    if ppo_data:
        records = st.session_state.get("rlhf-ppo_step_records", [])
        if records and len(records) > 1:
            r_first = records[0].get("reward", float("nan"))
            r_last = records[-1].get("reward", float("nan"))
            vl_first = records[0].get("value_loss", float("nan"))
            vl_last = records[-1].get("value_loss", float("nan"))
            pl_first = records[0].get("policy_loss", float("nan"))
            pl_last = records[-1].get("policy_loss", float("nan"))
            kl_first = records[0].get("avg_kl", float("nan"))
            kl_last = records[-1].get("avg_kl", float("nan"))

            def _arrow(a: float, b: float) -> str:
                if math.isnan(a) or math.isnan(b) or abs(b - a) < 0.01:
                    return " →"
                return " ↑" if b > a else " ↓"

            st.markdown(
                f'<div class="summary-card">'
                f"<h4>{len(records)} PPO steps — before → after</h4>"
                f'<table class="summary-table">'
                f'<tr><th style="text-align:left;font-weight:500;color:var(--text-muted)">metric</th>'
                f'<th style="text-align:right;font-weight:500;color:var(--text-muted)">step 1</th>'
                f'<th style="text-align:right;font-weight:500;color:var(--text-muted)">step {len(records)}</th></tr>'
                f"<tr><td>Reward</td>"
                f'<td style="text-align:right">{r_first:.2f}</td>'
                f'<td style="text-align:right">{r_last:.2f}'
                f'<span class="{"summary-good" if r_last > r_first else "summary-bad" if r_last < r_first else ""}">{_arrow(r_first, r_last)}</span></td></tr>'
                f"<tr><td>Value loss</td>"
                f'<td style="text-align:right">{vl_first:.1f}</td>'
                f'<td style="text-align:right">{vl_last:.1f}'
                f'<span class="summary-{"good" if vl_last < vl_first else "bad" if vl_last > vl_first else ""}">{_arrow(vl_first, vl_last)}</span></td></tr>'
                f"<tr><td>Policy loss</td>"
                f'<td style="text-align:right">{pl_first:+.3f}</td>'
                f'<td style="text-align:right">{pl_last:+.3f}</td></tr>'
                f"<tr><td>KL divergence</td>"
                f'<td style="text-align:right">{kl_first:.2f}</td>'
                f'<td style="text-align:right">{kl_last:.2f}'
                f'<span class="">{_arrow(kl_first, kl_last)}</span></td></tr>'
                f"</table></div>",
                unsafe_allow_html=True,
            )
        st.info(ppo_data["note"])
        with st.expander("Raw response"):
            st.json(ppo_data)

# --- Write-up + diagrams + stats below the demo, so users can try it immediately ---

render_week_details(_PAGE)
