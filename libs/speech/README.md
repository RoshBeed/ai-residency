# speech

Loading, feature extraction and decoding for Whisper — shared by `whisper-fine-tuning`
and `speech-emotion-recognition`.

Both services fine-tune Whisper. They had entirely separate implementations, one on
`openai-whisper` and one on `transformers`, because different teams built them in
different weeks. There was never a technical reason for the split.

```python
from speech import features_from_file, load_whisper, transcribe

whisper = load_whisper("openai/whisper-tiny")
features = features_from_file(whisper.processor, "clip.wav")
print(transcribe(whisper, features))
```

## Why `transformers`, not `openai-whisper`

`transformers` does everything `openai-whisper` did here, plus one thing it cannot.

| | `openai-whisper` | `transformers` |
|---|---|---|
| log-mel features | `log_mel_spectrogram` | **bit-identical** output, verified in the tests |
| decoder targets | prefix assembled by hand from special-token ids | tokenizer emits it |
| loss | manual cross-entropy | `model(labels=…).loss` |
| decode | `whisper.decode` | `generate()` |
| **extend the vocabulary** | **not supported cleanly** | `add_special_tokens` + `resize_token_embeddings` |

That last row decides it: `speech-emotion-recognition` exists to make Whisper's decoder
emit an emotion token it was never trained on. Standardising the other way round was
never an option.

Two things went away with the dependency:

- **The system `ffmpeg` requirement.** `openai-whisper` shells out to the binary in
  `load_audio`; librosa does not.
- **A device workaround.** `whisper.load_model` chooses between CUDA and CPU only — no
  MPS branch — so on Apple Silicon it ran on CPU with the GPU idle, and the service
  carried a comment explaining that. Here you pass a device, and the problem does not
  exist.

## The invariant this holds

**The tokenizer and the model's embedding table must agree about how many tokens exist.**

```python
whisper = load_whisper("openai/whisper-tiny", extra_tokens=["<|happy|>", "<|sad|>"])
```

Adding tokens to the tokenizer without resizing the model leaves the new ids indexing
past the end of the embedding table — a crash if you are lucky, and silently wrong if
they happen to land inside it. `load_whisper` does both or neither, and refuses a
checkpoint whose vocabulary does not match the tokenizer it is being loaded against.

## What is not here

Training loops. They genuinely differ: one takes five gradient steps on a single clip
and streams a decode after each, the other runs epochs over a corpus with evaluation.
What they share is loading, features, labels and decoding — which is all of this.
