# Signal Flow: LLM-Depression Pipeline

End-to-end view of how a raw audio recording becomes a depression prediction,
from dataset files on disk through model inference and metrics.

---

## Overview

```
Raw Dataset (WAV + CSV + labels)
    │
    ▼
[Stage 1]  Manifest Build      — catalogue chunks, join labels & transcripts
    │
    ▼
[Stage 2]  Example Building    — build prompts, route by dataset/mode
    │
    ▼
[Stage 3]  Audio Loading       — read WAV, resample, optional augmentation
    │
    ▼
[Stage 4]  Processor / Collation  — mel spectrogram + tokenization + label mask
    │
    ▼
[Stage 5]  Model Forward Pass  — Whisper encoder → projector → Qwen2 decoder
    │
    ├──► [Training]   cross-entropy loss on label tokens → LoRA backprop
    │
    └──► [Inference]  likelihood scoring or generation → prediction
    │
    ▼
[Stage 6]  Aggregation & Metrics  — subject-level vote → macro-F1
```

---

## Stage 1 — Manifest Build

**Code:** `src/data/build_manifest.py`, `src/data/{daic,edaic,cmdc,eatd}.py`

The pipeline does **not** perform audio segmentation. Chunks arrive pre-cut from
an offline preprocessing step outside this repository.

```
dataset_root/
├── train_audio_segments/
│   └── {subject_id}/
│       ├── 300_segment_0.wav      ← already-cut chunk
│       ├── 300_segment_1.wav
│       └── ...
├── train_preprocessing_summary.csv
│     subject_id | PHQ8_Binary | segment_files              | full_transcript
│     300        | 1           | ["300_segment_0.wav", ...] | "entire interview..."
└── dev_preprocessing_summary.csv
```

What manifest build does: **join + catalogue**

```
preprocessing_summary.csv  +  audio segment files on disk
              │
              ▼
    For each segment_file listed per subject:
      one JSONL row → {
        subject_id:  "300"
        sample_id:   "300_segment_0"
        audio_path:  ".../300_segment_0.wav"
        transcript:  "full interview text"   ← identical for all chunks of subject
        label:       1
        split:       "train"
      }
              │
              ▼
    outputs/manifests/daic_manifest.jsonl   (one row per chunk)
    outputs/splits/daic_subject_partitions.json
```

No audio I/O, no signal processing occurs here.

---

## Stage 2 — Example Building

**Code:** `src/data/runtime.py :: build_examples()`

`build_examples()` is a **router**. It reads `dataset` and `sample_mode` from
config and delegates to one of four paths. It also loads the optional SECap
emotion caption cache before routing.

### Routing table

| dataset | sample_mode / modality | result |
|---|---|---|
| daic / edaic | `text_only` | 1 subject → 1 example, no audio |
| daic / edaic | `subject_audio` | 1 subject → 1 example, K chunks |
| eatd | `subject` (default) | 1 subject → 1 example, 3 responses |
| any | `chunk` / `response` | 1 manifest row → 1 example |

### Chunk mode (default DAIC/EDAIC)

```
manifest row (300_segment_0)
    │
    ▼
1 example: audio_paths = [300_segment_0.wav]
```

### Subject-audio mode (K-chunk, DAIC/EDAIC)

```
48 chunk rows for subject 300
    │
    ├─ Eval:  deterministic evenly-spaced K indices → [seg_0, seg_16, seg_32, seg_47]
    └─ Train: K randomly sampled per epoch → fresh view each epoch

    ▼
1 example: audio_paths = [seg_A.wav, seg_B.wav, seg_C.wav, seg_D.wav]
```

Designed to reduce audio overfitting by preventing the model from memorising
subject identity across many near-identical chunk examples.

### EATD subject mode

```
EATD on disk per subject:
  negative.wav / negative.txt
  neutral.wav  / neutral.txt
  positive.wav / positive.txt
  label.txt  (SDS score → binary at threshold 53.0)

    ▼
1 example: audio_paths = [negative.wav, neutral.wav, positive.wav]
           transcript  = "[Negative response]\n...\n\n[Neutral response]\n..."
```

Captures affect variation across emotional contexts.

### Prompt construction

After routing, every example ends up with two text fields:

**`prompt_text`** — what the model sees at inference:
```
<|im_start|>system
You are a psychologist analyzing speech and transcript information for depression screening.<|im_end|>
<|im_start|>user
Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>
Audio 2: <|audio_bos|><|AUDIO|><|audio_eos|>
The subject's speech audio is provided in 2 segments sampled from the interview.
The transcript of the subject's speech is:
{transcript}

Based on audio and transcript, determine whether the subject is Depressed or Non-depressed.
Answer with exactly one label.<|im_end|>
<|im_start|>assistant
```

**`training_text`** — prompt + label appended for SFT:
```
{prompt_text}Depressed<|im_end|>
```

---

## Stage 3 — Audio Loading & Augmentation

**Code:** `src/data/runtime.py :: load_audio_array()`, `apply_audio_augment()`

Called inside `AudioTextDataset.__getitem__()` at batch time — **not** during
example building.

```
audio_path (e.g. 300_segment_0.wav)
    │
    ├─ sf.read() → float32 waveform  [T_samples]
    │
    ├─ Stereo → Mono: mean(channels)
    │
    ├─ Trim: keep min(max_seconds, duration) → [T_kept]
    │
    ├─ Resample (librosa): src_sr → 16000 Hz → [T_16k]
    │
    └─ [TRAIN ONLY] Augmentation (apply_audio_augment):
         ├─ Pitch shift    ±N semitones        (librosa.effects)
         ├─ Time stretch   rate ∈ [lo, hi]     (librosa.effects)
         ├─ Gain           ±dB                 (NumPy)
         └─ Additive noise SNR dB              (NumPy)
       Clipped to [-1.0, 1.0], length ≤ original

Output: np.float32 array @ 16kHz
```

Augmentation is **train-only**. Eval loads audio through `src/evaluate.py`
directly, never through `AudioTextDataset`, preserving determinism.

---

## Stage 4 — Processor & Collation

**Code:** `src/model/collator.py :: Qwen2AudioSFTCollator`

### Audio path

```
np.float32 waveform @ 16kHz
    │
    └─ Whisper FeatureExtractor (inside AutoProcessor)
         ├─ Pad / trim to 30s window (480,000 samples)
         ├─ STFT: hop=160, win=400, n_fft=400
         ├─ 128-band Mel filterbank
         └─ Log-mel spectrogram

Output: input_features  [1, 128, 3000]   (128 mels × 3000 frames per chunk)
```

### Text path

```
training_text / prompt_text  (string)
    │
    └─ Qwen2 Tokenizer
         └─ input_ids  [seq_len]
              <|AUDIO|> placeholder tokens mark where audio embeddings slot in
```

### Label mask

```
labels = input_ids.copy()
labels[:prompt_len] = -100     ← prompt tokens: ignored by cross-entropy
labels[prompt_len:] = token ids for "Depressed" / "Non-depressed"
```

### Batch output (torch tensors)

| tensor | shape | notes |
|---|---|---|
| `input_ids` | `[B, max_seq_len]` | right-padded |
| `attention_mask` | `[B, max_seq_len]` | |
| `labels` | `[B, max_seq_len]` | `-100` on prompt tokens |
| `input_features` | `[B×K, 128, 3000]` | K audios per example |
| `feature_attention_mask` | `[B×K, 3000]` | |

---

## Stage 5 — Model Forward Pass

**Model:** `Qwen2AudioForConditionalGeneration` + LoRA (PEFT)

### Audio branch (Whisper encoder)

```
input_features [B×K, 128, 3000]
    │
    └─ Whisper Encoder  (frozen by default)
         ├─ Conv1d stride-2  → 1500 frames
         ├─ Conv1d stride-2  → 750 frames
         └─ Transformer layers
              → audio_hidden  [B×K, 750, 1280]

    [optional] DepAdapter (bottleneck on encoder output)
         down_proj 1280→512 → GELU → up_proj 512→1280
         + LayerNorm + residual
              → adapted_audio  [B×K, 750, 1280]

    Multi-Modal Projector (linear)
         1280 → 4096
              → audio_embeds  [B×K, 750, 4096]
```

### Text branch + merge

```
input_ids [B, seq_len]
    │
    └─ Qwen2 Embedding table
         → text_embeds [B, seq_len, 4096]

<|AUDIO|> token positions replaced with audio_embeds
    → merged_embeds [B, seq_len + K×750 − K, 4096]
```

### LLM decoder

```
merged_embeds
    │
    └─ 28× Qwen2 Transformer Decoder layers
         LoRA applied to: q_proj, k_proj, v_proj, o_proj,
                          gate_proj, up_proj, down_proj
         rank=16, alpha=32, dropout=0.05

         → logits  [B, full_seq_len, vocab_size]
```

### Shape summary

| step | input | output |
|---|---|---|
| WAV read | variable T @ src_sr | `[T_16k]` float32 |
| Mel spectrogram | `[480000]` @16kHz | `[128, 3000]` |
| Whisper encoder | `[128, 3000]` | `[750, 1280]` |
| DepAdapter (opt.) | `[750, 1280]` | `[750, 1280]` |
| Projector | `[750, 1280]` | `[750, 4096]` |
| Merged embeds | `[seq_len]` tokens | `[seq_len+K×750, 4096]` |
| LLM decoder | `[full_seq, 4096]` | `[full_seq, vocab_size]` |

---

## Training Path

```
logits vs labels  (cross-entropy, prompt tokens masked with -100)
    │
    └─ Backprop through:
         LoRA weights only          (always)
         DepAdapter weights         (if audio_adapter.enabled=true)
         Multi-modal projector      (if audio_adapter.train_projector=true)
         Whisper encoder            (frozen by default)

Optimiser:  AdamW + linear LR warmup
Checkpoint: saved when val macro-F1 improves
```

---

## Inference Path

### Mode A — Likelihood (headline, deterministic)

```
For each candidate label ("Depressed", "Non-depressed"):
    full_text = prompt_text + candidate_label
    → model forward pass
    → logits at label token positions
    → mean log-prob across label tokens → score

pred = argmax(dep_score, non_score)  → {0, 1}
```

No sampling, fully deterministic. Used for checkpoint selection and
all reported headline metrics.

### Mode B — Generation (secondary)

```
model.generate(prompt_text)
    greedy decode (num_beams=1, do_sample=false)
    → token ids → decode → text string
    → parse: "Depressed" → 1 / "Non-depressed" → 0 / else → INVALID
```

---

## Stage 6 — Aggregation & Metrics

**Code:** `src/aggregate.py`, `src/metrics.py`

```
Per-sample predictions (one row per chunk / segment)
    │
    └─ Group by subject_id
         majority vote  OR  mean likelihood score
              → subject-level prediction {0, 1}

classification_metrics(y_true, y_pred):
    accuracy, macro-F1, weighted-F1
    precision / recall per class
    confusion matrix [[TN, FP], [FN, TP]]
```

### Output files

```
eval/best_checkpoint/
    predictions.csv               ← per-sample rows
    subject_predictions.csv       ← one row per subject
    metrics.json                  ← all classification metrics
best_vs_last_checkpoint_metrics.json
```

---

## Input Modality Modes

| mode | audio | text | prompt audio placeholders |
|---|---|---|---|
| `audio_text` | yes | yes | K × `<\|AUDIO\|>` |
| `audio_only` | yes | no | K × `<\|AUDIO\|>` |
| `text_only` | no | yes | none (Qwen2-7B-Instruct used instead) |
