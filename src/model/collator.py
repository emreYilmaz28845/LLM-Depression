from __future__ import annotations

from typing import Any

import numpy as np
import torch


class Qwen2AudioSFTCollator:
    def __init__(self, processor, debug: bool = False):
        self.processor = processor
        self.debug = debug
        self.last_debug_example: dict[str, Any] | None = None

    def _processor_kwargs(self, text: str, audio: list[np.ndarray] | None) -> dict[str, Any]:
        kwargs = {
            "text": text,
            "return_tensors": None,
            "padding": False,
        }
        if audio is not None:
            kwargs["audio"] = audio
            kwargs["sampling_rate"] = int(self.processor.feature_extractor.sampling_rate)
        return kwargs

    def _process_single(self, example: dict[str, Any]) -> dict[str, Any]:
        audio = example["audio_arrays"] if example["audio_arrays"] else None
        processor_kwargs = self._processor_kwargs(example["training_text"], audio)
        prompt_kwargs = self._processor_kwargs(example["prompt_text"], audio)
        processed_full = self.processor(
            **processor_kwargs,
        )
        processed_prompt = self.processor(
            **prompt_kwargs,
        )
        input_ids = np.asarray(processed_full["input_ids"], dtype=np.int64)
        attention_mask = np.asarray(processed_full["attention_mask"], dtype=np.int64)
        prompt_ids = np.asarray(processed_prompt["input_ids"], dtype=np.int64)
        if input_ids.ndim == 2:
            input_ids = input_ids[0]
        if attention_mask.ndim == 2:
            attention_mask = attention_mask[0]
        if prompt_ids.ndim == 2:
            prompt_ids = prompt_ids[0]
        prompt_len = int(len(prompt_ids))
        labels = input_ids.copy()
        labels[:prompt_len] = -100
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_len": prompt_len,
            "sample_id": example["sample_id"],
            "subject_id": example["subject_id"],
            "label": int(example["label"]),
        }
        if audio is not None:
            output["input_features"] = np.asarray(processed_full["input_features"], dtype=np.float32)
            output["feature_attention_mask"] = np.asarray(processed_full["feature_attention_mask"], dtype=np.int64)
        if self.debug and self.last_debug_example is None:
            self.last_debug_example = {
                "sample_id": example["sample_id"],
                "decoded_training_text": example["training_text"],
                "decoded_prompt_text": example["prompt_text"],
                "input_ids": input_ids.tolist(),
                "labels": labels.tolist(),
                "unmasked_token_ids": [int(token_id) for token_id in labels[prompt_len:] if token_id != -100],
            }
        return output

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        processed_items = [self._process_single(example) for example in batch]
        max_seq_len = max(len(item["input_ids"]) for item in processed_items)
        pad_token_id = int(self.processor.tokenizer.pad_token_id)
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        for item in processed_items:
            pad_len = max_seq_len - len(item["input_ids"])
            batch_input_ids.append(np.pad(item["input_ids"], (0, pad_len), constant_values=pad_token_id))
            batch_attention_mask.append(np.pad(item["attention_mask"], (0, pad_len), constant_values=0))
            batch_labels.append(np.pad(item["labels"], (0, pad_len), constant_values=-100))

        batch_dict: dict[str, Any] = {
            "input_ids": torch.tensor(np.stack(batch_input_ids), dtype=torch.long),
            "attention_mask": torch.tensor(np.stack(batch_attention_mask), dtype=torch.long),
            "labels": torch.tensor(np.stack(batch_labels), dtype=torch.long),
        }
        if any("loss_weight" in example for example in batch):
            batch_dict["loss_weight"] = torch.tensor(
                [float(example.get("loss_weight", 1.0)) for example in batch],
                dtype=torch.float32,
            )

        if "input_features" in processed_items[0]:
            all_features = np.concatenate([item["input_features"] for item in processed_items], axis=0)
            all_feature_masks = np.concatenate([item["feature_attention_mask"] for item in processed_items], axis=0)
            batch_dict["input_features"] = torch.tensor(all_features, dtype=torch.float32)
            batch_dict["feature_attention_mask"] = torch.tensor(all_feature_masks, dtype=torch.long)
        return batch_dict


SFTCollator = Qwen2AudioSFTCollator
