# Copyright 2026 Liquid AI. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Multimodal processor for LFM2-VL models with SigLip2 NaFlex support."""

import asyncio
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.lfm2_vl import Lfm2VlForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import (
    MultimodalSpecialTokens,
)
from sglang.srt.utils import load_image, logger
from sglang.utils import get_exception_traceback


class Lfm2VlImageProcessor(SGLangBaseProcessor):
    """Multimodal processor for LFM2-VL vision-language models.

    Handles image preprocessing using the HuggingFace processor and
    prepares inputs for the LFM2-VL model with SigLip2 NaFlex support.

    NaFlex (Native Flexible resolution) processes images at their native
    resolution using variable-length packed sequences.
    """

    models = [Lfm2VlForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        self.IMAGE_TOKEN_ID = hf_config.image_token_id
        self.IMAGE_TOKEN = "<image>"

        # Image framing token IDs for multi-tile images
        tokenizer = _processor.tokenizer
        self.IMAGE_START_TOKEN_ID = tokenizer.convert_tokens_to_ids("<|image_start|>")
        self.IMAGE_END_TOKEN_ID = tokenizer.convert_tokens_to_ids("<|image_end|>")

        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.IMAGE_TOKEN,
            image_token_id=hf_config.image_token_id,
        ).build(_processor)

    @staticmethod
    def _process_single_image_task(
        image_data: Union[str, bytes],
        processor,
        image_token: str,
        image_token_id: int,
        image_start_token_id: int,
        image_end_token_id: int,
    ) -> Optional[Dict]:
        """Process a single image and return pixel values and metadata.

        This is a static method to allow execution in a ProcessPoolExecutor.
        """
        try:
            image, _ = load_image(image_data)

            if image.mode != "RGB":
                image = image.convert("RGB")

            # Process using HF processor with a dummy text containing image token
            inputs = processor(
                text=f"{image_token}\n",
                images=[image],
                return_tensors="pt",
            )

            pixel_values_padded = inputs["pixel_values"]
            pixel_attention_mask = inputs["pixel_attention_mask"]
            spatial_shapes = inputs["spatial_shapes"]
            num_tiles = pixel_values_padded.shape[0]

            # Pack pixel_values for each tile using its attention mask
            packed_pixel_values_list = []
            packed_attention_masks_list = []

            for tile_idx in range(num_tiles):
                tile_mask = pixel_attention_mask[tile_idx].bool()
                tile_pixel_values = pixel_values_padded[tile_idx][tile_mask]
                packed_pixel_values_list.append(tile_pixel_values)
                packed_attention_masks_list.append(pixel_attention_mask[tile_idx])

            pixel_values_packed = torch.cat(packed_pixel_values_list, dim=0)

            # Calculate projected token count (after 2x2 pixel unshuffle)
            downsample_factor = 2
            tokens_per_tile = []
            for i in range(num_tiles):
                h, w = spatial_shapes[i].tolist()
                tokens_per_tile.append(
                    (h // downsample_factor) * (w // downsample_factor)
                )
            num_projected_tokens = sum(tokens_per_tile)

            # Extract image token sequence from HF processor output (includes framing tokens)
            hf_input_ids = inputs["input_ids"].squeeze(0).tolist()

            start_idx = None
            end_idx = None
            for i, tid in enumerate(hf_input_ids):
                if tid == image_start_token_id and start_idx is None:
                    start_idx = i
                if tid == image_end_token_id:
                    end_idx = i + 1
                    break

            if start_idx is not None and end_idx is not None:
                image_token_sequence = hf_input_ids[start_idx:end_idx]
            else:
                logger.warning(
                    "Could not find image framing tokens, falling back to simple expansion"
                )
                image_token_sequence = [image_token_id] * num_projected_tokens

            return {
                "pixel_values": pixel_values_packed,
                "pixel_attention_mask": packed_attention_masks_list,
                "spatial_shapes": spatial_shapes,
                "num_image_tokens": num_projected_tokens,
                "num_tiles": num_tiles,
                "tokens_per_tile": tokens_per_tile,
                "image_token_sequence": image_token_sequence,
            }
        except Exception:
            logger.error(
                "Exception in Lfm2VlImageProcessor:\n" + get_exception_traceback()
            )
            return None

    async def _process_single_image_async(
        self, image_data: Union[str, bytes]
    ) -> Optional[Dict]:
        """Process a single image asynchronously using cpu_executor."""
        if self.cpu_executor is not None:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self.cpu_executor,
                Lfm2VlImageProcessor._process_single_image_task,
                image_data,
                self._processor,
                self.IMAGE_TOKEN,
                self.IMAGE_TOKEN_ID,
                self.IMAGE_START_TOKEN_ID,
                self.IMAGE_END_TOKEN_ID,
            )
        else:
            return Lfm2VlImageProcessor._process_single_image_task(
                image_data,
                self._processor,
                self.IMAGE_TOKEN,
                self.IMAGE_TOKEN_ID,
                self.IMAGE_START_TOKEN_ID,
                self.IMAGE_END_TOKEN_ID,
            )

    def _text_only_result(self, input_text: str) -> Dict:
        """Return result for text-only request (no images)."""
        input_ids = self._processor.tokenizer(
            input_text, return_tensors="pt", add_special_tokens=False
        ).input_ids
        return {
            "input_ids": input_ids.squeeze(0).tolist(),
            "mm_items": [],
            "im_token_id": self.IMAGE_TOKEN_ID,
        }

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, Dict]],
        input_text: str,
        request_obj,
        *args,
        **kwargs,
    ):
        """Process multimodal data asynchronously."""
        if not image_data:
            return self._text_only_result(input_text)

        # Separate dict items (precomputed) from raw images that need processing
        dict_items = []
        raw_images = []
        raw_image_indices = []

        for idx, img in enumerate(image_data):
            if isinstance(img, dict):
                dict_items.append((idx, img))
            else:
                raw_images.append(img)
                raw_image_indices.append(idx)

        # Process all raw images in parallel using asyncio.gather
        processed_results = []
        if raw_images:
            tasks = [self._process_single_image_async(img) for img in raw_images]
            processed_results = await asyncio.gather(*tasks)

        # Combine results in original order
        image_results = [None] * len(image_data)

        # Place dict items
        for idx, item in dict_items:
            image_results[idx] = item

        # Place processed results
        for i, result in enumerate(processed_results):
            if result is not None:
                image_results[raw_image_indices[i]] = result

        # Filter out None results
        image_results = [r for r in image_results if r is not None]

        if not image_results:
            return self._text_only_result(input_text)

        # Tokenize the text
        input_ids = self._processor.tokenizer(
            input_text, return_tensors="pt", add_special_tokens=False
        ).input_ids.squeeze(0)

        image_token_sequences = [
            result["image_token_sequence"] for result in image_results
        ]
        image_token_counts = [result["num_image_tokens"] for result in image_results]

        # Expand input_ids: replace each <image> token with the full image token sequence
        expanded_input_ids = []
        img_idx = 0
        for token_id in input_ids.tolist():
            if token_id == self.IMAGE_TOKEN_ID and img_idx < len(image_token_sequences):
                expanded_input_ids.extend(image_token_sequences[img_idx])
                img_idx += 1
            else:
                expanded_input_ids.append(token_id)

        input_ids = torch.tensor(expanded_input_ids, dtype=torch.long)

        # Calculate offsets for each image's tokens (may be non-contiguous for multi-tile)
        all_offsets = SGLangBaseProcessor.get_mm_items_offset(
            input_ids, self.IMAGE_TOKEN_ID
        )

        offsets_per_image = []
        region_idx = 0
        for num_tokens in image_token_counts:
            image_offsets = []
            tokens_assigned = 0
            while tokens_assigned < num_tokens and region_idx < len(all_offsets):
                start, end = all_offsets[region_idx]
                region_tokens = end - start + 1
                image_offsets.append((start, end))
                tokens_assigned += region_tokens
                region_idx += 1
            offsets_per_image.append(image_offsets)

        # Create MultimodalDataItem for each image
        mm_items = []
        for i, result in enumerate(image_results):
            if isinstance(result["pixel_attention_mask"], list):
                pixel_attention_mask = torch.stack(
                    [
                        m if isinstance(m, torch.Tensor) else torch.from_numpy(m)
                        for m in result["pixel_attention_mask"]
                    ]
                )
            else:
                pixel_attention_mask = result["pixel_attention_mask"]

            pixel_attn_mask_np = (
                pixel_attention_mask.numpy()
                if isinstance(pixel_attention_mask, torch.Tensor)
                else pixel_attention_mask
            )
            spatial_shapes_np = (
                result["spatial_shapes"].numpy()
                if isinstance(result["spatial_shapes"], torch.Tensor)
                else result["spatial_shapes"]
            )

            mm_item = MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=result["pixel_values"].numpy().astype(np.float16),
                offsets=offsets_per_image[i] if i < len(offsets_per_image) else None,
                model_specific_data={
                    "pixel_attention_mask": pixel_attn_mask_np,
                    "spatial_shapes": spatial_shapes_np,
                },
            )
            mm_items.append(mm_item)

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "im_token_id": self.IMAGE_TOKEN_ID,
        }
