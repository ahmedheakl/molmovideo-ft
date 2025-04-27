import copy
import os
from dataclasses import dataclass, field
from typing import Dict
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset
from PIL import Image
from decord import VideoReader, cpu

from .params import DataArguments
from .constants import *
from .utils import SYSTEM_PROMPT, PROMPT_TEMPLATES, get_coords, compute_mse_points, plot_metric, extract_caption, get_points_in_xml_format
from glob import glob
import random 


MAX_PATCHES = 26
MAX_INPUT_IDS = 3000

def sample_frames(video_path, frame_indices, max_num_frames=2):
    start_index = random.sample(range(len(frame_indices)), 1)[0]
    ok = False
    while not ok:
        image_path = os.path.join(video_path, f"{frame_indices[start_index]:05d}.jpg")
        if os.path.exists(image_path):
            ok = True
        start_index = random.sample(range(len(frame_indices)), 1)[0]
    selected_indices = []
    for i in range(max_num_frames):
        if start_index - i >= 0:
            selected_indices.append(start_index - i)
    selected_indices = selected_indices[::-1]
    selected_frame_idxs = [frame_indices[i] for i in selected_indices]
    images = [f"{video_path}/{frame_indices[i]:05d}.jpg" for i in selected_indices]
    images = [Image.open(v).convert("RGB") for v in images]
    if len(images) < max_num_frames:
        black_image = Image.new('RGB', images[0].size, (0, 0, 0))
        images = [black_image] * (max_num_frames - len(images)) + images
        selected_frame_idxs = [0] * (max_num_frames - len(images)) + selected_frame_idxs
    return images, selected_frame_idxs


def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        padding=True,
    ):
        super(SupervisedDataset, self).__init__()
        extention = os.path.splitext(data_path)[-1] 
        if isinstance(data_path, str) and extention == '.json':
            list_data_dict = json.load(open(data_path, "r"))
        elif isinstance(data_path, str) and extention == '.jsonl':
            list_data_dict = []
            with open(data_path, "r") as f:
                for line in f:
                    list_data_dict.append(json.loads(line))
        else:
            # it is a folder with .jsonl inside of it
            list_data_dict = []
            for file in os.listdir(data_path):
                if file.endswith('.jsonl'):
                    with open(os.path.join(data_path, file), "r") as f:
                        for line in f:
                            list_data_dict.append(json.loads(line))

        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.fps = data_args.fps
        self.eos_token_id = processor.tokenizer.eos_token_id
        self.max_num_frames = 2

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        is_video = False
        num_frames = None

        processor = self.processor
        if "image" in sources:
            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]

            images = []
           
            for image_file in image_files:
                if not os.path.exists(image_file):
                    image_file = os.path.join(image_folder, image_file)
                images.append(Image.open(image_file).convert("RGB"))

        # Molmo does not support viedos for now, and it's for future update.
        elif "video" in sources:
            video_file = sources["video"]
            video_folder = self.data_args.image_folder

            if not os.path.exists(video_file):
                video_file = os.path.join(video_folder, video_file)

            images, selected_frame_idxs = sample_frames(video_file, sources['frame_idxs'], self.max_num_frames)
            caption = sources['caption']
            question = SYSTEM_PROMPT.format(num_frames=self.max_num_frames, selected_frame_idxs=selected_frame_idxs) + ' ' + random.choice(PROMPT_TEMPLATES).format(label=caption)
            selected_points = {i: sources['points'][i] for i in selected_frame_idxs[-1:]}
            answer = get_points_in_xml_format(selected_points, caption)
            sources['conversations'] = [
                {
                    "from": "human",
                    "value": f"{LLAVA_VIDEO_TOKEN}\n{question}"
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ]
            is_video = True
            num_frames = len(images)

        else:
            images = None

        sources = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=is_video))

        all_input_ids = [torch.tensor([self.eos_token_id])] # bos token id = eos token id
        all_labels = [torch.tensor([-100])] # ignore bos token
        all_images = []
        all_image_masks = []
        all_image_input_idx = []

        for idx, j in enumerate(range(0, len(sources), 2)):
            user_input = sources[j]
            gpt_response = sources[j + 1]
            
            gpt_prompt = f" {gpt_response['content']}"
            
            if idx == 0:
                user_prompt = user_input['content']
                inputs = processor.process(text=user_prompt, images=images)
                inputs['input_ids'] = torch.cat(
                    [inputs['input_ids'], torch.full((MAX_INPUT_IDS - inputs['input_ids'].shape[0],), self.processor.tokenizer.pad_token_id)],
                    dim=0
                )

                # pad images (only 3 dims)
                inputs['images'] = torch.cat(
                    [inputs['images'], torch.full((MAX_PATCHES - inputs['images'].shape[0], inputs['images'].shape[1], inputs['images'].shape[2]), 0)],
                    dim=0
                )

                # pad image_input_idx
                pad_image_input_idx = torch.full(
                    (MAX_PATCHES - inputs['image_input_idx'].shape[0], inputs['image_input_idx'].shape[1]),
                    0
                )
                inputs['image_input_idx'] = torch.cat([inputs['image_input_idx'], pad_image_input_idx], dim=0)

                # pad image_masks
                pad_image_masks = torch.full(
                    (MAX_PATCHES - inputs['image_masks'].shape[0], inputs['image_masks'].shape[1]),
                    -1
                )
                inputs['image_masks'] = torch.cat([inputs['image_masks'], pad_image_masks], dim=0)

                prompt_input_ids = inputs['input_ids'].unsqueeze(0)
                all_images.append(inputs['images'].unsqueeze(0))
                all_image_input_idx.append(inputs['image_input_idx'].unsqueeze(0))
                all_image_masks.append(inputs['image_masks'].unsqueeze(0))

            else:
                user_prompt = f" {user_input['role'].capitalize()}: {user_input['content']} {gpt_response['role'].capitalize()}"
                prompt_input_ids = processor.tokenizer(user_prompt, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            response_input_ids = processor.tokenizer(gpt_prompt, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),
                    response_input_ids.squeeze(0),
                ],
                dim=0,
            )

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        
        all_input_ids.append(torch.tensor([self.eos_token_id]))  # eos token id
        all_labels.append(torch.tensor([self.eos_token_id]))  # eos token id
        
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)

        images = torch.cat(all_images, dim=0)
        image_input_idx = torch.cat(all_image_input_idx, dim=0)
        image_masks = torch.cat(all_image_masks, dim=0)


        attention_mask = (input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            images=images,
            image_input_idx=image_input_idx,
            image_masks=image_masks,
        )
        
        return data_dict

class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_images = []
        batch_image_input_idx = []
        batch_image_mask = []

        for example in examples:
            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])
            batch_images.append(example["images"])
            batch_image_input_idx.append(example["image_input_idx"])
            batch_image_mask.append(example["image_masks"])
        
        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )

        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX)
        
        images = torch.cat(batch_images, dim=0)
        image_input_idx = torch.cat(batch_image_input_idx, dim=0)
        image_masks = torch.cat(batch_image_mask, dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images": images,
            "image_input_idx": image_input_idx,
            "image_masks": image_masks
        }

def replace_image_tokens(input_string, is_video=False):

    if is_video:
        input_string = input_string.replace(LLAVA_VIDEO_TOKEN+'\n', '')

    else:
        input_string = input_string.replace(LLAVA_IMAGE_TOKEN+'\n', '')

    return input_string

def llava_to_openai(conversations, is_video=False):
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    for conversation in conversations:
        transformed_content = replace_image_tokens(conversation["value"], is_video=is_video)
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        transformed_data.append(transformed_entry)

    return transformed_data

def make_supervised_data_module(processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = SupervisedDataset(
        data_path=data_args.data_path, processor=processor, data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id)

    return dict(train_dataset=sft_dataset,
                eval_dataset=None,
                data_collator=data_collator)