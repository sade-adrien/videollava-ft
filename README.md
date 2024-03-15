# videollava-ft

This repo proposes to fine-tune videollava for a specific application (fine-tuned with lora applied to the language model, and releasing the multimodal projector).

In training, we add random crop, random tilt to the images and try to select non-blurry images. The images are selected randomly from available time segments provided in the dataset.
In inference, we use original videollava implementation (no crop, no tilt, uniform sampling).

To implement this new training behavior the following files have been modified:
```
/videollava/model/multimodal_encoder/languagebind/video/processing_video_FT.py
/videollava/model/multimodal_encoder/languagebind/__init__.py
/videollava/model/builder.py
/videollava/model/llava_arch.py
/videollava/train//train_FT.py
/videollava/train/train_mem.py
```

We build a dataset from an existing one: we have videos of recipes along with relevant time segments and associated labels describing the recipe step in a very short sentence.
We augment this data using llava-34b to complete labels when we have less than 8. Then we use mixtral to create a recipe-like paragraph from the list of steps.
Data used is then videos, along with relevant segments (at least 8), and a recipe paragraph.

The training process presents good results with the following parameters:
```
lora_r 128
lora_alpha 256
lora_dropout 0.05
mm_projector_lr 5e-6
num_train_epochs 1
per_device_train_batch_size 1
gradient_accumulation_steps 1
weight_decay 0.
warmup_ratio 0.03
lr_scheduler_type "cosine"
model_max_length 2048
tokenizer_model_max_length 3072
gradient_checkpointing False
```
