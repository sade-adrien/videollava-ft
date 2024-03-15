
import torch
import cv2
import decord
import random
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from torchvision import transforms
from transformers import ProcessorMixin, BatchEncoding
from transformers.image_processing_utils import BatchFeature
from pytorchvideo.data.encoded_video import EncodedVideo
from  torchvision.transforms import InterpolationMode
from torchvision.transforms import Compose, Lambda, ToTensor
from torchvision.transforms._transforms_video import NormalizeVideo, RandomCropVideo, RandomHorizontalFlipVideo, CenterCropVideo
from torchvision.transforms._functional_video import crop
from torchvision.transforms.functional import rotate
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample

######
from torchvision.transforms import ToPILImage

decord.bridge.set_bridge('torch')

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

def is_blurry(image, threshold=250):
    image = image.float().permute(2, 0, 1) #(H, W, C) --> (C, H, W)
    image = np.array(ToPILImage()(image))  ##Yes this is stupid but it converts in the right format/precision
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

def build_frame_list(video, segments):
    nb_frames = len(segments)
    fps = video.get_avg_fps()
    frame_id_list = []

    for i in range(nb_frames):
        blurry = True
        cnt = 0
        while blurry and cnt < 10:
            index = np.random.randint(int(fps * segments[i][0]), int(fps * segments[i][1]) + 1)
            image = video[index]
            blurry = is_blurry(image)
            cnt += 1
        frame_id_list.append(index)
    return frame_id_list

def make_list_of_images(x):
    if not isinstance(x, list):
        return [x]
    return x

def get_video_transform(config):
    training = getattr(config, 'training', False)
    config = config.vision_config
    if config.video_decode_backend == 'pytorchvideo':
        transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(config.num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                    ShortSideScale(size=224),
                    CenterCropVideo(224),
                    RandomHorizontalFlipVideo(p=0.5),
                ]
            ),
        )

    elif config.video_decode_backend == 'decord':   ###this is the one we actually use -et
        if training:
            transform = Compose(
                [
                    # UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                    Lambda(lambda x: crop(x, torch.randint(0, int(.07*x.shape[-2]), (1,)).item(), torch.randint(0, int(.07*x.shape[-1]), (1,)).item(), int(.93*x.shape[-2])-1, int(.93*x.shape[-1])-1)), #random crop
                    Lambda(lambda x: rotate(x, torch.rand(1).item() * 10 - 5, interpolation=InterpolationMode.BILINEAR, expand=True)), #random tilt
                    ShortSideScale(size=224),
                    CenterCropVideo(224),
                    RandomHorizontalFlipVideo(p=0.5),
                ]
            )

        else:
            transform = Compose(
                [
                    # UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                    ShortSideScale(size=224),
                    CenterCropVideo(224),
                    RandomHorizontalFlipVideo(p=0.5),
                ]
            )

    elif config.video_decode_backend == 'opencv':
        transform = Compose(
            [
                # UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                ShortSideScale(size=224),
                CenterCropVideo(224),
                RandomHorizontalFlipVideo(p=0.5),
            ]
        )
    else:
        raise NameError('video_decode_backend should specify in (pytorchvideo, decord, opencv)')
    return transform


def load_and_transform_video(
        video_path,
        transform,
        video_decode_backend='opencv',
        clip_start_sec=0.0,
        clip_end_sec=None,
        num_frames=8,
        segments=None,
        display_images=False,
        training=False,
):

    if video_decode_backend == 'pytorchvideo':
        #  decord pyav
        video = EncodedVideo.from_path(video_path, decoder="decord", decode_audio=False)
        duration = video.duration
        start_sec = clip_start_sec  # secs
        end_sec = clip_end_sec if clip_end_sec is not None else duration  # secs
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
        video_outputs = transform(video_data)

    elif video_decode_backend == 'decord':
        decord.bridge.set_bridge('torch')
        decord_vr = VideoReader(video_path, ctx=cpu(0))

        if training:
            assert(segments is not None and len(segments) >= num_frames), f'Error with segments: {segments}'

            selected_segments = random.sample(segments, k=num_frames)
            selected_segments = sorted(selected_segments, key=lambda x: (x[0] + x[1]) / 2)
            frame_id_list = build_frame_list(video=decord_vr, segments=selected_segments)

        else:
            duration = len(decord_vr)
            frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)

        video_data = decord_vr.get_batch(frame_id_list)
        video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
        video_outputs = transform(video_data)

        if display_images:
            import matplotlib.pyplot as plt
            images = video_outputs.permute(1, 2, 3, 0)
            num_rows = int(np.ceil(np.sqrt(images.shape[0])))
            num_cols = int(np.ceil(images.shape[0] / num_rows))

            # Create a subplot grid
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

            # Flatten the axes if needed
            if num_rows == 1 and num_cols == 1:
                axes = np.array([axes])

            # Loop through the images and plot them
            for i in range(images.shape[0]):
                row_idx = i // num_cols
                col_idx = i % num_cols
                axes[row_idx, col_idx].imshow(images[i])
                axes[row_idx, col_idx].axis('off')  # Optional: Turn off axis labels

            # Adjust layout for better spacing
            plt.tight_layout()
            plt.show()


    elif video_decode_backend == 'opencv':
        cv2_vr = cv2.VideoCapture(video_path)
        # = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)

        video_data = []
        for frame_idx in frame_id_list:
            cv2_vr.set(1, frame_idx)
            _, frame = cv2_vr.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_data.append(torch.from_numpy(frame).permute(2, 0, 1))
        cv2_vr.release()
        video_data = torch.stack(video_data, dim=1)
        video_outputs = transform(video_data)
    else:
        raise NameError('video_decode_backend should specify in (pytorchvideo, decord, opencv)')
    return video_outputs

class LanguageBindVideoProcessor(ProcessorMixin):
    attributes = []
    tokenizer_class = ("LanguageBindVideoTokenizer")

    def __init__(self, config, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.transform = get_video_transform(config)
        self.image_processor = load_and_transform_video
        self.tokenizer = tokenizer

    def __call__(self, images=None, text=None, context_length=77, return_tensors=None, segments=None, display_images=False, **kwargs):
        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be none.")

        if text is not None:
            encoding = self.tokenizer(text, max_length=context_length, padding='max_length',
                                      truncation=True, return_tensors=return_tensors, **kwargs)

        if images is not None:
            images = make_list_of_images(images)
            image_features = [self.image_processor(image, self.transform,
                                                   video_decode_backend=self.config.vision_config.video_decode_backend,
                                                   num_frames=self.config.vision_config.num_frames,
                                                   segments=segments,
                                                   display_images=display_images,
                                                   training=getattr(self.config, 'training', False),) for image in images]
            image_features = torch.stack(image_features)

        if text is not None and images is not None:
            encoding["pixel_values"] = image_features
            return encoding
        elif text is not None:
            return encoding
        else:
            return {"pixel_values": image_features}

    def preprocess(self, images, return_tensors):
        return self.__call__(images=images, return_tensors=return_tensors)

    def batch_decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)

    def decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)
