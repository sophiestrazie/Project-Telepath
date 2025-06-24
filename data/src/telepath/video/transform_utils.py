from torchvision.transforms import Compose, Lambda, CenterCrop
from pytorchvideo.transforms import Normalize, UniformTemporalSubsample, ShortSideScale

def define_frames_transform():
    return Compose([
        UniformTemporalSubsample(8),
        Lambda(lambda x: x / 255.0),
        Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
        ShortSideScale(size=256),
        CenterCrop(256),
    ])

