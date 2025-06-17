# pytorch-image-models/imagenet_msgpack_dataset.py
import msgpack, io
from torch.utils.data import Dataset
from PIL import Image

class ImageNetMsgpackDataset(Dataset):
    def __init__(self, path, transform=None, max_samples=None):
        self.transform = transform
        self.samples = []
        with open(path, 'rb') as f:
            unpacker = msgpack.Unpacker(f, raw=False)
            for i, sample in enumerate(unpacker):
                self.samples.append(sample)
                if max_samples and i >= max_samples:
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(io.BytesIO(sample["image"])).convert("RGB")
        label = sample["label"]
        if self.transform:
            img = self.transform(img)
        return img, label
