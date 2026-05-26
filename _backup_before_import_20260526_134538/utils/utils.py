from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def get_transform(image_size, crop=True, final_size=None):
    transform_steps = [transforms.Resize(image_size)]

    if crop and final_size:
        transform_steps.append(transforms.CenterCrop(final_size))

    transform_steps.append(transforms.ToTensor())
    return transforms.Compose(transform_steps)


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.paths = sorted(
            path
            for path in self.root_dir.rglob('*')
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not self.root_dir.exists():
            raise FileNotFoundError(f'Dataset directory does not exist: {self.root_dir}')
        if not self.paths:
            raise RuntimeError(f'No images found in dataset directory: {self.root_dir}')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image_path = self.paths[index]
        with Image.open(image_path) as image:
            image = image.convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            return image
