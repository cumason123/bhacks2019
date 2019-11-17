from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import Image

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

TEST_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_image(file):
    image = Image.open(file)
    return TEST_TRANSFORMS(image)


def load_dataset(data_folder, split='train'):
    if split == 'train':
        transform = TRAIN_TRANSFORMS
    else:
        transform = TEST_TRANSFORMS

    return ImageFolder(data_folder, transform=transform)
