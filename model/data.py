from torchvision.datasets import ImageFolder


def load_dataset(data_folder):
    image_dataset = ImageFolder(data_folder)
    return image_dataset
