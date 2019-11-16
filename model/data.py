import torchvision


def load_dataset(data_folder):
    image_dataset = torchvision.datasets.ImageFolder(data_folder)
    return image_dataset