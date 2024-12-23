from torchvision import transforms


def get_transform(config):
    """get transformer
    ```
    get_transform(config: Namespace) -> tuple(train_transform: transforms.Compose, test_transform: transforms.Compose)
    ```

    Args:
        config(Namespace): Namespace of config
            This function uses following values in `config`

                config.data.random_flip: Whether use random flip
                config.data.image_size: Image size in dataset
    """
    if config.data.random_flip is False:
        train_transform = test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.CenterCrop(config.data.image_size),
                transforms.Lambda(lambda x: x * 2 - 1),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.CenterCrop(config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Lambda(lambda x: x * 2 - 1),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.CenterCrop(config.data.image_size),
                transforms.Lambda(lambda x: x * 2 - 1),
            ]
        )

    return train_transform, test_transform
