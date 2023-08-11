
import os
import PIL
import torch
import random
import logging
import numpy as np
from glob import glob
from PIL import Image
from PIL import ImageFile
import torchvision.transforms as T
from collections import defaultdict

ImageFile.LOAD_TRUNCATED_IMAGES = True


def open_image(path):
    return Image.open(path).convert("RGB")


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, test_folder, M=10, N=5, image_size=256):
        super().__init__()
        logging.debug(f"Searching test images in {test_folder}")

        images_paths = sorted(glob(f"{test_folder}/**/*.jpg", recursive=True))

        logging.debug(f"Found {len(images_paths)} images")
        images_metadatas = [p.split("@") for p in images_paths]
        # field 1 is UTM east, field 2 is UTM north
        self.utmeast_utmnorth = np.array([(m[1], m[2]) for m in images_metadatas]).astype(np.float64)

        class_id_group_id = [TrainDataset.get__class_id__group_id(*m, M, N) for m in self.utmeast_utmnorth]
        self.images_paths = images_paths
        self.class_id = [(id[0][0]+ M // 2, id[0][1]+ M // 2) for id in class_id_group_id]
        self.group_id = [id[1] for id in class_id_group_id]

        self.normalize = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        image_path = self.images_paths[index]
        # class_id = self.class_id[index]

        pil_image = open_image(image_path)
        # pil_image = T.functional.resize(pil_image, self.shapes[index])
        image = self.normalize(pil_image)
        if isinstance(image, tuple):
            image = torch.stack(image, dim=0)
        return image, tuple(self.utmeast_utmnorth[index])

    def __len__(self):
        return len(self.images_paths)

    def get_classes_num(self):
        return len(self.dict__cell_id__class_num)


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, train_path, dataset_name, group_num, M=10, N=5, min_images_per_class=10, transform=None):
        """
        Parameters
        ----------
        M : int, the length of the side of each cell in meters.
        N : int, distance (M-wise) between two classes of the same group.
        classes_ids : list of IDs of each class within this group. Each ID is a tuple
                with the center of the cell in UTM coords, e.g: (549900, 4178820).
        images_per_class : dict where the key is a class ID, and the value is a list
                containing the paths of the images withing the class.
        transform : a transform for data augmentation
        """
        super().__init__()

        cache_filename = f"cache/{dataset_name}_M{M}_N{N}_mipc{min_images_per_class}.torch"
        if not os.path.exists(cache_filename):
            classes_per_group, images_per_class_per_group = initialize(train_path, dataset_name, M, N, min_images_per_class)
            torch.save((classes_per_group, images_per_class_per_group), cache_filename)
        else:
            classes_per_group, images_per_class_per_group = torch.load(cache_filename)
        classes_ids = classes_per_group[group_num]
        images_per_class = images_per_class_per_group[group_num]

        self.train_path = train_path
        self.M = M
        self.N = N
        self.transform = transform
        self.classes_ids = classes_ids
        self.images_per_class = images_per_class
        self.class_centers = [(cl_id[0] + M // 2, cl_id[1] + M // 2) for cl_id in self.classes_ids]
    
    def __getitem__(self, _):
        # The index is ignored, and each class is sampled uniformly
        class_num = random.randint(0, len(self.classes_ids)-1)
        class_id = self.classes_ids[class_num]
        class_center = self.class_centers[class_num]

        # Pick a random image among the ones in this class.
        image_path = self.train_path + random.choice(self.images_per_class[class_id])
        
        try:
            pil_image = open_image(image_path)
            tensor_image = self.transform(pil_image)
        except PIL.UnidentifiedImageError:
            logging.info(f"ERR: There was an error while reading image {image_path}, it is probably corrupted")
            tensor_image = torch.zeros([3, 224, 224])

        return tensor_image, class_num, class_center

    def get_images_num(self):
        """Return the number of images within this group."""
        return sum([len(self.images_per_class[c]) for c in self.classes_ids])

    def get_classes_num(self):
        """Return the number of classes within this group."""
        return len(self.classes_ids)

    def __len__(self):
        """Return a large number. This is because if you return the number of
        classes and it is too small (like in pitts30k), the dataloader within
        InfiniteDataLoader is often recreated (and slows down training).
        """
        return 1000000

    @staticmethod
    def get__class_id__group_id(utm_east, utm_north, M, N):
        """Return class_id and group_id for a given point.
            The class_id is a tuple of UTM_east, UTM_north (e.g. (396520, 4983800)).
            The group_id represents the group to which the class belongs
            (e.g. (0, 1), and it is between (0, 0) and (N, N).
        """
        rounded_utm_east = int(utm_east // M * M)  # Rounded to nearest lower multiple of M
        rounded_utm_north = int(utm_north // M * M)

        class_id = (rounded_utm_east, rounded_utm_north)
        # group_id goes from (0, 0) to (N, N)
        group_id = (rounded_utm_east % (M * N) // M,
                    rounded_utm_north % (M * N) // M)
        return class_id, group_id


def initialize(dataset_folder, dataset_name, M, N, min_images_per_class):
    paths_file = f"cache/paths_{dataset_name}.torch"
    # Search paths of dataset only the first time, and save them in a cached file
    if not os.path.exists(paths_file):
        logging.info(f"Searching training images in {dataset_folder}")
        images_paths = sorted(glob(f"{dataset_folder}/**/*.jpg", recursive=True))
        # Remove folder_path from images_path, so that the same cache file can be used on any machine
        images_paths = [p.replace(dataset_folder, "") for p in images_paths]
        os.makedirs("cache", exist_ok=True)
        torch.save(images_paths, paths_file)
    else:
        images_paths = torch.load(paths_file)

    logging.info(f"Found {len(images_paths)} images")

    images_metadatas = [p.split("@") for p in images_paths]
    # field 1 is UTM east, field 2 is UTM north
    utmeast_utmnorth = [(m[1], m[2]) for m in images_metadatas]
    utmeast_utmnorth = np.array(utmeast_utmnorth).astype(np.float64)
    del images_metadatas
    logging.info("For each image, get its UTM east, UTM north from its path")
    logging.info("For each image, get class and group to which it belongs")
    class_id__group_id = [TrainDataset.get__class_id__group_id(*m, M, N) for m in utmeast_utmnorth]

    logging.info("Group together images belonging to the same class")
    images_per_class = defaultdict(list)
    images_per_class_per_group = defaultdict(dict)
    for image_path, (class_id, _) in zip(images_paths, class_id__group_id):
        images_per_class[class_id].append(image_path)

    # Images_per_class is a dict where the key is class_id, and the value
    # is a list with the paths of images within that class.
    images_per_class = {k: v for k, v in images_per_class.items() if len(v) >= min_images_per_class}

    logging.info("Group together classes belonging to the same group")
    # Classes_per_group is a dict where the key is group_id, and the value
    # is a list with the class_ids belonging to that group.
    classes_per_group = defaultdict(set)
    for class_id, group_id in class_id__group_id:
        if class_id not in images_per_class:
            continue  # Skip classes with too few images
        classes_per_group[group_id].add(class_id)

    for group_id, group_classes in classes_per_group.items():
        for class_id in group_classes:
            images_per_class_per_group[group_id][class_id] = images_per_class[class_id]
    # Convert classes_per_group to a list of lists.
    # Each sublist represents the classes within a group.
    classes_per_group = [list(c) for c in classes_per_group.values()]
    images_per_class_per_group = [c for c in images_per_class_per_group.values()]

    return classes_per_group, images_per_class_per_group

