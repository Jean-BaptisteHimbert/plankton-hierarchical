"""
(c) 2023 Inria
"""

import json
import logging
import os
import shutil
import tarfile
from pathlib import Path
from typing import List, Optional, Callable
from zipfile import ZipFile
import numpy as np
from PIL import ImageFile, Image
import pickle 

import torch
import requests
from pytorch_lightning.core import LightningDataModule
from torch import Generator
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, Subset
from torchvision.datasets import ImageFolder, DatasetFolder
from tqdm.auto import tqdm
from sklearn import preprocessing
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from collections import Counter

#import inria.public_data as public_data

logger = logging.getLogger(__file__)

from dataclasses import dataclass, field


def _download_to_dir(uri, target_dir, file_name, force_download):
    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / file_name
    if not force_download and file_path.exists():
        logger.info(f"{file_name} already exists in {target_dir}, skipping download")
        return
    with requests.get(uri, stream=True) as r:
        if r.status_code != 200:
            raise ValueError(f"Request failed with status code {r.status_code}")
        if r.headers.get("Content-Range") != None: # for ZooLake dataset
            total_length = int(r.headers.get("Content-Range").split(sep='/')[1])
        else:
            total_length = int(r.headers.get("Content-Length"))
        # implement progress bar via tqdm
        with tqdm.wrapattr(r.raw, "read", total=total_length, desc=f"Downloading {file_name}", leave=False, position=1) as raw:
            # save the output to a file
            with open(file_path, 'wb') as f:
                shutil.copyfileobj(raw, f)
    logger.info(f"Downloaded {file_name} to {target_dir}")
    
@dataclass
class ImageFolderLightningDataModule(LightningDataModule):
    """A Lightning data module that reads an image classification
    dataset that is organized as a ImageFolder
    (i.e. root_dir/class_name/image.png)"""

    data_dir: Path
    mean_std_path: Path = None
    image_size: int = 224
    train_size: float = 0.6
    val_size: float = 0.2
    split_seed: float = None
    num_workers: int = 0
    persistent_workers: bool = True
    train_transforms: Callable = None
    val_transforms: Callable = None
    batch_size: int = 64
    shuffle: bool = True
    sampler: bool = True
    pin_memory: bool = False
    manual_augmentations: bool = True
    number_of_augmentations: int = 14
    magnitude_of_augmentations: int = 10

    def __post_init__(self):
        super().__init__()

        ImageFile.LOAD_TRUNCATED_IMAGES = True

    def _cleanup_empty_dirs(self, imagefolder_dir: Path) -> None:
        """Delete empty subfolders as torchvision ImageFolder crashes otherwise."""
        rewind = True
        while rewind:
            try:
                rewind = False
                for dir in imagefolder_dir.glob("**"):
                    if dir.is_dir() and not os.listdir(dir):
                        shutil.rmtree(dir)
            except:
                rewind = True

    def prepare_data(self) -> None:
        pass
    
    def compute_mean_std(self, data_dir, save_to_path):
        
        img_height = img_width = self.image_size
        
        mean = []
        std = []

        logger.info("Calculating Mean and Standard deviation")
        class_names = sorted(os.listdir(data_dir))
        for class_name in class_names:
            if not class_name.startswith('.'):
                logger.info(f'Starting class {class_name}')
                class_dir = os.path.join(data_dir, class_name)
                for img_name in os.listdir(class_dir):
                    if img_name.endswith('.jpeg') or img_name.endswith('.jpg') or img_name.endswith('.png'):
                        img_path = os.path.join(class_dir, img_name)
                        img = Image.open(img_path)
                        img = img.resize((img_height, img_width), resample=Image.BILINEAR)
                        img = np.array(img) / 255.0
                        mean.append(np.mean(img, axis=(0, 1)))
                        std.append(np.std(img, axis=(0, 1)))

        # Calculate the mean and standard deviation for the entire dataset
        mean = np.mean(mean, axis=0)
        std = np.mean(std, axis=0)
        torch.save({mean, std}, save_to_path)
        logger.info("Calculating Mean and Standard deviation: DONE")
        return(mean, std)

    def setup(self, stage=None):

        img_height = img_width = self.image_size
        N = self.number_of_augmentations
        M = self.magnitude_of_augmentations

        # Check if a mean and standard deviation file exists, if not calculate them for the dataset
        
        if self.mean_std_path.is_file() == False:   

            self.mean, self.std = self.compute_mean_std(self.data_dir, self.mean_std_path)

        else:
 
            self.mean, self.std = torch.load(self.mean_std_path)


        val_test_transforms = transforms.Compose([
                                                    transforms.Resize(size=(img_height, img_width)), 
                                                    transforms.ToTensor(), 
                                                    transforms.Normalize(mean=self.mean, std=self.std),
                                            ]) 
        if self.manual_augmentations:
            train_transforms = transforms.Compose([
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomVerticalFlip(),
                                                    transforms.RandomRotation(degrees=(0, 180)),
                                                    transforms.Resize(size=(img_height, img_width)), 
                                                    #transforms.GaussianBlur(kernel_size=(5,5), sigma=(0.5,2)),
                                                    #transforms.RandomAffine(degrees=(30,90), translate=(0.1, 0.3), scale=(0.8, 1.2)),
                                                    #transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                                                    transforms.ToTensor(), 
                                                    transforms.Normalize(mean=self.mean, std=self.std),
                                                ])
            
        else:
            train_transforms = transforms.Compose([
                                                    transforms.Resize(size=(img_height, img_width)), 
                                                    transforms.RandAugment(N,M),
                                                    transforms.ToTensor(), 
                                                    transforms.Normalize(mean=self.mean, std=self.std),
                                               ])
        
        logger.info("Creating Dataset")
        # Create the ImageFolder dataset
        self.dataset = HierarchicalFolder(self.data_dir) #, transform=val_test_transforms)
        

        self.num_classes = self.dataset.num_classes
        self.num_samples = len(self.dataset)


        train_size = int(self.train_size * len(self.dataset))
        val_size = int(self.val_size * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(self.dataset, [train_size, val_size, test_size])

        self.train_dataset = TransformingDataset(train_dataset, train_transforms)
        self.val_dataset = TransformingDataset(val_dataset, val_test_transforms)
        self.test_dataset = TransformingDataset(test_dataset, val_test_transforms)

        logger.info("Calculating inverse weights")


        """ Weighted Random Sampler """
        y_train_indices = train_dataset.indices
        y_train = [self.dataset.maping_target_to_int_target[str(self.dataset.labels[i])] for i in y_train_indices]

        class_sample_count = {t:len(np.where(y_train == t)[0]) for t in np.unique(y_train)}
        samples_weight = np.array([1./class_sample_count[t] for t in y_train])
        self.weights = torch.tensor(samples_weight, dtype=torch.float)
        


        if self.sampler is True:

            logger.info("Weighted Random Sampler")
            self.sampler = WeightedRandomSampler(self.weights, len(self.weights), replacement=True)
            self.shuffle = False

        else:

            self.sampler = None
            self.shuffle = True


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            sampler = self.sampler,
            pin_memory= self.pin_memory,          

        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            batch_size=self.batch_size,
            shuffle=False,
            sampler = None,
            #pin_memory=True,
        )

    def test_dataloader(self):
        #dataset = torch.utils.data.Subset(self.test_dataset, indices=list(range(1000)))
        dataset = self.test_dataset
        # Implement the following code if you need to test results, it prevents having to wait for the entire dataset to be processed
        #subset_indices = list(range(100))  # Create a list of the first 100 sample indices
        #subset = torch.utils.data.Subset(dataset, indices=subset_indices)  # Create a new subset of the original dataset
        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            batch_size=self.batch_size,
            shuffle=False,
            sampler = None,
            drop_last=True,
            #pin_memory=True,
        )

class TransformingDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, torch.Tensor(y)
    def __len__(self):
        return len(self.subset)
    

class HierarchicalFolder(ImageFolder):
    def __init__(self,root_dir):
        #super().__init__(root_dir)

        self.root_dir = root_dir
        
        self.classes, self.maping_class_to_target, self.maping_target_to_class, self.maping_target_to_int_target, self.maping_int_target_to_target = self.get_classes_names_and_map()

        # Initialize empty lists to store the data
        self.image_paths = []
        self.labels = []
        num_classes = [0, 0, 0]

        max1 = 0
        max2 = 0
        max3 = 0

        label1 = 0
        label2 = 0
        label3 = 0
        # Iterate over the image files and their directories
        first_level_classes = sorted(os.listdir(self.root_dir))
        max1 = len(first_level_classes)
        for i, first_level_class in enumerate(first_level_classes):
            second_level_classes = sorted(os.listdir(os.path.join(self.root_dir, first_level_class)))
            max2 += len(second_level_classes)
            label1 += 1
            for j, second_level_class in enumerate(second_level_classes):
                third_level_classes = sorted(os.listdir(os.path.join(self.root_dir, first_level_class, second_level_class)))
                max3 += len(third_level_classes)
                label2 += 1
                for k, third_level_class in enumerate(third_level_classes):
                    images_list = sorted(os.listdir(os.path.join(self.root_dir, first_level_class, second_level_class, third_level_class)))
                    label3 += 1
                    for img_name in images_list:
                        if img_name.endswith('.jpeg') or img_name.endswith('.jpg') or img_name.endswith('.png'):
                            img_path = os.path.join(self.root_dir, first_level_class, second_level_class, third_level_class, img_name)
                            self.image_paths.append(img_path)
                            self.labels.append([label1 - 1, label2 - 1, label3 - 1])
        num_classes[0] = max1
        num_classes[1] = max2
        num_classes[2] = max3


        # Find the number of unique classes
        self.num_classes = num_classes


    def get_classes_names_and_map(self):
        first_level_classes = sorted(os.listdir(self.root_dir))

        class_names = []
        maping_class_to_target = {}
        maping_target_to_class = {}
        maping_target_to_int_target = {}
        maping_int_target_to_target = {}

        label1 = 0
        label2 = 0
        label3 = 0

        for i, first_level_class in enumerate(first_level_classes):
            second_level_classes = sorted(os.listdir(os.path.join(self.root_dir, first_level_class)))
            label1 += 1
            for j, second_level_class in enumerate(second_level_classes):
                third_level_classes = sorted(os.listdir(os.path.join(self.root_dir, first_level_class, second_level_class)))
                label2 += 1
                for k, third_level_class in enumerate(third_level_classes):
                    label3 += 1
                    class_name = str(first_level_class + '___' + second_level_class + '___' + third_level_class)
                    class_names.append(class_name)
                    target = [label1 - 1, label2 - 1, label3 - 1]
                    int_target = i + (10*j) + (1000*k)
                    maping_class_to_target[class_name] = str(target)
                    maping_target_to_class[str(target)] = class_name
                    maping_target_to_int_target[str(target)] = int_target
                    maping_int_target_to_target[int_target] = str(target)
                    
        
        return class_names, maping_class_to_target, maping_target_to_class, maping_target_to_int_target, maping_int_target_to_target



    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[index]
        return image, torch.Tensor(label)

    def __len__(self):
        return len(self.image_paths)

@dataclass
class WhoiDataModule(ImageFolderLightningDataModule):
    """A Pytorch Lightning DataModule for the WHOI plankton image dataset.
    Automates downloading the different releases and doing preliminary formatting.

    Heidi M. Sosik, Emily E. Peacock, Emily F. Brownlee (2014) Annotated Plankton Images - Data Set for
    Developing and Evaluating Classification Methods. WHOI. doi: 10.1575/1912/7341,
    url: https://hdl.handle.net/10.1575/1912/7341

    More information and example images associated with each category label can be found here: https://github.com/hsosik/WHOI-Plankton/wiki.
    """

    release_years: List[str] = field(default_factory=list)
    force_download: bool = False
    keep_files: bool = True

    __LOCAL_DATASET_INFO_FILE: str = "whoi-local-info.json"
    __LOCAL_WHOI_IMAGEFOLDER: str = "WHOI_imagefolder"
    __LOCAL_WHOI_RAW_DIR: str = "whoi_raw_releases"
    __WHOI_RELEASE_URIS = {
        "2014": "https://darchive.mblwhoilibrary.org/bitstream/handle/1912/7350/2014.zip?sequence=1&isAllowed=y",
        "2013": "https://darchive.mblwhoilibrary.org/bitstream/handle/1912/7349/2013.zip?sequence=1&isAllowed=y",
        "2012": "https://darchive.mblwhoilibrary.org/bitstream/handle/1912/7344/2012.zip?sequence=1&isAllowed=y",
        "2011": "https://darchive.mblwhoilibrary.org/bitstream/handle/1912/7347/2011.zip?sequence=1&isAllowed=y",
        "2010": "https://darchive.mblwhoilibrary.org/bitstream/handle/1912/7348/2010.zip?sequence=1&isAllowed=y",
        "2009": "https://darchive.mblwhoilibrary.org/bitstream/handle/1912/7346/2009.zip?sequence=1&isAllowed=y",
        "2008": "https://darchive.mblwhoilibrary.org/bitstream/handle/1912/7345/2008.zip?sequence=1&isAllowed=y",
        "2007": "https://darchive.mblwhoilibrary.org/bitstream/handle/1912/7343/2007.zip?sequence=1&isAllowed=y",
        "2006": "https://darchive.mblwhoilibrary.org/bitstream/handle/1912/7342/2006.zip?sequence=1&isAllowed=y",
    }

    def __post_init__(self):
        super().__post_init__()

    def prepare_data(self, force_download):
        if self.release_years:
            invalid_years = [year for year in self.release_years if str(year) not in self.__WHOI_RELEASE_URIS]
            if invalid_years:
                logger.error(f"Years: {invalid_years} are not valid WHOI release years.")
                raise ValueError(f"Years: {invalid_years} are not valid WHOI release years.")

        release_years = sorted(list(self.release_years if self.release_years else self.__WHOI_RELEASE_URIS.keys()))

        raw_release_zips_dir = self.data_dir / self.__LOCAL_WHOI_RAW_DIR
        raw_release_zips_dir.mkdir(parents=True, exist_ok=True)

        imagefolder_dir = self.data_dir / self.__LOCAL_WHOI_IMAGEFOLDER
        imagefolder_dir.mkdir(parents=True, exist_ok=True)

        local_info_file = imagefolder_dir / self.__LOCAL_DATASET_INFO_FILE

        local_releases = []

        if not force_download:
            # If not forcing dowload attempt to skip releases that are already downloaded.
            try:
                with open(local_info_file) as f:
                    local_releases = json.load(f)

                if [rel for rel in local_releases if rel not in release_years]:
                    # local_releases should be a subset of release_years, imagefolder_dir must be re-created
                    shutil.rmtree(imagefolder_dir)
                    imagefolder_dir.mkdir(parents=True, exist_ok=True)
                    local_releases = []
            except FileNotFoundError:
                pass

        release_downloads = [year for year in release_years if year not in local_releases]

        if release_downloads:
            logger.info(f"Downloading missing releases {release_downloads}.")

            for release_year in tqdm(release_downloads, desc="Release download progress", leave=False, position=0):
                _download_to_dir(
                    self.__WHOI_RELEASE_URIS[release_year], raw_release_zips_dir, f"{release_year}.zip", force_download
                )
                local_releases += [release_year]

            logger.info(f"Unzipping releases {release_downloads}.")
            for zip_file in tqdm(
                [f"{year}.zip" for year in release_downloads], desc="Release unzip progress", leave=False, position=0
            ):
                with ZipFile(raw_release_zips_dir / zip_file, "r") as zip_ref:
                    for file in tqdm(
                        iterable=zip_ref.namelist(),
                        total=len(zip_ref.namelist()),
                        desc=f"Extracting {zip_file}",
                        leave=False,
                        position=1,
                    ):
                        zip_ref.extract(member=file, path=raw_release_zips_dir)

                if not self.keep_files:
                    logger.info(f"Removing release zip {zip_file}. Set keep_files=True to prevent this.")
                    Path(zip_file).unlink()

            logger.info(f"Clean-up ImageFolder structure.")
            self._cleanup_empty_dirs(raw_release_zips_dir)

            logger.info(f"Moving to ImageFolder dir.")
            for whoi_release in tqdm(
                sorted([item for item in raw_release_zips_dir.glob("*") if item.is_dir()]),
                desc="Release imagefolder move progress",
                leave=False,
                position=0,
            ):
                for folder in tqdm(
                    [item for item in whoi_release.glob("*") if item.is_dir()],
                    desc=f"Moving release {whoi_release.name}",
                    leave=False,
                    position=1,
                ):
                    (imagefolder_dir / folder.name).mkdir(exist_ok=True)
                    for img_file in folder.glob("*.png"):
                        try:
                            shutil.move(folder / img_file, imagefolder_dir / folder.name)
                        except OSError:
                            logger.debug(f"File {folder / img_file} already in {imagefolder_dir / folder.name}.")
                shutil.rmtree(whoi_release)

            with open(local_info_file, "w") as f:
                json.dump(self.release_years, f)

        logger.info(f"Done preparing WHOI dataset!")

    def setup(self, stage: Optional[str] = None) -> None:
        self.setup_on_dir(self.data_dir / self.__LOCAL_WHOI_IMAGEFOLDER)


@dataclass
class ZooscanDataModule(ImageFolderLightningDataModule):
    """A Pytorch Lightning DataModule for training on the Zooscan dataset.

    Elineau Amanda, Desnos Corinne, Jalabert Laetitia, Olivier Marion, Romagnan Jean-Baptiste,
    Costa Brandao Manoela, Lombard Fabien, Llopis Natalia, Courboulès Justine,
    Caray-Counil Louis, Serranito Bruno, Irisson Jean-Olivier, Picheral Marc,
    Gorsky Gaby, Stemmann Lars (2018). ZooScanNet: plankton images captured with the ZooScan.
    SEANOE. doi: 10.17882/55741, url: https://www.seanoe.org/data/00446/55741/
    """

    force_download: bool = False
    keep_files: bool = True

    __ZOOSCAN_URI: str = "https://www.seanoe.org/data/00446/55741/data/57398.tar"
    __LOCAL_DATASET_FILE: str = "57398.tar"
    __LOCAL_RAW_DIR: str = "zooscan_raw"
    __LOCAL_IMAGEFOLDER: str = "Zooscan_imagefolder"

    def __post_init__(self):
        super().__post_init__()

    def prepare_data(self, force_download):
        raw_zooscan_dir = self.data_dir / self.__LOCAL_RAW_DIR
        raw_zooscan_dir.mkdir(parents=True, exist_ok=True)

        imagefolder_dir = self.data_dir / self.__LOCAL_IMAGEFOLDER
        imagefolder_dir.mkdir(parents=True, exist_ok=True)

        if os.listdir(imagefolder_dir):
            if force_download:
                # clean-up imagefolder dir
                shutil.rmtree(imagefolder_dir)
                imagefolder_dir.mkdir(parents=True, exist_ok=True)
            else:
                logger.warn(
                    f"Zooscan ImageFolder dir {imagefolder_dir} is not empty, using its content as dataset. Set force_download=True to prevent this."
                )
                return

        _download_to_dir(self.__ZOOSCAN_URI, raw_zooscan_dir, self.__LOCAL_DATASET_FILE, force_download)

        logger.info(f"Unpacking {raw_zooscan_dir/self.__LOCAL_DATASET_FILE}.")
        with tarfile.open(name=raw_zooscan_dir / self.__LOCAL_DATASET_FILE) as tar:
            for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers()), desc="Unpack progress"):
                tar.extract(member=member, path=raw_zooscan_dir)

        if not self.keep_files:
            logger.info(
                f"Removing release file {raw_zooscan_dir/self.__LOCAL_DATASET_FILE}. Set keep_files=True to prevent this."
            )
            Path(raw_zooscan_dir / self.__LOCAL_DATASET_FILE).unlink()

        logger.info(f"Clean-up ImageFolder structure.")
        self._cleanup_empty_dirs(imagefolder_dir)

        logger.info(f"Moving to ImageFolder dir.")
        shutil.move(raw_zooscan_dir / "ZooScanSet" / "imgs", imagefolder_dir)

        # removing used files
        shutil.rmtree(raw_zooscan_dir / "ZooScanSet")

        logger.info(f"Done preparing ZooScan dataset!")

    def setup(self, stage: Optional[str] = None) -> None:
        self.setup_on_dir(self.data_dir / self.__LOCAL_IMAGEFOLDER)


@dataclass
class LenslessDataModule(ImageFolderLightningDataModule):
    """Pastore, V.P., Zimmerman, T.G., Biswas, S.K. et al. Annotation-free learning of
    plankton for classification and anomaly detection. Sci Rep 10, 12142 (2020).
    <https://doi.org/10.1038/s41598-020-68662-3>

    Original dataset is available online at <https://ibm.ent.box.com/v/PlanktonData>.
    """

    test_split: int = -1
    force_download: bool = False

    __LOCAL_IMAGEFOLDER: str = "Lensless_imagefolder"
    __LOCAL_DATASET_FILE: str = "lensless_dataset.zip"

    def __post_init__(self):
        super().__post_init__()

    def prepare_data(self, force_download):
        imagefolder_dir = self.data_dir / self.__LOCAL_IMAGEFOLDER
        imagefolder_dir.mkdir(parents=True, exist_ok=True)

        if os.listdir(imagefolder_dir):
            if force_download:
                # clean-up imagefolder dir
                shutil.rmtree(imagefolder_dir)
                imagefolder_dir.mkdir(parents=True, exist_ok=True)
            else:
                logger.warn(
                    f"Lensless ImageFolder dir {imagefolder_dir} is not empty, using its content as dataset. Set force_download=True to prevent this."
                )
                return

        dataset_path = Path(public_data.__file__).parent

        logger.info(f"Unzipping {dataset_path/self.__LOCAL_DATASET_FILE}.")

        with ZipFile(dataset_path / self.__LOCAL_DATASET_FILE, "r") as zip_ref:
            for file in tqdm(
                iterable=zip_ref.namelist(),
                total=len(zip_ref.namelist()),
                desc=f"Extracting {self.__LOCAL_DATASET_FILE}",
                leave=False,
                position=0,
            ):
                zip_ref.extract(member=file, path=imagefolder_dir)

        logger.info(f"Done preparing Lensless dataset!")

    def setup(self, stage: Optional[str] = None) -> None:
        assert self.test_split == -1, "Lensless dataset has a predefined test set. Test split is not used."

        self.main_dataset = ImageFolder(
            self.data_dir / self.__LOCAL_IMAGEFOLDER / "lensless_dataset" / "TRAIN_IMAGE", transform=self.transform
        )
        self.test_dataset = ImageFolder(
            self.data_dir / self.__LOCAL_IMAGEFOLDER / "lensless_dataset" / "TEST_IMAGE", transform=self.transform
        )

        self.num_classes = len(self.main_dataset.classes)

        val_size = int(len(self.main_dataset) * self.val_split)

        train_size = len(self.main_dataset) - val_size

        self.train_dataset, self.val_dataset = random_split(
            self.main_dataset,
            [train_size, val_size],
            generator=Generator().manual_seed(self.split_seed) if self.split_seed else None,
        )

@dataclass
class ZooLakeDataModule(ImageFolderLightningDataModule):
    """ AUTHOR: Kyathanahally Sreenath P., Hardeman Thomas, Merz Ewa, Bulas Thea, Reyes Marta, Isles Peter, Pomati Francesco, Baity-Jesi Marco
	    TITLE:  'Deep Learning Classification of Lake Zooplankton'  
	    JOURNAL:Frontiers in Microbiology     
	    VOLUME: 12      
	    YEAR:   2021   
	    URL:    https://www.frontiersin.org/articles/10.3389/fmicb.2021.746297     
	    DOI:    10.3389/fmicb.2021.746297    
	    ISSN:   1664-302X   

    ABSTRACT:   Plankton are effective indicators of environmental change and ecosystem health in freshwater habitats, but collection of plankton data using manual 
                microscopic methods is extremely labor-intensive and expensive. Automated plankton imaging offers a promising way forward to monitor plankton communities 
                with high frequency and accuracy in real-time. Yet, manual annotation of millions of images proposes a serious challenge to taxonomists. Deep learning 
                classifiers have been successfully applied in various fields and provided encouraging results when used to categorize marine plankton images. 
                Here, we present a set of deep learning models developed for the identification of lake plankton, and study several strategies to obtain optimal performances, 
                which lead to operational prescriptions for users. To this aim, we annotated into 35 classes over 17900 images of zooplankton and large phytoplankton colonies, 
                detected in Lake Greifensee (Switzerland) with the Dual Scripps Plankton Camera. Our best models were based on transfer learning and ensembling, which classified 
                plankton images with 98% accuracy and 93% F1 score. When tested on freely available plankton datasets produced by other automated imaging tools 
                (ZooScan, Imaging FlowCytobot, and ISIIS), our models performed better than previously used models. Our annotated data, code and classification models are freely 
                available online.
    """ 
    
    force_download: bool = False
    keep_files: bool = True

    __ZOOLAKE_URI: str = "https://opendata.eawag.ch/dataset/52b6ba86-5ecb-448c-8c01-eec7cb209dc7/resource/1cc785fa-36c2-447d-bb11-92ce1d1f3f2d/download/data.zip"
    __LOCAL_DATASET_FILE: str = "Zoolake.zip"
    __LOCAL_RAW_DIR: str = "Zoolake_raw"
    __LOCAL_IMAGEFOLDER: str = "Zoolake_imagefolder"

    def __post_init__(self):
        super().__post_init__()

    def prepare_data(self, force_download):
        raw_zoolake_dir = self.data_dir / self.__LOCAL_RAW_DIR
        raw_zoolake_dir.mkdir(parents=True, exist_ok=True)
        imagefolder_dir = self.data_dir / self.__LOCAL_IMAGEFOLDER
        imagefolder_dir.mkdir(parents=True, exist_ok=True)

        if os.listdir(imagefolder_dir):
            if force_download:
                # clean-up imagefolder dir
                shutil.rmtree(imagefolder_dir)
                imagefolder_dir.mkdir(parents=True, exist_ok=True)
            else:
                logger.warn(
                    f"Zoolake ImageFolder dir {imagefolder_dir} is not empty, using its content as dataset. Set force_download=True to prevent this."
                )
                return

        _download_to_dir(self.__ZOOLAKE_URI, raw_zoolake_dir, self.__LOCAL_DATASET_FILE, force_download)

        logger.info(f"Unzipping Zoolake.zip")

        zip_path = raw_zoolake_dir/"Zoolake.zip"
       
        with ZipFile(zip_path, "r") as zip_ref:
            for file in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())):
                zip_ref.extract(path=raw_zoolake_dir, member=file)

        image_files = raw_zoolake_dir/"data"/"zooplankton_0p5x"

        for file in os.listdir(image_files):
            path = os.path.join(image_files, file)
            if os.path.isdir(path):

                current_path = os.path.join(path, 'training_data')
                class_name = current_path.split('/')[-2]
                destination_path = os.path.join(imagefolder_dir, class_name)
                os.makedirs(destination_path, exist_ok=True)
        
                for image_file in os.listdir(current_path):
                    image_file_path = os.path.join(current_path, image_file)
                    shutil.move(image_file_path, destination_path)
        
        shutil.rmtree(current_path)

        logger.info(f"Zoolake dataset complete")

    def setup(self, stage: Optional[str] = None) -> None:
        self.setup_on_dir(self.data_dir / self.__LOCAL_IMAGEFOLDER)