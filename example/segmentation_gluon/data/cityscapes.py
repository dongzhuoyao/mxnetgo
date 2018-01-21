from mxnet.gluon.data import Dataset, DataLoader

import os
from PIL import Image
import cfg
import augment


class CityScapes(Dataset):
    def __init__(self, root, split, transform):
        super(CityScapes, self).__init__()
        self.root = os.path.join(root, 'leftImg8bit', split)
        self.transform = transform

        self.img_paths = []

        self._img = os.path.join(root, 'leftImg8bit', split,
                                 '{}_leftImg8bit.png')
        self._lbl = os.path.join(root, 'gtFine', split,
                                 '{}_gtFine_labelIds.png')

        cities = [
            city for city in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, city))
        ]

        for city in cities:
            for img in os.listdir(os.path.join(self.root, city)):
                if len(img) > 3 and img[-4:] == '.png':
                    self.img_paths.append(os.path.join(city, img[:-16]))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, lbl_path = self._img.format(
            self.img_paths[idx]), self._lbl.format(self.img_paths[idx])

        img = Image.open(img_path)
        lbl = Image.open(lbl_path)

        img, lbl = self.transform(img, lbl)
        return img, lbl


def cs_test():
    dataset = CityScapes(cfg.cityscapes_root, 'train',
                         augment.cityscapes_train)
    print(len(dataset))
    loader = DataLoader(dataset, batch_size=24, shuffle=True)
    for data in loader:
        print(data)
        break


if __name__ == '__main__':
    cs_test()
