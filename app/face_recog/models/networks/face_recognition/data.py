import os
import torch
from torch.utils.data import Dataset

from facenet_pytorch import MTCNN

import PIL

from collections import defaultdict
import random
import glob


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)


class Base(DataSet):
    def __init__(self, data_path, data_size, transforms=None):
        super(Base, self).__init__()
        self.data_path = data_path
        self.data_size = data_size
        self.transforms = transforms
        self.data_dict = self.read_files()
    
    def read_files(self):
      work_dir = os.getcwd()
      os.chdir(self.data_path)
      data_dict = defaultdict(dict)

      try:
          for file in glob.glob('./*'):
              path_split = file.split('/')[1:]
              data_dict[path_split[0]] = glob.glob(path_split[0]+'/*')
      except Exception as e:
          print(e)
          os.chdir(work_dir)
      os.chdir(work_dir)
      return data_dict
    
    def __len__(self):
        return self.data_size

class TestData(Base):
    def __init__(self, data_path, data_size, n_ways, transforms=None):
        super(Base, self).__init__(data_path, data_size, transforms)
        self.nways = n_ways
    
    def __getitem__(self, index):
        cats = random.sample(self.data_dict.keys(), self.n_ways)
        
        main_img = pos_img = None
        while None in [main_img, pos_img]:
            main_img, pos_img = [mtcnn(PIL.Image.open(self.data_path + '/' + img)) for img in \
                                random.sample(self.data_dict[cats[0]], 2)]
        
        test_set = []
        while (len(test_set)!=(self.n_ways-1)):
            cat = random.choice(cats[1:])
            img = mtcnn(PIL.Image.open(self.data_path + '/' + random.choice(self.data_dict[cat])))
            if img is None:
                continue
            test_set.append(img)
        test_set.append(pos_img)
        
        test_set = torch.stack(test_set)
        return (main_img, test_set)