from torch.utils.data import DataLoader
from torch import nn

import numpy as np
from data import TestData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval(model, n_ways, transforms):
  data = TestData('test', 1000, n_ways, transforms)
  loader = DataLoader(data, shuffle=True)
  d = nn.PairwiseDistance(p=2)
  model = model.to(device)
  predictions = []
  with torch.no_grad():
    model.eval()
    for main_img, test_set in loader:
      main = main_img.to(device)
      target = test_set.to(device).squeeze(0)
      main_emb = model(main)
      test_emb = model(target)
      dist = d(main_emb.repeat(n_ways,1), test_emb)
      pred_label = torch.argmin(dist)
      if (pred_label.item() == n_ways-1):
        predictions.append(1)
      else:
        predictions.append(0)
  return np.mean(predictions) * 100