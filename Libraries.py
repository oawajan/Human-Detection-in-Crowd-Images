import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch.cuda
import torch.nn as nn
from facenet_pytorch import MTCNN as torchmtcnn
from mtcnn.mtcnn import MTCNN as MTCNN
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torch.utils.data.dataloader import default_collate
import tensorflow as tf
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.ops import nms
from torchvision.ops import box_iou
from torchvision.transforms import Resize
from torchvision.transforms.functional import to_tensor
from torchvision.io import read_image
import torch.optim as optim
from torchsummary import summary
from PIL import Image
from sklearn.decomposition import PCA
from collections import defaultdict