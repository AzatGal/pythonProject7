import os
from easydict import EasyDict
from configs.oxford_pet_cfg import cfg as dataset_cfg

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cfg = EasyDict()
cfg.seed = 0

cfg.batch_size = 64
cfg.lr = 1e-3

cfg.model_name = 'VGG16'  # ['VGG16', 'ResNet50']
cfg.optimizer_name = 'Adam'  # ['SGD', 'Adam']

cfg.device = 'cpu'  # ['cpu', 'cuda']

cfg.model_cfg = ...
cfg.dataset_cfg = dataset_cfg

cfg.exp_dir = os.path.join(ROOT_DIR, 'train_vgg16')

cfg.num_epoch = 5
