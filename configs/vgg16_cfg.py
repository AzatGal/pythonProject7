from easydict import EasyDict

cfg = EasyDict()

cfg.conv_blocks = [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]]

cfg.full_conn_blocks = [4096, 4096]
