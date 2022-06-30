from .builder import DATASETS
from .ade import ADE20KDataset
@DATASETS.register_module()
class FoodSeg103Dataset(ADE20KDataset):
    CLASSES = ("Meat", "Nuts/seeds", "Eggs", "Beans/lentils/peas", "Fruit", "Grain", "Vegetables", "Dairy", "Sauce/Spread", "Soup", "Beverage")
    PALETTE = [[255, 66, 79], [191, 122, 57], [255, 255, 0], 
            [255, 0, 204], [120, 79, 255], [211, 255, 117], [5, 245, 17], [255, 255, 255],[107, 7, 0], [97, 80, 5], [250, 87, 236]]
    def __init__(self, **kwargs):
        super().__init__(img_suffix='.jpg', seg_map_suffix='.png', **kwargs)

from mmcv import Config
cfg = Config.fromfile('mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x512_40k_ade20k.py')