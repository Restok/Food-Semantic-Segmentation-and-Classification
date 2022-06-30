# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import mmcv
import numpy as np
import pandas as pd
from .builder import DATASETS
from .multi_label import MultiLabelDataset


@DATASETS.register_module()
class FoodSeg103(MultiLabelDataset):

    CLASSES = ("Meat", "Nuts/seeds", "Eggs", "Beans/lentils/peas", "Fruit", "Grain", "Vegetables", "Dairy", "Dessert", "Sauce/Spread", "Soup", "Drink")


    def __init__(self, **kwargs):
        super(FoodSeg103, self).__init__(**kwargs)

    def load_annotations(self):
        """Load annotations.

        Returns:
            list[dict]: Annotation info from XML file.
        """
        data_infos = []
        img_ids = mmcv.list_from_file(self.ann_file)
        for img_id in img_ids:
            filename = f'JPEGImages/{img_id}'
            pd_path = osp.join(self.data_prefix, 'food103labels.csv')
            labels_db = pd.read_csv(pd_path)
            LABELS = list(self.CLASSES)

            gt_label = labels_db.loc[labels_db['filename'] == img_id][LABELS].to_numpy()
            info = dict(
                img_prefix=self.data_prefix,
                img_info=dict(filename=filename),
                gt_label=gt_label.astype(np.int8))
            data_infos.append(info)

        return data_infos
