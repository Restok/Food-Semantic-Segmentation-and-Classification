# Copyright (c) OpenMMLab. All rights reserved.
import base64
import os

import cv2
import mmcv
import torch
from mmcv.cnn.utils.sync_bn import revert_sync_batchnorm
from ts.torch_handler.base_handler import BaseHandler

from mmcv.parallel import collate, scatter
from mmseg.datasets.pipelines import Compose
from mmseg.apis import init_segmentor
from mmseg.apis.inference import LoadImage
import json
import matplotlib.pyplot as plt
import numpy as np
import io
def inference_segmentor(model, img):
        """Inference image(s) with the segmentor.

        Args:
            model (nn.Module): The loaded segmentor.
            imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
                images.

        Returns:
            (list[Tensor]): The segmentation result.
        """
        cfg = model.cfg
        device = next(model.parameters()).device  # model device
        # build the data pipeline
        test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
        test_pipeline = Compose(test_pipeline)
        # prepare data
        data = dict(img=img)
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        if next(model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [device])[0]
        else:
            data['img_metas'] = [i.data[0] for i in data['img_metas']]
        # forward the model
        with torch.no_grad():
            result, labels = model(return_loss=False, rescale=True, **data)

        return result, labels
from PIL import Image
def return_img_array(model,
                       img,
                       result,
                       palette=None,
                       fig_size=(10, 5),
                       opacity=0.6,
                       title='',
                       block=True,
                       show_legend=False,
                       LABELS=[""]):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=fig_size)
    uniques_set = set(np.unique(result))
    fig.suptitle(title)
    if(show_legend):
        custom_lines = []
        labels = []
        for i in uniques_set:
            labels.append(LABELS[i])
            rgb_color = [v/255 for v in palette[i]]
            custom_lines.append(plt.Line2D([0], [0], color=tuple(rgb_color), lw=4))
        fig.legend(custom_lines, labels)
    axes[0].imshow(mmcv.bgr2rgb(img))
    axes[0].axis('off')
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(
        img, result, palette=palette, show=False, opacity=opacity)
    axes[1].imshow(mmcv.bgr2rgb(img))
    axes[1].axis('off')
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=72)
    buf.seek(0)
    content = buf.read()
    buf.close()
    return content
class MMsegHandler(BaseHandler):
    def initialize(self, context):
        properties = context.system_properties
        self.map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.map_location + ':' +
                                   str(properties.get('gpu_id')) if torch.cuda.
                                   is_available() else self.map_location)
        self.manifest = context.manifest
        model_dir = properties.get('model_dir')
        serialized_file = self.manifest['model']['serializedFile']
        checkpoint = os.path.join(model_dir, serialized_file)
        self.config_file = os.path.join(model_dir, 'config.py')
        self.model = init_segmentor(self.config_file, checkpoint, self.device)
        self.model = revert_sync_batchnorm(self.model)
        self.initialized = True

    def preprocess(self, data):
        images = []

        for row in data:
            image = row.get('data') or row.get('body')
            if isinstance(image, str):
                image = base64.b64decode(image)
            image = mmcv.imfrombytes(image)
            images.append(image)

        return images
        
    def inference(self, data, *args, **kwargs):
        results = [(inference_segmentor(self.model, img),img) for img in data]
        return results

    def postprocess(self, data):
        classes = ["Background", "Meat", "Nuts/seeds", "Eggs", "Beans/lentils/peas", "Fruit", "Grain", "Vegetables", "Dairy", "Sauce/Spread", "Soup/Drink"]
        labels = []
        palette = np.random.randint(0, 255, size=(len(classes), 3))
        palette[0, :] = 0
        for result in data:
            sig_activation = torch.nn.Sigmoid()(result[0][1])
            sig_activation[sig_activation>=0.5] = 1
            sig_activation[sig_activation<0.5] = 0
            title = "Multilabel head predictions: "
            for i in range(len(sig_activation)):
                if(sig_activation[i]==1):
                    title += (classes[i+1] + " ")
            labels.append(return_img_array(self.model, result[1], result[0][0], palette, title=title, LABELS=classes, show_legend=True))
        return labels
