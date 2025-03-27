import json

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.utils import draw_bounding_boxes


def get_img_shape(image_grid_thw, processor):
    """Tamanho da imagem que foi de fato processada pelo modelo."""

    grid_t, grid_h, grid_w = image_grid_thw
    ps = processor.image_processor.patch_size
    return grid_h.item()*ps, grid_w.item()*ps

class DetectionOutputParser:
    """Processa a saída do VLM Qwen-VL para detecção de objetos."""

    def __init__(self, img, processor):

        self.img = img
        self.processor = processor

    def parse_output(self, inputs, output_text):
        """Processa a saída do modelo e retorna um dicionário com os objetos detectados."""

        input_shape = get_img_shape(inputs["image_grid_thw"][0], self.processor)

        # Fator para redimensionar as coordenadas
        factor_h = self.img.height/input_shape[0]
        factor_w = self.img.width/input_shape[1]

        # Transforma a saída em um dicionário
        clean_txt = "".join(output_text.split("\n")[1:-1])
        model_objects = json.loads(clean_txt)

        # Cria um dicionário com os objetos detectados
        objects = {}
        for obj in model_objects:
            bbox = obj["bbox_2d"]
            label = obj["label"]

            bbox = [bbox[0]*factor_h, bbox[1]*factor_w, bbox[2]*factor_h, bbox[3]*factor_w]

            objects[label] = bbox

        return objects
    
    def plot(self, objects):
        """Plota uma imagem com as caixas delimitadoras dos objetos."""
        
        labels = list(objects.keys())
        bboxes = torch.tensor(list(objects.values()))

        img = torch.from_numpy(np.array(self.img)).permute(2, 0, 1)
        img_bb = draw_bounding_boxes(img, bboxes, labels, width=3)

        plt.imshow(img_bb.permute(1, 2, 0))