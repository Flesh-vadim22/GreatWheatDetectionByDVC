import matplotlib.pyplot as plt
import cv2
import numpy as np


def vizualize_box(img_tensor, target):
    fig, ax = plt.subplots(1,1, figsize=(16,10))
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    for box in target['boxes'].cpu().detach().numpy().astype(np.int32):
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)

    ax.set_axis_off()
    ax.imshow(img)