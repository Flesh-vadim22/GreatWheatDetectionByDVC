import matplotlib.pyplot as plt
import cv2

def vizualize_box(img_tensor, target):
    fig, ax = plt.subplots(1,1, figsize=(16,10))
    #     print(img_tensor.shape)
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    #     print(img.shape)
    #     print(img)
    for box in target['boxes'].cpu().detach().numpy().astype(np.int32):
        xmin, ymin, xmax, ymax = box
        #         print(xmin, ymin, xmax, ymax)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)

    ax.set_axis_off()
    ax.imshow(img)