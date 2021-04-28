import cv2
import skimage.io
import os
import numpy as np
import matplotlib.pyplot as plt


def image_predict_convert():
    img_id = os.listdir('predict/')
    load_path = 'predict/' + img_id[0]
    save_path = 'predict/png_converted/' + 'predict.png'

    biopsy = skimage.io.MultiImage(load_path)
    img = cv2.resize(biopsy[-1], (0, 0), fx=0.1, fy=0.1)
    cv2.imwrite(save_path, img)


def prediction(model, image_path, categories, pixels=258):
    """
    Analyses new cell images given a loaded or just trained model
    :param model: Model to use
    :param image_path: Image to analyse
    :param categories: List. Defined previously
    :param pixels: Image dimension. Default 258
    :return: Shows up the original cell image with a caption, Parasitized or Uninfected
    """
    try:
        img = cv2.imread(image_path)
        resized_img = cv2.resize(img, (pixels, pixels))
    except Exception as e:
        print(f"Error reading file {image_path}")
        exit()

    to_predict = np.array(resized_img).reshape(-1, pixels, pixels, 3)  # 1 grayscale, 3 colored images
    y = model.predict(to_predict)

    # Display results
    result = categories[int(y[0][0])]

    fig, ax = plt.subplots()
    label_font = {"fontname": "Arial", "fontsize": 9}
    plt.imshow(img)
    fig.suptitle(result, fontsize=20)
    ax.set_title(image_path, fontdict=label_font)

    return img, result, image_path


