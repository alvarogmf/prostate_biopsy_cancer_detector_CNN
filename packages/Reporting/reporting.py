import cv2
import skimage.io
import os
import numpy as np
import matplotlib.pyplot as plt


def image_predict_convert():
    """
    Takes the image from the predict folder and converts it into a PNG file
    :return: None
    """
    img_id = os.listdir('predict/')
    load_path = 'predict/' + img_id[0]
    save_path = 'predict/png_converted/predict.png'

    biopsy = skimage.io.MultiImage(load_path)
    img = cv2.resize(biopsy[-1], (0, 0), fx=0.1, fy=0.1)
    cv2.imwrite(save_path, img)


def prediction(model, image_path, categories, pixels=258):
    """
    Prediction function.
    :param model: Model to use
    :param image_path: Image to analyse
    :param categories: List. Defined previously
    :param pixels: Image dimension. Default 258
    :return: the image, the result and the path to the image
    """
    try:
        img = cv2.imread(image_path)
        resized_img = cv2.resize(img, (pixels, pixels))
    except Exception as e:
        print(f"Error reading file {image_path}")
        exit()

    to_predict = np.array(resized_img).reshape(-1, pixels, pixels, 3)
    y = model.predict(to_predict)

    # Display results
    result = categories[int(y[0][0])]

    fig, ax = plt.subplots()
    label_font = {"fontname": "Arial", "fontsize": 9}
    plt.imshow(img)
    fig.suptitle(result, fontsize=20)
    ax.set_title(image_path, fontdict=label_font)

    return img, result, image_path

def to_pdf(img, result, name = 'results'):
    """
    Saves the analyzed biopsy into a PDF
    :param img: Image array that was previously analysed
    :param result: Prediction result
    :param name: name to provide the PDF, default is results
    :return: PDF file with the analysed image and its prediction
    """
    pdf_path = "results/"
    name = name

    # Matplotlib and pdf generation
    fig, ax = plt.subplots()
    label_font = {"fontname": "Arial", "fontsize": 12}
    img_plot = plt.imshow(img)
    fig.suptitle(result, fontsize=18)
    #ax.set_title(image_path, fontdict=label_font)
    plt.savefig(f"results/{name}.pdf")
    print(f"Image saved as a pdf at {pdf_path}{name}.pdf")
