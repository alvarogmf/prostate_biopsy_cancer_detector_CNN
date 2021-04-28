import packages.Acquisition.acquisition as ac
import packages.Wrangling.wrangling as wr
import packages.Model.model as ml
import packages.Reporting.reporting as rp


train_csv_path = "train.csv"
load_path = "train_images/"
save_path = "png_images/"
CATEGORIES = ['Benign', 'Malign']
model_name = 'model'
prediction_image_path = 'predict/png_converted/predict.png'

import argparse


def terminal_parser():
    parser = argparse.ArgumentParser(description="Determines whether a biopsy image has cancer or not")
    parser.add_argument("-c", "--convert", action="store_true",
						help="Converts the Train images from .TIFF format to .PNG")
    parser.add_argument("-t", "--train", action="store_true",
						help="Reads the PNG images in the images_png folder, trains the model and saves it as a h5 file.")
    parser.add_argument("-p", "--predict", action="store_true",
						help="Reads a TIFF image placed in the predict folder and brings the prediction to a pdf file"))

    return parser.parse_args()

def main():
    # Argparse
    args = terminal_parser()

    if args.convert:
        train_csv = ac.load_csv(path=train_csv_path)
        ac.transform_images(load_path=load_path, save_path=save_path, csv_name=train_csv, format = '.png')

    if args.train:
        train_csv = ac.load_csv(path=train_csv_path)
        clean_train_df = wr.create_clean_csv(original_csv=train_csv, format='.png')

        train_generator, validation_generator = wr.image_generator(dataframe=clean_train_df,directory=save_path, target_size = (258, 258), batch_size = 64)

        model = ml.model(train_generator, validation_generator)

        ml.save_model_h5(model, model_path="model/", model_name="model")

    if args.predict:
        rp.image_predict_convert()
        model = tf.keras.models.load_model(os.path.join('model/', f"{model_name}.h5"))
        img, result, image_path = rp.prediction(model = model , image_path = prediction_image_path, categories, pixels=258)
        print(f'Prediction Complete! The result is: {result}')
        # MISSING REPORT TO PDF




# -------------------------
if __name__ == "__main__":
    main()
