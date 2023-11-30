import tensorflow as tf
from flask import Flask, render_template, request, jsonify
import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


model = tf.keras.models.load_model("C:\\Users\\ASUS\\PycharmProjects\\bangkit_capstone\\bounding_box_segmentation_5.h5")
# model.summary()


@app.route("/ml-api/helloWorld")
def home():
    return "hello world"

@app.route("/ml-api/getOcr", methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    # print(imagefile)
    uid = request.form.get('uid')
    # image_path = "C:\\Users\\ASUS\\PycharmProjects\\bangkit_capstone\\test.jpg"
    image_path = "C:\\Users\\ASUS\\PycharmProjects\\bangkit_capstone\\images\\"+imagefile.filename
    # image = imagefile.filename
    imagefile.save(image_path)

    img = cv2.imread(image_path, 0)
    ret, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    img = cv2.resize(img, (512, 512))
    img = np.expand_dims(img, axis=-1)
    img = img / 255

    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    pred = np.squeeze(np.squeeze(pred, axis=0), axis=-1)
    # plt.imsave('C:\\Users\\ASUS\\PycharmProjects\\bangkit_capstone\\segmentation\\segmentation.png', pred)
    plt.imsave('segmentation.png', pred)

    img = cv2.imread('segmentation.png', 0)
    cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, img)
    ori_img = cv2.imread(image_path)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    ori_img = cv2.resize(ori_img, (512, 512))

    roi_img = []

    roi_number = 0
    contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0] + cv2.boundingRect(x)[1] * img.shape[1])
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        if w > 40:
            # draw a white rectangle to visualize the bounding rect
            cv2.rectangle(ori_img, (x, y), (x + w, y + h), (36, 255, 12), 2)
            ROI = ori_img[y:y + h, x:x + w]
            roi_img.append(ROI)
            roi_number += 1

    if len(roi_img) > 1:
        name = pytesseract.image_to_string(roi_img[0], lang='eng', config='--psm 7')
        jenis_kelamin = pytesseract.image_to_string(roi_img[1], lang='eng', config='--psm 7')

        verified = bool(True)

    wrong_name = ["\n", "\f"]

    for i in range(len(wrong_name)):
        if wrong_name[i] in name:
            new_name = name.strip(wrong_name[i])
    new_name = new_name.strip("\n")

    for i in range(len(wrong_name)):
        if wrong_name[i] in jenis_kelamin:
            new_jenis_kelamin = jenis_kelamin.strip(wrong_name[i])
    new_jenis_kelamin = new_jenis_kelamin.strip("\n")

    if jenis_kelamin == "PEREMPUAN":
        new_jenis_kelamin = "female"
    elif jenis_kelamin == "LAKI-LAKI":
        new_jenis_kelamin = "male"
    else:
        new_jenis_kelamin = new_jenis_kelamin


        response_json = {
            "name": new_name.title(),
            "gender": new_jenis_kelamin.title(),
            "verified": verified
        }

        return jsonify(response_json)

    # elif len(roi_img) == 1:
    #     text = pytesseract.image_to_string(roi_img[0], lang='eng', config='--psm 7')
    #     return "success"
    # else:
    #     return "success"
    # # return "Success"


if __name__ == "__main__":
    # app.run(host="localhost", debug=True)
    app.run(debug=True, host="0.0.0.0")