import os
import glob
import re
import cv2
import numpy as np
from PIL import Image
import pytesseract
import json
from flask import Flask, render_template, send_file
from flask import request, redirect, url_for
import shutil


# Load owner details 
with open("db.json") as f:
    owner_details = json.load(f)

app = Flask(__name__)


# Detecting number plate
def number_plate_detection(img):
    def clean2_plate(plate):
        gray_img = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_img, 110, 255, cv2.THRESH_BINARY)
        num_contours, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        if num_contours:
            contour_area = [cv2.contourArea(c) for c in num_contours]
            max_cntr_index = np.argmax(contour_area)
            max_cnt = num_contours[max_cntr_index]
            x, y, w, h = cv2.boundingRect(max_cnt)

            if not ratioCheck(cv2.contourArea(max_cnt), w, h):
                return plate, None

            final_img = thresh[y : y + h, x : x + w]
            return final_img, [x, y, w, h]
        else:
            return plate, None

    def ratioCheck(area, width, height):
        ratio = float(width) / float(height)
        if ratio < 1:
            ratio = 1 / ratio
        return (1063.62 < area < 73862.5) and (3 < ratio < 6)

    def isMaxWhite(plate):
        avg = np.mean(plate)
        return avg >= 115

    def ratio_and_rotation(rect):
        (x, y), (width, height), rect_angle = rect
        angle = -rect_angle if width > height else 90 + rect_angle
        return (
            abs(angle) <= 15
            and width > 0
            and height > 0
            and ratioCheck(width * height, width, height)
        )

    img2 = cv2.GaussianBlur(img, (5, 5), 0)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2 = cv2.Sobel(img2, cv2.CV_8U, 1, 0, ksize=3)
    _, img2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(17, 3))
    morph_img_threshold = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel=element)

    num_contours, _ = cv2.findContours(
        morph_img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    for cnt in num_contours:
        min_rect = cv2.minAreaRect(cnt)
        if ratio_and_rotation(min_rect):
            x, y, w, h = cv2.boundingRect(cnt)
            plate_img = img[y : y + h, x : x + w]

            if isMaxWhite(plate_img):
                clean_plate, rect = clean2_plate(plate_img)
                if rect:
                    plate_im = Image.fromarray(clean_plate)
                    text = pytesseract.image_to_string(plate_im, lang="eng")
                    return text
    return None


@app.route("/")
def index():
    detected = False
    dir_path = os.path.dirname(__file__)
    car_images = []

    # Detect number plates in the images
    for img_file in glob.glob(os.path.join(dir_path, "Images", "*.jpeg")):
        img = cv2.imread(img_file)
        img2 = cv2.resize(img, (600, 600))
        number_plate = number_plate_detection(img)

        if number_plate is not None:
            res2 = str("".join(re.split("[^a-zA-Z0-9]*", number_plate))).upper()

            if res2 in owner_details:
                owner_info = owner_details[res2]
                car_images.append(
                    {
                        "plate": res2,
                        "name": owner_info["Name"],
                        "phone": owner_info["Phone"],
                        "address": owner_info["Address"],
                        "fine_details": owner_info["Fine Details"],
                        "location": owner_info["Location"],
                        "near": owner_info["Near"],
                        "date": owner_info["Date"],
                        "time": owner_info["Time"],
                        "fine_amount": owner_info["Fine Amount"],
                        "paid_or_unpaid": owner_info["Paid or Unpaid"],
                        "img_file": img_file,
                    }
                )
                detected = True

    return render_template("index.html", car_images=car_images, detected=detected)


@app.route("/image/<path:filename>")
def get_image(filename):
    return send_file(filename, mimetype='image/jpeg')
    # return send_file(os.path.join("Images", filename), mimetype="image/jpeg")


@app.route("/upload", methods=["POST"])
def upload_image():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"
    
    if file and file.filename.endswith(".jpeg"):
        # Clear the 'Images' folder before saving the new image
        dir_path = os.path.join("Images")
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)  # Remove all previous images
        os.makedirs(dir_path)  # Recreate the folder after deletion

        file_path = os.path.join(dir_path, file.filename)
        file.save(file_path)

        return redirect(url_for("index"))  # After upload, redirect to the index to display results
    
    return "Invalid file format, only JPEG is allowed"


if __name__ == "__main__":
    app.run(debug=True)
    






