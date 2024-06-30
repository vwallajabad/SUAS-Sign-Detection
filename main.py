import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt

img = cv2.imread("image.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

i = 0

for contour in contours:
    if i == 0:
        i = 1
        continue

    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

    cv2.drawContours(img, [contour], 0, (0, 0, 255), 5)

    M = cv2.moments(contour)
    if M["m00"] != 0.0:
        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])

    if len(approx) == 3:
        cv2.putText(
            img, "Triangle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
    elif len(approx) == 4:
        cv2.putText(
            img,
            "Quadrilateral",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
    elif len(approx) == 5:
        cv2.putText(
            img, "Pentagon", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
    elif len(approx) == 6:
        cv2.putText(
            img, "Hexagon", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
    else:
        pass


ocr_img = cv2.imread("image.jpg")


gray_ocr = cv2.cvtColor(ocr_img, cv2.COLOR_BGR2GRAY)


ret, thresh1 = cv2.threshold(gray_ocr, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)


rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))


dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)


ocr_contours, hierarchy = cv2.findContours(
    dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
)


im2 = ocr_img.copy()


file = open("recognized.txt", "w+")
file.write("")
file.close()


for cnt in ocr_contours:
    x, y, w, h = cv2.boundingRect(cnt)

    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cropped = im2[y : y + h, x : x + w]

    file = open("recognized.txt", "a")

    text = pytesseract.image_to_string(cropped)

    file.write(text)
    file.write("\n")

    file.close()


plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Processed Image with Shape Labels")
plt.show()


plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))
plt.title("OCR Image with Detected Text Regions")
plt.show()
