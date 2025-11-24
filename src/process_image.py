import cv2
from PIL import Image
import numpy as np

# Сегментация
def segment_image(image_path):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Инвент цвета
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    # Контуры
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    symbols = []
    # Сортировка контуров
    for cnt in sorted(contours, key=lambda c: cv2.boundingRect(c)[0]):
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 5 or h < 5:
            continue
        symbol_img = img[y:y + h, x:x + w]
        symbol_img = Image.fromarray(symbol_img)
        symbols.append(symbol_img)
    return symbols
