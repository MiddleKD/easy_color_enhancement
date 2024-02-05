import cv2
import numpy as np
from PIL import Image


def resize_image(img):
    height, width = img.shape[:2]

    min_dim = min(height, width)

    start_h = (height - min_dim) // 2
    end_h = start_h + min_dim
    start_w = (width - min_dim) // 2
    end_w = start_w + min_dim

    cropped_resized_img = cv2.resize(img[start_h:end_h, start_w:end_w], (512, 512))

    return cropped_resized_img


def image_agcwd(img, a=0.25, truncated_cdf=False, normalize=1.5):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])  # Y 채널에 대한 hist

    prob_normalized = hist / hist.sum()  # Y 채널에 대한 hist 총합 1

    unique_intensity = np.unique(img)

    prob_min = prob_normalized.min()
    prob_max = prob_normalized.max()

    pn_temp = (prob_normalized - prob_min) / (prob_max - prob_min) # Y 채널에 대한 hist [0:1]
    pn_temp[pn_temp > 0] = prob_max * (pn_temp[pn_temp > 0] ** a)
    pn_temp[pn_temp < 0] = prob_max * (-((-pn_temp[pn_temp < 0]) ** a))
    prob_normalized_wd = pn_temp / pn_temp.sum() * normalize # normalize to [0,normalize]
    cdf_prob_normalized_wd = prob_normalized_wd.cumsum()

    if truncated_cdf:
        inverse_cdf = np.maximum(normalize / 2, normalize - cdf_prob_normalized_wd)
    else:
        inverse_cdf = normalize - cdf_prob_normalized_wd

    img_new = img.copy()
    for i in unique_intensity:
        img_new[img == i] = np.round(255 * (i / 255) ** inverse_cdf[i])

    return img_new


def process_bright(img):  # 밝은 이미지를 어둡게
    img_negative = 255 - img
    agcwd = image_agcwd(img_negative, a=0.25, truncated_cdf=False)
    reversed_agcwd = 255 - agcwd
    return reversed_agcwd


def process_dimmed(img):  # 어두운 이미지를 밝게
    agcwd = image_agcwd(img, a=0.75, truncated_cdf=True)
    return agcwd


def image_brightness(img):
    YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y = YCrCb[:, :, 0]

    threshold = 0.3
    exp_in = 112
    M, N = img.shape[:2]
    mean_in = np.sum(Y / (M * N))
    t = (mean_in - exp_in) / exp_in

    img_output = None
    if t < -threshold:  # 어두운 이미지
        print("Dimmed Image")
        result = process_dimmed(Y)
        YCrCb[:, :, 0] = result
        img_output = cv2.cvtColor(YCrCb, cv2.COLOR_YCrCb2BGR)
    elif t > threshold:  # 밝은 이미지
        print("Bright Image")
        result = process_bright(Y)
        YCrCb[:, :, 0] = result
        img_output = cv2.cvtColor(YCrCb, cv2.COLOR_YCrCb2BGR)
    else:
        img_output = img
    return img_output


def enhance_color(img, factor=1.2, count=1):
    dst = img

    for _ in range(count):
        max_rgb = np.max(dst, axis=-1, keepdims=True)
        max_rgb_nonzero = np.where(max_rgb == 0, 1, max_rgb)
        dst = (dst / max_rgb_nonzero) ** factor * max_rgb

        dst = dst.astype(np.uint8)

    return dst.astype(np.uint8)


def color_enhancement(image):
    src = np.array(image)
    src = resize_image(src)
    cv2.imshow("src", src)

    brightened = image_brightness(src)
    cv2.imshow("brightened", brightened)

    enhanced = enhance_color(brightened, factor=1.7)
    cv2.imshow("enhanced", enhanced)
    return enhanced


if __name__ == "__main__":
    import os

    pwd = os.path.dirname(os.path.realpath(__file__))
    img_path = os.path.join(pwd, "Images", "cup.jpg")
    src = cv2.imread(img_path)
    pil_image = Image.fromarray(src)

    color_enhancement(pil_image)
    cv2.waitKey()
    cv2.destroyAllWindows()
