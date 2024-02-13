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

    pn_temp = (prob_normalized - prob_min) / (
        prob_max - prob_min
    )  # Y 채널에 대한 hist [0:1]
    pn_temp[pn_temp > 0] = prob_max * (pn_temp[pn_temp > 0] ** a)
    pn_temp[pn_temp < 0] = prob_max * (-((-pn_temp[pn_temp < 0]) ** a))
    prob_normalized_wd = (
        pn_temp / pn_temp.sum() * normalize
    )  # normalize to [0,normalize]
    cdf_prob_normalized_wd = prob_normalized_wd.cumsum()

    if truncated_cdf:
        inverse_cdf = np.maximum(normalize / 2, normalize - cdf_prob_normalized_wd)
    else:
        inverse_cdf = normalize - cdf_prob_normalized_wd

    img_new = img.copy()
    for i in unique_intensity:
        img_new[img == i] = np.round(255 * (i / 255) ** inverse_cdf[i])

    return img_new


def process_bright(img, a=0.25):  # 밝은 이미지를 어둡게
    img_negative = 255 - img
    agcwd = image_agcwd(img_negative, a=a, truncated_cdf=False)
    reversed_agcwd = 255 - agcwd
    return reversed_agcwd


def process_dimmed(img, a=0.75):  # 어두운 이미지를 밝게
    agcwd = image_agcwd(img, a=a, truncated_cdf=True)
    return agcwd


def is_bright(Y, img_size):
    threshold = 0.3
    exp_in = 112
    M, N = img_size
    mean_in = np.sum(Y / (M * N))
    t = (mean_in - exp_in) / exp_in
    if t < -threshold:  # 어두운 이미지
        return "Dimmed"
    elif t > threshold:  # 밝은 이미지
        return "Bright"
    else:
        return "Plain"


def enhance_color(img, factor=1.2, count=1):
    dst = img

    for _ in range(count):
        max_rgb = np.max(dst, axis=-1, keepdims=True)
        max_rgb_nonzero = np.where(max_rgb == 0, 1, max_rgb)
        dst = (dst / max_rgb_nonzero) ** factor * max_rgb

        dst = dst.astype(np.uint8)

    return dst.astype(np.uint8)


def handler(image, gamma=0.75, factor=1.7):
    """
    Applies a series of image processing operations to enhance the brightness and color of an input image.

    Args:
    - image (PIL.Image.Image): Input image in PIL format.
    - gamma (float): Gamma correction factor for adjusting brightness. Must be in the range [0.0, 1.0].
    - factor (float): Color enhancement factor. Should be in the range [1.0, 3.0].

    Note:
    - The gamma parameter controls the adjustment of brightness in dimmed or bright areas of the image.
    - The factor parameter enhances the color saturation.
    """
    src = np.array(image)
    # src = resize_image(src)

    YCrCb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
    Y = YCrCb[:, :, 0]

    brightness = is_bright(Y, img_size=src.shape[:2])
    # print(brightness)

    brightened = src
    if brightness == "Dimmed":
        result = process_dimmed(Y, a=gamma)
        YCrCb[:, :, 0] = result
        brightened = cv2.cvtColor(YCrCb, cv2.COLOR_YCrCb2BGR)
    elif brightness == "Bright":
        result = process_bright(Y, a=1.0-gamma)
        YCrCb[:, :, 0] = result
        brightened = cv2.cvtColor(YCrCb, cv2.COLOR_YCrCb2BGR)

    enhanced = enhance_color(brightened, factor=factor)

    # cv2.imshow("src", src)
    # cv2.imshow("brightened", brightened)
    # cv2.imshow("enhanced", enhanced)
    return enhanced

if __name__ == "__main__":
    import os

    pwd = os.path.dirname(os.path.realpath(__file__))
    img_path = os.path.join(pwd, "Images", "perfume.jpg")
    src = cv2.imread(img_path)
    pil_image = Image.fromarray(src)

    dst = handler(pil_image, 0.75, 1.7)
    cv2.waitKey()
    cv2.destroyAllWindows()
