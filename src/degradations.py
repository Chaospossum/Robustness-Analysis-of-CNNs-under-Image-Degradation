import numpy as np
import cv2

def to_uint8(img_float01):
    return (np.clip(img_float01, 0, 1) * 255).astype(np.uint8)

def to_float01(img_uint8):
    return img_uint8.astype(np.float32) / 255.0

def gaussian_noise(img_float01, sigma):
    noise = np.random.normal(0.0, sigma, img_float01.shape).astype(np.float32)
    out = img_float01 + noise
    return np.clip(out, 0, 1)

def gaussian_blur(img_float01, ksize):
    # ksize must be odd: 3,5,7,...
    img_u8 = to_uint8(img_float01)
    out = cv2.GaussianBlur(img_u8, (ksize, ksize), 0)
    return to_float01(out)

def darken(img_float01, factor):
    # factor in (0,1], e.g. 0.8 darker, 0.5 very dark
    return np.clip(img_float01 * factor, 0, 1)
def contrast(img_float01, factor):
    """
    factor < 1 reduces contrast, factor > 1 increases contrast.
    Typical low-contrast: 0.8, 0.6, 0.4
    """
    mean = np.mean(img_float01, axis=(0, 1), keepdims=True)
    out = (img_float01 - mean) * factor + mean
    return np.clip(out, 0, 1)

def jpeg_compress(img_float01, quality):
    """
    JPEG compression artifacts. quality in [1..100].
    Lower quality = worse artifacts. Try 50, 30, 10.
    """
    img_u8 = to_uint8(img_float01)
    # OpenCV expects BGR for encoding/decoding, but artifacts are fine either way.
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, enc = cv2.imencode(".jpg", img_u8, encode_param)
    if not ok:
        return img_float01
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return to_float01(dec)

def motion_blur(img_float01, ksize):
    """
    Simple horizontal motion blur. ksize must be odd: 3,5,7,...
    """
    img_u8 = to_uint8(img_float01)
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0
    kernel /= kernel.sum()
    out = cv2.filter2D(img_u8, -1, kernel)
    return to_float01(out)
