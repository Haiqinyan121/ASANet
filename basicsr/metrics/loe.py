import cv2
import numpy as np

from basicsr.metrics.metric_util import reorder_image


def _to_lightness_map(img):
    if img.ndim == 2:
        return img.astype(np.float32)
    if img.shape[2] == 1:
        return img[..., 0].astype(np.float32)
    return np.max(img.astype(np.float32), axis=2)


def _resize_for_loe(lightness_map, downsample_to):
    h, w = lightness_map.shape[:2]
    short_side = min(h, w)
    if short_side <= 0:
        raise ValueError('Invalid image shape for LOE calculation.')

    scale = float(downsample_to) / float(short_side)
    resized_w = max(1, int(round(w * scale)))
    resized_h = max(1, int(round(h * scale)))
    return cv2.resize(lightness_map, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)


def calculate_loe(img_low,
                  img_enhanced,
                  crop_border=0,
                  input_order='HWC',
                  downsample_to=50):
    """Calculate LOE (Lightness Order Error).

    Ref: LIME: Low-Light Image Enhancement via Illumination Map Estimation.

    Args:
        img_low (ndarray): Original low-light image in range [0, 255] or [0, 1].
        img_enhanced (ndarray): Enhanced image in range [0, 255] or [0, 1].
        crop_border (int): Cropped pixels in each edge of an image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
        downsample_to (int): Resize the shorter side to this value before LOE.

    Returns:
        float: LOE result. Lower is better.
    """

    assert img_low.shape == img_enhanced.shape, (
        f'Image shapes are different: {img_low.shape}, {img_enhanced.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')

    img_low = reorder_image(img_low, input_order=input_order)
    img_enhanced = reorder_image(img_enhanced, input_order=input_order)

    if crop_border != 0:
        img_low = img_low[crop_border:-crop_border, crop_border:-crop_border, ...]
        img_enhanced = img_enhanced[crop_border:-crop_border, crop_border:-crop_border, ...]

    low_lightness = _to_lightness_map(img_low)
    enhanced_lightness = _to_lightness_map(img_enhanced)

    low_lightness = _resize_for_loe(low_lightness, downsample_to)
    enhanced_lightness = _resize_for_loe(enhanced_lightness, downsample_to)

    low_vec = low_lightness.reshape(-1)
    enhanced_vec = enhanced_lightness.reshape(-1)

    low_order = low_vec[:, None] >= low_vec[None, :]
    enhanced_order = enhanced_vec[:, None] >= enhanced_vec[None, :]
    relative_order_error = np.logical_xor(low_order, enhanced_order)

    return float(relative_order_error.sum(axis=1).mean())
