import numpy as np

def calc_mse(original: np.ndarray, compressed: np.ndarray):
    """计算MSE均方误差，数值越大图像失真越严重"""
    return np.mean(np.square(original.astype(np.float64) - compressed.astype(np.float64)))


def calc_psnr(original: np.ndarray, compressed: np.ndarray):
    """
    PSNR峰值信噪比，单位dB
    越高质量越好；30dB以上肉眼基本看不出差异
    """
    mse = calc_mse(original, compressed)
    if mse < 1e-10:
        return float("inf")   # 完全无损，无穷大
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr


def calc_compression_ratio(original_img_np: np.ndarray, compressed_dict):
    """
    粗略计算压缩比：原始字节 / 压缩后字节
    RLE、Huffman只是内存字典，这里做**估算**，仅用于参考
    """
    h, w = original_img_np.shape[0], original_img_np.shape[1]
    if len(original_img_np.shape) ==3:
        original_bytes = h * w *3
    else:
        original_bytes = h * w

    # --------RLE压缩比估算--------
    if "data" in compressed_dict:
        if compressed_dict["type"] == "rgb":
            total = len(compressed_dict["r"]) + len(compressed_dict["g"]) + len(compressed_dict["b"])
        else:
            total = len(compressed_dict["data"])
        comp_bytes = total * 2
    # --------Huffman压缩比估算（比特串总比特转字节）--------
    elif "r_bits" in compressed_dict:
        if compressed_dict["type"] == "rgb":
            total_bits = len(compressed_dict["r_bits"]) + len(compressed_dict["g_bits"]) + len(compressed_dict["b_bits"])
        else:
            total_bits = len(compressed_dict["bits"])
        comp_bytes = total_bits / 8.0
    else:
        comp_bytes = original_bytes

    ratio = original_bytes / comp_bytes
    return ratio
