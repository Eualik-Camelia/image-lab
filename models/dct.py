import numpy as np


class DCTCompressor:
    def __init__(self, quality=50):
        """
        :param quality: JPEG质量系数 1~100；越大画质越高，压缩程度越小
        """
        self.quality = quality
        # JPEG标准亮度量化表
        self.base_y_quant = np.array([
            [16,11,10,16,24,40,51,61],
            [12,12,14,19,26,58,60,55],
            [14,13,16,24,40,57,69,56],
            [14,17,22,29,51,87,80,62],
            [18,22,37,56,68,109,103,77],
            [24,35,55,64,81,104,113,92],
            [49,64,78,87,103,121,120,101],
            [72,92,95,98,112,100,103,99]
        ], dtype=np.float64)
        # JPEG标准色度量化表
        self.base_c_quant = np.array([
            [17,18,24,47,99,99,99,99],
            [18,21,26,66,99,99,99,99],
            [24,26,56,99,99,99,99,99],
            [47,66,99,99,99,99,99,99],
            [99,99,99,99,99,99,99,99],
            [99,99,99,99,99,99,99,99],
            [99,99,99,99,99,99,99,99],
            [99,99,99,99,99,99,99,99]
        ], dtype=np.float64)
        self.scale_quant_table()
        # 预计算8点正交归一DCT矩阵
        self.dct_mat = self._make_dct_matrix(8)
        self.idct_mat = self.dct_mat.T

    def _make_dct_matrix(self, N):
        """生成N点正交归一DCT变换矩阵"""
        mat = np.zeros((N, N), dtype=np.float64)
        for u in range(N):
            for x in range(N):
                if u == 0:
                    cu = np.sqrt(1.0 / N)
                else:
                    cu = np.sqrt(2.0 / N)
                mat[u, x] = cu * np.cos((2 * x + 1) * u * np.pi / (2 * N))
        return mat

    def scale_quant_table(self):
        """根据quality参数缩放量化表"""
        q = np.clip(self.quality, 1, 100)
        if q < 50:
            scale = 5000.0 / q
        else:
            scale = 200.0 - q * 2.0
        self.y_quant = np.round(self.base_y_quant * scale / 100.0).clip(1, 255)
        self.c_quant = np.round(self.base_c_quant * scale / 100.0).clip(1, 255)

    def rgb2ycbcr(self, rgb: np.ndarray):
        """RGB uint8 → YCbCr浮点"""
        r = rgb[:, :, 0].astype(np.float64)
        g = rgb[:, :, 1].astype(np.float64)
        b = rgb[:, :, 2].astype(np.float64)
        Y  = 0.299 * r + 0.587 * g + 0.114 * b
        Cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
        Cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
        return np.dstack([Y, Cb, Cr])

    def ycbcr2rgb(self, ycbcr: np.ndarray):
        """YCbCr浮点 → RGB uint8，裁剪0‑255"""
        Y  = ycbcr[:, :, 0]
        Cb = ycbcr[:, :, 1]
        Cr = ycbcr[:, :, 2]
        R = Y + 1.402 * (Cr - 128)
        G = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
        B = Y + 1.772 * (Cb - 128)
        rgb = np.dstack([R, G, B])
        return np.clip(rgb, 0, 255).astype(np.uint8)

    def block_dct(self, block: np.ndarray):
        """8×8块DCT，输入已经完成‑128中心化"""
        return self.dct_mat @ block @ self.dct_mat.T

    def block_idct(self, block: np.ndarray):
        """8×8逆DCT，输出后外部+128恢复偏移"""
        return self.idct_mat @ block @ self.idct_mat.T

    def process_channel(self, channel: np.ndarray, quant_table: np.ndarray):
        """单通道处理：分8×8块 → DCT → 量化 → 反量化 → IDCT"""
        h, w = channel.shape
        out = np.zeros_like(channel, dtype=np.float64)
        h8 = h - (h % 8)
        w8 = w - (w % 8)
        for i in range(0, h8, 8):
            for j in range(0, w8, 8):
                blk = channel[i:i+8, j:j+8]
                dct_blk = self.block_dct(blk - 128)
                q_blk = np.round(dct_blk / quant_table)
                iq_blk = q_blk * quant_table
                rec_blk = self.block_idct(iq_blk) + 128
                out[i:i+8, j:j+8] = rec_blk
        return out[:h8, :w8]

    def compress(self, image: np.ndarray):
        """
        DCT压缩入口
        :param image: (H,W,3) uint8 RGB原图
        :return dict: type,h,w,quality,reconstructed_rgb重建图像
        """
        h, w = image.shape[0], image.shape[1]
        h8 = h - (h % 8)
        w8 = w - (w % 8)
        img_crop = image[:h8, :w8, :]
        ycbcr = self.rgb2ycbcr(img_crop)

        Y_rec  = self.process_channel(ycbcr[:, :, 0], self.y_quant)
        Cb_rec = self.process_channel(ycbcr[:, :, 1], self.c_quant)
        Cr_rec = self.process_channel(ycbcr[:, :, 2], self.c_quant)

        ycbcr_rec = np.dstack([Y_rec, Cb_rec, Cr_rec])
        rgb_rec = self.ycbcr2rgb(ycbcr_rec)

        return {
            "type": "dct",
            "h": h8,
            "w": w8,
            "quality": self.quality,
            "reconstructed_rgb": rgb_rec
        }

    def decompress(self, comp_dict: dict) -> np.ndarray:
        """简化实现：DCT重建已经完成，直接返回图像"""
        return comp_dict["reconstructed_rgb"]


# ----------------测试入口----------------
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))

    from PIL import Image
    from utils.metrics import calc_psnr

    print("====测试DCT有损压缩====")
    input_path  = "test/input/mao.jpg"
    output_path = "test/output/dct_out.png"

    img = Image.open(input_path).convert("RGB")
    img_np = np.array(img)

    dct = DCTCompressor(quality=30)
    comp = dct.compress(img_np)
    restore = dct.decompress(comp)

    print("原图shape", img_np.shape)
    print("DCT重建shape", restore.shape)
    psnr = calc_psnr(img_np[:restore.shape[0], :restore.shape[1], :], restore)
    print(f"PSNR = {psnr:.2f} dB")

    Image.fromarray(restore).save(output_path)
    print(f"输出 {output_path}")
