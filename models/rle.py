import numpy as np

class RLECompressor:
    def __init__(self):
        pass

    def encode_channel(self, channel: np.ndarray):
        """
        对单通道二维图像做RLE编码
        :param channel: (H,W) 单通道uint8数组
        :return: list[(count, value)] 游程列表
        """
        # 铺平成一维
        data = channel.flatten()
        encoded = []
        if len(data) == 0:
            return encoded

        current_val = data[0]
        count = 1
        for val in data[1:]:
            if val == current_val:
                count += 1
            else:
                encoded.append((count, current_val))
                current_val = val
                count = 1
        # 把最后一组加入
        encoded.append((count, current_val))
        return encoded

    def decode_channel(self, encoded: list, height:int, width:int) -> np.ndarray:
        """
        RLE解码单通道，还原(H,W)数组
        :param encoded: encode_channel输出的游程列表
        :param height: 图像高
        :param width: 图像宽
        :return: np.ndarray shape(H,W) uint8
        """
        buffer = []
        for cnt, val in encoded:
            buffer.extend([val]*cnt)
        arr = np.array(buffer, dtype=np.uint8)
        return arr.reshape(height, width)

    def compress(self, image: np.ndarray):
        """
        对外压缩接口
        :param image: np.ndarray (H,W)灰度 / (H,W,3)彩色 RGB
        :return: dict 保存编码结果 + 图像尺寸信息，用于后续解码
        """
        h, w = image.shape[0], image.shape[1]
        if len(image.shape) == 2:
            # 灰度图，单通道
            enc = self.encode_channel(image)
            return {
                "type":"gray",
                "h":h,
                "w":w,
                "data": enc
            }
        elif len(image.shape)==3 and image.shape[2]==3:
            # RGB三通道分别编码
            r_enc = self.encode_channel(image[:,:,0])
            g_enc = self.encode_channel(image[:,:,1])
            b_enc = self.encode_channel(image[:,:,2])
            return {
                "type":"rgb",
                "h":h,
                "w":w,
                "r": r_enc,
                "g": g_enc,
                "b": b_enc
            }
        else:
            raise ValueError("不支持的图像数组格式")

    def decompress(self, compressed_dict: dict) -> np.ndarray:
        """
        对外解压接口，返回还原图像numpy数组
        """
        h = compressed_dict["h"]
        w = compressed_dict["w"]
        if compressed_dict["type"] == "gray":
            ch = self.decode_channel(compressed_dict["data"], h, w)
            return ch
        elif compressed_dict["type"] == "rgb":
            r_ch = self.decode_channel(compressed_dict["r"], h, w)
            g_ch = self.decode_channel(compressed_dict["g"], h, w)
            b_ch = self.decode_channel(compressed_dict["b"], h, w)
            return np.dstack([r_ch, g_ch, b_ch])
        else:
            raise ValueError("未知压缩数据类型")


# ----------------直接在这里写测试代码，不需要GUI----------------
if __name__ == "__main__":
    from PIL import Image
    import os
    print("====测试RLE====")
    # 输入路径
    input_path = "test/input/mao.jpg"
    # 输出路径
    output_path = "test/output/rle_test_out.png"

    img = Image.open(input_path).convert("RGB")
    img_np = np.array(img)
    # 创建对象rle
    rle = RLECompressor()
    comp = rle.compress(img_np)
    restore = rle.decompress(comp)
    print("原图shape", img_np.shape)
    print("还原图shape", restore.shape)
    print("是否完全无损：", np.array_equal(img_np, restore))
    Image.fromarray(restore).save(output_path)
    print(f"输出文件 {output_path} 已生成，请打开核对图片")
