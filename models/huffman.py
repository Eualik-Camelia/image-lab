import numpy as np
import heapq
from collections import Counter


class HuffmanCompressor:
    class Node:
        """霍夫曼树叶子/非叶子节点"""
        def __init__(self, value, freq):
            self.value = value
            self.freq = freq
            self.left = None
            self.right = None

        # heapq 需要重载小于运算符
        def __lt__(self, other):
            return self.freq < other.freq


    def build_tree(self, freq_dict):
        """根据像素频率构建霍夫曼最小堆树"""
        heap = []
        for val, f in freq_dict.items():
            heapq.heappush(heap, self.Node(val, f))

        while len(heap) > 1:
            n1 = heapq.heappop(heap)
            n2 = heapq.heappop(heap)
            parent = self.Node(None, n1.freq + n2.freq)
            parent.left = n1
            parent.right = n2
            heapq.heappush(heap, parent)
        root = heap[0] if heap else None
        return root


    def build_code_map(self, root):
        """DFS遍历树，生成 {像素值:"01码字"}"""
        code_map = {}
        def dfs(node, code_str):
            if node is None:
                return
            if node.value is not None:
                code_map[node.value] = code_str
                return
            dfs(node.left, code_str + "0")
            dfs(node.right, code_str + "1")
        dfs(root, "")
        return code_map


    def encode_channel(self, channel: np.ndarray):
        """单通道编码，返回比特字符串 + 霍夫曼树根节点"""
        data = channel.flatten()
        freq = Counter(data)
        tree_root = self.build_tree(freq)
        code_map = self.build_code_map(tree_root)
        bit_str = "".join([code_map[p] for p in data])
        return bit_str, tree_root


    def decode_channel(self, bit_str: str, tree_root, height:int, width:int) -> np.ndarray:
        """单通道解码比特串还原图像数组"""
        total_pixel = height * width
        res = []
        cur_node = tree_root

        for bit in bit_str:
            if bit == "0":
                cur_node = cur_node.left
            else:
                cur_node = cur_node.right
            if cur_node.value is not None:
                res.append(cur_node.value)
                cur_node = tree_root
                if len(res) >= total_pixel:
                    break
        arr = np.array(res, dtype=np.uint8)
        return arr.reshape(height, width)


    def compress(self, image: np.ndarray):
        """对外压缩接口，支持灰度、RGB彩色"""
        h, w = image.shape[0], image.shape[1]
        if len(image.shape) == 2:
            bits, tree = self.encode_channel(image)
            return {
                "type":"gray",
                "h":h,
                "w":w,
                "bits": bits,
                "tree": tree
            }
        elif len(image.shape)==3 and image.shape[2]==3:
            r_bits, r_tree = self.encode_channel(image[:,:,0])
            g_bits, g_tree = self.encode_channel(image[:,:,1])
            b_bits, b_tree = self.encode_channel(image[:,:,2])
            return {
                "type":"rgb",
                "h":h,
                "w":w,
                "r_bits": r_bits,
                "r_tree": r_tree,
                "g_bits": g_bits,
                "g_tree": g_tree,
                "b_bits": b_bits,
                "b_tree": b_tree
            }
        else:
            raise ValueError("不支持图像格式")


    def decompress(self, comp_dict: dict) -> np.ndarray:
        """解压返回numpy图像数组"""
        h = comp_dict["h"]
        w = comp_dict["w"]
        if comp_dict["type"] == "gray":
            ch = self.decode_channel(comp_dict["bits"], comp_dict["tree"], h, w)
            return ch
        elif comp_dict["type"] == "rgb":
            r_ch = self.decode_channel(comp_dict["r_bits"], comp_dict["r_tree"], h, w)
            g_ch = self.decode_channel(comp_dict["g_bits"], comp_dict["g_tree"], h, w)
            b_ch = self.decode_channel(comp_dict["b_bits"], comp_dict["b_tree"], h, w)
            return np.dstack([r_ch, g_ch, b_ch])
        else:
            raise ValueError("未知压缩数据类型")


# ----------------测试代码，路径统一 test/input → test/output ----------------
if __name__ == "__main__":
    import sys
    from pathlib import Path
    # 将项目根目录加入搜索路径
    sys.path.append(str(Path(__file__).parent.parent))

    from PIL import Image
    from utils.metrics import calc_psnr, calc_compression_ratio

    print("====测试霍夫曼编码====")
    input_path = "test/input/mao.jpg"
    output_path = "test/output/huffman_out.png"

    img = Image.open(input_path).convert("RGB")
    img_np = np.array(img)

    huf = HuffmanCompressor()
    comp_data = huf.compress(img_np)
    restore_img = huf.decompress(comp_data)

    print("原图shape", img_np.shape)
    print("还原shape", restore_img.shape)
    print("是否无损：", np.array_equal(img_np, restore_img))

    psnr_val = calc_psnr(img_np, restore_img)
    cr = calc_compression_ratio(img_np, comp_data)
    print(f"PSNR = {psnr_val:.2f} dB")
    print(f"估算压缩比 = {cr:.2f}")

    Image.fromarray(restore_img).save(output_path)
    print(f"输出 {output_path}")

