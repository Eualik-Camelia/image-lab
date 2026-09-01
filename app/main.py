import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np

from models.rle import RLECompressor
from models.huffman import HuffmanCompressor
from models.dct import DCTCompressor
from utils.metrics import calc_psnr, calc_compression_ratio


class ImageCompressGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("图像压缩实验平台｜RLE / Huffman / DCT")
        self.root.geometry("1100x720")

        self.img_origin_np = None
        self.img_recon_np = None
        self.origin_tk = None
        self.recon_tk = None

        # 顶部控制区
        frame_top = ttk.Frame(root, padding=8)
        frame_top.pack(fill=tk.X)

        ttk.Button(frame_top, text="选择图片", command=self.load_image).pack(side=tk.LEFT, padx=5)

        ttk.Label(frame_top, text="算法：").pack(side=tk.LEFT, padx=(15, 5))
        self.alg_var = tk.StringVar(value="RLE")
        self.combo_alg = ttk.Combobox(frame_top, textvariable=self.alg_var,
                                       values=["RLE", "Huffman", "DCT"], state="readonly", width=12)
        self.combo_alg.pack(side=tk.LEFT)
        self.combo_alg.bind("<<ComboboxSelected>>", self.on_alg_change)

        # DCT quality滑块
        self.quality_frame = ttk.Frame(frame_top)
        ttk.Label(self.quality_frame, text="Quality:").pack(side=tk.LEFT, padx=(15, 4))
        self.qual_var = tk.IntVar(value=30)
        self.scale_qual = ttk.Scale(self.quality_frame, from_=1, to=100,
                                    variable=self.qual_var, orient=tk.HORIZONTAL, length=160)
        self.scale_qual.pack(side=tk.LEFT)
        self.label_qual_val = ttk.Label(self.quality_frame, text="30")
        self.label_qual_val.pack(side=tk.LEFT, padx=4)
        self.scale_qual.configure(command=lambda v: self.label_qual_val.config(text=str(int(float(v)))))
        self.quality_frame.pack(side=tk.LEFT)
        self.quality_frame.pack_forget()

        ttk.Button(frame_top, text="执行压缩解压", command=self.run_compress).pack(side=tk.LEFT, padx=10)
        ttk.Button(frame_top, text="保存结果", command=self.save_result).pack(side=tk.LEFT)

        # 信息面板
        frame_info = ttk.LabelFrame(root, text="指标信息", padding=8)
        frame_info.pack(fill=tk.X, padx=8, pady=4)
        self.info_text = tk.Text(frame_info, height=4)
        self.info_text.pack(fill=tk.X)

        # 图片显示区
        frame_img = ttk.Frame(root)
        frame_img.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.label_origin = ttk.Label(frame_img, text="原图")
        self.label_origin.grid(row=0, column=0, sticky="nsew")
        self.label_recon = ttk.Label(frame_img, text="重建图")
        self.label_recon.grid(row=0, column=1, sticky="nsew")

        frame_img.columnconfigure(0, weight=1)
        frame_img.columnconfigure(1, weight=1)
        frame_img.rowconfigure(0, weight=1)

    def on_alg_change(self, event):
        alg = self.alg_var.get()
        if alg == "DCT":
            self.quality_frame.pack(side=tk.LEFT)
        else:
            self.quality_frame.pack_forget()

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        img = Image.open(path).convert("RGB")
        self.img_origin_np = np.array(img)
        self.img_recon_np = None
        self.info_text.delete(1.0, tk.END)
        self.show_thumbnail()

    def show_thumbnail(self):
        max_w, max_h = 520, 520
        # 原图缩略
        img_pil = Image.fromarray(self.img_origin_np)
        img_pil.thumbnail((max_w, max_h))
        self.origin_tk = ImageTk.PhotoImage(img_pil)
        self.label_origin.config(image=self.origin_tk)

        # 重建图
        if self.img_recon_np is not None:
            rec_pil = Image.fromarray(self.img_recon_np)
            rec_pil.thumbnail((max_w, max_h))
            self.recon_tk = ImageTk.PhotoImage(rec_pil)
            self.label_recon.config(image=self.recon_tk)
        else:
            self.label_recon.config(image="")

    def run_compress(self):
        if self.img_origin_np is None:
            messagebox.showwarning("提示", "请先加载图片")
            return
        alg = self.alg_var.get()
        origin = self.img_origin_np
        comp_data = None
        recon = None

        try:
            if alg == "RLE":
                comp = RLECompressor()
                comp_data = comp.compress(origin)
                recon = comp.decompress(comp_data)
            elif alg == "Huffman":
                comp = HuffmanCompressor()
                comp_data = comp.compress(origin)
                recon = comp.decompress(comp_data)
            elif alg == "DCT":
                q = self.qual_var.get()
                comp = DCTCompressor(quality=q)
                comp_data = comp.compress(origin)
                recon = comp.decompress(comp_data)
            else:
                return
        except Exception as e:
            messagebox.showerror("错误", f"运算异常：{e}")
            return

        self.img_recon_np = recon
        # 计算指标
        h_r, w_r = recon.shape[:2]
        orig_crop = origin[:h_r, :w_r, :]
        psnr = calc_psnr(orig_crop, recon)
        cr = calc_compression_ratio(compressed_dict=comp_data, original_img_np=origin)

        info = f"算法：{alg}\n"
        info += f"图像尺寸：{origin.shape[0]} × {origin.shape[1]}\n"
        info += f"PSNR：{psnr:.2f} dB | 估算压缩比：{cr:.3f}\n"
        if alg == "DCT":
            info += f"DCT Quality = {self.qual_var.get()}"

        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, info)
        self.show_thumbnail()

    def save_result(self):
        if self.img_recon_np is None:
            messagebox.showwarning("提示", "没有重建图像")
            return
        out_dir = Path(__file__).parent.parent / "test" / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "gui_output.png"
        Image.fromarray(self.img_recon_np).save(str(out_path))
        messagebox.showinfo("完成", f"已保存到：{out_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCompressGUI(root)
    root.mainloop()
