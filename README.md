# README.md

```
# 图像压缩实验平台
基于Python实现的图像压缩实验工具，包含 **RLE游程编码、Huffman霍夫曼编码、DCT离散余弦变换(JPEG模拟)**，提供Tkinter图形GUI界面。

## 项目结构
```

image‑lab/
├── app/
│   └── main.py          # Tkinter 图形界面主程序
├── models/
│   ├── rle.py           # RLE 游程编码 无损压缩
│   ├── huffman.py       # Huffman 霍夫曼编码 无损压缩
│   └── dct.py           # DCT 离散余弦变换 + JPEG 量化 有损压缩
├── utils/
│   └── metrics.py       # 评价指标：PSNR、压缩比计算
├── test/
│   ├── input/           # 放置待测试图片
│   └── output/          # 输出重建后的图片
├── .gitignore           # Git 忽略配置
└── README.md

```

## 环境依赖
```bash
pip install numpy pillow scipy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

- `numpy`：矩阵运算、图像数组处理
- `pillow`：图片读写
- `scipy`：DCT 变换（可选，项目内置手写 DCT 矩阵也可脱离 scipy 运行）

## 运行方式

### 方式 1：图形界面 GUI（推荐）

```
python app/main.py
```

GUI 功能：

1. **选择图片**：加载本地 jpg/png/bmp 图像
2. **算法选择**：RLE / Huffman / DCT
3. DCT 模式：拖动 Quality 滑块控制压缩质量（1‑100，数值越大画质越高，压缩率越低）
4. **执行压缩解压**：运算后显示 PSNR、压缩比，左右分栏预览原图与重建图像
5. **保存结果**：重建图像输出至 `test/output/gui_output.png`

### 方式 2：单独运行模块测试

```
# 测试RLE
python models/rle.py

# 测试Huffman
python models/huffman.py

# 测试DCT
python models/dct.py
```

测试脚本会读取 `test/input/mao.jpg`，输出重建图片到 `test/output/`，打印 PSNR 等指标。

## 算法说明

1. **RLE 游程编码（无损）**

> 
> 对连续重复像素做游程记录；**普通照片压缩效果差，适合大段纯色图像**。

2. **Huffman 霍夫曼编码（无损）**

> 
> 根据像素统计概率生成变长编码；照片类图像压缩比一般接近 1。

3. **DCT 离散余弦变换（有损，模拟 JPEG）**

> 
> 图像分块 8×8 DCT 变换 + JPEG 标准量化表；Quality 控制量化强度，会产生块效应。
> 
> 
> - Quality 小：压缩率高，PSNR 低，块效应明显
> - Quality 大：画质好，压缩率下降

## 评价指标

- **PSNR(dB)**：峰值信噪比，数值越高代表重建图像失真越小；无损压缩 PSNR 为`inf`。
- **压缩比**：原始数据大小 / 压缩后数据大小；比值越大压缩效果越好。

## 使用注意

1. 请把测试图片放入 `test/input/`
2. `test/output/` 为输出目录，`.gitignore` 已忽略输出图片，不会提交到版本库
3. RLE、Huffman 属于无损算法，自然照片很难得到高压缩比；DCT 是本项目主要有损压缩实验对象。