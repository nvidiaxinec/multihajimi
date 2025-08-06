import gradio as gr
from PIL import Image
import io
import zlib
import base91
import numpy as np
from math import sqrt, floor

# 自定义哈基米字典
CUSTOM_DICT = {
    'AA': '哈', 'BB': '基', 'CC': '米', 'DD': '呀',
    'EE': '啊', 'FF': '西', 'GG': '嘎', 'HH': '摸',
    'II': '库', 'JJ': '奶', 'KK': '龙', 'LL': '绿',
    'MM': '豆', 'NN': '南', 'OO': '北', 'PP': '曼',
    'AABB': '波', 'CCDD': '耶', 'EEFF': '没',
    'GGHH': '路', 'IIJJ': '多', 'KKLL': '哒'
}
# 反转字典用于解码
REVERSE_DICT = {v: k for k, v in CUSTOM_DICT.items()}

# --- 定义最大面积常量 ---
MAX_PIXEL_AREA = 16384
# TARGET_SIZE = int(sqrt(MAX_PIXEL_AREA)) # 不再需要固定的填充目标尺寸

def smart_resize(image, target_area=MAX_PIXEL_AREA):
    """
    智能缩放图像。
    - 如果原始面积 <= target_area：保持原始尺寸，不进行缩放或填充。
    - 如果原始面积 > target_area：等比例缩放，使新面积 <= target_area，不进行填充。
    """
    img = Image.fromarray(image.astype('uint8'), 'RGB')
    original_w, original_h = img.size
    original_area = original_w * original_h

    # --- 根据面积判断是否需要缩放 ---
    if original_area <= target_area:
        # 小图或刚好，不进行任何缩放或填充，直接返回原图
        print(f"Image is small/large enough ({original_w}x{original_h}), processing at original size.")
        # 返回处理后的图像和原始尺寸
        return img, (original_w, original_h) # processed_img 就是原 img
    else:
        # 大图，需要缩放
        # 计算缩放比例，使新面积接近但不超过 target_area
        scale_factor = sqrt(target_area / original_area)
        # 确保新尺寸至少为1
        new_w = max(1, floor(original_w * scale_factor)) # 使用 floor 更符合“小于等于”
        new_h = max(1, floor(original_h * scale_factor))
        # 高质量缩放
        resized_img = img.resize((new_w, new_h), Image.LANCZOS)
        print(f"Image is large ({original_w}x{original_h}), resized to ({new_w}x{new_h}) based on area.")
        # 返回缩放后的图像和原始尺寸
        return resized_img, (original_w, original_h)

# adaptive_compress 函数基本保持不变，但 now receives a PIL Image
def adaptive_compress(image_pil, quality=15):
    """
    根据图像特征自适应压缩
    """
    # image_pil is now a PIL Image object, already resized by smart_resize
    img = image_pil
    w, h = img.size
    # 计算图像复杂度（基于颜色变化）
    np_img = np.array(img)
    diff_x = np.abs(np_img[:, 1:] - np_img[:, :-1]).mean()
    diff_y = np.abs(np_img[1:, :] - np_img[:-1, :]).mean()
    complexity = (diff_x + diff_y) / 2
    # 根据图像特征调整压缩参数
    # --- 调整判断条件以适应可能变化的尺寸逻辑 ---
    # 使用面积判断可能更合理，或者保留原始逻辑（这里保留原始逻辑，但注释说明）
    if w * h < 5000:  # 小图像区域 (这个判断基于处理后的图像尺寸)
        # 小图像使用较高质量压缩
        adjusted_quality = max(quality, 20)
        colors = 256
    elif complexity < 5:  # 简单图像（如截图、文字）
        adjusted_quality = quality + 5
        colors = 64
    elif complexity > 30:  # 复杂图像（如照片）
        adjusted_quality = max(quality - 5, 5)
        colors = 128
    else:  # 中等复杂度
        adjusted_quality = quality
        colors = 128

    # 转换为调色板图像减少颜色数量
    if colors < 256:
        img = img.quantize(colors=colors, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG)
        img = img.convert('RGB') # 转换回 RGB 以便保存

    # 保存为WebP
    buffer = io.BytesIO()
    img.save(buffer, format='WEBP', quality=adjusted_quality, method=6)
    compressed_img = buffer.getvalue()
    return compressed_img, adjusted_quality, colors

def image_to_text(image, quality=15):
    """将图片转换为文本编码"""
    # 记录原始尺寸
    original_size = image.shape[1], image.shape[0] # (宽, 高)
    # 智能缩放 (根据新的面积逻辑)
    processed_img_pil, orig_size = smart_resize(image, target_area=MAX_PIXEL_AREA)
    # 自适应压缩 (传入 PIL Image)
    compressed_data, used_quality, used_colors = adaptive_compress(processed_img_pil, quality)
    # 二次压缩
    zlib_compressed = zlib.compress(compressed_data, level=9)
    # Base91编码
    base91_str = base91.encode(zlib_compressed)
    # 应用自定义字典替换（先替换长模式）
    text_data = base91_str
    for pattern in sorted(CUSTOM_DICT.keys(), key=len, reverse=True):
        text_data = text_data.replace(pattern, CUSTOM_DICT[pattern])
    # 添加原始尺寸信息作为前缀
    size_prefix = f"哈基片:{orig_size[0]}x{orig_size[1]}|"
    # --- 不再添加压缩参数信息 ---
    # param_info = f"质量:{used_quality} 颜色:{used_colors}|"
    # 添加可爱的分隔符
    lines = [text_data[i:i+24] for i in range(0, len(text_data), 24)]
    formatted_text = "\n".join(lines)
    # --- 返回时不包含 param_info ---
    return size_prefix + formatted_text # 只返回尺寸和编码文本

def text_to_image(text_data):
    """将文本编码还原为图片"""
    # --- 增加空输入检查 ---
    if not text_data or not text_data.strip():
        gr.Warning("输入的加密文本为空。")
        return None, "错误: 输入为空"

    # --- 初始化参数信息 ---
    param_info_str = "解码中..."
    orig_w = "未知"
    orig_h = "未知"
    processed_w = "未知"
    processed_h = "未知"
    used_quality = "未知"
    used_colors = "未知"

    # --- 修正点：正确解析包含一个 '|' 的文本 ---
    # 格式应为: "哈基片:宽x高|编码主体"
    clean_text_for_decoding = text_data # 初始化为完整文本
    if text_data.startswith("哈基片:"):
        parts = text_data.split("|", 1) # 只分割一次
        if len(parts) == 2:
            size_info_part = parts[0]
            text_data_body = parts[1]
            # 尝试解析尺寸 "哈基片:宽x高"
            try:
                _, size_str = size_info_part.split(":", 1)
                orig_w_str, orig_h_str = size_str.split("x", 1)
                orig_w = int(orig_w_str)
                orig_h = int(orig_h_str)
                param_info_str = f"原始尺寸: {orig_w}x{orig_h}\n"
            except (ValueError, IndexError) as e:
                 gr.Warning(f"加密文本中的尺寸信息格式不正确: {e}")
                 param_info_str = "尺寸信息解析失败\n"

            # 剩余部分是编码主体
            clean_text_for_decoding = text_data_body
        else: # 有"尺寸:"前缀但没有'|'，认为格式错误
            gr.Warning("加密文本格式错误：发现'尺寸:'前缀但缺少分隔符'|'。")
            param_info_str = "格式错误: 缺少分隔符\n"
    # else: # 没有尺寸前缀，也认为格式可能错误，但仍尝试解码
    #     gr.Warning("加密文本似乎缺少尺寸信息前缀。")
    #     param_info_str = "警告: 缺少尺寸前缀\n"

    # --- 移除换行和空格 (只对编码主体部分) ---
    clean_text = clean_text_for_decoding.replace("\n", "").replace(" ", "")
    
    if not clean_text:
        error_msg = "错误: 处理后的加密文本主体为空。"
        gr.Error(error_msg)
        return None, param_info_str + error_msg

    # --- 应用反向字典替换 ---
    temp_text = clean_text
    for char, pattern in REVERSE_DICT.items():
        temp_text = temp_text.replace(char, pattern)
    
    clean_text = temp_text

    # --- Base91解码 ---
    try:
        zlib_data = base91.decode(clean_text)
    except Exception as e:
        error_msg = f"Base91解码失败: {str(e)}"
        gr.Error(error_msg)
        return None, param_info_str + error_msg

    # --- zlib解压 ---
    try:
        webp_data = zlib.decompress(zlib_data)
    except Exception as e:
        error_msg = f"zlib解压失败: {str(e)}。输入文本可能已损坏或不完整。"
        gr.Error(error_msg) # 这个错误会直接显示给用户
        return None, param_info_str + error_msg

    # --- 重建图像 ---
    try:
        img = Image.open(io.BytesIO(webp_data))
        img = img.convert('RGB')
        processed_w, processed_h = img.size
        final_param_info = param_info_str + f"处理后尺寸: {processed_w}x{processed_h}\n质量: {used_quality}\n颜色: {used_colors}\n解码成功!"
        return np.array(img), final_param_info
    except Exception as e:
        error_msg = f"无法重建图像: {str(e)}"
        gr.Error(error_msg)
        return None, param_info_str + error_msg

# ==================== Gradio 界面定义 ====================
with gr.Blocks(title="哈基米图片编码器", theme=gr.themes.Soft()) as app: # <-- 定义了 'app' 变量
    gr.Markdown("## 🐱 哈基米图片编码器 ")
    gr.Markdown("上传任意尺寸图片，智能处理（大图缩放至面积<16384，小图不填充）并转换为有趣的文字编码！")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="上传图片", type="numpy")
            quality_slider = gr.Slider(1, 30, value=15, label="压缩质量",
                                      info="数值越低压缩率越高，图像质量越低")
            encode_btn = gr.Button("转换为文字", variant="primary")
        with gr.Column():
            # --- 修改点 1: 更改标签 ---
            output_text = gr.Textbox(label="加密文本", lines=10, elem_id="output-textbox",
                                    placeholder="这里将显示图片的加密文本...")
            copy_btn = gr.Button("复制文本")
            # --- 修改点 2: 添加显示解码参数的文本框 ---
            decode_params = gr.Textbox(label="解码参数", interactive=False, visible=True, lines=5) # 初始可见，方便查看
        with gr.Column():
            decoded_image = gr.Image(label="解码预览", interactive=False)
            decode_btn = gr.Button("从文本还原图片", variant="secondary")
    # 添加各种尺寸的示例 (建议使用本地路径)
    # gr.Examples(
    #     examples=[
    #         ["examples/sample1.jpg", 15],
    #         ["examples/small1.jpg", 10],
    #         ["examples/wide1.jpg", 15],
    #         ["examples/tall1.jpg", 20],
    #         ["examples/tiny1.jpg", 15]
    #     ],
    #     inputs=[input_image, quality_slider],
    #     outputs=output_text,
    #     fn=image_to_text,
    #     label="不同尺寸示例（点击试试看）"
    # )
    # 事件处理
    encode_btn.click(
        fn=image_to_text,
        inputs=[input_image, quality_slider],
        outputs=output_text
    )
    # --- 修改点 3: 修复 copy_btn (使用推荐的方法) ---
    copy_btn.click(
        fn=None,
        inputs=output_text, # 将 output_text 的值作为参数 't' 传递给 js 函数
        outputs=None,
        js="(t) => { if (t) { navigator.clipboard.writeText(t); console.log('Text copied!'); } else { console.log('No text to copy'); } }"
        # 添加了简单的检查和日志
    )
    # --- 修改点 4: 修改 decode_btn 的 outputs ---
    decode_btn.click(
        fn=text_to_image,
        inputs=output_text, # 从 "加密文本" 框读取
        outputs=[decoded_image, decode_params] # 输出到 "解码预览" 和 "解码参数" 框
    )
    # 添加处理说明 (更新说明)
    with gr.Accordion("图片处理说明", open=False):
        gr.Markdown(f"""
        ### 智能处理策略 (基于面积, 无强制填充)
        1. **面积判断**：
           - 图片分辨率（宽 x 高）<= {MAX_PIXEL_AREA}：**直接处理，不进行缩放或填充**。
           - 图片分辨率 > {MAX_PIXEL_AREA}：等比例缩放，使缩放后面积 <= {MAX_PIXEL_AREA}，**不进行填充**。
        2. **保持原始比例**：
           - 缩放时严格保持原始宽高比。
           - **不再添加灰色边框填充**。
        3. **自适应压缩**：
           - 小尺寸图片区域：使用更高画质。
           - 简单图像（如文字）：减少颜色数量。
           - 复杂照片：适当降低画质。
        4. **信息存储**：
           - 在编码开头添加 `尺寸:宽x高|` 信息。
           - **压缩参数信息 `质量:Q 颜色:C|` 已从加密文本中移除**。
           - 解码时，参数信息会显示在**独立的“解码参数”框**中。
        """)

# ==================== 启动应用 ====================
# 现在 'app' 变量已经定义，可以启动了
if __name__ == "__main__":
    app.launch()