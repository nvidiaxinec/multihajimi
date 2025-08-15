import gradio as gr
from PIL import Image
import io
import zlib
import base91
import numpy as np
from math import sqrt, floor
import librosa
import soundfile as sf

# 自定义哈基米字典 (保持不变)
CUSTOM_DICT = {
    'A': '哈', 'B': '蛤', 'C': '基', 'D': '鸡', 'E': '几', 'F': '季', 'G': '集', 'H': '寄',
    'I': '吉', 'J': '棘', 'K': '脊', 'L': '米', 'M': '密', 'N': '咪', 'O': '莫', 'P': '摸',
    'Q': '膜', 'R': '抹', 'S': '漠', 'T': '磨', 'U': '陌', 'V': '寞', 'W': '男', 'X': '南',
    'Y': '喃', 'Z': '夏', 'a': '那', 'b': '哪', 'c': '呐', 'd': '纳', 'e': '没', 'f': '美',
    'g': '每', 'h': '绿', 'i': '路', 'j': '鹿', 'k': '录', 'l': '下', 'm': '豆', 'n': '逗',
    'o': '斗', 'p': '多', 'q': '哆', 'r': '跺', 's': '朵', 't': '啊', 'u': '阿', 'v': '西',
    'w': '系', 'x': '噶', 'y': '嘎', 'z': '呀', '0': '压', '1': '雅', '2': '亚', '3': '丫',
    '4': '库', '5': '哭', '6': '酷', '7': '奶', '8': '乃', '9': '奈', '!': '耐', '#': '龙',
    '$': '咯', '%': '曼', '&': '波', '*': '漫', '(': '啵', ')': '偶', '+': '欧', ',': '吗',
    '"': '马', '.': '嘛', '/': '里', ':': '力', '<': '利', '=': '理', '>': '丽',
    '?': '历', '@': '也', '[': '耶', '~': '大', ']': '哒', '^': '达', '_': '嗒', '`': '不',
    '{': '布', '|': '打', '}': '哇', 'AA': '呵', 'BB': '挖', '9!': 'ccb', 'DD': '带手机', 'EE': '兴奋剂', 'FF': '操逼', 'GG': '一段', ':}': 'wow'
}
# 反转字典用于解码 (保持不变)
REVERSE_DICT = {v: k for k, v in CUSTOM_DICT.items()}

# --- 优化后的常量 (从新版本移植) ---
MAX_PIXEL_AREA = 12000  # 降低最大面积以获得更好的压缩

# --- 音频处理常量 ---
AUDIO_SAMPLE_RATE = 22050  # 用于重采样的目标采样率
AUDIO_N_FFT = 2048
AUDIO_HOP_LENGTH = 512
AUDIO_MAX_DURATION = 10.0 # 最大处理时长 (秒)，防止过大文件

# --- 优化后的图片处理函数 (从新版本移植) ---

def smart_resize(image, target_area=MAX_PIXEL_AREA):
    """智能缩放图像 (优化版本)"""
    # 处理不同的数据类型
    if image.dtype != np.uint8:
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
        else:
            image = np.clip((image / np.max(image)) * 255, 0, 255).astype(np.uint8)
            
    img = Image.fromarray(image, 'RGB')
    original_w, original_h = img.size
    original_area = original_w * original_h
    if original_area <= target_area:
        print(f"图片尺寸合适 ({original_w}x{original_h})，使用原始尺寸")
        return img, (original_w, original_h)
    else:
        # 添加0.95的安全系数确保不超过限制
        scale_factor = sqrt(target_area / original_area) * 0.95
        new_w = max(1, floor(original_w * scale_factor))
        new_h = max(1, floor(original_h * scale_factor))
        resized_img = img.resize((new_w, new_h), Image.LANCZOS)
        print(f"图片从 ({original_w}x{original_h}) 缩放为 ({new_w}x{new_h})")
        return resized_img, (original_w, original_h)

def rle_compress_colors(image_pil):
    """对纯色区域进行RLE压缩预处理 (新增优化)"""
    np_img = np.array(image_pil)
    diff_threshold = 10
    height, width, channels = np_img.shape
    for y in range(height):
        for x in range(width-1):
            if np.abs(np_img[y, x] - np_img[y, x+1]).max() < diff_threshold:
                np_img[y, x+1] = np_img[y, x]
    return Image.fromarray(np_img)

def adaptive_compress(image_pil, quality=15):
    """优化的自适应压缩 (增强版本)"""
    # 先进行RLE预处理
    img = rle_compress_colors(image_pil)
    w, h = img.size
    np_img = np.array(img)
    
    # 计算图像复杂度
    diff_x = np.abs(np_img[:, 1:] - np_img[:, :-1]).mean()
    diff_y = np.abs(np_img[1:, :] - np_img[:-1, :]).mean()
    complexity = (diff_x + diff_y) / 2

    # 更精细的颜色量化策略
    if w * h < 5000:
        colors = 128
    elif complexity < 5:
        colors = 32  # 简单图像使用更少颜色
    elif complexity < 15:
        colors = 64
    else:
        colors = 96  # 复杂图像适当减少颜色

    # 颜色量化
    if colors < 256:
        img = img.quantize(colors=colors, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG)
        img = img.convert('RGB')

    buffer = io.BytesIO()
    try:
        # 优先使用WEBP格式，添加更多优化参数
        img.save(buffer, format='WEBP', quality=quality, method=6, optimize=True, lossless=False)
    except Exception as e:
        print(f"WEBP编码失败，使用JPEG: {e}")
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality, optimize=True)
    
    compressed_img = buffer.getvalue()
    return compressed_img, quality, colors

def image_to_text(image, quality=15):
    """将图片转换为文本编码 (使用优化的处理函数)"""
    if image is None:
         return "错误: 未提供图片", ""
    try:
        # 使用优化后的处理函数
        processed_img_pil, orig_size = smart_resize(image, target_area=MAX_PIXEL_AREA)
        compressed_data, used_quality, used_colors = adaptive_compress(processed_img_pil, quality)
        
        # 保持原有的编码流程
        zlib_compressed = zlib.compress(compressed_data, level=9)
        base91_str = base91.encode(zlib_compressed)

        text_data = base91_str
        for pattern in sorted(CUSTOM_DICT.keys(), key=len, reverse=True):
            text_data = text_data.replace(pattern, CUSTOM_DICT[pattern])

        size_prefix = f"哈基片:{orig_size[0]}x{orig_size[1]}|"
        
        # 调整每行字符数以获得更好的显示效果
        lines = [text_data[i:i+40] for i in range(0, len(text_data), 40)]
        formatted_text = "\n".join(lines)
        
        # 返回详细的压缩信息
        param_info = f"压缩质量: {used_quality}\n颜色数: {used_colors}\nWebP大小: {len(compressed_data)} 字节\nZlib压缩后: {len(zlib_compressed)} 字节\n压缩比: {len(compressed_data)/len(zlib_compressed):.2f}"
        return size_prefix + formatted_text, param_info
        
    except Exception as e:
        gr.Error(f"图片编码失败: {e}")
        return f"错误: {e}", ""

def text_to_image(text_data):
    """将文本编码还原为图片 (保持原有逻辑)"""
    if not text_data or not text_data.strip():
        gr.Warning("输入的加密文本为空。")
        return None, "错误: 输入为空"

    param_info_str = "解码中..."
    orig_w = "未知"
    orig_h = "未知"
    processed_w = "未知"
    processed_h = "未知"

    clean_text_for_decoding = text_data
    if text_data.startswith("哈基片:"):
        parts = text_data.split("|", 1)
        if len(parts) == 2:
            size_info_part = parts[0]
            text_data_body = parts[1]
            try:
                _, size_str = size_info_part.split(":", 1)
                orig_w_str, orig_h_str = size_str.split("x", 1)
                orig_w = int(orig_w_str)
                orig_h = int(orig_h_str)
                param_info_str = f"原始尺寸: {orig_w}x{orig_h}\n"
            except (ValueError, IndexError) as e:
                 gr.Warning(f"加密文本中的尺寸信息格式不正确: {e}")
                 param_info_str = "尺寸信息解析失败\n"
            clean_text_for_decoding = text_data_body
        else:
            gr.Warning("加密文本格式错误：发现'哈基片:'前缀但缺少分隔符'|'。")
            param_info_str = "格式错误: 缺少分隔符\n"

    clean_text = clean_text_for_decoding.replace("\n", "").replace(" ", "")
    if not clean_text:
        error_msg = "错误: 处理后的加密文本主体为空。"
        gr.Error(error_msg)
        return None, param_info_str + error_msg

    temp_text = clean_text
    for char, pattern in REVERSE_DICT.items():
        temp_text = temp_text.replace(char, pattern)
    clean_text = temp_text

    try:
        zlib_data = base91.decode(clean_text)
    except Exception as e:
        error_msg = f"Base91解码失败: {str(e)}"
        gr.Error(error_msg)
        return None, param_info_str + error_msg

    try:
        webp_data = zlib.decompress(zlib_data)
    except Exception as e:
        error_msg = f"zlib解压失败: {str(e)}。输入文本可能已损坏或不完整。"
        gr.Error(error_msg)
        return None, param_info_str + error_msg

    try:
        img = Image.open(io.BytesIO(webp_data))
        img = img.convert('RGB')
        processed_w, processed_h = img.size
        final_param_info = param_info_str + f"处理后尺寸: {processed_w}x{processed_h}\n解码成功!"
        return np.array(img), final_param_info
    except Exception as e:
        error_msg = f"无法重建图像: {str(e)}"
        gr.Error(error_msg)
        return None, param_info_str + error_msg

# --- 音频处理函数 (保持原有逻辑) ---

def audio_to_text(audio_tuple, quality=15):
    """将音频转换为文本编码"""
    if audio_tuple is None:
        return "错误: 未提供音频", ""
    try:
        # audio_tuple 是 (sample_rate, audio_data)
        orig_sr, audio_data = audio_tuple
        # 确保是单声道
        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0] # 取第一个声道

        # 限制时长
        max_samples = int(AUDIO_MAX_DURATION * orig_sr)
        if len(audio_data) > max_samples:
             gr.Info(f"音频过长，仅处理前 {AUDIO_MAX_DURATION} 秒。")
             audio_data = audio_data[:max_samples]

        # 重采样
        audio_float = audio_data.astype(np.float32)
        if np.max(np.abs(audio_data)) > 0:
            audio_float = audio_float / np.max(np.abs(audio_data))
        audio_resampled = librosa.resample(y=audio_float, orig_sr=orig_sr, target_sr=AUDIO_SAMPLE_RATE)

        # 转换为频谱图 (例如 Mel 频谱)
        mel_spec = librosa.feature.melspectrogram(y=audio_resampled, sr=AUDIO_SAMPLE_RATE, n_fft=AUDIO_N_FFT, hop_length=AUDIO_HOP_LENGTH)
        # 转换为对数幅度
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # 序列化为字节
        spec_bytes = log_mel_spec.tobytes()

        # 压缩
        zlib_compressed = zlib.compress(spec_bytes, level=9)
        # Base91编码
        base91_str = base91.encode(zlib_compressed)

        # 应用自定义字典替换
        text_data = base91_str
        for pattern in sorted(CUSTOM_DICT.keys(), key=len, reverse=True):
            text_data = text_data.replace(pattern, CUSTOM_DICT[pattern])

        # 添加时长信息前缀
        duration = len(audio_resampled) / AUDIO_SAMPLE_RATE
        duration_prefix = f"大狗叫:{duration:.2f}s|"

        # 格式化输出
        lines = [text_data[i:i+24] for i in range(0, len(text_data), 24)]
        formatted_text = "\n".join(lines)

        param_info = f"音频时长: {duration:.2f}s\n采样率: {AUDIO_SAMPLE_RATE}Hz\nFFT大小: {AUDIO_N_FFT}\nHop长度: {AUDIO_HOP_LENGTH}"
        return duration_prefix + formatted_text, param_info
    except Exception as e:
        gr.Error(f"音频编码失败: {e}")
        return f"错误: {e}", ""

def text_to_audio(text_data):
    """将文本编码还原为音频 (简化版，仅演示流程)"""
    if not text_data or not text_data.strip():
        gr.Warning("输入的加密文本为空。")
        return None, "错误: 输入为空"

    param_info_str = "解码中..."
    duration = "未知"

    clean_text_for_decoding = text_data
    if text_data.startswith("大狗叫:"):
        parts = text_data.split("|", 1)
        if len(parts) == 2:
            size_info_part = parts[0]
            text_data_body = parts[1]
            try:
                _, duration_str = size_info_part.split(":", 1)
                duration = duration_str.replace("s", "") # 移除 's'
                duration = float(duration)
                param_info_str = f"音频时长: {duration:.2f}s\n"
            except (ValueError, IndexError) as e:
                 gr.Warning(f"加密文本中的时长信息格式不正确: {e}")
                 param_info_str = "时长信息解析失败\n"
            clean_text_for_decoding = text_data_body
        else:
            gr.Warning("加密文本格式错误：发现'大狗叫:'前缀但缺少分隔符'|'。")
            param_info_str = "格式错误: 缺少分隔符\n"

    clean_text = clean_text_for_decoding.replace("\n", "").replace(" ", "")
    if not clean_text:
        error_msg = "错误: 处理后的加密文本主体为空。"
        gr.Error(error_msg)
        return None, param_info_str + error_msg

    temp_text = clean_text
    for char, pattern in REVERSE_DICT.items():
        temp_text = temp_text.replace(char, pattern)
    clean_text = temp_text

    try:
        zlib_data = base91.decode(clean_text)
    except Exception as e:
        error_msg = f"Base91解码失败: {str(e)}"
        gr.Error(error_msg)
        return None, param_info_str + error_msg

    try:
        spec_bytes = zlib.decompress(zlib_data)
    except Exception as e:
        error_msg = f"zlib解压失败: {str(e)}。输入文本可能已损坏或不完整。"
        gr.Error(error_msg)
        return None, param_info_str + error_msg

    try:
        # 从字节重建频谱图形状 (这里需要知道原始形状，简化处理)
        # 实际应用中可能需要在编码时存储频谱图的形状信息
        # 这里假设一个典型的形状 (例如 128 频段, 44 个时间帧)
        # 注意：这会导致解码后的音频与原始音频不完全一致，仅用于演示
        spec_shape = (128, 44) # 示例形状
        log_mel_spec_flat = np.frombuffer(spec_bytes, dtype=np.float32)
        if log_mel_spec_flat.size != np.prod(spec_shape):
             # 尝试根据字节数推断 (可能不准确)
             expected_size = len(spec_bytes) // 4 # float32 is 4 bytes
             spec_shape = (128, expected_size // 128) # 简单估算
             if spec_shape[0] * spec_shape[1] * 4 != len(spec_bytes):
                  raise ValueError("无法推断频谱图形状")

        log_mel_spec = log_mel_spec_flat.reshape(spec_shape)

        # 转换回幅度谱
        mel_spec = librosa.db_to_power(log_mel_spec)

        # Griffin-Lim 算法重建音频
        audio_reconstructed = librosa.feature.inverse.mel_to_audio(mel_spec, sr=AUDIO_SAMPLE_RATE, n_fft=AUDIO_N_FFT, hop_length=AUDIO_HOP_LENGTH)

        # 转换为 Gradio 需要的格式 (采样率, 音频数据)
        # Gradio 期望 int16 类型的数据
        audio_int16 = (audio_reconstructed * 32767).astype(np.int16)

        final_param_info = param_info_str + f"采样率: {AUDIO_SAMPLE_RATE}Hz\n解码成功 (注意: 为演示，可能与原音频不同)!"
        return (AUDIO_SAMPLE_RATE, audio_int16), final_param_info
    except Exception as e:
        error_msg = f"无法重建音频: {str(e)}"
        gr.Error(error_msg)
        return None, param_info_str + error_msg

# --- 智能解码函数 ---
def smart_decode(text_data):
    """根据前缀智能判断并调用解码函数"""
    if not text_data or not text_data.strip():
        return None, "错误: 输入为空", None

    clean_text = text_data.strip()
    if clean_text.startswith("哈基片:"):
        img, param_info = text_to_image(clean_text)
        return img, param_info, None # 返回图片, 参数信息, 音频(无)
    elif clean_text.startswith("大狗叫:"):
        audio, param_info = text_to_audio(clean_text)
        return None, param_info, audio # 返回图片(无), 参数信息, 音频
    else:
        # 如果没有明确前缀，可以尝试判断或报错
        # 这里简单报错
        gr.Warning("无法识别文本类型，请确保文本以 '哈基片:' 或 '大狗叫:' 开头。")
        return None, "错误: 无法识别的文本类型", None

# --- 字数统计函数 ---
def count_chars(text):
    """计算文本中的字符数（不包括换行符）"""
    if not text:
        return 0
    return len(text.replace('\n', ''))

# ==================== Gradio 界面定义 ====================
with gr.Blocks(title="哈基米图片/音频编码器 - V 0.2.0", theme=gr.themes.Soft()) as app:
    gr.Markdown("## 🐱 哈基米图片/音频编码器 - V 0.2.0")
    gr.Markdown("上传图片或音频，智能处理并转换为有趣的文字编码！**图片压缩已优化，文字数量更少！**")

    # 功能选择
    mode_selector = gr.Radio(
        choices=["图片编码", "音频编码"],
        value="图片编码",
        label="选择功能"
    )

    with gr.Row():
        with gr.Column():
            # 图片输入组件
            input_image = gr.Image(
                label="上传图片", 
                type="numpy", 
                visible=True
            )
            
            # 音频输入组件 - 修复配置
            input_audio = gr.Audio(
                label="上传/录制音频",
                sources=["upload", "microphone"],  # 启用文件上传和麦克风录制
                type="numpy",  # 返回 (sample_rate, numpy_array) 格式
                visible=False
            )

            quality_slider = gr.Slider(1, 30, value=15, label="压缩质量",
                                      info="数值越低压缩率越高，图像质量越低")
            encode_btn = gr.Button("转换为文字", variant="primary")
            
        with gr.Column():
            # 字数统计显示
            char_count_display = gr.Number(label="字数", interactive=False, value=0)

            # 压缩信息显示
            compression_info = gr.Textbox(
                label="压缩信息", 
                interactive=False, 
                lines=4,
                placeholder="压缩统计信息将显示在这里..."
            )

            output_text = gr.Textbox(label="加密文本", lines=10, elem_id="output-textbox",
                                    placeholder="这里将显示图片/音频的加密文本...")
            copy_btn = gr.Button("📋 复制文本")

            # 解码参数显示
            decode_params = gr.Textbox(label="解码参数", interactive=False, visible=True, lines=5)
            
        with gr.Column():
            # 图片输出组件
            decoded_image = gr.Image(
                label="解码预览 (图片)", 
                interactive=False, 
                visible=True
            )
            
            # 音频输出组件 - 修复配置
            decoded_audio = gr.Audio(
                label="解码预览 (音频)",
                interactive=False,  # 只用于播放
                visible=False
            )
            
            decode_btn = gr.Button("🔄 从文本还原", variant="secondary")

    # --- 事件处理 ---

    # 功能切换逻辑 - 简化和修复
    def update_interface(mode):
        if mode == "图片编码":
            return [
                gr.update(visible=True),   # input_image
                gr.update(visible=False),  # input_audio  
                gr.update(visible=True),   # decoded_image
                gr.update(visible=False),  # decoded_audio
            ]
        else: # "音频编码"
            return [
                gr.update(visible=False),  # input_image
                gr.update(visible=True),   # input_audio
                gr.update(visible=False),  # decoded_image  
                gr.update(visible=True),   # decoded_audio
            ]

    mode_selector.change(
        fn=update_interface,
        inputs=mode_selector,
        outputs=[input_image, input_audio, decoded_image, decoded_audio]
    )

    # 字数统计逻辑
    output_text.change(
        fn=count_chars,
        inputs=output_text,
        outputs=char_count_display
    )

    # 编码逻辑 - 修复输入处理
    def encode_wrapper(mode, image_input, audio_input, quality):
        if mode == "图片编码":
            result_text, param_info = image_to_text(image_input, quality)
            compression_info_text = f"图片编码完成\n{param_info}"
            return result_text, param_info, compression_info_text
        elif mode == "音频编码":
            if audio_input is None:
                return "错误: 未提供音频", "", "错误: 未提供音频"
            result_text, param_info = audio_to_text(audio_input, quality)
            compression_info_text = f"音频编码完成\n{param_info}"
            return result_text, param_info, compression_info_text
        else:
            return "错误: 未知模式", "", "错误: 未知模式"

    encode_btn.click(
        fn=encode_wrapper,
        inputs=[mode_selector, input_image, input_audio, quality_slider],
        outputs=[output_text, decode_params, compression_info]
    )

    # 复制按钮逻辑 (保持不变)
    copy_btn.click(
        fn=None,
        inputs=output_text,
        outputs=None,
        js="(t) => { if (t) { navigator.clipboard.writeText(t); console.log('Text copied to clipboard!'); alert('文本已复制到剪贴板!'); } else { alert('没有文本可复制!'); } }"
    )

    # 解码逻辑 (智能解码)
    def decode_wrapper(text_input):
        try:
            img, param_info, audio = smart_decode(text_input)
            return img, param_info, audio
        except Exception as e:
            error_msg = f"解码过程发生未预期错误: {e}"
            gr.Error(error_msg)
            return None, error_msg, None

    decode_btn.click(
        fn=decode_wrapper,
        inputs=output_text,
        outputs=[decoded_image, decode_params, decoded_audio]
    )

    # 处理说明
    with gr.Accordion("优化说明", open=False):
        gr.Markdown(f"""
        ### 🚀 本版本的优化内容：
        
        **图片处理优化：**
        1. **降低最大面积**: 从 16384 降至 {MAX_PIXEL_AREA}，显著减少文字数量
        2. **RLE预处理**: 对相似颜色区域进行预处理，提高压缩效率
        3. **智能颜色量化**: 根据图像复杂度动态调整颜色数量
        4. **优化WEBP参数**: 使用更好的压缩参数，减小文件体积
        5. **更长的文本行**: 每行40字符（原来24），减少换行符数量
        6. **详细压缩信息**: 显示完整的压缩统计数据
       
        
        **预期效果：**
        - 同等质量图片的文字数量减少约 20-40%
        - 保持100%的编码/解码兼容性
        
        **使用建议：**
        - 简单图片（如截图、文字图片）可以使用较低质量设置（5-10）
        - 复杂照片建议使用中等质量设置（15-25）
        - 如需最小文字数量，可尝试质量1-5
        - github：https://github.com/nvidiaxinec/multihajimi
        """)

# ==================== 启动应用 ====================
if __name__ == "__main__":
    app.launch(inbrowser=True)