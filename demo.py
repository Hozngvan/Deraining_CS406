import streamlit as st
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
# from io import BytesIO
# import glob
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

# --- IMPORT ARCHITECTURES ---
from MPRNet import MPRNet 
from PReNet import PReNet
from DRT import DRT_Inference_Wrapper 

# --- CẤU HÌNH ---
st.set_page_config(page_title="Multi-Model Deraining Comparison", layout="wide", page_icon="🌧️")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ĐƯỜNG DẪN DATASET
DATASET_PATH = "./../Test1200/Test1200" 

# DANH SÁCH MODELS
MODEL_CONFIGS = {
    "MPRNet": {
        "path": "./pretrained_models/model_epoch_80.pth", 
        "class": MPRNet,
        "args": {},
        "pad_factor": 8
    },
    "PReNet": {
        "path": "./pretrained_models/prenet_best_final.pth", 
        "class": PReNet,
        "args": {"recurrent_iter": 6, "use_GPU": torch.cuda.is_available()},
        "pad_factor": 8
    },
    "DRT (Transformer)": {
        "path": "./pretrained_models/best_model_from_scratch.pt", 
        "class": DRT_Inference_Wrapper,
        "args": {
            "dim": 96, 
            "patch_size": 1, 
            "l_recursion": 3,   
            "n_RTB": 6,         
            "num_heads": 2,     
            "window_size": 7   
        },
        "pad_factor": 7
    }
}

# --- HÀM LOAD 1 MODEL  ---
def load_single_model(name, config):
    print(f"Loading {name}...")
    model = config["class"](**config["args"])
    weight_path = config["path"]
    
    if not os.path.exists(weight_path):
        return None, f"File not found: {weight_path}"

    try:
        checkpoint = torch.load(weight_path, map_location=DEVICE)
        
        if name == "PReNet":
            if "model_state" in checkpoint:
                model.load_state_dict(checkpoint["model_state"])
            else:
                model.load_state_dict(checkpoint)
        # ==============================================================
        else:
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            new_state_dict = {}
            for k, v in state_dict.items():
                key_name = k[7:] if k.startswith('module.') else k
                new_state_dict[key_name] = v
            
            model.load_state_dict(new_state_dict, strict=False)
            
        model.to(DEVICE)
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)

# --- HÀM LOAD TẤT CẢ MODEL ---
@st.cache_resource
def load_all_models():
    loaded_models = {}
    errors = []
    for name, config in MODEL_CONFIGS.items():
        model, err = load_single_model(name, config)
        if model:
            loaded_models[name] = model
        else:
            errors.append(f"{name}: {err}")
    return loaded_models, errors

# --- HÀM TÍNH METRICS ---
def calculate_metrics(target_pil, output_pil):
    target_np = np.array(target_pil)
    output_np = np.array(output_pil)
    psnr_val = psnr_metric(target_np, output_np, data_range=255)
    ssim_val = ssim_metric(target_np, output_np, channel_axis=2, data_range=255)
    return psnr_val, ssim_val

# --- HÀM XỬ LÝ ẢNH (GIỮ NGUYÊN LOGIC PADDING) ---
def process_image(image_pil, model, model_name):
    config = MODEL_CONFIGS.get(model_name)
    factor = config["pad_factor"] if config else 8

    # Pre-processing
    img = TF.to_tensor(image_pil).to(DEVICE).unsqueeze(0) # BCHW

    # Padding
    h, w = img.shape[2], img.shape[3]
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh, padw = H - h, W - w
    img_padded = F.pad(img, (0, padw, 0, padh), 'reflect')

    # Inference
    with torch.no_grad():
        if "MPRNet" in model_name:
            restored = model(img_padded)[0]
        elif "PReNet" in model_name:
            restored, _ = model(img_padded)
        elif "DRT" in model_name:
            restored = model(img_padded)
        else:
            restored = model(img_padded)
            if isinstance(restored, (list, tuple)): restored = restored[0]

    # Post-processing
    restored = restored[:, :, :h, :w]
    restored = torch.clamp(restored, 0, 1)
    restored = restored.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    return Image.fromarray((restored * 255).astype(np.uint8))

# --- GIAO DIỆN CHÍNH  ---
st.title("Multi-Model Deraining Comparison")

# Load models
with st.spinner("Đang tải các model vào bộ nhớ..."):
    models_dict, load_errors = load_all_models()

if load_errors:
    for err in load_errors:
        st.warning(f"Warning: {err}")

if not models_dict:
    st.error("Không có model nào được tải thành công.")
    st.stop()

# --- TẠO TABS ---
tab1, tab2 = st.tabs(["Đánh giá trên tập Test100", "Upload ảnh ngoài"])

# TAB 1
with tab1:
    st.header("Đánh giá hiệu năng trên tập dữ liệu có ground truth")
    input_dir = os.path.join(DATASET_PATH, "input")
    target_dir = os.path.join(DATASET_PATH, "target")
    
    if not os.path.exists(input_dir) or not os.path.exists(target_dir):
        st.error(f"Không tìm thấy thư mục dataset: `{input_dir}`")
    else:
        image_files = sorted(os.listdir(input_dir))
        image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(image_files) == 0:
            st.warning("Thư mục input trống.")
        else:
            selected_file = st.selectbox("Chọn ảnh để test:", image_files)
            input_path = os.path.join(input_dir, selected_file)
            target_path = os.path.join(target_dir, selected_file)
            
            if os.path.exists(target_path):
                col_in, col_gt = st.columns(2)
                input_img = Image.open(input_path).convert("RGB")
                target_img = Image.open(target_path).convert("RGB")
                
                with col_in:
                    st.image(input_img, caption="Rainy Input", use_container_width=False)
                with col_gt:
                    st.image(target_img, caption="Ground Truth (Target)", use_container_width=False)
                
                if st.button("Chạy Đánh Giá", key="btn_eval"):
                    st.divider()
                    st.subheader("Kết quả so sánh")
                    cols = st.columns(len(models_dict))
                    
                    for idx, (name, model) in enumerate(models_dict.items()):
                        with cols[idx]:
                            with st.spinner(f"Running {name}..."):
                                try:
                                    output_img = process_image(input_img, model, name)
                                    psnr, ssim = calculate_metrics(target_img, output_img)
                                    st.image(output_img, caption=f"Output: {name}", use_container_width=True)
                                    st.success(f"**PSNR:** {psnr:.2f} dB\n\n**SSIM:** {ssim:.4f}")
                                except Exception as e:
                                    st.error(f"Lỗi {name}: {e}")
            else:
                st.error(f"Không tìm thấy file Ground Truth: `{target_path}`")

# TAB 2
with tab2:
    st.header("Chạy thử trên ảnh tải lên")
    uploaded_file = st.file_uploader("Upload ảnh đầu vào...", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        input_image = Image.open(uploaded_file).convert("RGB")
        st.subheader("Ảnh Gốc (Input)")
        st.image(input_image, use_container_width=True)

        if st.button("Chạy So Sánh Tất Cả Model", key="btn_inference"):
            st.divider()
            st.subheader("Kết Quả")
            cols = st.columns(len(models_dict))
            for idx, (name, model) in enumerate(models_dict.items()):
                with cols[idx]:
                    st.info(f"**{name}**")
                    try:
                        output = process_image(input_image, model, name)
                        st.image(output, caption=f"Output: {name}", use_container_width=True)
                    except Exception as e:
                        st.error(f"Lỗi: {e}")

st.sidebar.divider()
st.sidebar.info("Hệ thống hỗ trợ 2 chế độ:\n1. **Benchmark**: So sánh với Ground Truth.\n2. **Inference**: Chạy thực tế.")