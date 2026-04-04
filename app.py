import sys, os, io, re, time, warnings, csv
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Project-specific imports
from config import N_CLASSES, CLASS_NAMES, COLOR_PALETTE
from model import OffRoadSegNet

# ────────────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OffRoad Segmentation AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────
# Global CSS
# ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}

.stApp{background:linear-gradient(135deg,#07101d 0%,#0b1828 60%,#091420 100%);color:#e2e8f0;}

section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0c1520 0%,#0f1d2c 100%);
    border-right:1px solid rgba(56,189,248,.15);
}

/* tabs */
div[data-testid="stTabs"] button{
    font-family:'Inter',sans-serif;font-weight:600;font-size:15px;
    color:#64748b;border:none;background:transparent;
    padding:10px 22px;border-radius:10px 10px 0 0;transition:all .2s;
}
div[data-testid="stTabs"] button[aria-selected="true"]{
    color:#38bdf8;border-bottom:2px solid #38bdf8;background:rgba(56,189,248,.07);
}

/* cards */
.stat-card{
    background:linear-gradient(135deg,rgba(13,23,38,.92),rgba(18,33,55,.92));
    border:1px solid rgba(56,189,248,.2);border-radius:16px;
    padding:18px 22px;text-align:center;
    backdrop-filter:blur(10px);transition:transform .2s,border-color .2s;
    margin-bottom:10px;
}
.stat-card:hover{transform:translateY(-3px);border-color:rgba(56,189,248,.5);}
.stat-card .lbl{font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;margin-bottom:5px;}
.stat-card .val{font-size:26px;font-weight:700;color:#38bdf8;}
.stat-card .sub{font-size:11px;color:#536175;margin-top:3px;}

/* hero */
.hero{
    background:linear-gradient(135deg,rgba(14,165,233,.12),rgba(99,102,241,.12));
    border:1px solid rgba(56,189,248,.22);border-radius:20px;
    padding:28px 36px;margin-bottom:24px;text-align:center;
}
.hero h1{
    font-size:2.4rem;font-weight:800;margin:0;
    background:linear-gradient(90deg,#38bdf8,#818cf8,#34d399);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.hero p{color:#94a3b8;font-size:1rem;margin-top:6px;}

/* section title */
.sec-title{
    font-size:.95rem;font-weight:700;color:#38bdf8;letter-spacing:1px;
    text-transform:uppercase;border-bottom:2px solid rgba(56,189,248,.25);
    padding-bottom:6px;margin-bottom:14px;
}

/* upload box */
.upload-box{
    border:2px dashed rgba(56,189,248,.35);border-radius:14px;
    padding:32px;text-align:center;background:rgba(56,189,248,.03);
}

/* per class bar */
.cls-row{display:flex;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid rgba(255,255,255,.05);}
.cls-swatch{width:13px;height:13px;border-radius:3px;flex-shrink:0;border:1px solid rgba(255,255,255,.18);}
.cls-name{flex:1;font-size:12px;color:#e2e8f0;min-width:0;overflow:hidden;text-overflow:ellipsis;}
.cls-bar-bg{flex:2;background:rgba(255,255,255,.07);border-radius:4px;height:5px;}
.cls-pct{font-size:12px;color:#38bdf8;font-weight:600;flex-shrink:0;min-width:42px;text-align:right;}

/* perf metric big card */
.perf-card{
    background:linear-gradient(135deg,rgba(13,23,38,.95),rgba(18,33,55,.95));
    border:1px solid rgba(56,189,248,.18);border-radius:16px;
    padding:22px;text-align:center;margin-bottom:12px;
}
.perf-card .ttl{font-size:11px;color:#94a3b8;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;}
.perf-card .big{font-size:36px;font-weight:800;}
.perf-card .sml{font-size:12px;color:#64748b;margin-top:4px;}
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────
# Paths & device
# ────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH   = os.path.join(BASE_DIR, "models", "offroadnet_best.pth")
LOG_PATH    = os.path.join(BASE_DIR, "train_stats", "evaluation_metrics.txt")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_DISPLAY = 15
IMG_EXTS    = {".jpg",".jpeg",".png",".bmp",".tiff",".tif",".webp"}

# ────────────────────────────────────────────────────────
# Cached loaders
# ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_instance():
    model = OffRoadSegNet(N_CLASSES).to(DEVICE)
    if os.path.exists(CKPT_PATH):
        try:
            ckpt  = torch.load(CKPT_PATH, map_location=DEVICE)
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state)
        except Exception:
            pass
    model.eval()
    return model

# ────────────────────────────────────────────────────────
# Inference helpers
# ────────────────────────────────────────────────────────
def preprocess_image(image_np, target_size=(448, 256)):
    image_resized = cv2.resize(image_np, target_size, interpolation=cv2.INTER_LINEAR)
    img_tensor = torch.from_numpy(image_resized).float().permute(2, 0, 1) / 255.0
    return img_tensor.unsqueeze(0).to(DEVICE), image_resized

def colorize(mask, colors):
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(N_CLASSES):
        color[mask == c] = colors[c]
    return color

def blend(rgb, color_mask, alpha=0.45):
    return (rgb.astype(np.float32)*(1-alpha) +
            color_mask.astype(np.float32)*alpha).clip(0,255).astype(np.uint8)

def run_one(rgb: np.ndarray, model):
    t0 = time.perf_counter()
    img_tensor, right_sized_img = preprocess_image(rgb)
    with torch.no_grad():
        with torch.amp.autocast(DEVICE.type, enabled=(DEVICE.type=="cuda")):
            out = model(img_tensor)
            if isinstance(out, tuple): 
                out = out[0]
            # out might be [1, N_CLASSES, H, W]
            probs = F.softmax(out, dim=1)
            conf = probs.max(dim=1).values[0].cpu().numpy()
            pred = torch.argmax(out, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            
    ms = (time.perf_counter()-t0)*1000
    return pred, conf, ms, right_sized_img

def class_stats(mask, n_classes):
    total = mask.size
    pcts  = [round(int(np.sum(mask==i))/total*100,2) for i in range(n_classes)]
    return pcts

# ────────────────────────────────────────────────────────
# Chart helpers
# ────────────────────────────────────────────────────────
def make_pie(names, pcts, colors_rgb):
    hex_ = [f"rgb({r},{g},{b})" for r,g,b in colors_rgb]
    fig  = go.Figure(go.Pie(
        labels=names, values=pcts, hole=.55,
        marker=dict(colors=hex_,line=dict(color="#07101d",width=2)),
        textinfo="label+percent",
        textfont=dict(size=10,color="#e2e8f0"),
        hovertemplate="<b>%{label}</b><br>%{value:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter",color="#e2e8f0"),showlegend=False,
        margin=dict(t=20,b=20,l=10,r=10),height=320,
        annotations=[dict(text="Coverage",x=.5,y=.5,font_size=13,
                          showarrow=False,font_color="#94a3b8")]
    )
    return fig

def make_conf_heatmap(conf):
    fig = go.Figure(go.Heatmap(
        z=conf,colorscale=[[0,"#1e0a40"],[.35,"#6d28d9"],[.65,"#0ea5e9"],
                          [.85,"#34d399"],[1,"#fbbf24"]],
        zmin=0,zmax=1,showscale=True,
        colorbar=dict(title=dict(text="Conf",side="right",font=dict(color="#94a3b8",size=10)),
                      tickfont=dict(color="#94a3b8"),bgcolor="rgba(0,0,0,0)",thickness=10),
        hovertemplate="Conf: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=6,b=6,l=6,r=70),height=240,
        xaxis=dict(visible=False),yaxis=dict(visible=False,autorange="reversed"),
    )
    return fig

def cls_breakdown_html(names, pcts, colors):
    rows = ""
    for n,p,c in sorted(zip(names,pcts,colors),key=lambda x:-x[1]):
        if p==0: continue
        r,g,b = c
        bw = min(int(p*2),100)
        rows += f"""<div class="cls-row">
            <div class="cls-swatch" style="background:rgb({r},{g},{b});"></div>
            <div class="cls-name">{n}</div>
            <div class="cls-bar-bg"><div style="width:{bw}%;background:rgb({r},{g},{b});height:5px;border-radius:4px;"></div></div>
            <div class="cls-pct">{p:.1f}%</div>
        </div>"""
    return f"<div style='max-height:300px;overflow-y:auto;'>{rows}</div>"

# ────────────────────────────────────────────────────────
# Log parser
# ────────────────────────────────────────────────────────
def parse_training_log(log_path: str) -> dict:
    """Parse evaluation_metrics.txt format."""
    if not os.path.exists(log_path):
        return {}

    epochs, train_loss, val_iou, lr_vals = [], [], [], []
    best_iou, max_epoch = 0.0, 0
    reading_table = False

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if "Best Val IoU" in line:
                best_iou = float(line.split(":")[1].strip())
            elif "Total Epochs" in line:
                max_epoch = int(line.split(":")[1].strip())
            elif "Ep" in line and "Loss" in line:
                reading_table = True
                continue
            elif "-------" in line:
                continue
            elif reading_table:
                parts = line.split()
                if len(parts) >= 4:
                    epochs.append(int(parts[0]))
                    train_loss.append(float(parts[1]))
                    val_iou.append(float(parts[2]))
                    lr_vals.append(float(parts[3]))

    return {
        "best_iou": best_iou,
        "max_epoch": max_epoch,
        "epochs": epochs,
        "train_loss": train_loss,
        "miou": val_iou,
    }

# ────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:14px 0 6px;'>
            <div style='font-size:2rem;'>🌿</div>
            <div style='font-size:1.05rem;font-weight:700;color:#38bdf8;'>OffRoad Seg AI</div>
            <div style='font-size:.72rem;color:#64748b;margin-top:2px;'>Semantic Scene Segmentation</div>
        </div>
        <hr style='border-color:rgba(56,189,248,.12);margin:10px 0;'>
        """, unsafe_allow_html=True)

        dev_str = f"{'CUDA 🚀' if DEVICE.type=='cuda' else 'CPU 🖥️'}"
        cuda_warning = "" if torch.cuda.is_available() else "<div style='color:#f87171;font-size:11px;margin-bottom:8px;'>⚠️ PyTorch installed without CUDA support</div>"
        ckpt_ok = os.path.exists(CKPT_PATH)
        
        # Count params
        model = load_model_instance()
        params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        
        st.markdown(f"""
        <div class='stat-card'>
            <div class='lbl'>Architecture</div>
            <div class='val' style='font-size:15px;'>Custom Efficient UNet</div>
            <div class='sub'>OffRoadSegNet ({params:.2f}M)</div>
        </div>
        {cuda_warning}
        <div class='stat-card'>
            <div class='lbl'>Device</div>
            <div class='val' style='font-size:17px;'>{dev_str}</div>
        </div>
        <div class='stat-card'>
            <div class='lbl'>Best Model</div>
            <div class='val' style='font-size:15px;color:{"#34d399" if ckpt_ok else "#f87171"};'>
                {'✅ Loaded' if ckpt_ok else '⚠️ Missing'}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='sec-title' style='margin-top:14px;'>🎨 Class Legend</div>",
                    unsafe_allow_html=True)
        legend = ""
        for name, color in zip(CLASS_NAMES, COLOR_PALETTE):
            r, g, b = color
            legend += f"""<div style='display:flex;align-items:center;gap:7px;padding:3px 0;'>
                <div style='width:13px;height:13px;border-radius:3px;background:rgb({r},{g},{b});
                            border:1px solid rgba(255,255,255,.18);'></div>
                <span style='font-size:12px;color:#cbd5e1;'>{name}</span></div>"""
        st.markdown(legend, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────
# TAB 1 – SEGMENTATION
# ────────────────────────────────────────────────────────
def tab_segmentation(model):
    st.markdown("""
    <div class='hero'>
        <h1>🌿 OffRoad Terrain Segmentation</h1>
        <p>Upload one or multiple off-road images — the model identifies every terrain class with pixel-level precision</p>
    </div>""", unsafe_allow_html=True)

    # ── Upload section ──────────────────────────────────
    st.markdown("<div class='sec-title'>📁 Upload Images</div>", unsafe_allow_html=True)

    mode = st.radio(
        "Select upload mode:",
        ["📂 Select Files (one or more)", "📁 Folder Path (type directory)"],
        horizontal=True, label_visibility="collapsed",
    )

    uploaded_images = []  # list of (name, np.ndarray rgb)

    if mode.startswith("📂"):
        files = st.file_uploader(
            "Drag & drop images or click to browse",
            type=["jpg","jpeg","png","bmp","tiff","tif","webp"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if files:
            for f in files:
                try:
                    arr = np.array(Image.open(f).convert("RGB"))
                    uploaded_images.append((f.name, arr))
                except Exception:
                    st.warning(f"Could not load {f.name}")
    else:  # folder path
        c1, c2 = st.columns([1, 4])
        with c1:
            if st.button("📂 Browse Folder...", use_container_width=True):
                try:
                    import tkinter as tk
                    from tkinter import filedialog
                    root = tk.Tk()
                    root.withdraw()
                    root.attributes('-topmost', True)
                    sel_dir = filedialog.askdirectory(master=root, title="Select Image Folder")
                    try: root.destroy()
                    except: pass
                    if sel_dir:
                        st.session_state["sel_folder"] = sel_dir
                        st.rerun()
                except Exception as e:
                    st.error(f"Folder browser dialog not supported in this environment.")

        with c2:
            folder_input = st.text_input(
                "Enter full folder path:",
                value=st.session_state.get("sel_folder", ""),
                placeholder=r"e.g.  C:\My Dataset\test_images",
                label_visibility="collapsed",
            )
            if folder_input != st.session_state.get("sel_folder", ""):
                st.session_state["sel_folder"] = folder_input

        folder = st.session_state.get("sel_folder", "")
        if folder:
            folder = folder.strip().strip('"').strip("'")
            if os.path.isdir(folder):
                try:
                    paths = [os.path.join(folder, f) for f in os.listdir(folder)
                             if os.path.splitext(f)[1].lower() in IMG_EXTS]
                except Exception:
                    paths = []
                if paths:
                    st.info(f"Found **{len(paths)}** image(s) in folder.")
                    for p in paths:
                        try:
                            arr = np.array(Image.open(p).convert("RGB"))
                            uploaded_images.append((os.path.basename(p), arr))
                        except Exception:
                            st.warning(f"Skipped: {os.path.basename(p)}")
                else:
                    st.warning("No supported images found in that folder.")
            else:
                st.error("Directory not found — please check the path.")

    if not uploaded_images:
        st.markdown("""
        <div class='upload-box'>
            <div style='font-size:2.8rem;'>🖼️</div>
            <div style='font-size:1.05rem;color:#94a3b8;margin-top:10px;'>Waiting for images…</div>
            <div style='font-size:.83rem;color:#475569;margin-top:4px;'>Supports JPG · PNG · BMP · TIFF</div>
        </div>""", unsafe_allow_html=True)
        return

    n_imgs = len(uploaded_images)
    st.markdown(f"<div style='color:#94a3b8;font-size:13px;margin-bottom:14px;'>"
                f"🖼️ <b style='color:#38bdf8;'>{n_imgs}</b> image(s) queued for processing "
                f"{'| showing max 15 results' if n_imgs>MAX_DISPLAY else ''}</div>",
                unsafe_allow_html=True)

    # ── Process all images ──────────────────────────────
    batch_id = hash(tuple(name for name, _ in uploaded_images))

    if st.session_state.get("batch_id") != batch_id:
        results = []
        progress = st.progress(0, text="Processing images…")
        for i,(name,rgb) in enumerate(uploaded_images):
            mask, conf_map, inf_ms, rsize_img = run_one(rgb, model)
            pcts = class_stats(mask, N_CLASSES)
            # Make sure RGB fits the mask size
            col_mask = colorize(mask, COLOR_PALETTE)
            results.append({
                "name":    name,
                "rgb":     rsize_img, # use preprocessed right sized img
                "mask":    mask,
                "conf":    conf_map,
                "pcts":    pcts,
                "ms":      inf_ms,
                "col_mask":col_mask,
            })
            progress.progress(min(1.0, (i+1)/n_imgs), text=f"Processed {i+1}/{n_imgs}: {name}")
        progress.empty()
        st.session_state["batch_id"] = batch_id
        st.session_state["cached_results"] = results
    else:
        results = st.session_state["cached_results"]

    if not results: return

    all_ms = [r["ms"] for r in results]
    agg_pcts = np.mean([r["pcts"] for r in results], axis=0)
    dom_idx  = int(np.argmax(agg_pcts))

    st.markdown("<div class='sec-title'>📊 Batch Summary</div>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    cs = """<div class='stat-card'><div class='lbl'>{l}</div><div class='val'>{v}</div><div class='sub'>{s}</div></div>"""
    c1.markdown(cs.format(l="Images Processed",v=n_imgs,s="total uploaded"),      unsafe_allow_html=True)
    c2.markdown(cs.format(l="Avg Inference",v=f"{np.mean(all_ms):.0f}ms",s="per image"), unsafe_allow_html=True)
    c3.markdown(cs.format(l="Fastest",v=f"{min(all_ms):.0f}ms",s="single image"), unsafe_allow_html=True)
    c4.markdown(cs.format(l="Dominant Class",v=CLASS_NAMES[dom_idx],s=f"{agg_pcts[dom_idx]:.1f}% avg"), unsafe_allow_html=True)

    if n_imgs > MAX_DISPLAY:
        import math
        total_pages = math.ceil(n_imgs / MAX_DISPLAY)
        col_p1, col_p2 = st.columns([1, 4])
        with col_p1:
            page_num = st.number_input("Select Page", min_value=1, max_value=total_pages, value=1)
        start_idx = (page_num - 1) * MAX_DISPLAY
        end_idx = start_idx + MAX_DISPLAY
        display_results = results[start_idx:end_idx]
    else:
        display_results = results
        start_idx = 0

    st.markdown("<div class='sec-title' style='margin-top: 15px;'>🖼️ Segmentation Results</div>", unsafe_allow_html=True)

    for idx, res in enumerate(display_results):
        actual_idx = start_idx + idx + 1
        with st.expander(f"{'🟢' if idx<5 else '🔵'} [{actual_idx}] {res['name']}  — {res['ms']:.0f}ms  |  Conf: {np.mean(res['conf'])*100:.1f}%",
                         expanded=(idx < 3)):
            ca, cb, cc = st.columns(3)
            with ca:
                st.markdown("**Original**")
                st.image(res["rgb"], use_container_width=True)
            with cb:
                st.markdown("**Segmentation Mask**")
                st.image(res["col_mask"], use_container_width=True)
            with cc:
                st.markdown("**Blended Overlay**")
                st.image(blend(res["rgb"], res["col_mask"]), use_container_width=True)

            d1, d2 = st.columns([1,1.5])
            with d1:
                st.plotly_chart(make_pie(CLASS_NAMES, res["pcts"], COLOR_PALETTE), use_container_width=True)
            with d2:
                st.markdown(cls_breakdown_html(CLASS_NAMES, res["pcts"], COLOR_PALETTE), unsafe_allow_html=True)
                st.plotly_chart(make_conf_heatmap(res["conf"]), use_container_width=True)


# ────────────────────────────────────────────────────────
# TAB 2 – MODEL PERFORMANCE
# ────────────────────────────────────────────────────────
def tab_performance():
    st.markdown("""
    <div class='hero'>
        <h1>📊 Model Performance Dashboard</h1>
        <p>Training metrics and convergence curves tracked from offline stats logs.</p>
    </div>""", unsafe_allow_html=True)

    log = parse_training_log(LOG_PATH)
    
    st.markdown("<div class='sec-title'>🏆 Best Checkpoint Metrics</div>", unsafe_allow_html=True)
    best_miou    = log.get("best_iou", 0)
    total_epochs = log.get("max_epoch", 0)
    
    # Just find best epoch roughly from mIoU list
    best_epoch = "N/A"
    if log.get("miou") and len(log["miou"]) > 0:
        best_epoch = log["epochs"][np.argmax(log["miou"])]

    p1,p2,p3 = st.columns(3)
    for col,ttl,val,clr,sub in [
        (p1,"Best Val mIoU",   f"{best_miou:.4f}",  "#38bdf8", f"Epoch {best_epoch}"),
        (p2,"Epochs Trained",  f"{len(log.get('epochs',[]))}", "#818cf8", f"of {total_epochs} max"),
        (p3,"Log Status",      "Loaded", "#34d399",  "evaluation_metrics.txt"),
    ]:
        col.markdown(f"""<div class='perf-card'>
            <div class='ttl'>{ttl}</div>
            <div class='big' style='color:{clr};'>{val}</div>
            <div class='sml'>{sub}</div>
        </div>""", unsafe_allow_html=True)

    if not log.get("epochs"):
        st.warning("No metrics found in log file yet.")
        return

    eps = log["epochs"]

    # ── mIoU & Loss chart ──
    st.markdown("<div class='sec-title'>📈 Loss & mIoU progression</div>",
                unsafe_allow_html=True)

    fig_miou = make_subplots(specs=[[{"secondary_y": True}]])
    fig_miou.add_trace(go.Scatter(
        x=eps, y=log["miou"], name="Val mIoU",
        mode="lines+markers",
        line=dict(color="#34d399",width=3),
        marker=dict(size=7,color="#34d399",
                    symbol=["star" if v==max(log["miou"]) else "circle" for v in log["miou"]]),
        hovertemplate="Epoch %{x}<br>mIoU: %{y:.4f}<extra></extra>",
    ), secondary_y=False)
    fig_miou.add_trace(go.Scatter(
        x=eps, y=log["train_loss"], name="Train Loss",
        mode="lines+markers",
        line=dict(color="#38bdf8",width=2,dash="dot"),
        marker=dict(size=5,color="#38bdf8"),
        hovertemplate="Epoch %{x}<br>Loss: %{y:.4f}<extra></extra>",
    ), secondary_y=True)

    best_idx = np.argmax(log["miou"])
    fig_miou.add_vline(x=eps[best_idx],line_dash="dash",
                       line_color="rgba(251,191,36,.5)",
                       annotation_text=f" Best (Ep {eps[best_idx]})",
                       annotation_font_color="#fbbf24")
    fig_miou.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter",color="#e2e8f0"),
        xaxis=dict(title="Epoch",color="#94a3b8",gridcolor="rgba(255,255,255,.05)",dtick=2),
        yaxis=dict(title="mIoU",color="#34d399",gridcolor="rgba(255,255,255,.05)"),
        yaxis2=dict(title="Train Loss",color="#38bdf8",overlaying="y",side="right",gridcolor="rgba(0,0,0,0)"),
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=12)),
        margin=dict(t=16,b=40,l=55,r=55),height=400,
    )
    st.plotly_chart(fig_miou, use_container_width=True)
    
    # ── Table output ──
    st.markdown("<div class='sec-title' style='margin-top:20px;'>📋 Epoch-by-Epoch Log</div>", unsafe_allow_html=True)
    df = pd.DataFrame({
        "Epoch": log["epochs"],
        "Train Loss": log["train_loss"],
        "Val mIoU": log["miou"],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.markdown("<div class='sec-title' style='margin-top:20px;'>📷 Training Curves Output</div>", unsafe_allow_html=True)
    plot_file = os.path.join(BASE_DIR, "train_stats", "training_curves.png")
    if os.path.exists(plot_file):
        st.image(plot_file, use_container_width=True)

# ────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────
def main():
    with st.spinner("Loading Network..."):
        model = load_model_instance()

    render_sidebar()
    tab1, tab2 = st.tabs(["🌿 Segmentation Inference", "📊 Model Performance"])

    with tab1:
        tab_segmentation(model)
    with tab2:
        tab_performance()

if __name__ == "__main__":
    main()
