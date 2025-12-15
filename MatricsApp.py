import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import io

st.set_page_config(page_title="Matrix Transformations & Image Filters", layout="wide")

# =================== Utility Image Loader =====================

def load_image(uploaded):
    if uploaded is None:
        return None
    try:
        image = Image.open(BytesIO(uploaded.read()))
        img = np.array(image)
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        img = np.clip(img, 0, 1)
        return img
    except Exception as e:
        st.error(f"Failed to load image: {str(e)}")
        return None

def show(img, caption=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    if img.ndim == 2:
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(img, vmin=0, vmax=1)
    ax.axis("off")
    if caption:
        ax.set_title(caption)
    st.pyplot(fig)
    plt.close(fig)

def get_image_bytes(img, format='PNG'):
    img_uint8 = (img * 255).astype(np.uint8)
    if img.ndim == 2:
        img_uint8 = np.stack([img_uint8] * 3, axis=-1)
    pil_img = Image.fromarray(img_uint8)
    buf = io.BytesIO()
    pil_img.save(buf, format=format)
    buf.seek(0)
    return buf.getvalue()

# =================== Affine Transform =====================

def transform(img, M, interp="nearest"):
    h, w = img.shape[:2]
    ys, xs = np.indices((h, w))
    coords = np.stack([xs.ravel(), ys.ravel(), np.ones(h*w)], axis=0)
    M_inv = np.linalg.inv(M)
    mapped = M_inv @ coords
    src_x, src_y = mapped[0].reshape(h, w), mapped[1].reshape(h, w)

    def sample(ch):
        if interp == "nearest":
            xi = np.round(src_x).astype(int)
            yi = np.round(src_y).astype(int)
            valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            output = np.zeros_like(src_x)
            output[valid] = ch[yi[valid], xi[valid]]
            return output
        else:
            x0 = np.floor(src_x).astype(int); x1 = x0 + 1
            y0 = np.floor(src_y).astype(int); y1 = y0 + 1
            x0 = np.clip(x0, 0, w-1); x1 = np.clip(x1, 0, w-1)
            y0 = np.clip(y0, 0, h-1); y1 = np.clip(y1, 0, h-1)
            Ia = ch[y0, x0]; Ib = ch[y1, x0]
            Ic = ch[y0, x1]; Id = ch[y1, x1]
            wa = (x1-src_x)*(y1-src_y)
            wb = (x1-src_x)*(src_y-y0)
            wc = (src_x-x0)*(y1-src_y)
            wd = (src_x-x0)*(src_y-y0)
            return wa*Ia + wb*Ib + wc*Ic + wd*Id

    if img.ndim == 2:
        return sample(img)
    else:
        return np.stack([sample(img[..., c]) for c in range(img.shape[2])], axis=2)

# =================== Convolution =====================

def apply_kernel(im, k):
    h, w = im.shape[:2]
    kh, kw = k.shape
    pad = kh // 2
    out = np.zeros_like(im)

    if im.ndim == 3:
        for c in range(im.shape[2]):
            padded = np.pad(im[..., c], pad, mode="edge")
            for i in range(h):
                for j in range(w):
                    out[i, j, c] = np.sum(padded[i:i+kh, j:j+kw] * k)
    else:
        padded = np.pad(im, pad, mode="edge")
        for i in range(h):
            for j in range(w):
                out[i, j] = np.sum(padded[i:i+kh, j:j+kw] * k)

    return np.clip(out, 0, 1)

kernels = {
    "Blur": np.ones((3, 3)) / 9,
    "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    "Edge": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
}

# =================== Sidebar Menu =====================

menu = st.sidebar.radio("Navigation Menu", ["Home", "Image Processing", "Team Members"])

# =================== HOME =====================

if menu == "Home":
    st.title("Matrix Transformations & Image Processing App")
    st.markdown("""
    **Introduction**  
    Aplikasi ini dibuat untuk mendemonstrasikan konsep *Matrix Transformation* dan *Convolution* 
    pada pengolahan citra digital menggunakan Python dan Streamlit.

    **What this app does:**
    - Transformasi geometri berbasis matriks (translasi, rotasi, scaling, dll)
    - Image filtering menggunakan operasi convolution
    - Visualisasi hasil transformasi citra

    **Matrix Transformation (Brief)**  
    Transformasi citra direpresentasikan menggunakan matriks affine 3×3 untuk
    memetakan koordinat piksel input ke output.

    **Convolution (Brief)**  
    Convolution adalah operasi matematika antara kernel dan citra untuk mengekstraksi
    fitur seperti tepi, blur, dan penajaman.
    """)

    st.image("https://upload.wikimedia.org/wikipedia/commons/2/28/Matrix_multiplication_diagram_2.svg",
             caption="Visual example of matrix transformation", width=400)

# =================== IMAGE PROCESSING =====================

elif menu == "Image Processing":
    st.title("Image Processing")
    uploaded = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    img = load_image(uploaded)

    tab1, tab2 = st.tabs(["Geometric Transformations", "Image Filtering"])

    with tab1:
        st.subheader("Geometric Transformations")
        if img is None:
            st.info("Upload image first.")
        else:
            show(img, "Original")
            t = st.selectbox("Select Transform", ["Translation", "Scaling", "Rotation"])
            h, w = img.shape[:2]
            if t == "Translation":
                tx = st.slider("Shift X", -200, 200, 0)
                ty = st.slider("Shift Y", -200, 200, 0)
                M = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
            elif t == "Scaling":
                s = st.slider("Scale", 0.2, 3.0, 1.0)
                M = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]])
            else:
                a = st.slider("Angle", -180, 180, 0)
                rad = np.radians(a)
                M = np.array([[np.cos(rad), -np.sin(rad), 0],
                              [np.sin(rad), np.cos(rad), 0],
                              [0, 0, 1]])
            out = transform(img, M)
            show(out, "Output")

    with tab2:
        st.subheader("Image Filtering (Convolution)")
        if img is None:
            st.info("Upload image first.")
        else:
            show(img, "Original")
            choice = st.selectbox("Filter", list(kernels.keys()))
            out = apply_kernel(img, kernels[choice])
            show(out, f"{choice} Output")

# =================== TEAM MEMBERS =====================

elif menu == "Team Members":
    st.title("Team Members – Group 7")

    members = [
        {
            "name": "Ahmad Galan Ali",
            "role": "Implementasi matrix transformation dan web app Streamlit",
            "photo": "images/ahmad.jpg"
        },
        {
            "name": "Kennedy Ibrahim Ubaldus",
            "role": "Image filtering & convolution",
            "photo": "images/kennedy.jpg"
        },
        {
            "name": "Raffi Ardiansyah Zulin",
            "role": "Documentation and report",
            "photo": "images/raffi.jpg"
        }
    ]

    cols = st.columns(3)

    for col, member in zip(cols, members):
        with col:
            st.image(member["photo"], width=200)
            st.subheader(member["name"])
            st.write(f"**Contribution:** {member['role']}")



