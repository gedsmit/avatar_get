import base64
import requests
import streamlit as st
from PIL import Image
import io
import zipfile

st.set_page_config(page_title="Anime Avatar Generator", page_icon=None)

def set_resolution(img, resolution):
    img = img.resize((resolution, resolution), Image.ANTIALIAS)
    return img

def app():
    # Banner or logo
    banner = Image.open("data/streamlit/banner.png")
    st.image(banner)

    st.title("👥 Avatar Generator")

    # Sidebar
    st.sidebar.title("Controls")
    num_images = st.sidebar.slider('Number of avatars', 1, 50, 5)
    generate_button = st.sidebar.button('Generate Avatars')
    resolution = st.sidebar.selectbox('Select image resolution for download', [64, 128, 256, 512, 1024])

    if "images" not in st.session_state or generate_button:
        data = {"num_images": num_images}
        response = requests.post("http://localhost:5000/generate", json=data)
        st.session_state.images = response.json().get("images")

    rows = num_images // 5
    if num_images % 5:
        rows += 1

    memory_file = io.BytesIO()

    with zipfile.ZipFile(memory_file, 'w') as zf:
        for i in range(rows):
            cols = st.columns(5)
            for j in range(5):
                idx = i * 5 + j
                if idx < num_images and idx < len(st.session_state.images):
                    img_data = base64.b64decode(st.session_state.images[idx])
                    img = Image.open(io.BytesIO(img_data))
                    img = set_resolution(img, 1024)
                    with cols[j]:
                        st.image(img, use_column_width=True, caption=f"Avatar {idx+1}")
                        img_download = set_resolution(img, resolution)
                        img_bytes = io.BytesIO()
                        img_download.save(img_bytes, format='PNG')
                        img_bytes = img_bytes.getvalue()
                        href = f'<a download="avatar_{idx+1}.png" href="data:image/png;base64,{base64.b64encode(img_bytes).decode()}">Download</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        zf.writestr(f"avatar_{idx+1}.png", img_bytes)

    memory_file.seek(0)
    if st.sidebar.button('Prepare'):
        href_all = f'<a download="avatars.zip" href="data:application/zip;base64,{base64.b64encode(memory_file.getvalue()).decode()}">Download All</a>'
        st.sidebar.markdown(href_all, unsafe_allow_html=True)

if __name__ == "__main__":
    app()
