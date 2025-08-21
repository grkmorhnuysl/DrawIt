import yaml, numpy as np, torch, streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from pathlib import Path
from model import SmallCNN

st.set_page_config(page_title='Doodle_Emoji', page_icon='🎨', layout='centered')
st.title('Doodle_Emoji/Komut')

@st.cache_resource
def load_model(ckpt_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(ckpt_path, map_location=device)
    model = SmallCNN(num_classes=len(ckpt['classes']))
    model.load_state_dict(ckpt['model']); model.to(device); model.eval()
    return model, ckpt['classes'], device

def rasterize(img_arr):
    img = Image.fromarray(img_arr.astype('uint8')).convert('L').resize((28,28))
    arr = (np.array(img).astype('float32')/255.0)[None,None,...]
    return torch.from_numpy(arr)

mapping_path = Path('app/mapping.yaml')
if mapping_path.exists():
    mapping = yaml.safe_load(mapping_path.read_text(encoding='utf-8'))
else:
    mapping = {}

ckpt_path = Path('artifacts/best.pt')
if not ckpt_path.exists():
    st.warning('Model bulunamadı (artifacts/best.pt). Önce eğitimi çalıştırın.')
else:
    model, classes, device = load_model(str(ckpt_path))
    st.write('Sınıflar:', ', '.join(classes))
    stroke_width = st.sidebar.slider('Kalem kalınlığı', 5, 30, 15)
    bg = st.sidebar.color_picker('Zemin', '#000000')
    fg = st.sidebar.color_picker('Renk', '#FFFFFF')
    canvas = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=stroke_width,
        stroke_color=fg,
        background_color=bg,
        height=280, width=280, drawing_mode="freedraw", key="canvas",
    )
    if st.button('Tahmin et') and canvas.image_data is not None:
        x = rasterize(canvas.image_data).to(device)
        with torch.no_grad(): out = model(x); prob = torch.softmax(out, dim=1)[0].cpu().numpy()
        idx = int(prob.argmax()); cls = classes[idx]; conf = float(prob[idx])
        st.subheader(f'Tahmin: {cls}  (p={conf:.2f})')
        emoji = mapping.get(cls, '✨')
        if conf >= float(mapping.get('_threshold', 0.75)):
            st.markdown(f'# {emoji}')
        else:
            st.info('Güven eşiği altında — komut tetiklenmedi.')
