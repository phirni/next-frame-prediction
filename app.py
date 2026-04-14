from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
from model_def import (
    ConvLSTM, BetaVAE,
    generate_frames_convlstm, PongFrameProcessor
)
from PIL import Image
import io, base64, os, glob

# -----------------------------------------------------------------------------
# 1. App setup
# -----------------------------------------------------------------------------
app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------
# Load models (resolve weight filenames with glob; RNN removed)
# ---------------------------------------------------------------------
convlstm_model = ConvLSTM()
vae_model = BetaVAE()

def find_weight(glob_patterns):
    for pat in glob_patterns:
        matches = glob.glob(pat)
        if matches:
            return matches[0]
    return None

available = os.listdir('model_weights') if os.path.isdir('model_weights') else []
convlstm_path = find_weight(['model_weights/*convlstm*.pth', 'model_weights/*convlstm*.pt'])
vae_path = find_weight(['model_weights/*beta*vae*.pth', 'model_weights/*vae*.pth', 'model_weights/*beta_vae*.pth'])

if convlstm_path is None or vae_path is None:
    raise FileNotFoundError(
        "Missing required model weight(s). Found in model_weights: "
        f"{available}. Expected files matching convlstm and beta/vae patterns."
    )

convlstm_model.load_state_dict(torch.load(convlstm_path, map_location=device))
vae_model.load_state_dict(torch.load(vae_path, map_location=device))

convlstm_model.to(device)
vae_model.to(device)

processor = PongFrameProcessor(target_size=(64, 64))

# -----------------------------------------------------------------------------
# 3. Routes
# -----------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')  # A simple HTML frontend

@app.route('/predict', methods=['POST'])
def predict():
    model_choice = request.form.get('model_choice')
    try:
        num_frames = int(request.form.get('num_frames', 1))
    except Exception:
        return jsonify({'error': 'num_frames must be an integer'}), 400

    if num_frames < 1 or num_frames > 100:
        return jsonify({'error': 'num_frames must be between 1 and 100'}), 400

    image_file = request.files.get('image')
    if image_file is None:
        return jsonify({'error': 'No image file uploaded'}), 400

    # Process uploaded image
    img_bytes = image_file.read()
    img = Image.open(io.BytesIO(img_bytes))
    img = img.convert('L')  # grayscale
    img_resized = img.resize((64, 64))
    img_arr = np.array(img_resized, dtype=np.float32) / 255.0
    img_tensor = torch.tensor(img_arr).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, T=1, C=1, H, W)

    frames_b64 = []

    # Run inference
    if model_choice == 'convlstm':
        out_frames = generate_frames_convlstm(convlstm_model, img_tensor, num_frames=num_frames, device=device)
        # out_frames shape: (num_frames, C, H, W) or (num_frames, 1, H, W)
        for i in range(out_frames.size(0)):
            frame = out_frames[i]
            # if shape is (C,H,W) or (1,H,W)
            if frame.dim() == 3:
                arr = frame[0].cpu().numpy() * 255
            else:
                arr = frame.cpu().numpy() * 255
            out_img = Image.fromarray(arr.astype(np.uint8))
            buf = io.BytesIO()
            out_img.save(buf, format='PNG')
            frames_b64.append(base64.b64encode(buf.getvalue()).decode('utf-8'))

    elif model_choice == 'vae':
        # VAE expects (B, C, H, W) — remove time dim
        vae_input = img_tensor.squeeze(1).to(device)  # (1, C, H, W)
        vae_model.eval()
        vae_model.to(device)
        with torch.no_grad():
            mu, logvar = vae_model.encode(vae_input)
            for _ in range(num_frames):
                # reparameterize will produce different samples each call
                z = vae_model.reparameterize(mu, logvar).to(device)  # (1, latent_dim)
                recon = vae_model.decode(z)  # (1, C, H, W)
                arr = recon[0, 0].cpu().numpy() * 255
                out_img = Image.fromarray(arr.astype(np.uint8))
                buf = io.BytesIO()
                out_img.save(buf, format='PNG')
                frames_b64.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
    else:
        return jsonify({'error': 'Invalid model choice'}), 400

    return jsonify({'generated_frames': frames_b64})
# ...existing code...
# 4. Run server
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # bind to all interfaces so ngrok can forward traffic
    app.run(host='0.0.0.0', port=5000, debug=False)
