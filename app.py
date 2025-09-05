import os
import io
import traceback
import numpy as np
from PIL import Image
from joblib import load as joblib_load

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
import flask as flask_module

# Feature extraction imports
from skimage.feature import hog, local_binary_pattern
try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops



APP_PORT = int(os.environ.get("PORT", 5000))
MODEL_FILENAME = os.environ.get("MODEL_FILE", "Recycling_project.joblib")
CLASS_LABELS = ["glass", "paper", "plastic", "metal"]

# Flask setup
app = Flask(__name__)
CORS(app)



def to_py(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="ignore")
    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_py(v) for v in obj]
    return obj

def json_response(data, status=200):
    return make_response(jsonify(to_py(data)), status)

@app.errorhandler(HTTPException)
def handle_http_exc(e):
    return json_response({"error": e.name, "message": e.description}, e.code)

@app.errorhandler(Exception)
def handle_any_exc(e):
    return json_response(
        {"error": "Internal Server Error", "message": str(e), "trace": traceback.format_exc()},
        500,
    )



model = None
model_meta = {}

def load_model():
    global model, model_meta
    if os.path.exists(MODEL_FILENAME):
        try:
            model = joblib_load(MODEL_FILENAME)
            classes = getattr(model, "classes_", None)
            if classes is not None:
                classes = [str(c) for c in classes]
            else:
                classes = CLASS_LABELS
            model_meta = {
                "model_file": MODEL_FILENAME,
                "loaded": True,
                "classes": classes,
                "hint": "Model loaded successfully.",
            }
        except Exception as e:
            model = None
            model_meta = {
                "model_file": MODEL_FILENAME,
                "loaded": False,
                "error": f"Failed to load model: {e}",
                "trace": traceback.format_exc(),
            }
    else:
        model_meta = {
            "model_file": MODEL_FILENAME,
            "loaded": False,
            "error": "Model file not found in current directory.",
        }

load_model()



def preprocess_image_to_vector(file_storage, size=(64, 64)):
    img_bytes = file_storage.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("L").resize(size)

    # Normalize
    img_float = np.array(image, dtype=np.float32) / 255.0
    img_int = (img_float * 255).astype(np.uint8)

    # HOG
    hog_features = hog(
        img_float,
        orientations=12,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        feature_vector=True,
        transform_sqrt=True,
        block_norm="L2-Hys"
    )

    # Multi-radius LBP
    lbp_parts = []
    for radius in (1, 2, 3):
        n_points = 8 * radius
        lbp = local_binary_pattern(img_int, n_points, radius, method="uniform")
        lbp_hist, _ = np.histogram(
            lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2), density=True
        )
        lbp_parts.append(lbp_hist)
    lbp_feat = np.concatenate(lbp_parts)

    # GLCM
    img_q = (img_int // 8).astype(np.uint8)
    glcm = graycomatrix(
        img_q,
        [1, 2],
        [0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=32,
        symmetric=True,
        normed=True
    )
    glcm_props = []
    for p in ("contrast", "homogeneity", "energy", "correlation"):
        vals = graycoprops(glcm, p).ravel()
        glcm_props.extend(vals)
    glcm_feat = np.array(glcm_props, dtype=np.float32)

    return np.concatenate([hog_features, lbp_feat, glcm_feat])



@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return json_response({"error": "No image uploaded", "code": "no_image"}, 400)

    try:
        file = request.files["image"]
        if not file or file.filename == "":
            return json_response({"error": "Empty file"}, 400)

        vec = preprocess_image_to_vector(file)

        if model is not None and model_meta.get("loaded"):
            try:
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba([vec])[0]
                else:
                    scores = model.decision_function([vec])
                    if scores.ndim == 1:  # binary case
                        scores = np.column_stack([-scores, scores])
                    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                    probs = (exp_scores / exp_scores.sum(axis=1, keepdims=True))[0]
            except Exception as e:
                return json_response({
                    "error": "Model prediction failed",
                    "message": str(e),
                    "hint": "Check if preprocessing matches training pipeline."
                }, 500)
            # Ensure we map to known labels (glass, paper, plastic, metal)
            classes = CLASS_LABELS  
            prob_map = {cls: float(p) for cls, p in zip(classes, probs)}

            top_idx = int(np.argmax(probs))
            result = {
                "success": True,
                "predicted_label": classes[top_idx],
                "top_confidence": float(probs[top_idx]),
                "probabilities": prob_map,
            }
            return json_response(result, 200)

        else:
            return json_response({
                "error": "Model not loaded",
                "message": model_meta.get("error", "Missing model file"),
                "model_meta": model_meta
            }, 500)

    except Exception as e:
        return json_response({
            "error": "Exception during prediction",
            "message": str(e),
            "trace": traceback.format_exc()
        }, 500)


@app.route("/test-upload", methods=["POST"])
def test_upload():
    if "image" not in request.files:
        return json_response({"error": "No image uploaded", "debug": "no_image"}, 400)
    f = request.files["image"]
    return json_response({
        "success": True,
        "filename": f.filename,
        "content_type": f.content_type,
        "message": "File received successfully"
    })

@app.route("/debug-info")
def debug_info():
    info = {
        "flask_version": flask_module.__version__,
        "python_version": os.sys.version,
        "cwd": os.getcwd(),
        "files_in_dir": os.listdir("."),
        "has_model_file": os.path.exists(MODEL_FILENAME),
        "model_meta": model_meta,
    }
    return json_response(info)

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Trash Classifier</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body { font-family: Arial, sans-serif; background: #0f1226; color: #eee; text-align:center; padding:40px; }
  .drop { border: 2px dashed #888; padding: 40px; border-radius: 12px; margin: 20px auto; width: 400px; }
  img { max-width: 200px; margin-top: 15px; border-radius: 8px; }
  #results { margin-top:20px; font-size:16px; text-align:left; display:inline-block; }
  .bar { background:#222; border-radius:8px; overflow:hidden; height:16px; margin-top:4px; }
  .fill { height:100%; transition:width 0.5s; }
</style>
</head>
<body>
  <h1>♻️ Trash Classification</h1>
  <div class="drop">
    <input type="file" id="file" accept="image/*"><br>
    <button onclick="upload()">Upload & Predict</button>
    <div id="status"></div>
    <img id="preview" style="display:none;">
  </div>
  <div id="results"></div>

<script>
async function upload() {
  const fileInput = document.getElementById('file');
  const status = document.getElementById('status');
  const results = document.getElementById('results');
  const preview = document.getElementById('preview');
  results.innerHTML = '';
  if (!fileInput.files[0]) { status.innerText = "Please select a file"; return; }
  const fd = new FormData();
  fd.append("image", fileInput.files[0]);
  preview.src = URL.createObjectURL(fileInput.files[0]);
  preview.style.display = "block";
  status.innerText = "Uploading...";
  try {
    const res = await fetch("/predict", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok || data.error) {
      status.innerText = "Error: " + (data.message || data.error);
      console.error(data);
      return;
    }
    status.innerText = "Prediction complete!";

    results.innerHTML = `<h3>Predicted: ${data.predicted_label}</h3>`;
    const probs = data.probabilities || {};

    // Find top prediction
    let topLabel = null, topVal = -1;
    for (const [label, value] of Object.entries(probs)) {
      if (value > topVal) {
        topVal = value;
        topLabel = label;
      }
    }

    // Render bars
    for (const [label, value] of Object.entries(probs)) {
      const pct = (value * 100).toFixed(1);
      const isTop = (label === topLabel);
      const barColor = isTop ? "#4caf50" : "#4cafef"; // green for top
      results.innerHTML += `
        <div style="margin:8px 0; text-align:left;">
          <strong>${label}</strong> — ${pct}%
          <div class="bar">
            <div class="fill" style="width:${pct}%; background:${barColor};"></div>
          </div>
        </div>
      `;
    }

  } catch(err) {
    status.innerText = "Network error";
    console.error(err);
  }
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return INDEX_HTML

if __name__ == "__main__":
    print("Starting Trash Classifier (JSON-safe) on http://localhost:%d" % APP_PORT)
    if not model_meta.get("loaded"):
        print("Model not loaded:", model_meta.get("error", "unknown"))
        print("Place your joblib model next to app.py as:", MODEL_FILENAME)
    app.run(host="0.0.0.0", port=APP_PORT, debug=True)
