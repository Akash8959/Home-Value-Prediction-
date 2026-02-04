import os
import sqlite3
from datetime import datetime
from typing import Optional
from flask import session
from werkzeug.security import generate_password_hash, check_password_hash

from flask import (
    Flask, render_template, request, redirect,
    url_for, send_from_directory, flash, jsonify
)
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# Optional Keras support
try:
    from tensorflow.keras.models import load_model as keras_load_model
    KERAS_AVAILABLE = True
except Exception as e:
    print("Keras/TensorFlow not available:", e)
    KERAS_AVAILABLE = False

# -------------------------
# Config
# -------------------------
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DB_PATH = os.path.join(APP_ROOT, "predictions.db")
MODEL_DIR = os.path.join(APP_ROOT, "model")
TABULAR_MODEL_PATH = os.path.join(MODEL_DIR, "home_value_model.pkl")
CNN_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model.h5")          # optional
FUSION_MODEL_PATH = os.path.join(MODEL_DIR, "fusion_model.pkl")  # optional

ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg", "tif", "tiff"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "replace-with-a-secure-key"

# -------------------------
# Load models
# -------------------------
tabular_model = None
cnn_model = None
fusion_model = None

if os.path.exists(TABULAR_MODEL_PATH):
    try:
        tabular_model = joblib.load(TABULAR_MODEL_PATH)
        print("Loaded tabular model:", TABULAR_MODEL_PATH)
    except Exception as e:
        print("Failed to load tabular model:", e)

if KERAS_AVAILABLE and os.path.exists(CNN_MODEL_PATH):
    try:
        cnn_model = keras_load_model(CNN_MODEL_PATH)
        print("Loaded CNN model:", CNN_MODEL_PATH)
    except Exception as e:
        print("Failed to load CNN model:", e)

if os.path.exists(FUSION_MODEL_PATH):
    try:
        fusion_model = joblib.load(FUSION_MODEL_PATH)
        print("Loaded fusion model:", FUSION_MODEL_PATH)
    except Exception as e:
        print("Failed to load fusion model:", e)

# -------------------------
# Helpers
# -------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXT

def secure_name(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in "._-").strip()

def save_image(file_storage) -> str:
    filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + secure_name(file_storage.filename or "image")
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file_storage.save(path)
    return filename

def preprocess_image_for_cnn(pil_img: Image.Image, target_size=(224,224)) -> np.ndarray:
    img = pil_img.convert("RGB").resize(target_size)
    arr = np.asarray(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

def predict_tabular_from_form(form) -> tuple[float, pd.DataFrame]:
    numeric_cols = ['longitude', 'latitude', 'housing_median_age',
                    'total_rooms', 'total_bedrooms', 'population',
                    'households', 'median_income']
    cat_cols = ['ocean_proximity']

    row = {}
    for c in numeric_cols:
        val = form.get(c)
        if val is None or val == "":
            raise ValueError(f"Missing numeric field: {c}")
        row[c] = float(val)

    row['ocean_proximity'] = form.get('ocean_proximity', '')

    df = pd.DataFrame([row], columns=numeric_cols + cat_cols)
    if tabular_model is None:
        raise RuntimeError("Tabular model not loaded")
    pred = tabular_model.predict(df)[0]
    return float(pred), df

def predict_cnn_from_file(filepath: str) -> Optional[float]:
    if cnn_model is None:
        return None
    try:
        pil_img = Image.open(filepath)
        x = preprocess_image_for_cnn(pil_img, target_size=(224,224))
        preds = cnn_model.predict(x)
        cnn_pred = float(np.squeeze(preds))
        return cnn_pred
    except Exception as e:
        print("CNN predict error:", e)
        return None

def combine_predictions(tab_pred: float, cnn_pred: Optional[float]=None,
                        weight_tab: float=0.6, weight_cnn: float=0.4) -> float:
    if cnn_pred is None:
        return tab_pred
    if fusion_model is not None:
        try:
            arr = np.array([[tab_pred, cnn_pred]])
            fused = fusion_model.predict(arr)[0]
            return float(fused)
        except Exception as e:
            print("Fusion model failed, falling back to weights:", e)
    return float(tab_pred * weight_tab + cnn_pred * weight_cnn)


# -------------------------
# Database: create users table
# -------------------------
def init_users_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            email TEXT UNIQUE,
            password_hash TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

init_users_db()

# -------------------------
# Routes
# -------------------------

# -------------------------
# Registration
# -------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        if not username or not email or not password:
            flash("Please fill in all fields", "warning")
            return redirect(url_for("register"))

        password_hash = generate_password_hash(password)
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("""
                INSERT INTO users (username, email, password_hash, created_at)
                VALUES (?, ?, ?, ?)
            """, (username, email, password_hash, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()
            conn.close()
            flash("Registration successful! Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username or email already exists", "danger")
            return redirect(url_for("register"))

    return render_template("register.html")

# -------------------------
# Login
# -------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id, password_hash FROM users WHERE username=?", (username,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[1], password):
            session["user_id"] = user[0]
            session["username"] = username
            flash(f"Welcome, {username}!", "success")
            return redirect(url_for("home"))
        else:
            flash("Invalid username or password", "danger")
            return redirect(url_for("login"))

    return render_template("login.html")

# -------------------------
# Logout
# -------------------------
@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully", "info")
    return redirect(url_for("login"))

# -------------------------
# Protect prediction page
# -------------------------
@app.before_request
def require_login():
    protected_routes = ["home", "predict", "dashboard"]
    if request.endpoint in protected_routes and "user_id" not in session:
        return redirect(url_for("login"))

@app.route("/")
def home():
    return render_template("index.html",
                           has_cnn=(cnn_model is not None),
                           has_tabular=(tabular_model is not None))

@app.route("/predict", methods=["POST"])
def predict():
    try:
        tab_pred, df_input = predict_tabular_from_form(request.form)
        file = request.files.get("satellite_image")
        image_filename = None
        cnn_pred = None

        if file and file.filename and allowed_file(file.filename):
            image_filename = save_image(file)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], image_filename)
            cnn_pred = predict_cnn_from_file(filepath)

        final_pred = combine_predictions(tab_pred, cnn_pred)

        # Save to DB
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            INSERT INTO predictions (
                timestamp, longitude, latitude, housing_median_age,
                total_rooms, total_bedrooms, population, households,
                median_income, ocean_proximity, satellite_image,
                model_used, predicted_value
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            float(request.form['longitude']), float(request.form['latitude']),
            float(request.form['housing_median_age']),
            float(request.form['total_rooms']), float(request.form['total_bedrooms']),
            float(request.form['population']), float(request.form['households']),
            float(request.form['median_income']), request.form['ocean_proximity'],
            image_filename,
            "fusion" if (cnn_pred is not None and fusion_model is not None) else ("cnn" if cnn_pred is not None else "tabular"),
            float(final_pred)
        ))
        conn.commit()
        conn.close()

        # Pass entered values to template
        input_values = {col: request.form.get(col) for col in [
            'longitude','latitude','housing_median_age','total_rooms',
            'total_bedrooms','population','households','median_income','ocean_proximity'
        ]}

        return render_template(
            "result.html",
            predicted_value=round(final_pred, 2),
            tabular_pred=round(tab_pred, 2),
            image_filename=image_filename,
            input_values=input_values
        )

    except Exception as e:
        print("Predict route error:", e)
        return f"Prediction Error: {e}", 400


@app.route("/dashboard")
def dashboard():
    try:
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 10))
        offset = (page - 1) * per_page

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM predictions")
        total = c.fetchone()[0]

        c.execute("""
            SELECT id, timestamp, longitude, latitude, housing_median_age,
                   total_rooms, total_bedrooms, population, households,
                   median_income, ocean_proximity, satellite_image, model_used, predicted_value
            FROM predictions
            ORDER BY id DESC LIMIT ? OFFSET ?
        """, (per_page, offset))
        rows = c.fetchall()
        conn.close()

        total_pages = (total + per_page - 1) // per_page

        # Chart data
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            "SELECT timestamp, predicted_value, ocean_proximity FROM predictions ORDER BY id DESC LIMIT 500", conn
        )
        conn.close()
        df = df[::-1]

        chart_times = df['timestamp'].astype(str).tolist()
        chart_preds = df['predicted_value'].tolist()
        prox_counts = df['ocean_proximity'].value_counts().to_dict()

        return render_template("dashboard.html",
                               rows=rows,
                               page=page,
                               total_pages=total_pages,
                               chart_times=chart_times,
                               chart_preds=chart_preds,
                               prox_labels=list(prox_counts.keys()),
                               prox_values=list(prox_counts.values()))
    except Exception as e:
        print("Dashboard error:", e)
        return f"Dashboard Error: {e}", 500

@app.route("/delete/<int:id>", methods=["POST", "GET"])
def delete(id):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT satellite_image FROM predictions WHERE id=?", (id,))
        row = c.fetchone()
        if row and row[0]:
            path = os.path.join(app.config["UPLOAD_FOLDER"], row[0])
            if os.path.exists(path):
                os.remove(path)
        c.execute("DELETE FROM predictions WHERE id=?", (id,))
        conn.commit()
        conn.close()
        flash("Record deleted", "success")
        return redirect(url_for("dashboard"))
    except Exception as e:
        print("Delete error:", e)
        return f"Delete Error: {e}", 500

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/api/models")
def api_models():
    return jsonify({
        "tabular": bool(tabular_model),
        "cnn": bool(cnn_model),
        "fusion": bool(fusion_model)
    })

if __name__ == "__main__":
    app.run(debug=True)
