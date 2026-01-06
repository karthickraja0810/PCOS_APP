# -*- coding: utf-8 -*-
import os
import io
import json
import base64
import sqlite3
import joblib
import numpy as np
import cv2
import time
import random
from dotenv import load_dotenv

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.utils import secure_filename

# TensorFlow
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Gemini Vision LLM
from google import genai
from google.genai import types
from google.genai.errors import APIError

# ====================================================
#                U-NET SEGMENTATION MODEL
# ====================================================

UNET_MODEL_PATH = "best_unet_checkpoint.keras"
TFLITE_MODEL_PATH = "best_unet_checkpoint.tflite"
UNET_IMG_SIZE = (256, 256)

load_dotenv()

# Custom Dice Loss (must match training)
def dice_loss(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    smooth = 1e-6
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    dice = (2.0 * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )
    return 1 - dice

# 3. The Optimized TFLite Loader
tflite_interpreter = None

def get_tflite_interpreter():
    """Lazy load the TFLite interpreter to save memory."""
    global tflite_interpreter
    if tflite_interpreter is None:
        print("Loading TFLite Model into memory...")
        try:
            # TFLite uses significantly less memory than full Keras models
            tflite_interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
            tflite_interpreter.allocate_tensors()
            print("TFLite Model Loaded Successfully.")
        except Exception as e:
            print(f"ERROR loading TFLite model: {e}")
    return tflite_interpreter

# Removed legacy Keras get_unet_model function

# Removed eager load to prevent OOM on app startup in Render's free tier


def unet_predict_mask(image_bytes):
    """
    Takes raw uploaded image bytes and returns (mask_b64, overlay_b64)
    """
    # Use TFLite for memory efficiency on Render
    interpreter = get_tflite_interpreter()
    if interpreter is None:
        raise Exception("TFLite model could not be loaded. Check memory limits or file path.")

    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise Exception("Failed to decode image.")
        
    orig_h, orig_w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, UNET_IMG_SIZE)

    # Prepare input for TFLite
    input_data = np.expand_dims(resized.astype("float32") / 255.0, axis=0)

    # Run inference
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])[0]

    mask = (pred > 0.5).astype(np.uint8)

    mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    mask_out = (mask_resized * 255).astype(np.uint8)

    overlay = img.copy()
    overlay[mask_out == 255] = (0, 255, 0)
    blended = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

    _, mask_png = cv2.imencode(".png", mask_out)
    mask_b64 = base64.b64encode(mask_png).decode("utf-8")

    _, overlay_png = cv2.imencode(".png", blended)
    overlay_b64 = base64.b64encode(overlay_png).decode("utf-8")

    return mask_b64, overlay_b64


# ====================================================
#                 GEMINI CLIENT (OCR + Q&A)
# ====================================================

# Use 'gemini-1.5-flash-latest' for better stability and to avoid 404 issues
GEMINI_MODEL = "gemini-1.5-flash-latest"

def get_gemini_client():
    # Clean the API key to handle common Render configuration errors (extra spaces or quotes)
    raw_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not raw_key:
        print("CRITICAL: GEMINI_API_KEY not found in environment!")
        return None
    
    api_key = raw_key.strip().replace('"', '').replace("'", "")
    try:
        # Pass the cleaned key explicitly to the client
        return genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Gemini init error: {e}")
        return None

def retry_gemini_call(func, *args, **kwargs):
    """
    Retries a Gemini API call with exponential backoff for 503 errors.
    """
    max_retries = 3
    base_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except APIError as e:
            if e.code == 503: # Service Unavailable/Overloaded
                if attempt < max_retries - 1:
                    sleep_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Gemini 503 Error. Retrying in {sleep_time:.2f}s... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(sleep_time)
                else:
                    print("Max retries reached for Gemini API.")
                    raise e
            else:
                raise e
        except Exception as e:
            raise e


# ====================================================
#               XGBOOST PCOS MODEL
# ====================================================

try:
    xgb_model = joblib.load("pcos_xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("PCOS ML Model Loaded.")
except Exception as e:
    print("Failed to load XGB/Scaler:", e)
    xgb_model = None
    scaler = None


# ====================================================
#                    FLASK APP
# ====================================================

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "a-very-long-random-string")

# Ensure Gemini uses the API Key from your .env or Environment Settings
client = None
try:
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    print("Gemini client initialized.")
except Exception as e:
    print("Gemini init error:", e)

UPLOAD_FOLDER = "temp_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ====================================================
#                DATABASE INITIALIZATION
# ====================================================

# Get the absolute path to ensure Render finds the file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Use environment variable for DB path if available (recommended for Persistent Disks)
db_path = os.environ.get("DATABASE_PATH", os.path.join(BASE_DIR, "pcos.db"))

def init_db():
    # Only create if it doesn't exist
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}. Creating new database...")
        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                email TEXT UNIQUE,
                password TEXT,
                role TEXT
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle TEXT,
                weight REAL,
                acne TEXT,
                risk TEXT
            )
        """)

        # Add default admin if not present
        c.execute("SELECT * FROM users WHERE email = 'admin@hospital.com'")
        if not c.fetchone():
            c.execute("INSERT INTO users (name, email, password, role) VALUES (?, ?, ?, ?)",
                      ("Admin", "admin@hospital.com", "admin123", "admin"))

        conn.commit()
        conn.close()
        print("Database initialized successfully.")
    else:
        print("Database already exists. Skipping initialization.")

# Run the initialization
init_db()

def get_db_connection():
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        print(f"Database Connection Error: {e}")
        return None

# ====================================================
#                    FLASK ROUTES
# ====================================================

@app.route("/")
def patient_home():
    return render_template("patient_home.html")


# ========================== U-NET ROUTE ==========================

@app.route("/unet_predict", methods=["GET", "POST"])
def unet_predict_route():
    if request.method == "GET":
        return redirect(url_for("doctor_dashboard"))
    if unet_model is None:
        return jsonify({"error": "U-Net model failed to load."})

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded."})

    try:
        image_bytes = file.read()
        mask_b64, overlay_b64 = unet_predict_mask(image_bytes)

        return jsonify({
            "unet_mask": mask_b64,
            "unet_overlay": overlay_b64
        })

    except Exception as e:
        return jsonify({"error": f"Segmentation Error: {str(e)}"})
    
# ---------------------------------------------
# GLOBAL FEATURE LIST (MUST MATCH XGboost training data)
# ---------------------------------------------
FINAL_MODEL_FEATURES = [
    ' Age (yrs)', 'Weight (Kg)', 'Height(Cm) ', 'BMI', 'Pulse rate(bpm) ',
    'RR (breaths/min)', 'Hb(g/dl)', 'Cycle(R/I)', 'Cycle length(days)',
    'Pregnant(Y/N)', 'No. of aborptions',
    ' I beta-HCG(mIU/mL)', 'II beta-HCG(mIU/mL)', 'FSH(mIU/mL)',
    'LH(mIU/mL)', 'FSH/LH', 'Hip(inch)', 'Waist(inch)', 'Waist:Hip Ratio',
    'TSH (mIU/L)', 'AMH(ng/mL)', 'PRL(ng/mL)', 'Vit D3 (ng/mL)',
    'PRG(ng/mL)', 'RBS(mg/dl)', 'Weight gain(Y/N)', 'hair growth(Y/N)',
    'Skin darkening (Y/N)', 'Hair loss(Y/N)', 'Pimples(Y/N)',
    'Fast food (Y/N)', 'Reg.Exercise(Y/N)', 'BP _Systolic (mmHg)',
    'BP _Diastolic (mmHg)', 'Follicle No. (L)', 'Follicle No. (R)',
    'Avg. F size (L) (mm)', 'Avg. F size (R) (mm)', 'Endometrium (mm)',
    'Blood Group_A-', 'Blood Group_AB+', 'Blood Group_AB-',
    'Blood Group_B+', 'Blood Group_B-', 'Blood Group_O+',
    'Blood Group_O-'
    # NOTE: 'Blood Group_A+' is the base case dropped by drop_first=True
]    



# ====================================================
#               LOGIN AND DASHBOARD ROUTES
# ====================================================
def safe_float(value):
    """Convert input safely to float, treating blank as 0"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
    
@app.route('/predict', methods=['GET', 'POST'])
def predict_patient_rule_based():
    """Rule-based prediction for general public (patient_home link)"""
    if request.method == 'POST':
        try:
            age = safe_float(request.form.get('age'))
            height = safe_float(request.form.get('height'))
            weight = safe_float(request.form.get('weight'))
            bmi = safe_float(request.form.get('bmi'))
            hb = safe_float(request.form.get('hb'))
            pulse_rate = safe_float(request.form.get('pulse_rate'))

            cycle_regular = request.form.get('cycle_regular', 'regular')
            hair_growth = request.form.get('hair_growth', 'normal')
            acne = request.form.get('acne', 'no')
            dark_patches = request.form.get('dark_patches', 'no')

            score = 0
            if bmi > 30:
                score += 2
            elif 25 <= bmi <= 30:
                score += 1
            if cycle_regular == 'irregular':
                score += 2
            if hb < 10:
                score += 1
            if pulse_rate > 100:
                score += 1
            if hair_growth == 'excessive':
                score += 2
            if acne == 'yes':
                score += 1
            if dark_patches == 'yes':
                score += 1

            if score >= 5:
                result = "High"
            elif 3 <= score < 5:
                result = "Moderate"
            else:
                result = "Low"

            return render_template('result.html', result=result)

        except Exception as e:
            return f"Error in rule-based prediction: {str(e)}"

    return render_template('predict.html')

@app.route("/hcw")
def hcw_home():
    return render_template("hcw_home.html")

@app.route("/hcw/login", methods=["GET", "POST"])
def hcw_login():
    if request.method == "GET":
        return redirect(url_for("hcw_home"))
        
    email = request.form["email"]
    password = request.form["password"]

    conn = get_db_connection()
    if conn is None:
        flash("Database connection error. Please try again later.", "error")
        return redirect(url_for("hcw_home"))

    user = conn.execute(
        "SELECT * FROM users WHERE email = ? AND password = ?",
        (email, password)
    ).fetchone()
    conn.close()

    if user:
        session["user_id"] = user["id"]
        session["role"] = user["role"]
        session["name"] = user["name"]

        if user["role"] == "admin":
            return redirect(url_for("admin_dashboard"))
        else:
            return redirect(url_for("doctor_dashboard"))
    else:
        flash("Invalid email or password", "error")
        return redirect(url_for("hcw_home"))
    
@app.route('/doctor/ocr_process', methods=['GET', 'POST'])
def ocr_process():
    if request.method == "GET":
        return redirect(url_for("doctor_dashboard"))
    
    if session.get('role') != 'doctor':
        return jsonify({'status': 'error', 'message': 'Access denied.'}), 403

    # --- UPDATED: Use the helper function to get the client ---
    current_client = get_gemini_client()
    if current_client is None:
        return jsonify({
            'status': 'error', 
            'message': 'Gemini Vision Model not initialized. Please verify the GEMINI_API_KEY in Render environment variables.'
        }), 503
    # -------------------------------------------------------

    if 'report_image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part in the request.'}), 400

    file = request.files['report_image']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file.'}), 400

    image_bytes = file.read()
    mime_type = file.mimetype or 'image/jpeg'
    filename = file.filename.lower()

    if filename.endswith('.pdf'):
        return jsonify({'status': 'error', 'message': 'PDF files require conversion to image. Upload PNG/JPG.'}), 400

    if not filename.endswith(('.png', '.jpg', '.jpeg', '.webp')):
        return jsonify({'status': 'error', 'message': 'Unsupported file type.'}), 400

    required_keys = [
        'Age', 'Height', 'Weight', 'BMI', 'Pulse_ratebpm', 'RR_breathsmin',
        'Hbgdl', 'Waist', 'Hip', 'BP_Systolic', 'BP_Diastolic', 'Cycle_RI',
        'Cycle_lengthdays', 'No_of_aborptions', 'beta_HCG_I', 'beta_HCG_II',
        'Follicle_No_L', 'Follicle_No_R', 'Avg_F_size_L', 'Avg_F_size_R',
        'Endometrium', 'FSH', 'LH', 'TSH', 'AMH', 'PRL', 'VitD3', 'PRG', 'RBS'
    ]

    prompt = (
        "You are an expert medical data extractor specializing in PCOS reports. Analyze the provided report image. "
        "Extract the lab values and patient vitals for the keys provided. "
        "Return the output ONLY as a single JSON object. "
        "The keys must match the required list exactly, and all values must be numeric (float or integer). "
        "If a value is not clearly present in the report, omit that key from the JSON object."
        f"\n\nRequired JSON keys: {', '.join(required_keys)}"
    )

    try:
        # 1. Create a Part object for the image
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

        # 2. Call Gemini using current_client and the stable GEMINI_MODEL
        response = retry_gemini_call(
            current_client.models.generate_content,
            model=GEMINI_MODEL,
            contents=[prompt, image_part],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=types.Schema(
                    type=types.Type.OBJECT,
                    properties={key: types.Schema(type=types.Type.NUMBER) for key in required_keys}
                ),
                temperature=0.0
            )
        )

        parsed_data = json.loads(response.text)
        final_data = {k: safe_float(v) for k, v in parsed_data.items()}

        if not final_data:
            return jsonify({
                'status': 'success',
                'message': 'No key values were reliably found in the image.',
                'data': {}
            })

        return jsonify({
            'status': 'success',
            'message': 'Data extracted and ready for auto-fill.',
            'data': final_data
        })

    except APIError as e:
        return jsonify({
            'status': 'error',
            'message': f'Gemini API Error: {e}. Ensure API Key is valid and billing is enabled if required.'
        }), 500
    except Exception as e:
        print(f"Vision Processing Error: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Processing error: {str(e)}'}), 500


# 6. Admin Dashboard
@app.route('/admin')
def admin_dashboard():
    if session.get('role') != 'admin':
        flash("Access denied.", "error")
        return redirect(url_for('hcw_home'))

    conn = get_db_connection()
    doctors = conn.execute("SELECT * FROM users WHERE role = 'doctor'").fetchall()
    conn.close()

    return render_template('admin_dashboard.html', doctors=doctors)


# 7. Add New Doctor
@app.route('/admin/add_doctor', methods=['GET', 'POST'])
def add_doctor():
    if request.method == "GET":
        return redirect(url_for("admin_dashboard"))
    if session.get('role') != 'admin':
        flash("Access denied.", "error")
        return redirect(url_for('hcw_home'))

    name = request.form['name']
    email = request.form['email']
    password = request.form['password']

    conn = get_db_connection()
    try:
        conn.execute("INSERT INTO users (name, email, password, role) VALUES (?, ?, ?, ?)",
                     (name, email, password, 'doctor'))
        conn.commit()
        flash("Doctor added successfully!", "success")
    except sqlite3.IntegrityError:
        flash("Email already exists!", "error")
    finally:
        conn.close()

    return redirect(url_for('admin_dashboard'))
# 8. Doctor Dashboard


@app.route("/doctor")
def doctor_dashboard():
    if session.get("role") != "doctor":
        return redirect(url_for("hcw_home"))

    conn = get_db_connection()
    patients = conn.execute("SELECT * FROM patients").fetchall()
    conn.close()
    return render_template("doctor_dashboard.html", patients=patients)

# 9. ML Prediction Input Form (NEW ROUTE)
@app.route('/doctor/ml_input')
def ml_predict_input():
    if session.get('role') != 'doctor':
        flash("Access denied.", "error")
        return redirect(url_for('hcw_home'))

    # Pass the feature list so the template can generate the fields correctly
    return render_template('ml_input_form.html', features=FINAL_MODEL_FEATURES)


# 10. ML Prediction Route (Doctor/XGBoost Model)
@app.route('/doctor/ml_predict', methods=['GET', 'POST'])
def ml_predict_doctor():
    if request.method == "GET":
        return redirect(url_for("doctor_dashboard"))
    if session.get('role') != 'doctor':
        flash("Access denied.", "error")
        return redirect(url_for('hcw_home'))

    if xgb_model is None or scaler is None:
        return render_template('predict_result.html', error="ML Model dependencies not loaded.")

    try:
        data = request.form
        feature_values = []

        # Collect values in the exact order of FINAL_MODEL_FEATURES
        for feature_name_in_model in FINAL_MODEL_FEATURES:
            # Handle Blood Group Dummies
            if feature_name_in_model.startswith('Blood Group_'):
                blood_group_category = feature_name_in_model.split('_', 1)[1]
                if data.get('blood_group') == blood_group_category:
                    feature_values.append(1.0)
                else:
                    feature_values.append(0.0)
                continue

            # Simplistic mapping from model feature name to form name
            # e.g., ' Age (yrs)' -> 'Ageyrs' (this must match your HTML input names)
            form_name = feature_name_in_model
            # Remove spaces and special characters to map to input names
            form_name = form_name.replace(' ', '').replace('(', '').replace(')', '').replace('.', '').replace('/', '').replace('-', '').strip()

            # Special-case for beta-HCG naming in your FINAL_MODEL_FEATURES
            if 'betaHCG' in form_name or 'IbetaHCG' in form_name or 'IIbetaHCG' in form_name:
                if 'I' in feature_name_in_model and 'II' not in feature_name_in_model:
                    form_name = 'beta_HCG_I'
                elif 'II' in feature_name_in_model:
                    form_name = 'beta_HCG_II'

            value = data.get(form_name, None)
            if value is None:
                # if not present, try fallback using original as-is stripped
                value = data.get(feature_name_in_model.strip(), 0.0)
            feature_values.append(safe_float(value))

        final_features = np.array([feature_values])

        if final_features.shape[1] != len(FINAL_MODEL_FEATURES):
            raise ValueError(
                f"Input features mismatch. Expected {len(FINAL_MODEL_FEATURES)} features but received {final_features.shape[1]}."
            )

        prediction_scaled = scaler.transform(final_features)
        prediction = xgb_model.predict(prediction_scaled)[0]
        result_text = "High Risk" if prediction == 1 else "Low Risk"

        conn = get_db_connection()
        conn.execute("INSERT INTO patients (cycle, weight, acne, risk) VALUES (?, ?, ?, ?)",
                     (data.get('Cycle_length', 'N/A'), data.get('Weight_Kg', 'N/A'),
                      data.get('Pimples', 'N/A'), result_text))
        conn.commit()
        conn.close()

        return render_template('predict_result.html', result=result_text)

    except Exception as e:
        error_message = f"ML Processing Error: {str(e)}. Please ensure the correct number of features are submitted. Expected {len(FINAL_MODEL_FEATURES)} features."
        print(f"Prediction Error: {error_message}")
        return render_template('predict_result.html', error=error_message)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("hcw_home"))

@app.route("/hcw/logout")
def hcw_logout():
    return redirect(url_for("logout"))

# ... (after your existing routes, before if __name__ == "__main__":)

# 11. LLM Chat Assistant Route
@app.route('/api/llm_query', methods=['GET', 'POST'])
def llm_query():
    if request.method == "GET":
        return redirect(url_for("doctor_dashboard"))
    
    if session.get('role') not in ['doctor', 'admin']:
        return jsonify({'status': 'error', 'message': 'Authentication required.'}), 403

    # FIX: Get client inside the route
    current_client = get_gemini_client()
    if current_client is None:
        return jsonify({'status': 'error', 'message': 'Gemini API Key missing.'}), 503

    try:
        data = request.get_json()
        prompt = data.get('prompt')
        if not prompt:
            return jsonify({'status': 'error', 'message': 'No prompt provided.'}), 400

        full_prompt = (
            "You are a helpful and medically knowledgeable AI assistant specializing in PCOS. "
            "Respond to the following doctor's query concisely: "
            f"'{prompt}'"
        )
        
        response = retry_gemini_call(
            current_client.models.generate_content,
            model=GEMINI_MODEL,
            contents=[full_prompt]
        )
        
        return jsonify({'status': 'success', 'llm_response': response.text})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
# ====================================================
#        AGENTIC PCOS CLINICAL REPORT GENERATOR
# ====================================================

@app.route("/doctor/pcos_agent_report", methods=["GET", "POST"])
def pcos_agent_report():
    if request.method == "GET":
        return redirect(url_for("doctor_dashboard"))
    if session.get("role") != "doctor":
        return jsonify({"status": "error", "message": "Access denied."}), 403

    try:
        # 1️⃣ Receive all form data from frontend
        form = request.get_json()
        if not form:
            return jsonify({"status": "error", "message": "No data received."}), 400

        # 2️⃣ Build feature vector for ML model
        features = []
        for feat in FINAL_MODEL_FEATURES:
            if feat.startswith('Blood Group_'):
                bg = feat.split("_", 1)[1]
                features.append(1.0 if form.get("blood_group") == bg else 0.0)
                continue

            key = feat.replace(" ", "").replace("(", "").replace(")", "").replace("/", "").replace("-", "").replace(".", "")
            value = safe_float(form.get(key, 0))
            features.append(value)

        features = np.array([features])

        # 3️⃣ Run ML model
        scaled = scaler.transform(features)
        pred = xgb_model.predict(scaled)[0]
        risk_label = "High PCOS Risk" if pred == 1 else "Low PCOS Risk"
        probability = float(xgb_model.predict_proba(scaled)[0][1])

        # 4️⃣ Pull guideline text (simplified)
        guideline_text = """
        PCOS management generally includes lifestyle modification, cycle regulation, 
        metabolic screening, weight management, thyroid evaluation, and fertility planning.
        """

        # 5️⃣ Use Gemini to create a structured clinical report
        prompt = f"""
You are an expert gynecologist. Generate a detailed and structured clinical summary.

PATIENT VALUES (JSON):
{json.dumps(form, indent=2)}

ML MODEL OUTPUT:
Risk Category: {risk_label}
Probability Score: {probability}

GUIDELINE NOTES:
{guideline_text}

TASK:
Create a clinically formatted report with:
1. One-line summary
2. Risk Assessment (with reasoning)
3. Interpretation referencing PCOS diagnostic criteria
4. Recommendations
5. Follow-up Plan
6. Disclaimer

Write clearly and professionally.
"""

        ai_response = retry_gemini_call(
            client.models.generate_content,
            model=GEMINI_MODEL,
            contents=[prompt]
        )

        return jsonify({
            "status": "success",
            "risk": risk_label,
            "probability": probability,
            "report": ai_response.text
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# ====================================================
#                    MAIN
# ====================================================

if __name__ == "__main__":
    # Use the port assigned by the host, default to 5000 for local testing
    port = int(os.environ.get("PORT", 5000))
    # Never use debug=True in production
    app.run(host="0.0.0.0", port=port, debug=False)
