import os
import uuid
import ollama  # חיווט ישיר ומקצועי ל-API המקומי במק מיני
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename  # הגנה על שמות קבצים מפני תווים בעייתיים

# הגרת בסיס - פרויקט supervisor_AI_Image_analysis
app = Flask(__name__, static_folder='.', static_url_path='/')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME = "gemma4:e4b"

# תיקייה זמנית לשמירת התמונה שהמפקח מעלה
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def run_ollama_vision_api(prompt, system_prompt, image_path):
    """פנייה ישירה ל-API הרשמי של אולמה - מנקה את ה-Thinking ומאיצה את הביצועים"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt, "images": [image_path]}  # העברת התמונה בצורה מובנית
    ]

    try:
        # ה-API מדבר ישירות עם השרת שכבר טעון בזיכרון של ה-M4 - טס בפחות מ-30 שניות
        response = ollama.chat(model=MODEL_NAME, messages=messages)
        return response['message']['content'].strip()
    except Exception as e:
        return f"שגיאה בתקשורת מול מנוע ה-AI: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({"error": "לא הועלתה תמונה מהשטח"}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "קובץ לא תקין או ריק"}), 400

    # ניקוי שם הקובץ והוספת ה-UUID הייחודי שלך
    safe_name = secure_filename(file.filename)
    unique_filename = f"audit_{uuid.uuid4().hex[:6]}_{safe_name}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
    
    try:
        # שמירה זמנית של הקובץ במק מיני
        file.save(filepath)

        # ה-System Prompt המעולה שלך
        system_instruction = (
            "You are a helpful, professional, and very friendly AI assistant for a construction site supervisor in Israel. "
            "Analyze the provided image with attention to visible elements, missing requirements, and infrastructure. "
            "CRITICAL DIRECTIONS FOR OUTPUT:\n"
            "1. Strictly prohibited to include any 'Thinking', chain-of-thought, or internal reasoning in English or Hebrew.\n"
            "2. Write the ENTIRE output in fluent, natural Hebrew.\n"
            "3. Use an encouraging, constructive, and gentle tone. Avoid alarming or threatening language (do NOT say 'stop work immediately' or 'critical failure'). Instead, use phrases like 'כדאי לתת תשומת לב', 'מומלץ לתגבר', 'למען השקט הנפשי שלך'.\n"
            "4. Structure the response beautifully using HTML bullet points (<ul> and <li> tags) for clean reading."
        )

        user_prompt = (
            "Identify the main subjects and hazards in this specific image. "
            "1. Start by describing exactly what the people in the image are doing right now. "
            "2. List 3 specific construction elements visible in the background or foreground. "
            "3. Provide safety and QA recommendations based ONLY on these observed elements. "
            "Do not use generic site safety phrases unless they directly apply to what is visible."
        )
        
        # הרצה מול ה-API המקומי
        analysis_res = run_ollama_vision_api(user_prompt, system_instruction, filepath)
        return jsonify({"analysis": analysis_res})

    except Exception as e:
        return jsonify({"error": f"שגיאה פנימית בשרת: {str(e)}"}), 500

    finally:
        # הבטחת מחיקת הקובץ מה-Mac mini בכל מצב (הצלחה או שגיאה)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass

if __name__ == '__main__':
    # רץ על פורט 5000 כמבוקש
    app.run(host='0.0.0.0', port=5000, debug=True)