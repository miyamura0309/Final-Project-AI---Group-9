import os
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import predict

app = Flask(__name__)
CORS(app)  # Cho phép web game (frontend) gọi API

# Thư mục lưu file WAV tạm thời
UPLOAD_FOLDER = 'temp_audio'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/health', methods=['GET'])
def health():
    """Kiểm tra server còn sống không."""
    return jsonify({"status": "ok", "message": "VoiceMaze API đang chạy!"})

@app.route('/predict', methods=['POST'])
def predict_route():
    """
    Nhận file WAV, trả về kết quả nhận diện.
    Request : multipart/form-data với field 'audio' là file WAV
    Response: JSON {"label": "up", "confidence": 99.9, "probabilities": {...}}
    """
    # Kiểm tra có file gửi lên không
    if 'audio' not in request.files:
        return jsonify({"error": "Không tìm thấy file audio"}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "File rỗng"}), 400

    # Lưu file tạm với tên ngẫu nhiên để tránh trùng
    temp_filename = f"{uuid.uuid4().hex}.wav"
    temp_path     = os.path.join(UPLOAD_FOLDER, temp_filename)

    try:
        file.save(temp_path)
        result = predict(temp_path)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Xóa file tạm sau khi xử lý xong
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == '__main__':
    print("🚀 VoiceMaze API đang khởi động...")
    print("   URL: http://localhost:5000")
    print("   Health check: http://localhost:5000/health")
    app.run(host='0.0.0.0', port=5000, debug=False)
