import sys
from pathlib import Path
from flask import Flask, request, jsonify, render_template


# root dir
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from src.inference import predict_iris


# init app flask
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """API Endpoint nhận dữ liệu và trả về kết quả dự đoán"""
    try:
        # Lấy dữ liệu từ form HTML hoặc từ API (JSON)
        if request.is_json:
            data = request.get_json()
            features = data.get("features", [])
        else:
            # Lấy dữ liệu từ thẻ <form> trong HTML
            features = [
                float(request.form.get("sepal_length", 0)),
                float(request.form.get("sepal_width", 0)),
                float(request.form.get("petal_length", 0)),
                float(request.form.get("petal_width", 0)),
            ]

        # check input
        if len(features) != 4:
            return jsonify({"ERROR": "Vui lòng nhập đầy đủ 4 thông số."}), 400

        # Gọi hàm inference
        result = predict_iris(features)

        # Xử lý nếu inference trả về lỗi (như bạn đã cấu hình trong inference.py)
        if "ERROR" in result:
            return jsonify(result), 500

        # Trả về kết quả thành công
        return jsonify(result)

    except ValueError:
        return jsonify({"ERROR": "Dữ liệu đầu vào phải là số hợp lệ."}), 400
    except Exception as e:
        return jsonify({"ERROR": f"Lỗi server: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
