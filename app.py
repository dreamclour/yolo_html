import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO

# 初始化Flask应用
app = Flask(__name__)

# 配置上传文件夹
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
  return '.' in filename and \
         filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# 加载YOLO模型 (这里使用YOLOv5示例)
# 第一次运行时会自动下载模型
try:
  import torch

  model = YOLO("weights/yolov8s.pt")
  #model = YOLO("weights/yolov5s.pt")
  #model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
except Exception as e:
  print(f"加载模型时出错: {e}")
  model = None


@app.route('/')
def index():
  return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
  if 'file' not in request.files:
    return jsonify({'error': '没有文件部分'}), 400

  file = request.files['file']

  if file.filename == '':
    return jsonify({'error': '没有选择文件'}), 400

  if file and allowed_file(file.filename):
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # 使用YOLO进行检测
    if model is not None:
      # 读取图像
      img = Image.open(filepath)

      # 进行检测
      results = model(img)

      # 渲染检测结果
      #results.render()  # 在原图上绘制检测框
      result = results[0] # 绘制检测框
      result_im = result.plot()  # 返回带标注的 BGR numpy 数组

      # 保存带检测结果的图像
      output_filename = f"detected_{filename}"
      output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
      #Image.fromarray(results.ims[0]).save(output_path)
      cv2.imwrite(output_path, result_im)


      # 获取检测结果信息
      # detections = []
      # for *xyxy, conf, cls in results.xyxy[0]:
      #   detections.append({
      #     'class': results.names[int(cls)],
      #     'confidence': float(conf),
      #     'bbox': [float(x) for x in xyxy]
      #   })

      # 获取检测结果信息
      detections = []
      for box in results[0].boxes:
        detections.append({
          "class": model.names[int(box.cls)],
          "confidence": float(box.conf),
          "bbox": box.xyxy.tolist()[0]  # [x1, y1, x2, y2]
        })

      return jsonify({
        'original': filename,
        'detected': output_filename,
        'results': detections
      })
    else:
      return jsonify({'error': '模型未加载'}), 500

  return jsonify({'error': '不允许的文件类型'}), 400


if __name__ == '__main__':
  app.run(debug=True)