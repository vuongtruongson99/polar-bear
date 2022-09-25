from flask import Flask, render_template, flash, request, url_for, redirect, current_app
import numpy as np
import os
from PIL import Image
import torch
from bbox.utils import voc2coco, draw_bboxes, load_image
import config as config

CKPT_PATH = './best.pt'
IMG_SIZE  = 1000
CONF      = 0.1
IOU       = 0.2
AUGMENT   = True

def get_bbox(annots):
    bboxes = [list(annot.values()) for annot in annots]
    return bboxes

def predict(model, img, size=768, augment=False):
    height, width = img.shape[:2]
    results = model(img, size=size, augment=augment)  # custom inference size
    preds   = results.pandas().xyxy[0]
    bboxes  = preds[['xmin','ymin','xmax','ymax']].values
    if len(bboxes):
        bboxes  = voc2coco(bboxes,height,width).astype(int)
        confs   = preds.confidence.values
        return bboxes, confs
    else:
        return [],[]

def show_img(img, bboxes, bbox_format='yolo'):
    names  = ['bear']*len(bboxes)
    labels = [0]*len(bboxes)
    img    = draw_bboxes(img = img,
                           bboxes = bboxes, 
                           classes = names,
                           class_ids = labels,
                           class_name = True, 
                           colors = colors, 
                           bbox_format = bbox_format,
                           line_thickness = 2)
    
    return Image.fromarray(img)

np.random.seed(32)
colors = [(np.random.randint(255), np.random.randint(255), np.random.randint(255))\
          for idx in range(1)]

def load_model(ckpt_path, conf=0.25, iou=0.5):
    model = torch.hub.load('./yolov5',
                           'custom',
                           path=ckpt_path,
                           source='local',
                           force_reload=True)
    model.conf = conf
    model.iou = iou
    model.classes = None
    model.multi_label = True
    model.max_det = 1000
    model.eval()
    return model

app = Flask(__name__, static_folder='static')

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

@app.after_request
def add_header(response):
    response.cache_control.max_age = 20
    return response

def allowed_file(filename):
    """
    Процесс проверки того, загрузил ли пользователь действительный файл или нет
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/image', methods=['POST'])
def upload_image():
    """
    Обработка загрузки фотографий пользователей
    """
    uploaded_files = request.files.getlist("upload_imgs[]")
    out_file = []

    for file in uploaded_files:
        if file.filename == '':
            flash('Не выбрано изображение для загрузки')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            out_file.append(filename)

            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #print('upload_image filename: ' + filename)
            flash('Изображение успешно загружено')

            img = np.array(Image.open(app.config["UPLOAD_FOLDER"] + filename).convert('RGB'))
            
            bboxes, confis = predict(model, img, size=IMG_SIZE, augment=AUGMENT)

            predicted_img = show_img(img, bboxes, bbox_format='coco')

            predicted_img.save(app.config["UPLOAD_FOLDER"] + filename)

        else:
            flash('Допустимые типы изображений: - png, jpg, jpeg')
            return redirect(request.url)
    
    return render_template('3rd_section.html', filenames=out_file, uploadfolder=UPLOAD_FOLDER)

@app.route('/display/<filename>')
def display_image(filename):
    """
    Вывод изображения после его обработки
    """
    print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/')
def index():
    """
    Получить основной раздел сайта
    """
    return render_template('1st_section.html')

@app.route('/image')
def image():
    """
    Получить вкладку для загрузки изображения
    """
    return render_template("3rd_section.html", percent=0)

@app.route('/teams')
def teams():
    """
    Получите вкладку, чтобы увидеть всю команду разработчиков
    """
    return render_template("4th_section.html")   


if __name__ == '__main__':
    global model
    model = load_model(CKPT_PATH, conf=CONF, iou=IOU)
    port = int(os.environ.get('PORT', 5000)) #Define port so we can map container port to localhost
    app.run(host='0.0.0.0') 
    