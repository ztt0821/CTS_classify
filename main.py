from flask import Flask, render_template, redirect, url_for, request, jsonify, make_response
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import timedelta
import os, json
from datetime import datetime
from demo import helper, get_model
from collections import defaultdict
import shutil
from openpyxl import Workbook
from io import BytesIO

# 初始化 Flask 和 Flask-Login
app = Flask(__name__, template_folder='./templates', static_folder='./static')
app.config['MAX_CONTENT_LENGTH'] = 512 * 1024 * 1024  # 限制上传文件大小为 5MB
app.secret_key = '123456'  # 用于会话加密
# 设置会话过期时间（例如 30 分钟）
# app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)
login_manager = LoginManager()
login_manager.init_app(app)
model, users = None, None
UPLOAD_FOLDER = os.path.join(app.static_folder, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# 用户模型示例
class User:
    def __init__(self, username, password, active=True):
        self.username = username
        self.password = generate_password_hash(password)  # 通常存储的是加密后的密码
        self.active = active  # 默认用户是活跃状态
    @property
    def is_authenticated(self):
        return True  # 根据实际情况返回是否认证成功
    @property
    def is_active(self):
        return self.active  # 如果用户活跃，则返回 True
    @property
    def is_anonymous(self):
        return False  # 系统中通常没有匿名用户
    def get_id(self):
        return self.username  # 返回用户的唯一标识符
    def check_password(self, password):
        return check_password_hash(self.password, password)


# 在 Flask 启动时加载模型
@app.before_request
def init_model():
    global model
    if model is None:
        model = get_model(['--resume_path', os.path.join(app.static_folder, 'resources', 'model_weight/save_96.pth'), '--image_path', '-1'])
    global users
    if users is None:
        account_file_path = os.path.join(app.static_folder, 'resources', 'account.txt')
        with open(account_file_path, "r") as f:
            lines = f.readlines()
            users = {}
            for line in lines:
                username, password = map(str, line.strip().split())
                print("账号", username, "密码", password)
                users[username] = User(username, password)

# 加载用户
@login_manager.user_loader
def load_user(username):
    return users.get(username)

# 登录页面
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = users.get(username)
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('detect'))  # 登录成功后跳转到检测页面
        else:
            return '登录失败，用户名或密码错误！'
    
    return render_template('pages/index.html')

# 自定义未授权处理
@login_manager.unauthorized_handler
def unauthorized():
    response = jsonify({"error": "未授权，请登录"})
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response

# 检测页面（登录后才能访问）
@app.route('/detect')
@login_required
def detect():
    return render_template('pages/classify.html')  # 假设这是检测页面

# def parseDCMDirectory(files):
#     # 一共四层目录，第一层为病人名称，第二层为病人序号，第三层为固定名称US，第四层为图像序号，第四层的目录下面都为.dcm文件
#     # 例如：./static/uploads/病人1/1/US/1/1.dcm
#     # save到指定的目录下面，返回所有的第四层文件序号的列表
#     dcm_paths = defaultdict(str)
#     for file in files:
#         file_names = file.filename.split('/')[1:]
#         print(file_names)
#         if len(file_names) != 5 or not file_names[-1].endswith('.dcm'):
#             continue
#         patient_name = file_names[0]
#         patient_id = file_names[1]
#         us_name = file_names[2]
#         image_id = file_names[3]
#         us_dir = os.path.join(app.config['UPLOAD_FOLDER'], patient_name, patient_id, us_name, image_id)
#         os.makedirs(us_dir, exist_ok=True)
#         file.save(os.path.join(us_dir, file_names[-1]))
#         # dcm_paths.add(us_dir)
#         dcm_paths['/'.join(file_names[:-1])] = us_dir
#     return dcm_paths

def parseDCMDirectory(parent_path, files):
    # 一共四层目录，第一层为病人名称，第二层为病人序号，第三层为固定名称US，第四层为图像序号，第四层的目录下面都为.dcm文件
    # 例如：./static/uploads/病人1/1/US/1/1.dcm
    # save到指定的目录下面，返回所有的第四层文件序号的列表
    dcm_paths = defaultdict(str)
    for file in files:
        file_names = file.split('/')[1:]
        print(file_names)
        if len(file_names) != 5 or not file_names[-1].endswith('.dcm'):
            continue
        dcm_paths['/'.join(file_names[:-1])] = os.path.join(parent_path, *file_names[:-1])
    return dcm_paths

@app.route('/upload', methods=['POST'])
def upload():
    parent_path = request.form["parent_path"]
    filenames = json.loads(request.form["filenames"])
    print(parent_path, filenames)
    dcm_paths = parseDCMDirectory(parent_path, filenames)
    result = {}
    try:
        for name, dcm_path in dcm_paths.items():
            _, className = helper(model, ['--resume_path', os.path.join(app.static_folder, 'resources', 'model_weight/save_96.pth'),
                        '--image_path', dcm_path])
            paitent_name = name.split('/')[0]
            seq_name = name.split('/')[-1]
            patient_id = name.split('/')[1]
            if paitent_name not in result:
                result[paitent_name] = {
                    "name": paitent_name,
                    "paient_id": patient_id,
                    "res": className,
                    "ok": True,
                    "seq": [{
                        "name": seq_name,
                        "class": className
                    }]
                }
            else:
                result[paitent_name]["ok"] &= className == result[paitent_name]["res"]
                result[paitent_name]["seq"].append({
                    "name": seq_name,
                    "class": className
                })
        for dirName in os.listdir(app.config['UPLOAD_FOLDER']):
            shutil.rmtree(os.path.join(app.config['UPLOAD_FOLDER'], dirName))  
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500
    return jsonify({"message": "Files accepted", "result": list(result.values())}), 200

# @app.route('/upload', methods=['POST'])
# def upload():
#     files = request.files.getlist("file")
#     print(files)
#     if not files:
#         return jsonify({"error": "没有上传文件"}), 400
#     # 判断是否是多个文件，且是否每个文件都是.dcm，或者只有一个文件且是.nii.gz
#     if len(files) > 1:
#         if not all(file.filename.endswith('.dcm') for file in files):
#             return jsonify({"error": "如果上传多个文件必须都得是.dcm的文件"}), 400
#     elif len(files) == 1:
#         if not files[0].filename.endswith('.nii.gz'):
#             return jsonify({"error": "如果只上传一个文件必须是.nii.gz的文件"}), 400
#     # 符合条件，继续后续逻辑
#     if len(files) == 1:
#         file = files[0]
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#         file.save(file_path)
#         try:
#             _, className = helper(model, ['--resume_path', os.path.join(app.static_folder, 'resources', 'model_weight/save_96.pth'),
#                             '--image_path', file_path])
#             result = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' ' + className + '\n'
#             with open(os.path.join(app.static_folder, 'resources', 'results.txt'), "a") as f:
#                 f.write(result)
#             # 删除上传的文件
#             os.remove(file_path)
#         except Exception as e:
#             print(e)
#             return jsonify({"error": str(e)}), 500
#     else:
#         # file_path = os.path.join(app.config['UPLOAD_FOLDER'], datetime.now().strftime('%Y-%m-%d-%H-%M-%S-') + 'dcm')
#         # os.makedirs(file_path, exist_ok=True)
#         # for file in files:
#         #     file.save(os.path.join(file_path, file.filename))
#         dcm_paths = parseDCMDirectory(files)
#         result = {}
#         try:
#             for name, dcm_path in dcm_paths.items():
#                 _, className = helper(model, ['--resume_path', os.path.join(app.static_folder, 'resources', 'model_weight/save_96.pth'),
#                             '--image_path', dcm_path])
#                 paitent_name = name.split('/')[0]
#                 seq_name = name.split('/')[-1]
#                 patient_id = name.split('/')[1]
#                 if paitent_name not in result:
#                     result[paitent_name] = {
#                         "name": paitent_name,
#                         "paient_id": patient_id,
#                         "res": className,
#                         "ok": True,
#                         "seq": [{
#                             "name": seq_name,
#                             "class": className
#                         }]
#                     }
#                 else:
#                     result[paitent_name]["ok"] &= className == result[paitent_name]["res"]
#                     result[paitent_name]["seq"].append({
#                         "name": seq_name,
#                         "class": className
#                     })
#             for dirName in os.listdir(app.config['UPLOAD_FOLDER']):
#                 shutil.rmtree(os.path.join(app.config['UPLOAD_FOLDER'], dirName))  

#         except Exception as e:
#             print(e)
#             return jsonify({"error": str(e)}), 500
        
#         return jsonify({"message": "Files accepted", "result": list(result.values())}), 200

@app.route('/download', methods=['POST'])
@login_required
def submit_form():
    form_data = request.form.to_dict()
    wb = Workbook()
    ws = wb.active
    headers = list(json.loads(form_data["headers"]))
    ws.append(headers)
    data = list(json.loads(form_data["data"]))
    for v in data:
        ws.append(list(json.loads(v)))
    excel_buffer = BytesIO()
    wb.save(excel_buffer)
    excel_buffer.seek(0)
    response = make_response(excel_buffer.getvalue())
    now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    response.headers['Content-Disposition'] = f"attachment; filename={now}.xlsx"
    response.headers['Content-Type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    return response

# 记录页面（登录后才能访问）
@app.route('/api/get_results')
@login_required
def get_results():
    file_path = os.path.join(app.static_folder, 'resources', 'results.txt')
    with open(file_path, "r") as f:
        lines = f.readlines()
        results = []
        for line in lines:
            results.append(line)
        results = ''.join(results[::-1])
    # 直接返回data
    return jsonify({"log": results}), 200

# 注销
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))  # 注销后跳转回登录页面

if __name__ == '__main__':
    app.run(debug=True)

