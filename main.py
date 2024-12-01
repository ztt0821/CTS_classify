from flask import Flask, render_template, redirect, url_for, request, jsonify
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import timedelta
import os
from datetime import datetime
from demo import helper, get_model

# 初始化 Flask 和 Flask-Login
app = Flask(__name__, template_folder='./templates', static_folder='./static')
app.secret_key = '123456'  # 用于会话加密
# 设置会话过期时间（例如 30 分钟）
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)
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

@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist("file")
    if not files:
        return jsonify({"error": "没有上传文件"}), 400
    # 判断是否是多个文件，且是否每个文件都是.dcm，或者只有一个文件且是.nii.gz
    if len(files) > 1:
        if not all(file.filename.endswith('.dcm') for file in files):
            return jsonify({"error": "如果上传多个文件必须都得是.dcm的文件"}), 400
    elif len(files) == 1:
        if not files[0].filename.endswith('.nii.gz'):
            return jsonify({"error": "如果只上传一个文件必须是.nii.gz的文件"}), 400
    # 符合条件，继续后续逻辑
    if len(files) == 1:
        file = files[0]
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
    else:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], datetime.now().strftime('%Y-%m-%d-%H-%M-%S-') + 'dcm')
        os.makedirs(file_path, exist_ok=True)
        for file in files:
            file.save(os.path.join(file_path, file.filename))
    try:
        _, className = helper(model, ['--resume_path', os.path.join(app.static_folder, 'resources', 'model_weight/save_96.pth'),
                        '--image_path', file_path])
        result = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' ' + className + '\n'
        with open(os.path.join(app.static_folder, 'resources', 'results.txt'), "a") as f:
            f.write(result)
        # 删除上传的文件
        if len(files) == 1:
            os.remove(file_path)
        else:
            for file in files:
                os.remove(os.path.join(file_path, file.filename))
            os.rmdir(file_path)
        return jsonify({"message": "Files accepted", "result": result}), 200
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

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

