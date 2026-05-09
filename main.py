import streamlit as st
import streamlit.components.v1 as components
import json
import sys
import os

# 将当前目录添加到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def _inject_js(code):
    """通过 iframe 注入 JS，避免 DOM removeChild 冲突"""
    components.html(f"<script>{code}</script>", height=0)

# 设置页面配置
st.set_page_config(
    page_title="抑郁症多维度心理特征分析系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化 session_state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None

# 用户数据文件路径
users_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'users.json')

def show_login_page():
    """显示登录页面"""
    # 自定义样式 - 不隐藏侧边栏
    st.markdown("""
    <style>
        body { background: linear-gradient(135deg, #F5F0FF 0%, #E8DCF8 30%, #DCD0F0 60%, #EDE0F8 100%); background-attachment: fixed; }
        .stApp { background: linear-gradient(135deg, #F5F0FF 0%, #E8DCF8 30%, #DCD0F0 60%, #EDE0F8 100%); background-attachment: fixed; }
        h1 { text-align: center; color: #9370DB !important; margin-bottom: 30px; }
        .stButton > button {
            background-color: #9370DB !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 10px 20px !important;
            width: 100%;
            font-size: 16px !important;
        }
        .stButton > button:hover { background-color: #7B68EE !important; }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1>用户登录</h1>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["登录", "注册"])
    
    with tab1:
        st.subheader("欢迎回来")
        login_username = st.text_input("用户名", key="login_username", placeholder="请输入用户名")
        login_password = st.text_input("密码", type="password", key="login_password", placeholder="请输入密码")
        
        if st.button("登录", key="main_login_btn"):
            if not login_username or not login_password:
                st.error("用户名和密码不能为空")
            else:
                try:
                    with open(users_path, 'r', encoding='utf-8') as f:
                        users = json.load(f)
                except:
                    users = {}
                
                if login_username not in users:
                    st.error("用户名不存在，请先注册")
                elif users[login_username] != login_password:
                    st.error("密码错误")
                else:
                    st.session_state.logged_in = True
                    st.session_state.username = login_username
                    # 正确设置localStorage
                    _inject_js(f'localStorage.setItem("isLoggedIn","true");localStorage.setItem("username","{login_username}");')
                    st.success("登录成功！正在进入系统...")
                    # 直接进入主页面
                    st.rerun()
    
    with tab2:
        st.subheader("创建新账户")
        reg_username = st.text_input("用户名", key="reg_username", placeholder="请输入用户名")
        reg_password = st.text_input("密码", type="password", key="reg_password", placeholder="请输入密码")
        reg_password_confirm = st.text_input("确认密码", type="password", key="reg_password_confirm", placeholder="请再次输入密码")
        
        if st.button("注册", key="main_register_btn"):
            if not reg_username or not reg_password:
                st.error("用户名和密码不能为空")
            elif reg_password != reg_password_confirm:
                st.error("两次输入的密码不一致")
            else:
                try:
                    with open(users_path, 'r', encoding='utf-8') as f:
                        users = json.load(f)
                except:
                    users = {}
                
                if reg_username in users:
                    st.error("用户名已被注册")
                else:
                    users[reg_username] = reg_password
                    with open(users_path, 'w', encoding='utf-8') as f:
                        json.dump(users, f, ensure_ascii=False, indent=2)
                    
                    st.session_state.logged_in = True
                    st.session_state.username = reg_username
                    # 正确设置localStorage
                    _inject_js(f'localStorage.setItem("isLoggedIn","true");localStorage.setItem("username","{reg_username}");')
                    st.success("注册成功！正在进入系统...")
                    # 直接进入主页面
                    st.rerun()
    
    st.markdown("---")
    st.info("💡 默认测试账号：用户名 admin / 密码 admin123")

def show_main_page():
    """显示主页面 - 使用模块导入方式替代exec()以提高性能"""
    # 确保 session_state 已设置
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    
    try:
        # 直接导入app模块并调用main函数
        # 这样可以利用Python的模块缓存机制，提高性能
        import app
        app.main()
        
    except Exception as e:
        st.error(f"加载主页面失败: {e}")
        import traceback
        st.error(traceback.format_exc())
        st.info("请刷新页面重试")

# 主逻辑
if not st.session_state.logged_in:
    show_login_page()
    st.stop()

# 登录成功后执行主页面
show_main_page()
