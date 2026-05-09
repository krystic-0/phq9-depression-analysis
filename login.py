import streamlit as st
import streamlit.components.v1 as components
import json
import os


def _inject_js(code):
    """通过 iframe 注入 JS，避免 DOM removeChild 冲突"""
    components.html(f"<script>{code}</script>", height=0)

# 页面配置 - 放在最前面
st.set_page_config(
    page_title="登录 - 抑郁症分析系统",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 自定义样式
st.markdown("""
<style>
    /* 页面背景 */
    body { background: linear-gradient(135deg, #F5F0FF 0%, #E8DCF8 30%, #DCD0F0 60%, #EDE0F8 100%); background-attachment: fixed; }
    .stApp { background: linear-gradient(135deg, #F5F0FF 0%, #E8DCF8 30%, #DCD0F0 60%, #EDE0F8 100%); background-attachment: fixed; }

    /* 隐藏默认的streamlit菜单 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 登录容器样式 */
    .login-container {
        background-color: white;
        padding: 40px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        max-width: 400px;
        margin: 50px auto;
    }
    
    /* 标题样式 */
    .login-title {
        color: #333333;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 30px;
    }
    
    /* 系统名称 */
    .system-name {
        color: #9370DB;
        text-align: center;
        font-size: 18px;
        margin-bottom: 40px;
        font-weight: 500;
    }
    
    /* 输入框样式 */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #E0E0E0;
        padding: 12px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #9370DB;
    }
    
    /* 按钮样式 */
    .stButton > button {
        background-color: #9370DB !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-size: 16px !important;
        font-weight: 500 !important;
        width: 100%;
        border: none !important;
    }
    
    .stButton > button:hover {
        background-color: #7B68EE !important;
    }
    
    /* 错误提示样式 */
    .error-message {
        background-color: #FFE6E6;
        color: #CC0000;
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
        text-align: center;
    }
    
    /* 成功提示样式 */
    .success-message {
        background-color: #E6FFE6;
        color: #006600;
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def load_users():
    """加载用户数据"""
    users_file = "users.json"
    if os.path.exists(users_file):
        with open(users_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"admin": "admin123"}  # 默认用户

def verify_login(username, password):
    """验证用户名和密码"""
    users = load_users()
    return username in users and users[username] == password

def main():
    """主函数"""
    # 检查是否已经登录
    if st.session_state.get('logged_in', False):
        # 如果已经登录，跳转到主应用
        st.switch_page("app.py")
    
    # 登录界面
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    
    # 系统标题
    st.markdown('<div class="system-name">基于Python的抑郁症多维度心理特征分析与可视化系统设计与实现</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-title">🔐 用户登录</div>', unsafe_allow_html=True)
    
    # 登录表单
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("用户名", placeholder="请输入用户名", key="login_username")
        password = st.text_input("密码", type="password", placeholder="请输入密码", key="login_password")
        
        submitted = st.form_submit_button("登录")
        
        if submitted:
            if not username or not password:
                st.markdown('<div class="error-message">⚠️ 请输入用户名和密码</div>', unsafe_allow_html=True)
            elif verify_login(username, password):
                # 登录成功
                st.session_state.logged_in = True
                st.session_state.username = username
                
                # 保存到localStorage
                _inject_js(f'localStorage.setItem("isLoggedIn","true");localStorage.setItem("username","{username}");window.location.href="/";')
                st.markdown('<div class="success-message">✅ 登录成功，正在跳转...</div>', unsafe_allow_html=True)
                st.stop()
            else:
                st.markdown('<div class="error-message">❌ 用户名或密码错误</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 底部版权信息
    st.markdown("""
    <div style="text-align: center; margin-top: 50px; color: #666666; font-size: 12px;">
        © 2024 抑郁症多维度心理特征分析系统 | 版本 1.0
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
