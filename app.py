import streamlit as st
import cv2
import numpy as np
import time
import os  # 新增：用於檔案管理
from PIL import Image, ImageDraw, ImageFont

# =================================================================
# 1. 系統核心配置
# =================================================================
st.set_page_config(
    page_title="Lip Expert V30 - 永久存檔版",
    page_icon="🗣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 定義影片存檔資料夾
VIDEO_SAVE_DIR = "user_videos"
if not os.path.exists(VIDEO_SAVE_DIR):
    os.makedirs(VIDEO_SAVE_DIR)  # 如果資料夾不存在，就建立一個

@st.cache_resource
def get_mp_tools():
    try:
        import mediapipe as mp
        return {
            "mesh": mp.solutions.face_mesh,
            "draw": mp.solutions.drawing_utils,
            "styles": mp.solutions.drawing_styles
        }
    except Exception as e:
        st.error(f"MediaPipe 啟動失敗：{e}")
        return None

MP_TOOLS = get_mp_tools()

# =================================================================
# 2. 課程資料庫
# =================================================================
COURSE_DATA = {
    "🟢 完整注音符號 (37音)": [
        "ㄅ", "ㄆ", "ㄇ", "ㄈ", "ㄉ", "ㄊ", "ㄋ", "ㄌ",
        "ㄍ", "ㄎ", "ㄏ", 
        "ㄐ", "ㄑ", "ㄒ",
        "ㄓ", "ㄔ", "ㄕ", "ㄖ",
        "ㄗ", "ㄘ", "ㄙ",
        "ㄧ", "ㄨ", "ㄩ",
        "ㄚ", "ㄛ", "ㄜ", "ㄝ", "ㄞ", "ㄟ", "ㄠ", "ㄡ",
        "ㄢ", "ㄣ", "ㄤ", "ㄥ", "ㄦ"
    ],
    "👋 日常問候與禮貌": [
        "你好", "早安", "晚安", "謝謝", "不客氣", 
        "對不起", "沒關係", "再見", "拜拜", "請問"
    ],
    "🗣️ 表達需求": [
        "我要", "不要", "好", "不好", 
        "肚子餓", "口渴", "喝水", "吃飯", 
        "上廁所", "想睡覺", "痛", "不舒服", "幫忙"
    ],
    "👨‍👩‍👧‍👦 家庭與稱謂": [
        "爸爸", "媽媽", "爺爺", "奶奶", 
        "哥哥", "姊姊", "弟弟", "妹妹", 
        "老師", "醫生", "護士", "我自己"
    ],
    "🔢 數字與數量": [
        "一", "二", "三", "四", "五", 
        "六", "七", "八", "九", "十", 
        "一百", "多少錢", "幾個", "一點點"
    ],
    "🏥 口腔復健動作": [
        "大張嘴 (啊)", "用力抿嘴 (一)", "圓唇嘟嘴 (嗚)", 
        "鼓腮 (像青蛙)", "左右撇嘴", "舌頭向上舔", "舌頭向下伸"
    ]
}

# =================================================================
# 3. 唇形分析核心演算法 (V29: 寬容評分 + 進度追蹤)
# =================================================================
class LipAnalyzer:
    
    @staticmethod
    def get_drawing_specs():
        import mediapipe as mp
        landmark_spec = mp.solutions.drawing_utils.DrawingSpec(
            color=(255, 255, 255), thickness=1, circle_radius=1
        )
        connection_spec = mp.solutions.drawing_utils.DrawingSpec(
            color=(255, 255, 255), thickness=1
        )
        return landmark_spec, connection_spec

    @staticmethod
    def get_mar(landmarks):
        p13 = np.array([landmarks[13].x, landmarks[13].y])
        p14 = np.array([landmarks[14].x, landmarks[14].y])
        p78 = np.array([landmarks[78].x, landmarks[78].y])
        p308 = np.array([landmarks[308].x, landmarks[308].y])
        v_dist = np.linalg.norm(p13 - p14)
        h_dist = np.linalg.norm(p78 - p308)
        return v_dist / (h_dist + 1e-6)

    @staticmethod
    def get_lip_shape_vector(landmarks):
        indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
        vector = []
        origin_x = (landmarks[78].x + landmarks[308].x) / 2
        origin_y = (landmarks[78].y + landmarks[308].y) / 2
        scale = np.linalg.norm(np.array([landmarks[78].x, landmarks[78].y]) - 
                               np.array([landmarks[308].x, landmarks[308].y])) + 1e-6

        for idx in indices:
            vector.append((landmarks[idx].x - origin_x) / scale)
            vector.append((landmarks[idx].y - origin_y) / scale)
        return np.array(vector)

    @staticmethod
    def calculate_lenient_score(std_vec, cur_vec, std_mar, cur_mar):
        dist = np.linalg.norm(std_vec - cur_vec)
        shape_score = max(0, 100 - (dist * 120)) 
        mar_diff = abs(std_mar - cur_mar)
        open_score = max(0, 100 - (mar_diff * 150))
        final_score = (shape_score * 0.6) + (open_score * 0.4)
        if final_score < 60 and final_score > 30:
            final_score += 20 
        return min(100, final_score)

    @staticmethod
    def analyze_video_sequence(video_path):
        cap = cv2.VideoCapture(video_path)
        sequence_vectors = []
        sequence_mars = []
        
        with MP_TOOLS["mesh"].FaceMesh(refine_landmarks=True, max_num_faces=1) as face_mesh:
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                frame_count += 1
                if frame_count % 3 != 0: continue 

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    vec = LipAnalyzer.get_lip_shape_vector(landmarks)
                    mar = LipAnalyzer.get_mar(landmarks)
                    sequence_vectors.append(vec)
                    sequence_mars.append(mar)
        
        cap.release()
        if sequence_vectors:
            return np.array(sequence_mars), np.array(sequence_vectors)
        return None, None

# =================================================================
# 4. 狀態管理
# =================================================================
def init_session_state():
    defaults = {
        'is_practice_mode': False,
        'standard_models': {},
        'uploaded_videos': {},
        'camera_index': 0,
        'current_category': list(COURSE_DATA.keys())[0],
        'current_word': COURSE_DATA[list(COURSE_DATA.keys())[0]][0],
        'smooth_score': 0.0,
        'progress_index': 0,
        'last_match_time': 0
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

# =================================================================
# 5. UI 繪圖 (含進度條)
# =================================================================
def draw_ui_overlay(frame, text_list, progress=0.0, color=(0, 255, 255)):
    h, w, _ = frame.shape
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("msjh.ttc", 36)
    except:
        font = ImageFont.load_default()
    
    for i, text in enumerate(text_list):
        x, y = 30, 30 + i*50
        draw.text((x-1, y), text, font=font, fill=(0,0,0))
        draw.text((x+1, y), text, font=font, fill=(0,0,0))
        draw.text((x, y-1), text, font=font, fill=(0,0,0))
        draw.text((x, y+1), text, font=font, fill=(0,0,0))
        draw.text((x, y), text, font=font, fill=color)
    
    bar_x, bar_y = 30, 10
    bar_w, bar_h = w - 60, 15
    draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], fill=(50, 50, 50))
    fill_w = int(bar_w * progress)
    draw.rectangle([bar_x, bar_y, bar_x + fill_w, bar_y + bar_h], fill=(0, 255, 0))
    
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# =================================================================
# 6. 主程式介面
# =================================================================

# --- 側邊欄 ---
with st.sidebar:
    st.title("🗂️ 課程選單")
    selected_category = st.selectbox("選擇分類", list(COURSE_DATA.keys()))
    st.session_state.current_category = selected_category
    
    word_list = COURSE_DATA[selected_category]
    selected_word = st.selectbox("選擇練習詞彙", word_list)
    
    if selected_word != st.session_state.current_word:
        st.session_state.current_word = selected_word
        st.session_state.is_practice_mode = False
        st.session_state.progress_index = 0

    st.divider()
    st.write("⚙️ 設定")
    st.session_state.camera_index = st.number_input("攝影機 ID", 0, 5, 0)
    
    if st.button("🔄 重置進度 (從頭開始)"):
        st.session_state.progress_index = 0
        st.session_state.smooth_score = 0.0

st.title(f"🗣️ 當前練習：{st.session_state.current_word}")
st.caption("V30 永久存檔版：影片會自動儲存在 user_videos 資料夾")

tab1, tab2 = st.tabs(["📺 1. 教學與建模", "🎯 2. 實戰練習評分"])

# =================================================
# TAB 1: 教學影片區 (核心修改處：永久存檔邏輯)
# =================================================
with tab1:
    col1, col2 = st.columns([1, 1])
    
    # 定義該詞彙的永久檔案路徑
    # Windows 系統不支援檔名有特殊符號，這裡假設詞彙都是中文或英文
    current_video_filename = f"{st.session_state.current_word}.mp4"
    current_video_path = os.path.join(VIDEO_SAVE_DIR, current_video_filename)

    with col1:
        st.subheader("步驟 1：影片管理")
        
        # 1. 檢查是否已經有存檔
        file_exists = os.path.exists(current_video_path)
        
        if file_exists:
            st.success(f"📂 已找到「{st.session_state.current_word}」的歷史影片！")
            # 如果還沒載入到 session，就載入
            if st.session_state.current_word not in st.session_state.uploaded_videos:
                st.session_state.uploaded_videos[st.session_state.current_word] = current_video_path
        else:
            st.info("尚未上傳此詞彙的影片。")

        # 2. 上傳介面 (無論有無舊檔，都可以上傳覆蓋)
        video_key = f"uploader_{st.session_state.current_word}"
        video_file = st.file_uploader("上傳新影片 (將覆蓋舊檔)", type=['mp4', 'mov'], key=video_key)
        
        if video_file:
            # === 永久存檔邏輯 ===
            with open(current_video_path, "wb") as f:
                f.write(video_file.getbuffer())
            
            # 更新 session
            st.session_state.uploaded_videos[st.session_state.current_word] = current_video_path
            st.toast(f"✅ 影片已永久儲存至 {current_video_path}")
            
            # 為了讓介面刷新顯示新影片，可以考慮 rerun，但這裡先手動更新變數
            file_exists = True 

        # 3. 顯示影片
        if file_exists:
            st.video(current_video_path)
        
    with col2:
        st.subheader("步驟 2：AI 序列建模")
        
        # 這裡的邏輯也改為讀取永久路徑
        target_video_path = st.session_state.uploaded_videos.get(st.session_state.current_word)
        
        # 如果 session 沒抓到，但硬碟有檔案，就用硬碟的
        if not target_video_path and os.path.exists(current_video_path):
            target_video_path = current_video_path

        if target_video_path:
            if st.button("🚀 建立追蹤模型", width='stretch', type="primary"):
                with st.spinner("分析中..."):
                    seq_mars, seq_vectors = LipAnalyzer.analyze_video_sequence(target_video_path)
                    
                    if seq_vectors is not None:
                        st.session_state.standard_models[st.session_state.current_word] = {
                            "mars": seq_mars,
                            "vectors": seq_vectors,
                            "length": len(seq_vectors)
                        }
                        st.success(f"✅ 模型已建立！包含 {len(seq_vectors)} 個連續動作點。")
                    else:
                        st.error("分析失敗，請檢查影片。")
        else:
            st.warning("👈 請先上傳影片")

# =================================================
# TAB 2: 練習評分區
# =================================================
with tab2:
    col_p1, col_p2 = st.columns([3, 1])
    
    with col_p2:
        st.subheader("控制面板")
        if st.button("🟢 開始練習", width='stretch'):
            if st.session_state.current_word in st.session_state.standard_models:
                st.session_state.is_practice_mode = True
                st.session_state.progress_index = 0 
                st.session_state.smooth_score = 0.0
            else:
                st.error("請先建立模型！")
                
        if st.button("🔴 停止練習", width='stretch'):
            st.session_state.is_practice_mode = False

        st.divider()
        st.write("### 即時評分")
        score_gauge = st.empty()
        status_box = st.empty()
        
    with col_p1:
        st.write("### 鏡頭畫面 (含進度條)")
        cam_placeholder = st.empty()
        lm_spec, conn_spec = LipAnalyzer.get_drawing_specs()
        
        if st.session_state.is_practice_mode:
            cap = cv2.VideoCapture(st.session_state.camera_index)
            
            model_data = st.session_state.standard_models[st.session_state.current_word]
            std_vectors = model_data["vectors"]
            std_mars = model_data["mars"]
            total_frames = model_data["length"]
            
            with MP_TOOLS["mesh"].FaceMesh(refine_landmarks=True, max_num_faces=1) as face_mesh:
                while st.session_state.is_practice_mode and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    frame = cv2.flip(frame, 1)
                    h, w, _ = frame.shape
                    
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb_frame)
                    
                    overlay = frame.copy()
                    display_text = ["等待動作..."]
                    current_progress = st.session_state.progress_index / total_frames
                    
                    if results.multi_face_landmarks:
                        MP_TOOLS["draw"].draw_landmarks(
                            image=overlay,
                            landmark_list=results.multi_face_landmarks[0],
                            connections=MP_TOOLS["mesh"].FACEMESH_LIPS,
                            landmark_drawing_spec=lm_spec,
                            connection_drawing_spec=conn_spec
                        )
                        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

                        landmarks = results.multi_face_landmarks[0].landmark
                        cur_mar = LipAnalyzer.get_mar(landmarks)
                        cur_vec = LipAnalyzer.get_lip_shape_vector(landmarks)
                        
                        current_idx = st.session_state.progress_index
                        search_window_size = 15 
                        
                        start_search = current_idx
                        end_search = min(current_idx + search_window_size, total_frames)
                        
                        if start_search < end_search:
                            window_vectors = std_vectors[start_search:end_search]
                            dists = np.linalg.norm(window_vectors - cur_vec, axis=1)
                            local_best_idx = np.argmin(dists) 
                            global_best_idx = start_search + local_best_idx 
                            
                            target_vec = std_vectors[global_best_idx]
                            target_mar = std_mars[global_best_idx]
                            
                            score = LipAnalyzer.calculate_lenient_score(target_vec, cur_vec, target_mar, cur_mar)
                            
                            if score > 60:
                                st.session_state.progress_index = global_best_idx
                            
                            st.session_state.smooth_score = (st.session_state.smooth_score * 0.8) + (score * 0.2)
                            final_score = int(st.session_state.smooth_score)
                            
                            score_gauge.metric("得分", f"{final_score} 分")
                            
                            if final_score > 80:
                                status_box.success("✨ 完美！跟上了！")
                                cv2.putText(frame, "GOOD!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                            elif final_score > 60:
                                status_box.info("👌 繼續保持")
                            else:
                                status_box.warning("⏳ 加油...")

                            display_text = [
                                f"詞彙：{st.session_state.current_word}",
                                f"進度：{int((st.session_state.progress_index / total_frames)*100)}%",
                                f"分數：{final_score}"
                            ]
                            
                            if st.session_state.progress_index >= total_frames - 2:
                                cv2.putText(frame, "FINISH!", (w//2-100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
                        else:
                            display_text = ["練習完成！", "請按重置"]
                            status_box.success("🎉 練習結束！")

                    else:
                        display_text = ["未偵測到臉部"]

                    frame = draw_ui_overlay(frame, display_text, progress=current_progress)
                    cam_placeholder.image(frame, channels="BGR", width='stretch')
                    time.sleep(0.01)
            
            cap.release()
            cam_placeholder.empty()

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    Lip Expert V30 | 永久存檔版 | 影片會儲存於 user_videos 資料夾
</div>
""", unsafe_allow_html=True)