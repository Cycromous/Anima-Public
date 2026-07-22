import streamlit as st
import streamlit.components.v1 as components
import tempfile
from CAIStyles import BOOT_HTML, MAIN_CSS, HIDE_CHROME_CSS

# --- UI LOGIC ---
class AnimaUI:
    @staticmethod
    def inject_global_styles():
        st.markdown(MAIN_CSS, unsafe_allow_html=True)

    @staticmethod
    def show_boot_screen():
        st.markdown(HIDE_CHROME_CSS, unsafe_allow_html=True)
        components.html(BOOT_HTML, height=800, scrolling=False)

    @staticmethod
    def render_header():
        st.markdown("""
        <div class="anima-header">
          <div class="anima-logo">&#11041;</div>
          <span class="anima-title">Anima</span>
          <span class="anima-badge">Local</span>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_empty_state():
        st.markdown("""
        <div class="empty-state">
          <h2>What can I help with?</h2>
          <p>Ask anything. Run tools. Analyze files.<br>Your local neural interface is ready.</p>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_message(role, content):
        with st.chat_message(role):
            if "[TOOL OUTPUT" in content:
                parts = content.split("[TOOL OUTPUT", 1)
                if parts[0].strip():
                    st.markdown(parts[0])
                st.markdown('<div class="tool-pill">&#9881; tool executed</div>', unsafe_allow_html=True)
                st.markdown("[TOOL OUTPUT" + parts[1])
            else:
                st.markdown(content)

# --- 1. PAGE CONFIG ---
st.set_page_config(
    page_title="Anima",
    page_icon="⬡",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- 2. BOOT SEQUENCE ---
if "brain_loaded" not in st.session_state:
    AnimaUI.show_boot_screen()
    from CAI import get_gemma_response
    st.session_state.brain_loaded = True
    st.rerun()

AnimaUI.inject_global_styles()

# --- 3. SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "backend_history" not in st.session_state:
    st.session_state.backend_history = []

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("### Anima")
    st.caption("Local inference · Experimental")
    st.divider()

    st.markdown("<p style='color:#2a2a2a;font-size:11px;letter-spacing:.5px;text-transform:uppercase;margin-bottom:6px'>Attachment</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("attach", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    
    current_image_path = None
    if uploaded_file:
        st.image(uploaded_file, use_column_width=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.getvalue())
            current_image_path = tmp.name

    st.divider()

    col1, col2 = st.columns([3, 2])
    with col1:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.backend_history = []
            st.rerun()
    with col2:
        msg_count = len([m for m in st.session_state.chat_history if m["role"] == "user"])
        st.markdown(f"<div style='text-align:right;padding:7px 2px;color:#222;font-size:11px;font-family:DM Mono,monospace'>{msg_count} msg</div>", unsafe_allow_html=True)

# --- 5. MAIN INTERFACE ---
AnimaUI.render_header()

if not st.session_state.chat_history:
    AnimaUI.render_empty_state()
else:
    for msg in st.session_state.chat_history:
        AnimaUI.render_message(msg["role"], msg["content"])

# --- 6. CHAT INPUT & LOGIC ---
user_input = st.chat_input("Ask Anima anything...")

if user_input:
    from CAI import get_gemma_response
    display_content = f"📎 *Image attached*\n\n{user_input}" if current_image_path else user_input

    AnimaUI.render_message("user", display_content)
    st.session_state.chat_history.append({"role": "user", "content": display_content})

    with st.chat_message("assistant"):
        with st.spinner(""):
            try:
                reply, updated_history = get_gemma_response(
                    user_input,
                    st.session_state.backend_history,
                    image_path=current_image_path
                )
                st.session_state.backend_history = updated_history
                
                AnimaUI.render_message("assistant", reply)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})

            except Exception as e:
                st.error(f"System error: {e}")
