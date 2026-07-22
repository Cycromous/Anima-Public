# --- BOOT SCREEN HTML ---
BOOT_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400&display=swap');
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    background:#080808;
    width:100vw; height:100vh;
    display:flex; align-items:center; justify-content:center;
    font-family:'DM Sans',sans-serif;
    overflow:hidden;
  }
  svg.waves {
    position:absolute; bottom:0; left:0;
    width:100%; height:62%; overflow:visible;
  }
  .wave-line {
    fill:none; stroke-linecap:round;
    stroke-dasharray:2200; stroke-dashoffset:2200;
    animation: draw-line 3s cubic-bezier(0.4,0,0.2,1) forwards,
                glow-pulse 2.5s ease-in-out infinite;
  }
  .w1{stroke:#fff;    stroke-width:1.5px;filter:url(#g1);animation-delay:0s,    3s;}
  .w2{stroke:#d8d8d8; stroke-width:1.3px;filter:url(#g1);animation-delay:0.14s, 3.14s;}
  .w3{stroke:#aaa;    stroke-width:1.1px;filter:url(#g2);animation-delay:0.28s, 3.28s;}
  .w4{stroke:#777;    stroke-width:1px;  filter:url(#g2);animation-delay:0.42s, 3.42s;}
  .w5{stroke:#4a4a4a; stroke-width:.9px; filter:url(#g3);animation-delay:0.56s, 3.56s;}
  .w6{stroke:#2e2e2e; stroke-width:.8px; filter:url(#g3);animation-delay:0.70s, 3.70s;}
  .w7{stroke:#1a1a1a; stroke-width:.7px; filter:url(#g3);animation-delay:0.84s, 3.84s;}
  @keyframes draw-line{
    0%{stroke-dashoffset:2200;opacity:0;}8%{opacity:1;}100%{stroke-dashoffset:0;}
  }
  @keyframes glow-pulse{0%,100%{opacity:1;}50%{opacity:.4;}}
  .content{position:relative;z-index:10;text-align:center;margin-bottom:220px;}
  .logo{
    width:48px;height:48px;border:1.5px solid #1e1e1e;border-radius:11px;
    display:flex;align-items:center;justify-content:center;
    font-size:22px;color:#444;margin:0 auto 22px auto;
    opacity:0;animation:fade-up .9s ease .1s forwards;box-shadow:0 0 40px #ffffff0d;
  }
  .title{font-size:22px;font-weight:400;color:#d8d8d8;letter-spacing:.2px;
    opacity:0;animation:fade-up .9s ease .3s forwards;}
  .sub{font-family:'DM Mono',monospace;font-size:10.5px;color:#282828;
    letter-spacing:3px;text-transform:uppercase;margin-top:14px;
    opacity:0;animation:fade-up .9s ease .5s forwards;}
  .dots{display:flex;gap:5px;justify-content:center;margin-top:28px;
    opacity:0;animation:fade-up .9s ease .7s forwards;}
  .dots span{width:3px;height:3px;border-radius:50%;background:#222;
    animation:dot-pulse 1.5s ease-in-out infinite;}
  .dots span:nth-child(2){animation-delay:.25s;}
  .dots span:nth-child(3){animation-delay:.5s;}
  @keyframes dot-pulse{0%,100%{background:#1e1e1e;transform:scale(1);}50%{background:#666;transform:scale(1.6);}}
  @keyframes fade-up{from{opacity:0;transform:translateY(8px);}to{opacity:1;transform:translateY(0);}}
</style>
</head>
<body>
<svg class="waves" viewBox="0 0 1440 380" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <filter id="g1" x="-60%" y="-60%" width="220%" height="220%">
      <feGaussianBlur stdDeviation="8" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <filter id="g2" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="4" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <filter id="g3" x="-40%" y="-40%" width="180%" height="180%">
      <feGaussianBlur stdDeviation="2" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <path class="wave-line w1" d="M-80,290 C200,290 260,55 480,105 C700,155 740,305 960,190 C1180,75 1300,250 1520,160"/>
  <path class="wave-line w2" d="M-80,322 C185,322 272,92 492,140 C712,188 750,335 965,222 C1180,109 1304,278 1520,194"/>
  <path class="wave-line w3" d="M-80,348 C172,348 284,124 504,170 C724,216 762,360 972,250 C1182,140 1307,304 1520,224"/>
  <path class="wave-line w4" d="M-80,370 C160,370 296,152 516,196 C736,240 774,382 980,274 C1186,166 1310,328 1520,252"/>
  <path class="wave-line w5" d="M-80,389 C150,389 308,178 528,220 C748,262 785,402 988,296 C1191,190 1312,350 1520,278"/>
  <path class="wave-line w6" d="M-80,405 C141,405 320,202 540,242 C760,282 796,420 996,318 C1196,216 1314,371 1520,302"/>
  <path class="wave-line w7" d="M-80,420 C133,420 332,224 552,262 C772,300 807,436 1004,338 C1201,240 1316,390 1520,324"/>
</svg>
<div class="content">
  <div class="logo">&#11041;</div>
  <div class="title">Anima</div>
  <div class="sub">Loading neural weights</div>
  <div class="dots"><span></span><span></span><span></span></div>
</div>
<script>
  try {
    var f = window.frameElement;
    f.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;border:none;z-index:99999;';
  } catch(e){}
</script>
</body>
</html>"""

# --- MAIN APP CSS ---
MAIN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background-color: #0d0d0d !important;
    color: #e0e0e0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

*:focus, *:focus-visible, *:focus-within {
    outline: none !important;
    box-shadow: none !important;
}

::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1e1e1e; border-radius: 2px; }

[data-testid="stAppViewContainer"] > .main {
    background: #0d0d0d !important;
    padding: 0 !important;
}
.block-container {
    max-width: 720px !important;
    margin: 0 auto !important;
    padding: 0 20px 140px 20px !important;
}

[data-testid="stChatMessage"] .stMarkdown p,
[data-testid="stChatMessage"] .stMarkdown li,
[data-testid="stChatMessage"] .stMarkdown span {
    color: #d0d0d0 !important;
}

.empty-state h2 { color: #444 !important; }
.empty-state p  { color: #333 !important; }

.anima-badge     { color: #555 !important; }

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] caption { color: #555 !important; }

:root {
    --primary-color: #3a3a3a !important;
}

[data-testid="stChatInputContainer"] > div:focus-within,
[data-testid="stChatInputContainer"] > div:focus,
[data-testid="stChatInputContainer"] > div *:focus,
[data-testid="stChatInputContainer"] > div *:focus-within,
div:has(> textarea[data-testid="stChatInputTextArea"]):focus-within {
    border-color: #3a3a3a !important;
    box-shadow: none !important;
    outline: none !important;
}

textarea[data-testid="stChatInputTextArea"]:focus {
    box-shadow: none !important;
    border: none !important;
    outline: none !important;
}

[data-testid="stSidebar"] div[style] { color: #444 !important; }

.anima-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 26px 0 6px 0;
}
.anima-logo {
    width: 26px; height: 26px;
    border: 1px solid #1e1e1e;
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; color: #444; flex-shrink: 0;
}
.anima-title {
    font-size: 16px; font-weight: 600;
    letter-spacing: -0.3px; color: #e8e8e8;
}
.anima-badge {
    font-size: 9.5px; color: #2e2e2e;
    background: #0e0e0e; border: 1px solid #1a1a1a;
    padding: 2px 6px; border-radius: 20px;
    letter-spacing: 0.8px; text-transform: uppercase;
    margin-left: 2px;
}

[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 8px 0 !important;
    gap: 14px !important;
}
[data-testid="chatAvatarIcon-user"] {
    background: #141414 !important;
    border: 1px solid #1e1e1e !important;
    color: #3a3a3a !important;
    border-radius: 7px !important;
    width: 28px !important; height: 28px !important;
    font-size: 11px !important;
}
[data-testid="chatAvatarIcon-assistant"] {
    background: #0f0f0f !important;
    border: 1px solid #202020 !important;
    color: #666 !important;
    border-radius: 7px !important;
    width: 28px !important; height: 28px !important;
    font-size: 11px !important;
}
[data-testid="stChatMessage"] .stMarkdown p {
    font-size: 14.5px !important;
    line-height: 1.8 !important;
    color: #c5c5c5 !important;
}
[data-testid="stChatMessage"] .stMarkdown h1,
[data-testid="stChatMessage"] .stMarkdown h2,
[data-testid="stChatMessage"] .stMarkdown h3 {
    color: #ddd !important;
    font-weight: 500 !important;
    margin: 16px 0 8px 0 !important;
}
[data-testid="stChatMessage"] .stMarkdown li {
    font-size: 14.5px !important;
    color: #b8b8b8 !important;
    line-height: 1.8 !important;
}

code {
    font-family: 'DM Mono', monospace !important;
    font-size: 12.5px !important;
    background: #111 !important;
    border: 1px solid #1c1c1c !important;
    padding: 1px 5px !important;
    border-radius: 4px !important;
    color: #aaa !important;
}
pre {
    background: #0a0a0a !important;
    border: 1px solid #181818 !important;
    border-radius: 8px !important;
}
pre code {
    padding: 14px !important;
    display: block;
    border: none !important;
    background: transparent !important;
}

[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottom"] > div > div,
.stChatFloatingInputContainer,
.stChatFloatingInputContainer > div,
div[class*="chatInputContainer"],
div[class*="stChatInput"] {
    background: #0d0d0d !important;
    border: none !important;
    box-shadow: none !important;
}

[data-testid="stChatInputContainer"]::before,
[data-testid="stChatInputContainer"]::after {
    display: none !important;
}

div[data-testid="stChatInputContainer"] {
    background: transparent !important;
}

[data-testid="stChatInputContainer"] {
    position: fixed !important;
    bottom: 0 !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    width: min(720px, 100vw) !important;
    padding: 12px 20px 22px 20px !important;
    background: linear-gradient(to top, #0d0d0d 70%, #0d0d0d00) !important;
    border: none !important;
    z-index: 100 !important;
}

[data-testid="stChatInputContainer"] > div {
    background: transparent !important;
    border: 1px solid #ffffff !important;
    border-radius: 16px !important;
    padding: 6px 10px !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
    box-shadow: none !important;
}

[data-testid="stChatInputContainer"] > div:focus-within {
    border-color: #ffffff !important;
    box-shadow: 0 0 0 1px #ffffff44 !important;
}

textarea[data-testid="stChatInputTextArea"] {
    background: transparent !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    color: #ddd !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14.5px !important;
    caret-color: #fff !important;
    resize: none !important;
}

textarea[data-testid="stChatInputTextArea"]::placeholder {
    color: #ffffff !important;
    opacity: 0.7 !important; 
}

[data-testid="stChatInputSubmitButton"] button {
    background: #1a1a1a !important;
    border: 1px solid #272727 !important;
    border-radius: 9px !important;
    color: #555 !important;
    transition: all 0.12s ease !important;
    box-shadow: none !important;
}
[data-testid="stChatInputSubmitButton"] button:hover {
    background: #efefef !important;
    border-color: #efefef !important;
    color: #0d0d0d !important;
}

[data-testid="stChatInputContainer"] *,
[data-testid="stChatInputContainer"] *:focus,
[data-testid="stChatInputContainer"] *:focus-within {
    outline: none !important;
    border-color: inherit !important;
}

[data-testid="stSpinner"] > div {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #333 !important;
    font-size: 12px !important;
}
.stSpinner > div > div {
    border-color: #222 #222 #222 #666 !important;
    width: 13px !important;
    height: 13px !important;
}

.tool-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #0e0e0e;
    border: 1px solid #1a1a1a;
    border-radius: 6px;
    padding: 3px 9px;
    font-size: 10.5px;
    color: #383838;
    font-family: 'DM Mono', monospace;
    margin-bottom: 8px;
}

.empty-state {
    text-align: center;
    padding: 90px 0 40px 0;
}
.empty-state h2 {
    font-size: 24px;
    font-weight: 400;
    color: #1e1e1e;
    letter-spacing: -0.5px;
    margin-bottom: 10px;
}
.empty-state p {
    font-size: 13px;
    color: #181818;
    line-height: 1.8;
}

[data-testid="stSidebar"] {
    background: #080808 !important;
    border-right: 1px solid #111 !important;
    width: 240px !important;
}
[data-testid="stSidebar"] > div {
    padding: 20px 14px !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #666 !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    letter-spacing: -0.2px !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] small {
    color: #2a2a2a !important;
    font-size: 11.5px !important;
    line-height: 1.5 !important;
}

[data-testid="stSidebar"] .stButton button {
    background: #0e0e0e !important;
    border: 1px solid #191919 !important;
    color: #444 !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12.5px !important;
    width: 100%;
    padding: 6px 10px !important;
    transition: all 0.12s ease;
    text-align: left !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: #141414 !important;
    border-color: #222 !important;
    color: #888 !important;
}

[data-testid="stFileUploader"] {
    background: transparent !important;
}
[data-testid="stFileUploader"] section {
    background: #0c0c0c !important;
    border: 1px dashed #1e1e1e !important;
    border-radius: 10px !important;
    padding: 14px !important;
    transition: border-color 0.15s ease;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #2e2e2e !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] div span {
    color: #2e2e2e !important;
    font-size: 11.5px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] div small {
    color: #1e1e1e !important;
    font-size: 10.5px !important;
}
[data-testid="stFileUploader"] button {
    background: #141414 !important;
    border: 1px solid #1e1e1e !important;
    color: #555 !important;
    border-radius: 7px !important;
    font-size: 11.5px !important;
    padding: 4px 10px !important;
}
[data-testid="stFileUploader"] button:hover {
    background: #1a1a1a !important;
    color: #888 !important;
}

hr {
    border: none !important;
    border-top: 1px solid #111 !important;
    margin: 14px 0 !important;
}

[data-testid="stAlert"] {
    background: #0f0808 !important;
    border: 1px solid #2a1010 !important;
    border-radius: 8px !important;
    color: #aa3333 !important;
    font-size: 13px !important;
}

[data-testid="collapsedControl"] {
    background: #0a0a0a !important;
    border-right: 1px solid #111 !important;
}
[data-testid="collapsedControl"] svg { color: #333 !important; }
</style>
"""

# --- INJECTED STYLE FOR BOOT ---
HIDE_CHROME_CSS = """
    <style>
    #MainMenu, footer, header,
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    [data-testid="stAppViewContainer"] > .main {
        background: #080808 !important;
        display: none !important;
    }
    html, body, [data-testid="stApp"] {
        background: #080808 !important;
    }
    .appview-container { margin-top: 0 !important; padding-top: 0 !important; }
    section[tabindex="0"] { padding-top: 0 !important; }
    </style>
"""