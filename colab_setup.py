"""
colab_setup.py  –  RBS Finance Dashboard: Colab + Ngrok 啟動腳本
================================================================
在 Colab cell 貼上並執行：

    exec(open('/content/drive/MyDrive/RBS_app/colab_setup.py').read())

或直接把整份貼到 cell 執行。
================================================================
"""
import subprocess
import sys
import socket
import time
import urllib.request
import shutil
from pathlib import Path

# ════════════════════════════════════════════════════════════════
# 設定區  ← 只需改這裡
# ════════════════════════════════════════════════════════════════
NGROK_AUTH_TOKEN = "31UVEcuXcsxDf69fAhH7e4qBLFO_6wotasJhN1ZLTwwrj2NSW"
NGROK_REGION     = "ap"          # ap / us / eu / jp / in / au / sa
STREAMLIT_PORT   = 8501
DRIVE_BASE       = Path("/content/drive/MyDrive/RBS_app")
GITHUB_BRANCH    = "claude/optimize-analysis-dashboard-NZUKB"
GITHUB_REPO      = "RoyARVSA/RBS_claude_finance"
SYNC_FILES       = ["app.py", "rbs_lib.py"]   # 每次啟動從 GitHub 拉取
# ════════════════════════════════════════════════════════════════


# ────────────────────────────────────────────────────────────────
# STEP 1  安裝套件
# ────────────────────────────────────────────────────────────────
print("━" * 55)
print("STEP 1 │ Installing packages")
print("━" * 55)

PACKAGES = [
    "streamlit", "yfinance", "numpy", "pandas",
    "matplotlib", "seaborn", "scikit-learn", "scipy",
    "adjustText", "feedparser", "newspaper3k",
    "requests", "plotly", "openai", "pyngrok",
]

result = subprocess.run(
    [sys.executable, "-m", "pip", "-q", "install", *PACKAGES],
    capture_output=True, text=True
)
if result.returncode != 0:
    print("⚠️  pip stderr:", result.stderr[-500:])
else:
    print(f"✅ Installed: {', '.join(PACKAGES)}")


# ────────────────────────────────────────────────────────────────
# STEP 2  掛載 Google Drive
# ────────────────────────────────────────────────────────────────
print("\n" + "━" * 55)
print("STEP 2 │ Mounting Google Drive")
print("━" * 55)

try:
    from google.colab import drive as _gd
    _gd.mount("/content/drive", force_remount=False)
    print("✅ Google Drive mounted")
except ImportError:
    print("⚠️  Not in Colab – skipping Drive mount")

# Drive 掛好之後再建目錄
DRIVE_BASE.mkdir(parents=True, exist_ok=True)
print(f"✅ Working directory: {DRIVE_BASE}")

if str(DRIVE_BASE) not in sys.path:
    sys.path.insert(0, str(DRIVE_BASE))


# ────────────────────────────────────────────────────────────────
# STEP 3  從 GitHub 同步最新程式碼到 Drive
# ────────────────────────────────────────────────────────────────
print("\n" + "━" * 55)
print("STEP 3 │ Syncing files from GitHub")
print("━" * 55)

RAW_BASE = (
    f"https://raw.githubusercontent.com/"
    f"{GITHUB_REPO}/{GITHUB_BRANCH}"
)

for fname in SYNC_FILES:
    dst = DRIVE_BASE / fname
    url = f"{RAW_BASE}/{fname}"
    try:
        urllib.request.urlretrieve(url, dst)
        size_kb = dst.stat().st_size / 1024
        print(f"  ✅  {fname:<12} ({size_kb:.1f} KB)  →  {dst}")
    except Exception as e:
        if dst.exists():
            print(f"  ⚠️  {fname}: download failed ({e}), using cached version")
        else:
            print(f"  ❌  {fname}: download failed and no cache – {e}")
            sys.exit(1)


# ────────────────────────────────────────────────────────────────
# STEP 4  啟動 Streamlit
# ────────────────────────────────────────────────────────────────
print("\n" + "━" * 55)
print(f"STEP 4 │ Starting Streamlit on port {STREAMLIT_PORT}")
print("━" * 55)

APP_PATH = DRIVE_BASE / "app.py"
LOG_PATH = DRIVE_BASE / "streamlit.log"

# 清除殘留進程
subprocess.run(["fuser", "-k", f"{STREAMLIT_PORT}/tcp"], capture_output=True)
time.sleep(0.5)

_log_file = open(LOG_PATH, "w")
st_proc = subprocess.Popen(
    [
        sys.executable, "-m", "streamlit", "run", str(APP_PATH),
        "--server.port",               str(STREAMLIT_PORT),
        "--server.address",            "0.0.0.0",
        "--server.headless",           "true",
        "--server.enableCORS",         "false",
        "--server.enableXsrfProtection", "false",
        "--browser.gatherUsageStats",  "false",
    ],
    stdout=_log_file,
    stderr=_log_file,
)

# 等待 Streamlit 就緒（最多 60 秒）
t0 = time.time()
while time.time() - t0 < 60:
    try:
        socket.create_connection(("127.0.0.1", STREAMLIT_PORT), 1).close()
        elapsed = time.time() - t0
        print(f"✅ Streamlit is up ({elapsed:.1f}s)  │  log → {LOG_PATH}")
        break
    except OSError:
        time.sleep(0.3)
else:
    print("❌ Streamlit did not start within 60s. Last log lines:")
    try:
        print(LOG_PATH.read_text()[-800:])
    except Exception:
        pass
    sys.exit(1)


# ────────────────────────────────────────────────────────────────
# STEP 5  建立 pyngrok tunnel
# ────────────────────────────────────────────────────────────────
print("\n" + "━" * 55)
print("STEP 5 │ Creating ngrok tunnel")
print("━" * 55)

from pyngrok import conf as _nc, ngrok as _ng   # noqa: E402

_nc.get_default().auth_token = NGROK_AUTH_TOKEN
_nc.get_default().region     = NGROK_REGION

# 關閉舊 tunnel
for _t in _ng.get_tunnels():
    _ng.disconnect(_t.public_url)

_tunnel    = _ng.connect(STREAMLIT_PORT, "http")
public_url = _tunnel.public_url.replace("http://", "https://")

print(f"✅ Tunnel ready")

# ────────────────────────────────────────────────────────────────
# 完成
# ────────────────────────────────────────────────────────────────
print()
print("╔" + "═" * 53 + "╗")
print(f"║  🌐  Public URL  : {public_url:<33}║")
print(f"║  📂  App         : {str(APP_PATH):<33}║")
print(f"║  📋  Log         : {str(LOG_PATH):<33}║")
print(f"║  🔢  Port        : {STREAMLIT_PORT:<33}║")
print("╠" + "═" * 53 + "╣")
print("║  To stop  →  st_proc.terminate()              ║")
print("║  To restart →  exec(open(__file__).read())    ║")
print("╚" + "═" * 53 + "╝")
