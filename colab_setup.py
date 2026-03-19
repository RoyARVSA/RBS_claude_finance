"""
colab_setup.py  –  RBS Finance Dashboard: Colab + Ngrok 啟動腳本
================================================================
在 Google Colab 中，把以下所有 cell 依序執行：

  STEP 1 – 安裝套件
  STEP 2 – 掛載 Google Drive
  STEP 3 – 把最新版 app.py / rbs_lib.py 同步到 Drive
  STEP 4 – 啟動 Streamlit + pyngrok 取得公開網址

或者直接在一個 cell 執行：
  exec(open("colab_setup.py").read())

================================================================
"""

# ════════════════════════════════════════════════════════════════
# STEP 1 – 安裝套件
# ════════════════════════════════════════════════════════════════
import subprocess, sys

def _pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", *pkgs])

print("📦 Installing dependencies…")
_pip(
    "streamlit",
    "yfinance",
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "scipy",
    "adjustText",
    "feedparser",
    "newspaper3k",
    "requests",
    "plotly",
    "openai",        # for sentiment LLM tab
    "pyngrok",
)
print("✅ All packages installed")

# ════════════════════════════════════════════════════════════════
# STEP 2 – 掛載 Google Drive
# ════════════════════════════════════════════════════════════════
from pathlib import Path

DRIVE_BASE = Path("/content/drive/MyDrive/RBS_app")
DRIVE_BASE.mkdir(parents=True, exist_ok=True)

try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    print(f"✅ Google Drive mounted → {DRIVE_BASE}")
except ImportError:
    print("⚠️  Not running in Colab – skipping Drive mount")

# ════════════════════════════════════════════════════════════════
# STEP 3 – 同步 app.py / rbs_lib.py 到 Drive
#          (從 GitHub 或本地複製)
# ════════════════════════════════════════════════════════════════
import shutil, os

# 來源：優先用當前目錄（git clone 下來的），找不到則從 GitHub 下載
_SCRIPT_DIR = Path(__file__).parent if "__file__" in dir() else Path(".")

def _sync_file(fname: str):
    src = _SCRIPT_DIR / fname
    dst = DRIVE_BASE / fname
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  copied  {fname}  →  {dst}")
    else:
        # 從 GitHub raw 下載
        import urllib.request
        RAW = f"https://raw.githubusercontent.com/RoyARVSA/RBS_claude_finance/claude/optimize-analysis-dashboard-NZUKB/{fname}"
        try:
            urllib.request.urlretrieve(RAW, dst)
            print(f"  fetched {fname}  →  {dst}")
        except Exception as e:
            print(f"  ⚠️  could not sync {fname}: {e}")

print("📂 Syncing files to Drive…")
_sync_file("app.py")
_sync_file("rbs_lib.py")
print("✅ Sync done")

# 確保 Drive path 在 sys.path
if str(DRIVE_BASE) not in sys.path:
    sys.path.insert(0, str(DRIVE_BASE))

# ════════════════════════════════════════════════════════════════
# STEP 4 – 啟動 Streamlit + pyngrok
# ════════════════════════════════════════════════════════════════
import socket
import time
import threading

STREAMLIT_PORT = 8501
APP_PATH = DRIVE_BASE / "app.py"

# --- Ngrok 設定 ---
NGROK_AUTH_TOKEN = "31UVEcuXcsxDf69fAhH7e4qBLFO_6wotasJhN1ZLTwwrj2NSW"  # ← 換成你的 token
NGROK_REGION    = "ap"   # ap=Asia Pacific; us/eu/jp/in/au/sa/...

# --- 殺掉舊的 8501 進程 ---
subprocess.run(["fuser", "-k", f"{STREAMLIT_PORT}/tcp"], capture_output=True)
time.sleep(0.5)

# --- 啟動 Streamlit 子進程 ---
print(f"🚀 Starting Streamlit on port {STREAMLIT_PORT}…")
st_proc = subprocess.Popen(
    [
        sys.executable, "-m", "streamlit", "run", str(APP_PATH),
        "--server.port", str(STREAMLIT_PORT),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
        "--browser.gatherUsageStats", "false",
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

# --- 等待 Streamlit 就緒 ---
t0 = time.time()
while time.time() - t0 < 60:
    try:
        socket.create_connection(("127.0.0.1", STREAMLIT_PORT), 1).close()
        print("✅ Streamlit is up")
        break
    except OSError:
        time.sleep(0.25)
else:
    print("❌ Streamlit did not start within 60 s")
    sys.exit(1)

# --- pyngrok tunnel ---
from pyngrok import conf as _ngrok_conf, ngrok as _ngrok

_ngrok_conf.get_default().auth_token = NGROK_AUTH_TOKEN
_ngrok_conf.get_default().region     = NGROK_REGION

# 關閉舊 tunnel（避免重複）
for t in _ngrok.get_tunnels():
    _ngrok.disconnect(t.public_url)

tunnel = _ngrok.connect(STREAMLIT_PORT, "http")
public_url = tunnel.public_url.replace("http://", "https://")

print()
print("=" * 60)
print(f"  🌐  Public URL : {public_url}")
print(f"  📂  App file   : {APP_PATH}")
print(f"  🔢  Port       : {STREAMLIT_PORT}")
print("=" * 60)
print()
print("To stop:  st_proc.terminate()")
