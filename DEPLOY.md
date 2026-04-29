# RBS Finance Dashboard — 部署指引

四種部署方式由易到難：

---

## 1. Streamlit Community Cloud（推薦，免費，永久線上）

最簡單也最穩定的方案。每次 push 自動部署。

### 步驟

1. 把 repo 變成 **Public**（Streamlit Cloud 免費版需要公開 repo）
2. 前往 https://share.streamlit.io/ 用 GitHub 登入
3. 點 **New app**
4. 設定：
   - **Repository**: `RoyARVSA/RBS_claude_finance`
   - **Branch**: `claude/optimize-analysis-dashboard-NZUKB` 或 `main`
   - **Main file path**: `streamlit_app.py`
5. 按 **Deploy**

完成後會拿到一個固定網址，例如 `https://rbs-finance.streamlit.app`，永久可用。

### 注意事項

- 免費版 1GB RAM，足夠跑此 dashboard
- `alerts_config.json` 會儲存在容器內（重啟可能消失），建議改用 Streamlit `st.secrets` 存敏感資料
- API Keys 請用 **Secrets** 介面管理（App settings → Secrets）

---

## 2. Hugging Face Spaces（免費，支援 Streamlit）

1. 註冊 https://huggingface.co/
2. **New Space** → Streamlit SDK
3. 把 repo 內容上傳（或設定 git remote 推送到 HF）
4. 自動偵測 `streamlit_app.py` 並啟動

優點：硬體規格可以付費升級到 GPU。

---

## 3. Colab + cloudflared（目前用法，免費但不穩定）

優點：
- 不用註冊外部服務
- 可訪問 Google Drive 檔案

缺點：
- Colab runtime 90 分鐘無互動會自動關閉
- `trycloudflare.com` URL 每次都會變
- iPad/某些 ISP 可能擋

執行步驟見 `RBS_Finance_Colab.ipynb`，依序跑 Cell 1 → 2 → 3。

---

## 4. 自架 VPS（最穩定，付費）

DigitalOcean / Linode / Vultr 最便宜方案 ~$5/月。

```bash
# Ubuntu 22.04
sudo apt update && sudo apt install -y python3-pip nginx
git clone https://github.com/RoyARVSA/RBS_claude_finance.git
cd RBS_claude_finance
pip install -r requirements.txt

# 用 systemd 常駐
sudo nano /etc/systemd/system/rbs.service
```

```ini
[Unit]
Description=RBS Finance Dashboard
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/RBS_claude_finance
ExecStart=/usr/bin/python3 -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable rbs && sudo systemctl start rbs
```

再用 nginx 反向代理 + Let's Encrypt 上 HTTPS。

---

## 各方案比較

| 方案 | 費用 | 穩定性 | 上手難度 | URL 持續性 |
|------|------|--------|----------|------------|
| Streamlit Cloud | 免費 | ⭐⭐⭐⭐⭐ | 極簡單 | 永久固定 |
| Hugging Face Spaces | 免費 | ⭐⭐⭐⭐ | 簡單 | 永久固定 |
| Colab + cloudflared | 免費 | ⭐⭐ | 中 | 每次變動 |
| 自架 VPS | ~$5/月 | ⭐⭐⭐⭐⭐ | 較難 | 永久固定 |

---

## 通知功能設定

### Telegram Bot

1. Telegram 找 `@BotFather`，傳 `/newbot` → 取得 **Bot Token**
2. 把你的 Bot 加為好友後傳一句話給它
3. 瀏覽器訪問：
   ```
   https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
   ```
   回傳 JSON 中的 `chat.id` 就是你的 **Chat ID**
4. 在 Dashboard「🚨 即時警報 → 通知設定」填入並測試

### Email (Gmail SMTP)

1. Google 帳戶 → **安全性** → 開啟 **兩步驟驗證**
2. 兩步驟驗證頁面下方 → **應用程式密碼** → 建立 16 字密碼
3. Dashboard 設定：
   - SMTP Host: `smtp.gmail.com`
   - Port: `465`
   - 帳號: 你的 Gmail
   - 密碼: 上一步的 16 字應用程式密碼（不是 Gmail 登入密碼！）

---

## 進階整合方向

### 排程自動掃描訊號 + 推播

Streamlit Cloud 不支援 cron job，但可以用：

- **GitHub Actions**：寫一個 Python script 每小時跑一次掃描，直接 call Telegram API 推播
- **Cron-Job.org**：免費 cron 服務，定期 GET 你的 webhook URL
- **AWS Lambda + EventBridge**：每分鐘觸發一次（免費額度 100 萬次/月）

### 進階交易功能（需付費 API）

- **Polygon.io**：真即時數據（$29/月）
- **Alpaca**：免費 paper trading + 可下真單（API 接入）
- **Interactive Brokers**：透過 `ib-insync` Python 套件接入專業券商
