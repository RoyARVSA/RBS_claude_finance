"""
state_crypto.py – 公開 repo 的敏感區塊加密（純標準庫、離線可測）

背景：cron 的持久化=把 state commit 回公開 repo，等於把個人投資論點、
paper 帳戶淨值、引擎簿記、全部策略參數攤在陽光下（審查團資安場次揭露、
使用者拍板「重要資訊不能 commit」）。轉私有 repo 會撞 Actions 免費額度牆，
整檔搬離 git 工程太大——折衷：**敏感區塊加密後才落檔**。

機制（標準庫 hmac/hashlib，無第三方依賴——cron 環境裝不了 cryptography）：
  • 金鑰：環境變數 STATE_ENC_KEY（GitHub Secrets + Streamlit Secrets 同步設）
  • keystream = HMAC-SHA256(key, nonce ‖ counter) 區塊串接，XOR 明文
  • 完整性：tag = HMAC-SHA256(key, nonce ‖ ciphertext)，解密先驗 tag
  • 每次加密隨機 nonce——同內容每次密文不同（公開 diff 看不出「沒變」）

誠實邊界（文件同步聲明）：
  • 這是「防瀏覽級」保護（擋公開 repo 的路人與爬蟲），非審計級 AEAD；
    威脅模型不含拿得到 key 的對手
  • **key 遺失＝加密區塊永久無法還原**——請把 key 抄在密碼管理器
  • 過去已 commit 的明文歷史仍在 git 裡；要抹除需改寫歷史（另行決定）
  • 未設 key 時一切照舊明文（opt-in，不會弄壞任何現有部署）
  • **換 key＝舊密文鎖死**——輪替前務必先在舊 key 環境下解密取回；
    key 設定後解密失敗會發 Telegram 告警（每製程一次）
  • 隱含安全前提：autotrade_enabled 預設 False——無 key 輪 thresholds 降級
    為預設值時 autotrade 自動關閉、該輪不下單。**勿把此預設改 True**

用法：save 端 encrypt_state()、load 端 decrypt_state()；journal 整檔走
encrypt_block/decrypt_block。app/bot/cron 統一經 read_state() 讀。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os

# 加密的 state 頂層鍵（其餘留明文：watchlist/weather/signal_history 等
# 皆為公開資料的衍生，加密徒增體積）
SENSITIVE_KEYS = ("theses", "shadow", "engine", "thresholds",
                  "plan_calib", "reflections", "calibration",
                  "mirror",    # 鏡像帳：使用者真實布建與資金量，絕不落明文
                  "eng_opt")   # /engtest apply 的參數紀錄（策略參數）

_MAGIC = "__enc__"


def get_key() -> str:
    """讀 STATE_ENC_KEY 並消毒（secrets 換行坑，PITFALLS D5/B10）。"""
    v = os.environ.get("STATE_ENC_KEY", "")
    for ch in ("\r", "\n", "\t", "​", "‌", "‍", "﻿", " "):
        v = v.replace(ch, "")
    return v.strip()


def _keystream(key: bytes, nonce: bytes, n: int) -> bytes:
    out = b""
    counter = 0
    while len(out) < n:
        out += hmac.new(key, nonce + counter.to_bytes(8, "big"),
                        hashlib.sha256).digest()
        counter += 1
    return out[:n]


def encrypt_block(obj, key: str) -> dict:
    """任意 JSON 可序列化物件 → {"__enc__":"v1", nonce, data, tag}。"""
    kb = hashlib.sha256(key.encode("utf-8")).digest()
    nonce = os.urandom(16)
    pt = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    ct = bytes(a ^ b for a, b in zip(pt, _keystream(kb, nonce, len(pt))))
    # b"tag" 前綴＝域分隔：tag 與 keystream 同 key，不加前綴在極端邊角
    # （len(ct)=8 且恰等於 counter bytes）會洩 keystream 塊（對抗驗證 Low-2）
    tag = hmac.new(kb, b"tag" + nonce + ct, hashlib.sha256).hexdigest()
    return {_MAGIC: "v1",
            "nonce": base64.b64encode(nonce).decode(),
            "data": base64.b64encode(ct).decode(),
            "tag": tag}


def is_enc(blob) -> bool:
    return isinstance(blob, dict) and blob.get(_MAGIC) == "v1"


def decrypt_block(blob: dict, key: str):
    """解密。key 錯 / tag 不符 / 格式壞 → None（呼叫端決定降級行為）。"""
    if not is_enc(blob) or not key:
        return None
    try:
        kb = hashlib.sha256(key.encode("utf-8")).digest()
        nonce = base64.b64decode(blob["nonce"])
        ct = base64.b64decode(blob["data"])
        want = hmac.new(kb, b"tag" + nonce + ct, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(want, str(blob.get("tag", ""))):
            return None
        pt = bytes(a ^ b for a, b in zip(ct, _keystream(kb, nonce, len(ct))))
        return json.loads(pt.decode("utf-8"))
    except Exception:
        return None


def encrypt_state(state: dict, key: str) -> dict:
    """回新 dict：SENSITIVE_KEYS 換成密文塊（無 key 原樣回傳）。"""
    if not key:
        return state
    out = dict(state)
    for k in SENSITIVE_KEYS:
        # None 不加密：encrypt(None)→decrypt 回 None 與「tag 失敗」無法區分，
        # 會被誤判 locked 永久鎖死（對抗驗證 Low-1）
        if k in out and out[k] is not None and not is_enc(out[k]):
            out[k] = encrypt_block(out[k], key)
    return out


def decrypt_state(state: dict, key: str) -> tuple[dict, list]:
    """
    回 (新 dict, 無法解密的鍵清單)。
    有 key：逐塊解密，tag 失敗的鍵列入清單並**保留密文塊**（絕不拿壞資料覆蓋）。
    無 key 但有密文塊：全部保留原樣並列入清單——呼叫端該大聲警告＋功能降級，
    但密文必須原封不動傳回 save，**否則一次無 key 的執行就把加密資料洗掉**。
    """
    out = dict(state)
    failed = []
    for k, v in state.items():
        if is_enc(v):
            dec = decrypt_block(v, key)
            if dec is None:
                failed.append(k)
            else:
                out[k] = dec
    return out, failed


def read_state(path) -> dict:
    """app/工具端便捷讀取：json 讀檔＋自動解密（無 key 時敏感塊為密文原樣）。"""
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    dec, failed = decrypt_state(raw, get_key())
    if failed:
        print(f"state_crypto: {len(failed)} 個區塊無法解密（未設 STATE_ENC_KEY？）：{failed}")
    return dec


# ── 自我測試 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    KEY = "test-key-123"
    obj = {"a": 1, "中文": [1.5, None, "論點：AI 供應鏈"], "n": {"x": True}}

    # 1) round-trip + 每次密文不同（隨機 nonce）
    b1, b2 = encrypt_block(obj, KEY), encrypt_block(obj, KEY)
    assert b1["data"] != b2["data"] and b1["nonce"] != b2["nonce"]
    assert decrypt_block(b1, KEY) == obj and decrypt_block(b2, KEY) == obj
    print("✅ 1 round-trip + nonce 隨機")

    # 2) 錯 key / 竄改 → None（絕不回壞資料）
    assert decrypt_block(b1, "wrong-key") is None
    tampered = dict(b1)
    ct = bytearray(base64.b64decode(tampered["data"]))
    if ct:
        ct[0] ^= 0xFF
    tampered["data"] = base64.b64encode(bytes(ct)).decode()
    assert decrypt_block(tampered, KEY) is None
    assert decrypt_block({"__enc__": "v1"}, KEY) is None
    print("✅ 2 錯 key/竄改/壞格式防護")

    # 3) state 級：敏感鍵加密、其他明文；解密還原
    st = {"watchlist": ["AAPL"], "thresholds": {"risk_pct": 0.01},
          "theses": [{"t": "NVDA", "note": "私人論點"}], "weather": {"score": 60}}
    enc = encrypt_state(st, KEY)
    assert is_enc(enc["thresholds"]) and is_enc(enc["theses"])
    assert enc["watchlist"] == ["AAPL"] and enc["weather"] == {"score": 60}
    assert "私人論點" not in json.dumps(enc, ensure_ascii=False)   # 明文不落檔
    dec, failed = decrypt_state(enc, KEY)
    assert dec == st and failed == []
    print("✅ 3 state 級加密（敏感鍵才加密、明文不外洩）")

    # 4) 無 key 讀到密文：保留原樣＋回報清單（絕不洗掉加密資料）
    dec2, failed2 = decrypt_state(enc, "")
    assert sorted(failed2) == ["theses", "thresholds"]
    assert dec2["thresholds"] == enc["thresholds"]        # 密文原封不動
    # 再加密（模擬無 key 輪的 save）不得破壞既有密文塊
    re_enc = encrypt_state(dec2, KEY)
    assert decrypt_state(re_enc, KEY)[0]["theses"] == st["theses"]
    print("✅ 4 無 key 降級不毀資料")

    # 5) 無 key 全程 passthrough（未啟用者零影響）
    assert encrypt_state(st, "") == st
    d3, f3 = decrypt_state(st, "")
    assert d3 == st and f3 == []
    print("✅ 5 未設 key 完全透明")

    # 6) 大物件（模擬 journal 500 筆）round-trip
    big = [{"time": f"2026-08-{i%28+1:02d}", "symbol": "AAPL", "qty": i,
            "price": 100.0 + i} for i in range(500)]
    eb = encrypt_block(big, KEY)
    assert decrypt_block(eb, KEY) == big
    print("✅ 6 journal 級大物件")

    print("\nstate_crypto selftest OK ✅")
