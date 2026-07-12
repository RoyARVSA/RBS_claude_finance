"""
sentiment_fg.py — 雙恐懼貪婪指數（傳統市場 CNN + 加密 alternative.me）
RBS Finance Dashboard

- CNN Fear & Greed：非官方端點（需瀏覽器 UA），失敗時退回社群鏡像
  （whit3rabbit/fear-greed-data 以 GitHub Actions 每日抓存 JSON）
- Crypto Fear & Greed：alternative.me 官方 API（免金鑰、明示免費可用）
- 雙訊號：兩者同時極端恐懼，歷史上常出現在風險資產短期底部附近
 （統計傾向而非保證——輸出永遠附免責）

純邏輯（classify / parse_cnn / parse_crypto / dual_signal / fg_text）離線可測；
fetch_* 需網路。教育用途，非投資建議。
"""
from __future__ import annotations

_UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/124.0 Safari/537.36")}

CNN_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
# 鏡像：whit3rabbit/fear-greed-data 每交易日 23:00 UTC 以 GitHub Actions 更新（已查證），
# 根目錄 fear-greed.csv 格式：date,score,rating（如 2026-07-10,49.49,neutral）
CNN_FALLBACK = ("https://raw.githubusercontent.com/whit3rabbit/fear-greed-data/"
                "main/fear-greed.csv")
CRYPTO_URL = "https://api.alternative.me/fng/?limit=30"


# ── 純邏輯 ────────────────────────────────────────────────────────────────────

def classify(score: float | None) -> tuple[str, str]:
    """0-100 → (中文標籤, emoji)。None → 無資料。"""
    if score is None:
        return "無資料", "❔"
    if score < 25:
        return "極度恐懼", "😱"
    if score < 45:
        return "恐懼", "😨"
    if score <= 55:
        return "中性", "😐"
    if score <= 75:
        return "貪婪", "🤑"
    return "極度貪婪", "🤯"


def parse_cnn(j: dict) -> dict | None:
    """CNN graphdata JSON → {score, rating, week_ago, month_ago}。形狀不符回 None。"""
    try:
        fg = j.get("fear_and_greed") or {}
        score = float(fg["score"])
        return {"score": round(score, 1),
                "rating": fg.get("rating", ""),
                "prev_close": _f(fg.get("previous_close")),
                "week_ago": _f(fg.get("previous_1_week")),
                "month_ago": _f(fg.get("previous_1_month")),
                "year_ago": _f(fg.get("previous_1_year"))}
    except (KeyError, TypeError, ValueError):
        return None


def parse_cnn_mirror(text: str) -> dict | None:
    """鏡像 CSV（date,score,rating 逐行；已查證格式）→ 取最新一筆 + 週前值。"""
    try:
        rows = []
        for line in (text or "").replace("\r", "").split("\n"):
            parts = [c.strip() for c in line.split(",")]
            if len(parts) < 2 or not parts[0][:4].isdigit():
                continue                            # 跳過表頭/壞行
            try:
                rows.append((parts[0], float(parts[1]),
                             parts[2] if len(parts) > 2 else ""))
            except ValueError:
                continue
        if not rows:
            return None
        last = rows[-1]
        week = rows[-6][1] if len(rows) >= 6 else None   # 5 個交易日前
        return {"score": round(last[1], 1), "rating": last[2],
                "prev_close": rows[-2][1] if len(rows) >= 2 else None,
                "week_ago": round(week, 1) if week is not None else None,
                "month_ago": None, "year_ago": None, "mirror": True}
    except (TypeError, ValueError, IndexError):
        return None


def parse_crypto(j: dict) -> dict | None:
    """alternative.me /fng JSON → {score, label, week_ago}。value 是字串。"""
    try:
        data = j.get("data") or []
        if not data:
            return None
        score = float(data[0]["value"])
        week = float(data[6]["value"]) if len(data) > 6 else None
        return {"score": round(score, 1),
                "label": data[0].get("value_classification", ""),
                "week_ago": week}
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def dual_signal(cnn_score: float | None, crypto_score: float | None) -> dict:
    """雙指數組合判讀。"""
    out = {"both_extreme_fear": False, "both_extreme_greed": False, "note": None}
    if cnn_score is None or crypto_score is None:
        return out
    if cnn_score < 25 and crypto_score < 25:
        out["both_extreme_fear"] = True
        out["note"] = "傳統+加密同時極度恐懼——歷史上常見於風險資產短期底部附近（統計傾向非保證）"
    elif cnn_score > 75 and crypto_score > 75:
        out["both_extreme_greed"] = True
        out["note"] = "傳統+加密同時極度貪婪——追高風險升高，留意回檔"
    return out


def fg_text(cnn: dict | None, crypto: dict | None, compact: bool = False) -> str:
    """晨報/Bot 文字。compact=True 出一行版（晨報用）。"""
    parts = []
    if cnn:
        lbl, emo = classify(cnn["score"])
        seg = f"{emo} 美股恐貪 {cnn['score']:.0f}（{lbl}"
        if cnn.get("week_ago") is not None:
            seg += f"，週前 {cnn['week_ago']:.0f}"
        seg += "）"
        parts.append(seg)
    if crypto:
        lbl, emo = classify(crypto["score"])
        parts.append(f"{emo} 加密恐貪 {crypto['score']:.0f}（{lbl}）")
    if not parts:
        return "恐懼貪婪指數：暫無資料"
    sig = dual_signal(cnn["score"] if cnn else None,
                      crypto["score"] if crypto else None)
    if compact:
        line = "　".join(parts)
        return line + (f"\n⚡ {sig['note']}" if sig["note"] else "")
    lines = ["🎭 *恐懼貪婪指數*"] + parts
    if sig["note"]:
        lines.append(f"⚡ {sig['note']}")
    lines.append("_情緒為反向參考指標之一，非投資建議。_")
    return "\n".join(lines)


def _f(x):
    try:
        return round(float(x), 1) if x is not None else None
    except (TypeError, ValueError):
        return None


# ── 抓取層（需網路）───────────────────────────────────────────────────────────

def fetch_cnn(timeout: int = 12) -> dict | None:
    """CNN 主端點 → 失敗退鏡像（逐年檔名嘗試）。"""
    import requests
    try:
        r = requests.get(CNN_URL, headers=_UA, timeout=timeout)
        if r.ok:
            out = parse_cnn(r.json())
            if out:
                return out
    except Exception:
        pass
    # 鏡像備援：whit3rabbit/fear-greed-data 根目錄 fear-greed.csv（每交易日更新）
    try:
        r = requests.get(CNN_FALLBACK, headers=_UA, timeout=timeout)
        if r.ok:
            return parse_cnn_mirror(r.text)
    except Exception:
        pass
    return None


def fetch_crypto(timeout: int = 12) -> dict | None:
    import requests
    try:
        r = requests.get(CRYPTO_URL, headers=_UA, timeout=timeout)
        if r.ok:
            return parse_crypto(r.json())
    except Exception:
        pass
    return None


def fetch_all() -> dict:
    """回 {cnn, crypto}——各自獨立容錯。"""
    return {"cnn": fetch_cnn(), "crypto": fetch_crypto()}


# ── CLI 自我測試（離線純邏輯）─────────────────────────────────────────────────

if __name__ == "__main__":
    # 分級邊界
    assert classify(10)[0] == "極度恐懼" and classify(24.9)[0] == "極度恐懼"
    assert classify(25)[0] == "恐懼" and classify(50)[0] == "中性"
    assert classify(56)[0] == "貪婪" and classify(80)[0] == "極度貪婪"
    assert classify(None)[0] == "無資料"

    # CNN 主端點 JSON 形狀
    cnn = parse_cnn({"fear_and_greed": {
        "score": 22.4, "rating": "extreme fear", "previous_close": 24.0,
        "previous_1_week": 35.2, "previous_1_month": 55.0, "previous_1_year": 70.1}})
    assert cnn and cnn["score"] == 22.4 and cnn["week_ago"] == 35.2
    assert parse_cnn({"unexpected": 1}) is None

    # 鏡像 CSV（已查證真實格式：date,score,rating）
    m = parse_cnn_mirror(
        "Date,Fear Greed,Rating\n"                       # 表頭要能跳過
        "2026-07-01,30,fear\n2026-07-02,28,fear\n2026-07-03,26,fear\n"
        "2026-07-06,24,extreme fear\n2026-07-07,23,extreme fear\n"
        "2026-07-08,22,extreme fear\n2026-07-09,21.5,extreme fear\n")
    assert m and m["score"] == 21.5 and m["week_ago"] == 28 and m["mirror"], m
    assert m["prev_close"] == 22
    assert parse_cnn_mirror("") is None and parse_cnn_mirror("junk,no,dates") is None

    # 加密（value 是字串、timestamp unix——official shape）
    cr = parse_crypto({"data": [{"value": "18", "value_classification": "Extreme Fear",
                                 "timestamp": "1751846400"}] +
                               [{"value": str(20 + i)} for i in range(10)]})
    assert cr and cr["score"] == 18.0 and cr["week_ago"] == 25.0
    assert parse_crypto({"data": []}) is None

    # 雙訊號
    s = dual_signal(18, 20)
    assert s["both_extreme_fear"] and "底部" in s["note"]
    s2 = dual_signal(80, 90)
    assert s2["both_extreme_greed"]
    assert dual_signal(50, 10)["note"] is None
    assert dual_signal(None, 10)["note"] is None

    # 文字輸出
    t = fg_text(cnn, cr)
    assert "恐懼貪婪指數" in t and "非投資建議" in t and "美股恐貪 22" in t
    t1 = fg_text(cnn, cr, compact=True)
    assert "\n⚡" in t1 or "⚡" not in t1        # compact 單行 + 可選訊號行
    assert fg_text(None, None) == "恐懼貪婪指數：暫無資料"

    print(f"✅ sentiment_fg 離線自我測試通過（雙恐懼訊號: {s['note'][:20]}…）")
