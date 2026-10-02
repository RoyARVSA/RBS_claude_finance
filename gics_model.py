"""gics_model.py – GICS 階層式神經網路（Hierarchical multi-head MLP, pure NumPy；移植自使用者 gics_nn 專案）。

存檔改為 npz（allow_pickle=False）：權重、TF-IDF 詞彙/idf、SVD、標準化參數全部是純陣列，
讀檔不會執行任何程式碼（原版 pickle 在公開 repo 是程式碼注入風險）。

架構
    features ─► Dense(H, ReLU) ─► Dropout ─┬─► softmax head L1 (11  sectors)
                                           ├─► softmax head L2 (25  industry groups)
                                           ├─► softmax head L3 (74  industries)
                                           └─► softmax head L4 (163 sub-industries)

訓練：四個 head 的 cross-entropy 加權相加 (L1 權重最高 → 先把大分類學穩)。
推論：不是四個 head 各自取 argmax (那樣可能得到「L1=Energy、L4=Semiconductors」
這種不合法組合)，而是對 163 條合法路徑 L1→L2→L3→L4 計分：

    score(path) = Σ_l  w_l · log p_l(path_l)

取分數最高的路徑，保證「由大分類一路遞延到小分類」且上下層一致。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

import gics_taxonomy as tx

LEVELS = (1, 2, 3, 4)


def _level_codes(code8: str) -> dict[int, str]:
    return {l: code8[: tx.LEVEL_DIGITS[l]] for l in LEVELS}


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@dataclass
class HierMLP:
    hidden: int = 256
    dropout: float = 0.3
    lr: float = 2e-3
    weight_decay: float = 1e-4
    epochs: int = 250
    batch: int = 64
    level_w: tuple = (1.0, 0.8, 0.6, 0.5)     # loss weights for L1..L4
    decode_w: tuple = (1.0, 1.0, 1.0, 1.0)    # path-scoring weights
    seed: int = 7
    classes: dict = field(default_factory=dict)   # level -> list[code]
    params: dict = field(default_factory=dict)

    # ----------------------------------------------------------------- setup
    def _init(self, d_in: int):
        rng = np.random.default_rng(self.seed)
        self.classes = {l: tx.codes_at(l) for l in LEVELS}
        P = {"W1": rng.normal(0, np.sqrt(2 / d_in), (d_in, self.hidden)),
             "b1": np.zeros(self.hidden)}
        for l in LEVELS:
            k = len(self.classes[l])
            P[f"W{l}h"] = rng.normal(0, np.sqrt(1 / self.hidden), (self.hidden, k))
            P[f"b{l}h"] = np.zeros(k)
        self.params = P
        # path matrix: for each of 163 L4 codes, its class index at every level
        self._paths = np.array([[self.classes[l].index(_level_codes(c8)[l]) for l in LEVELS]
                                for c8 in self.classes[4]])

    # --------------------------------------------------------------- forward
    def _forward(self, X, train=False, rng=None):
        P = self.params
        z1 = X @ P["W1"] + P["b1"]
        h = np.maximum(z1, 0)
        mask = None
        if train and self.dropout > 0:
            mask = (rng.random(h.shape) > self.dropout) / (1 - self.dropout)
            h = h * mask
        probs = {l: _softmax(h @ P[f"W{l}h"] + P[f"b{l}h"]) for l in LEVELS}
        return z1, h, mask, probs

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, y_code8: list[str], verbose=False):
        X = np.asarray(X, dtype=np.float64)
        self._init(X.shape[1])
        Y = {l: np.array([self.classes[l].index(_level_codes(c)[l]) for c in y_code8]) for l in LEVELS}
        rng = np.random.default_rng(self.seed)
        m = {k: np.zeros_like(v) for k, v in self.params.items()}
        v = {k: np.zeros_like(v) for k, v in self.params.items()}
        b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
        n = len(X)
        for ep in range(self.epochs):
            order = rng.permutation(n)
            tot = 0.0
            for s in range(0, n, self.batch):
                idx = order[s:s + self.batch]
                xb = X[idx]
                z1, h, mask, probs = self._forward(xb, train=True, rng=rng)
                g = {}
                dh = np.zeros_like(h)
                for i, l in enumerate(LEVELS):
                    yb = Y[l][idx]
                    p = probs[l]
                    tot += -self.level_w[i] * np.log(p[np.arange(len(idx)), yb] + 1e-12).sum()
                    dz = p.copy()
                    dz[np.arange(len(idx)), yb] -= 1
                    dz *= self.level_w[i] / len(idx)
                    g[f"W{l}h"] = h.T @ dz
                    g[f"b{l}h"] = dz.sum(0)
                    dh += dz @ self.params[f"W{l}h"].T
                if mask is not None:
                    dh *= mask
                dz1 = dh * (z1 > 0)
                g["W1"] = xb.T @ dz1
                g["b1"] = dz1.sum(0)
                t += 1
                for k in self.params:                      # AdamW
                    m[k] = b1 * m[k] + (1 - b1) * g[k]
                    v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
                    mh, vh = m[k] / (1 - b1 ** t), v[k] / (1 - b2 ** t)
                    self.params[k] -= self.lr * (mh / (np.sqrt(vh) + eps) + self.weight_decay * self.params[k])
            if verbose and (ep % 50 == 0 or ep == self.epochs - 1):
                print(f"  epoch {ep:4d}  loss {tot / n:.4f}")
        return self

    # -------------------------------------------------------------- predict
    def level_probs(self, X):
        return self._forward(np.asarray(X, dtype=np.float64))[3]

    def path_scores(self, X):
        """(n, 163) log-score of each legal L1→L4 path."""
        probs = self.level_probs(X)
        S = np.zeros((len(X), len(self._paths)))
        for i, l in enumerate(LEVELS):
            S += self.decode_w[i] * np.log(probs[l][:, self._paths[:, i]] + 1e-12)
        return S

    def predict(self, X, top_k=3):
        """Hierarchically consistent prediction. Returns list of dicts per sample."""
        S = self.path_scores(X)
        conf = _softmax(S)
        out = []
        for r in range(len(S)):
            best = np.argsort(-S[r])[:top_k]
            cands = []
            for j in best:
                c8 = self.classes[4][j]
                cands.append({"code8": c8, "confidence": float(conf[r, j]), **tx.path(c8)})
            out.append({"best": cands[0], "alternatives": cands[1:]})
        return out

    def predict_code8(self, X):
        return [self.classes[4][j] for j in self.path_scores(X).argmax(1)]

    # ------------------------------------------------------------- persist（npz，無 pickle）
    HYPER = ("hidden", "dropout", "lr", "weight_decay", "epochs", "batch", "level_w", "decode_w", "seed")

    def to_arrays(self) -> dict:
        out = {f"p_{k}": v for k, v in self.params.items()}
        meta = {k: getattr(self, k) for k in self.HYPER}
        meta["classes"] = {str(l): self.classes[l] for l in LEVELS}
        out["mlp_meta"] = np.array(json.dumps(meta))
        return out

    @classmethod
    def from_arrays(cls, arrs) -> "HierMLP":
        meta = json.loads(str(arrs["mlp_meta"]))
        obj = cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in meta.items() if k in cls.HYPER})
        obj.classes = {int(l): list(v) for l, v in meta["classes"].items()}
        obj.params = {k[2:]: np.asarray(arrs[k], dtype=np.float64) for k in arrs.files if k.startswith("p_")}
        obj._paths = np.array([[obj.classes[l].index(_level_codes(c8)[l]) for l in LEVELS] for c8 in obj.classes[4]])
        return obj

    def save(self, path, fb: "FeatureBuilder | None" = None, extra: dict | None = None):
        """權重 + （可選）特徵管線 + 附註 json 存成單一 npz。"""
        arrs = self.to_arrays()
        if fb is not None:
            arrs.update(fb.to_arrays())
        arrs["info"] = np.array(json.dumps(extra or {}))
        tmp = str(path) + ".tmp.npz"
        np.savez_compressed(tmp, **arrs)
        import os
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        """回 (model, feature_builder | None, info dict)。allow_pickle=False：檔案只能含數字/字串陣列。"""
        with np.load(path, allow_pickle=False) as arrs:
            mdl = cls.from_arrays(arrs)
            fb = FeatureBuilder.from_arrays(arrs) if "fb_idf" in arrs.files else None
            info = json.loads(str(arrs["info"])) if "info" in arrs.files else {}
        return mdl, fb, info


# =========================================================== feature pipeline
class FeatureBuilder:
    """Text (business summary + Yahoo sector/industry) → TF-IDF → SVD,
    concatenated with return-correlation to the 11 SPDR sector ETFs.

    訓練用 sklearn；推論可只靠存下來的陣列重算（詞彙、idf、SVD 成分、標準化參數），
    與 sklearn 的 transform 逐位一致（自測），所以載入模型不需要 pickle。"""

    SECTOR_ETFS = ["XLE", "XLB", "XLI", "XLY", "XLP", "XLV", "XLF", "XLK", "XLC", "XLU", "XLRE"]
    TFIDF_KW = dict(stop_words="english", ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True)

    def __init__(self, n_components=192):
        self.n_components = n_components

    @staticmethod
    def _text(meta: dict) -> str:
        yf_tags = f"{meta.get('yf_sector', '')} {meta.get('yf_industry', '')}"
        # repeat the tags so the short Yahoo labels are not drowned by long summaries
        return f"{yf_tags} {yf_tags} {meta.get('summary', '')}"

    def fit_transform(self, metas, corr):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import StandardScaler
        self.tfidf = TfidfVectorizer(**self.TFIDF_KW)
        T = self.tfidf.fit_transform([self._text(m) for m in metas])
        k = min(self.n_components, T.shape[1] - 1, T.shape[0] - 1)
        self.svd = TruncatedSVD(k, random_state=0)
        Ts = self.svd.fit_transform(T)
        self.scaler = StandardScaler().fit(np.hstack([Ts, corr]))
        self._from_fitted()
        return self.scaler.transform(np.hstack([Ts, corr]))

    def _from_fitted(self):
        vocab = self.tfidf.vocabulary_
        self.terms = np.array(sorted(vocab, key=vocab.get))
        self.idf = np.asarray(self.tfidf.idf_, dtype=np.float64)
        self.components = np.asarray(self.svd.components_, dtype=np.float64)
        self.mean = np.asarray(self.scaler.mean_, dtype=np.float64)
        self.scale = np.asarray(self.scaler.scale_, dtype=np.float64)
        self._index = {t: i for i, t in enumerate(self.terms)}

    def _tfidf_rows(self, texts):
        """純 NumPy 重算 TF-IDF（sublinear tf、idf、L2 正規化），分詞器用 sklearn 同參數的 analyzer。"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        an = TfidfVectorizer(**self.TFIDF_KW).build_analyzer()
        M = np.zeros((len(texts), len(self.terms)))
        for r, txt in enumerate(texts):
            for tok in an(txt):
                j = self._index.get(tok)
                if j is not None:
                    M[r, j] += 1.0
        nz = M > 0
        M[nz] = 1.0 + np.log(M[nz])
        M *= self.idf
        n = np.linalg.norm(M, axis=1, keepdims=True)
        return M / np.where(n > 0, n, 1.0)

    def transform(self, metas, corr):
        Ts = self._tfidf_rows([self._text(m) for m in metas]) @ self.components.T
        return (np.hstack([Ts, np.asarray(corr, dtype=np.float64)]) - self.mean) / np.where(self.scale > 0, self.scale, 1.0)

    def to_arrays(self) -> dict:
        return {"fb_terms": self.terms.astype(str), "fb_idf": self.idf, "fb_components": self.components,
                "fb_mean": self.mean, "fb_scale": self.scale, "fb_ncomp": np.array(self.n_components)}

    @classmethod
    def from_arrays(cls, arrs) -> "FeatureBuilder":
        fb = cls(int(arrs["fb_ncomp"]))
        fb.terms = np.asarray(arrs["fb_terms"]).astype(str)
        fb.idf, fb.components = np.asarray(arrs["fb_idf"]), np.asarray(arrs["fb_components"])
        fb.mean, fb.scale = np.asarray(arrs["fb_mean"]), np.asarray(arrs["fb_scale"])
        fb._index = {t: i for i, t in enumerate(fb.terms)}
        return fb


def cross_validate(metas, corr, y_code8, folds=5, **mlp_kw):
    """K-fold accuracy per level: hierarchical path decoding vs independent argmax."""
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y_code8))
    parts = np.array_split(idx, folds)
    hits = {"path": {l: 0 for l in LEVELS}, "flat": {l: 0 for l in LEVELS}, "consistent_flat": 0}
    for k in range(folds):
        te = parts[k]
        tr = np.concatenate([parts[j] for j in range(folds) if j != k])
        fb = FeatureBuilder()
        Xtr = fb.fit_transform([metas[i] for i in tr], corr[tr])
        Xte = fb.transform([metas[i] for i in te], corr[te])
        mdl = HierMLP(**mlp_kw).fit(Xtr, [y_code8[i] for i in tr])
        pred = mdl.predict_code8(Xte)
        probs = mdl.level_probs(Xte)
        for r, i in enumerate(te):
            truth = _level_codes(y_code8[i])
            pc = _level_codes(pred[r])
            flat = {l: mdl.classes[l][probs[l][r].argmax()] for l in LEVELS}
            for l in LEVELS:
                hits["path"][l] += pc[l] == truth[l]
                hits["flat"][l] += flat[l] == truth[l]
            hits["consistent_flat"] += all(flat[l + 1].startswith(flat[l]) for l in (1, 2, 3))
        print(f"  fold {k + 1}/{folds} done")
    n = len(y_code8)
    return {
        "n": n,
        "hierarchical_path_acc": {f"L{l}": hits["path"][l] / n for l in LEVELS},
        "independent_argmax_acc": {f"L{l}": hits["flat"][l] / n for l in LEVELS},
        "independent_argmax_consistent_rate": hits["consistent_flat"] / n,
    }


# ------------------------------------------------------------------ selftest
if __name__ == "__main__":
    import tempfile
    rng = np.random.default_rng(1)
    subs = tx.SUB_INDUSTRIES
    metas, y, corr = [], [], []
    vocab = ["global", "leading", "provider", "company", "customers", "solutions", "products", "markets"]
    for c8 in subs:
        for _ in range(4):
            nm = " ".join(tx.NAMES[x] for x in (c8[:2], c8[:4], c8[:6], c8)).lower()
            metas.append({"summary": f"{nm} " + " ".join(rng.choice(vocab, 6)), "yf_sector": tx.NAMES[c8[:2]],
                          "yf_industry": tx.NAMES[c8[:6]]})
            y.append(c8)
            v = np.zeros(11); v[tx.codes_at(1).index(c8[:2])] = 0.8
            corr.append(v + rng.normal(0, 0.1, 11))
    corr = np.array(corr)
    idx = rng.permutation(len(y)); tr, te = idx[:520], idx[520:]
    fb = FeatureBuilder(64)
    Xtr = fb.fit_transform([metas[i] for i in tr], corr[tr])
    # 1) 純 NumPy transform 與 sklearn 逐位一致
    ref = fb.scaler.transform(np.hstack([fb.svd.transform(fb.tfidf.transform([fb._text(metas[i]) for i in te])), corr[te]]))
    mine = fb.transform([metas[i] for i in te], corr[te])
    assert np.max(np.abs(ref - mine)) < 1e-9, np.max(np.abs(ref - mine))
    # 2) 訓練 + 路徑解碼一致（L1..L4 前綴一致）+ 準確率合理
    mdl = HierMLP(epochs=60, hidden=128).fit(Xtr, [y[i] for i in tr])
    pred = mdl.predict(mine)
    for p in pred:
        c8 = p["best"]["code8"]
        assert all(p["best"][f"L{l}"]["code"] == c8[:tx.LEVEL_DIGITS[l]] for l in LEVELS)
    acc1 = np.mean([p["best"]["L1"]["code"] == y[i][:2] for p, i in zip(pred, te)])
    assert acc1 > 0.8, acc1
    # 3) npz 存讀（allow_pickle=False）→ 預測逐位一致
    with tempfile.TemporaryDirectory() as td:
        f = f"{td}/m.npz"
        mdl.save(f, fb, {"cv_l4": 0.5})
        m2, fb2, info = HierMLP.load(f)
        assert info["cv_l4"] == 0.5 and fb2 is not None
        X2 = fb2.transform([metas[i] for i in te], corr[te])
        assert np.max(np.abs(X2 - mine)) < 1e-12
        assert m2.predict_code8(X2) == mdl.predict_code8(mine)
        with np.load(f, allow_pickle=False) as a:
            assert all(a[k].dtype != object for k in a.files)
    print(f"gics_model selftest OK ✅（held-out L1 acc {acc1:.2f}）")
