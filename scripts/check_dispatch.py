"""CI 檢查（Bot 指令層，離線）：
1. scan_signals.process_commands 的指令分派不得有「被前面無條件分支遮蔽」的分支（#61）。
   用語法樹解析 if/elif 鏈：`cmd == "/x"`（不論單雙引號、有無額外條件）、`cmd in ("/x", "/y")`；
   同一指令一旦出現無條件分支，後面任何同指令分支都執行不到 → 失敗。
2. /help 與長訊息會被 _split_tg 切到 Telegram 上限內（#63）。
"""
import ast
import sys

src = open("scan_signals.py", encoding="utf-8").read()
tree = ast.parse(src)
fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "process_commands")


def cmds_of(test):
    """回 (指令集合, 是否無條件)。"""
    if isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "cmd" and len(test.ops) == 1:
        c = test.comparators[0]
        if isinstance(test.ops[0], ast.Eq) and isinstance(c, ast.Constant) and isinstance(c.value, str):
            return {c.value}, True
        if isinstance(test.ops[0], ast.In) and isinstance(c, (ast.Tuple, ast.List, ast.Set)):
            return {e.value for e in c.elts if isinstance(e, ast.Constant)}, True
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
        cs, _ = cmds_of(test.values[0])
        return cs, False
    return set(), False


problems, n_branches = [], 0
for node in ast.walk(fn):
    if not isinstance(node, ast.If):
        continue
    first, _ = cmds_of(node.test)
    if not first:
        continue
    seen_uncond = set()
    cur = node
    while isinstance(cur, ast.If):
        cs, uncond = cmds_of(cur.test)
        if cs:
            n_branches += 1
            shadowed = cs & seen_uncond
            if shadowed:
                problems.append(f"line {cur.lineno}: {sorted(shadowed)} 被前面的無條件分支遮蔽")
            if uncond:
                seen_uncond |= cs
        cur = cur.orelse[0] if len(cur.orelse) == 1 and isinstance(cur.orelse[0], ast.If) else None
    break                                   # 只檢查最外層分派鏈

if problems:
    print("dispatch problems:\n  " + "\n  ".join(problems))
    sys.exit(1)

sys.path.insert(0, ".")
import scan_signals as ss  # noqa: E402

h = ss._cmd_help()
parts = ss._split_tg(h)
assert all(len(p) <= ss.TG_LIMIT for p in parts) and "".join(p.replace("\n", "") for p in parts) == h.replace("\n", "")
long_line = "x" * 9000
assert [len(p) for p in ss._split_tg(long_line)] == [4000, 4000, 1000]
assert ss._split_tg("short") == ["short"]
print(f"dispatch OK ({n_branches} command branches, none shadowed); /help {len(h)} chars → {len(parts)} parts")
