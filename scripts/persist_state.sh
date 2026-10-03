#!/usr/bin/env bash
# 把本輪的 state 類檔案 commit + push 回目前分支（衝突安全：push 失敗就同步遠端、只重放本輪實際變更的檔案再推）。
# 由 signal_scan.yml 的最後一步與 actions_loop.py 每輪呼叫；冪等，無變更時什麼都不做。
# 結束碼：0 = 已推送或無變更（工作樹與遠端一致）；2 = 推送最終失敗（呼叫端不可同步遠端，免得丟掉本輪 state）。
set -u
BR="${GITHUB_REF_NAME:-}"
if [ -z "$BR" ]; then BR="$(git rev-parse --abbrev-ref HEAD)"; fi
if [ "$BR" = "HEAD" ] || [ -z "$BR" ]; then
  echo "persist: 找不到分支名（detached HEAD 且未設 GITHUB_REF_NAME），不推送"; exit 2
fi
git config user.email "github-actions[bot]@users.noreply.github.com"
git config user.name  "github-actions[bot]"
# 上次被逾時殺掉留下的 index.lock（超過 10 分鐘才清，避免誤刪進行中的）
if [ -f .git/index.lock ] && [ -n "$(find .git/index.lock -mmin +10 2>/dev/null)" ]; then rm -f .git/index.lock; fi

add_all() {
  # 分開 add：多 pathspec 是原子的，任一檔不存在會整批中止
  git add watchlist_state.json 2>/dev/null || true
  git add trade_journal.json 2>/dev/null || true
  git add estimates_ledger.json 2>/dev/null || true
  git add data/universe 2>/dev/null || true
  git add data/fin 2>/dev/null || true
}

add_all
if git diff --staged --quiet; then
  echo "persist: no state changes."
  exit 0
fi
# 只備份「本輪實際變更」的檔案（整包重放 data/fin 會把夜間工作流剛更新的檔改回舊版，#60 對抗驗證 M1）
B=$(mktemp -d)
git diff --staged --name-only > "$B/touched.txt"
while IFS= read -r f; do
  [ -n "$f" ] && [ -f "$f" ] && mkdir -p "$B/files/$(dirname "$f")" && cp "$f" "$B/files/$f"
done < "$B/touched.txt"
git commit -q -m "chore: update bot state + trade journal [skip ci]"

for i in 1 2 3; do
  if git push -q origin HEAD:"$BR"; then echo "persist: pushed."; rm -rf "$B"; exit 0; fi
  echo "persist: push 失敗，同步遠端後重放本輪檔案 ($i)…"
  git rebase --abort 2>/dev/null || true
  git fetch -q origin "$BR"
  git reset -q --hard FETCH_HEAD
  while IFS= read -r f; do
    if [ -n "$f" ] && [ -f "$B/files/$f" ]; then mkdir -p "$(dirname "$f")" && cp "$B/files/$f" "$f" && git add "$f"; fi
  done < "$B/touched.txt"
  git diff --staged --quiet || git commit -q -m "chore: update bot state + trade journal [skip ci]"
done
if git push -q origin HEAD:"$BR"; then echo "persist: pushed."; rm -rf "$B"; exit 0; fi
echo "::warning::persist: push 最終失敗（狀態未持久化，下一輪重試）"
rm -rf "$B"
exit 2
