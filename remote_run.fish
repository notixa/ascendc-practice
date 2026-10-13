#!/usr/bin/env fish

# Atlas 200I DK A2 连接配置（对应 ~/.ssh/config 中的 atlas）
set REMOTE_HOST atlas
set REMOTE_DIR /home/HwHiAiUser/projects/ascendc

echo ">>> 正在增量同步源码到开发套件 ($REMOTE_HOST:$REMOTE_DIR)..."

# 自动在远端预建目录并执行增量同步
# 严格排除本地 x86 虚拟环境与 build 临时构建产物，保持板端干净
rsync -avz --delete \
    --rsync-path="mkdir -p $REMOTE_DIR && rsync" \
    --exclude '.venv' \
    --exclude '.git' \
    --exclude 'build' \
    --exclude 'out' \
    --exclude '*.run' \
    --exclude '.zed' \
    --exclude '*.log' \
    --exclude '*.pdf' \
    --exclude '*.md' \
    --exclude '.clangd_stub.h' \
    --exclude '.cache' \
    --exclude '*.so' \
    --exclude '*.a' \
    ./ $REMOTE_HOST:$REMOTE_DIR

if test $status -eq 0
    echo ">>> [OK] 代码同步完成！已实时更新至板端: $REMOTE_DIR"
else
    echo ">>> [ERROR] 同步失败，请检查网络或 SSH 连接！"
end
