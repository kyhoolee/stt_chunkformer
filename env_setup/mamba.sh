#!/usr/bin/env bash
set -euo pipefail

# 1. Xác định kiến trúc máy
ARCH=$(uname -m)
if [[ "$ARCH" == "x86_64" ]]; then
  PLATFORM="linux-64"
elif [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
  PLATFORM="linux-aarch64"
else
  echo "ERROR: Unsupported architecture: $ARCH"
  exit 1
fi

# 2. Tạo thư mục tạm và tải Micromamba
TMPDIR=$(mktemp -d)
echo "Downloading Micromamba for $PLATFORM..."
curl -Ls "https://micromamba.snakepit.net/api/micromamba/$PLATFORM/latest" \
  | tar -xvj -C "$TMPDIR" bin/micromamba

# 3. Đặt quyền thực thi và di chuyển vào /usr/local/bin
chmod +x "$TMPDIR/bin/micromamba"
sudo mv "$TMPDIR/bin/micromamba" /usr/local/bin/

# 4. Dọn dẹp
rm -rf "$TMPDIR"

# 5. Khởi tạo shell integration (bash, zsh, fish… tuỳ bạn)
echo "Initializing micromamba shell integration for bash..."
micromamba shell init -s bash

# 6. (Tuỳ chọn) Tự động activate môi trường base khi mở shell
# Thêm vào cuối ~/.bashrc (hoặc ~/.bash_profile) dòng này:
echo '[[ -f ~/.bashrc ]] && source ~/.bashrc' >/dev/null 
grep -qxF 'micromamba activate base' ~/.bashrc || \
  echo 'micromamba activate base' >> ~/.bashrc

echo "✅ Micromamba đã được cài đặt thành công!"
echo "Mở lại terminal hoặc chạy 'source ~/.bashrc' để áp dụng."
echo "Kiểm tra: micromamba --version"
