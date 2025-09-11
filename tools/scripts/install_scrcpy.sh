#!/usr/bin/env bash
# Re-exec with bash if not already running under bash (e.g., when invoked via `sh`)
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

# Install latest scrcpy release (Linux tarball + server), with SHA256 validation
# Usage: VER=3.3.2 ./install_scrcpy.sh

VER=${VER:-3.3.2}
ARCH_RAW=$(uname -m)
case "$ARCH_RAW" in
  x86_64|amd64)
    ARCH=x86_64
    ;;
  aarch64|arm64)
    ARCH=aarch64
    ;;
  *)
    echo "[WARN] Unsupported arch '$ARCH_RAW', defaulting to x86_64"
    ARCH=x86_64
    ;;
esac

BASE_URL="https://github.com/Genymobile/scrcpy/releases/download/v$VER"
TARBALL="scrcpy-linux-$ARCH-v$VER.tar.gz"
SERVER_ASSET="scrcpy-server-v$VER"   # server filename has no .jar in releases
SUMS="SHA256SUMS.txt"

TMP_DIR=$(mktemp -d)
cleanup() { rm -rf "$TMP_DIR" || true; }
trap cleanup EXIT

echo "[INFO] Installing scrcpy v$VER for $ARCH (detected: $ARCH_RAW)"
echo "[INFO] Using base URL: $BASE_URL"

echo "[STEP] Downloading checksum file ($SUMS)"
curl -L --progress-bar "$BASE_URL/$SUMS" -o "$TMP_DIR/$SUMS"

echo "[STEP] Downloading tarball ($TARBALL)"
curl -L --progress-bar "$BASE_URL/$TARBALL" -o "$TMP_DIR/$TARBALL"

echo "[STEP] Downloading server asset ($SERVER_ASSET)"
curl -L --progress-bar "$BASE_URL/$SERVER_ASSET" -o "$TMP_DIR/$SERVER_ASSET"

echo "[STEP] Validating SHA256 checksums"
exp_tarball=$(grep -E "\s$TARBALL$" "$TMP_DIR/$SUMS" | awk '{print $1}')
exp_server=$(grep -E "\s$SERVER_ASSET$" "$TMP_DIR/$SUMS" | awk '{print $1}')

if [[ -z "$exp_tarball" || -z "$exp_server" ]]; then
  echo "[ERROR] Failed to find expected entries in $SUMS"
  exit 1
fi

act_tarball=$(sha256sum "$TMP_DIR/$TARBALL" | awk '{print $1}')
act_server=$(sha256sum "$TMP_DIR/$SERVER_ASSET" | awk '{print $1}')

if [[ "$exp_tarball" != "$act_tarball" ]]; then
  echo "[ERROR] Tarball checksum mismatch"; echo "expected: $exp_tarball"; echo "actual:   $act_tarball"; exit 1
fi
if [[ "$exp_server" != "$act_server" ]]; then
  echo "[ERROR] Server checksum mismatch"; echo "expected: $exp_server"; echo "actual:   $act_server"; exit 1
fi
echo "[OK] Checksums valid"

INSTALL_PREFIX="/usr/local/share/scrcpy/$VER"
BIN_LINK="/usr/local/bin/scrcpy"
SERVER_DST_DIR="/usr/local/share/scrcpy"
SERVER_DST="$SERVER_DST_DIR/scrcpy-server.jar"

echo "[STEP] Extracting tarball to $INSTALL_PREFIX"
sudo mkdir -p "$INSTALL_PREFIX"
sudo tar -xzf "$TMP_DIR/$TARBALL" -C "$INSTALL_PREFIX"

# Try to find the scrcpy binary inside extracted tree
BIN_SRC=$(find "$INSTALL_PREFIX" -type f -name scrcpy -perm -111 | head -n1 || true)
if [[ -z "$BIN_SRC" ]]; then
  # Some archives unpack without nested dir
  if [[ -f "$INSTALL_PREFIX/scrcpy" ]]; then
    BIN_SRC="$INSTALL_PREFIX/scrcpy"
  else
    echo "[WARN] Could not locate 'scrcpy' binary in $INSTALL_PREFIX; continuing (server-only install)"
  fi
fi

if [[ -n "$BIN_SRC" ]]; then
  echo "[STEP] Installing scrcpy binary to $BIN_LINK"
  sudo install -m 0755 "$BIN_SRC" "$BIN_LINK"
fi

echo "[STEP] Installing server to $SERVER_DST"
sudo mkdir -p "$SERVER_DST_DIR"
sudo install -m 0644 "$TMP_DIR/$SERVER_ASSET" "$SERVER_DST"

echo "[INFO] Exporting SCRCPY_SERVER_PATH for current shell"
export SCRCPY_SERVER_PATH="$SERVER_DST"
echo "export SCRCPY_SERVER_PATH=$SERVER_DST" | tee "$TMP_DIR/.scrcpy_env" >/dev/null
echo "[HINT] To persist, add the following to your shell profile:"
echo "export SCRCPY_SERVER_PATH=$SERVER_DST"

echo "[DONE] scrcpy v$VER installed. Validate with: scrcpy --version || true"