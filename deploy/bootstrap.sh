#!/usr/bin/env bash
#
# Synthio Voice AI — one-run EC2 bootstrap (Ubuntu 24.04, t3.small / t4g.small).
#
# Prereqs on the box BEFORE running:
#   1. git clone <repo> /opt/synthio
#   2. cp /opt/synthio/deploy/.env.production.example /opt/synthio/.env
#      -> fill in the API keys, then: chmod 600 /opt/synthio/.env
#   3. sudo bash /opt/synthio/deploy/bootstrap.sh
#
# Idempotent-ish: safe to re-run (it skips ingest if the graph already exists).

set -euo pipefail

APP_DIR=/opt/synthio
RUN_USER=ubuntu
cd "$APP_DIR"

echo "==> 1/8  System packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3 python3-venv python3-pip curl git debian-keyring debian-archive-keyring apt-transport-https
# Node 20
if ! command -v node >/dev/null; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y nodejs
fi
# Caddy
if ! command -v caddy >/dev/null; then
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | tee /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -y && apt-get install -y caddy
fi

echo "==> 2/8  Load .env + derive dummy domain (sslip.io)"
[ -f "$APP_DIR/.env" ] || { echo "ERROR: $APP_DIR/.env missing (copy deploy/.env.production.example)"; exit 1; }
set -a; . "$APP_DIR/.env"; set +a
# Public IPv4 via IMDSv2 -> dashed sslip.io domain (valid HTTPS, mic works)
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 60" || true)
PUBIP=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/public-ipv4 || true)
[ -n "${PUBIP:-}" ] || { echo "ERROR: could not read public IP from metadata"; exit 1; }
SITE_ADDRESS="${PUBIP//./-}.sslip.io"
echo "    domain: https://$SITE_ADDRESS"

echo "==> 3/8  Backend venv + deps (lean: no torch/transformers)"
python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install --quiet --upgrade pip
"$APP_DIR/.venv/bin/pip" install --quiet -r "$APP_DIR/requirements.txt"

echo "==> 4/8  LightRAG venv + local server config"
python3 -m venv "$APP_DIR/.lightrag-venv"
"$APP_DIR/.lightrag-venv/bin/pip" install --quiet --upgrade pip
"$APP_DIR/.lightrag-venv/bin/pip" install --quiet "lightrag-hku[api]"
mkdir -p "$APP_DIR/lightrag_local/rag_storage_hcp" "$APP_DIR/lightrag_local/rag_storage_patient"
# Generate LIGHTRAG_API_KEY if the operator left it blank
if [ -z "${LIGHTRAG_API_KEY:-}" ]; then
  LIGHTRAG_API_KEY="lr-$(openssl rand -hex 12)"
  sed -i "s|^LIGHTRAG_API_KEY=.*|LIGHTRAG_API_KEY=${LIGHTRAG_API_KEY}|" "$APP_DIR/.env"
  echo "    generated LIGHTRAG_API_KEY"
fi
cat > "$APP_DIR/lightrag_local/.env" <<EOF
HOST=127.0.0.1
LIGHTRAG_API_KEY=${LIGHTRAG_API_KEY}
LLM_BINDING=openai
LLM_MODEL=gpt-4o-mini
LLM_BINDING_HOST=https://api.openai.com/v1
LLM_BINDING_API_KEY=${OPENAI_API_KEY}
OPENAI_API_KEY=${OPENAI_API_KEY}
EMBEDDING_BINDING=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536
EMBEDDING_BINDING_HOST=https://api.openai.com/v1
EMBEDDING_BINDING_API_KEY=${OPENAI_API_KEY}
MAX_ASYNC=4
EOF
chmod 600 "$APP_DIR/lightrag_local/.env"
# The LightRAG services run as $RUN_USER, so the config + storage they read/write
# must be owned by $RUN_USER (bootstrap runs as root). Do this BEFORE starting
# the services, not just in the later chown -R.
chown -R "$RUN_USER:$RUN_USER" "$APP_DIR/lightrag_local"

echo "==> 5/8  Install + start systemd services"
cp "$APP_DIR/deploy/systemd/"*.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now synthio-lightrag-hcp synthio-lightrag-patient
# Wait for both LightRAG servers to answer /health
for p in 9621 9622; do
  for i in $(seq 1 40); do
    curl -sf -H "X-API-Key: ${LIGHTRAG_API_KEY}" "http://127.0.0.1:$p/health" >/dev/null && break
    sleep 3
  done
done

echo "==> 6/8  Ingest Dupixent label (skip if graph already populated)"
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
export LIGHTRAG_API_KEY LIGHTRAG_BASE_URL_HCP=http://localhost:9621 LIGHTRAG_BASE_URL_PATIENT=http://localhost:9622
HCP_DOCS=$(curl -sf -H "X-API-Key: ${LIGHTRAG_API_KEY}" http://127.0.0.1:9621/documents/status_counts | grep -o '"all":[0-9]*' | grep -o '[0-9]*' || echo 0)
if [ "${HCP_DOCS:-0}" -lt 5 ]; then
  sudo -u "$RUN_USER" env PYTHONPATH="$APP_DIR" NO_PROXY=localhost,127.0.0.1 \
    LIGHTRAG_API_KEY="$LIGHTRAG_API_KEY" LIGHTRAG_BASE_URL_HCP=http://localhost:9621 LIGHTRAG_BASE_URL_PATIENT=http://localhost:9622 \
    "$APP_DIR/.venv/bin/python" "$APP_DIR/scripts/ingest_dupixent.py" --fetch
  for scope in hcp patient; do
    sudo -u "$RUN_USER" env PYTHONPATH="$APP_DIR" NO_PROXY=localhost,127.0.0.1 \
      LIGHTRAG_API_KEY="$LIGHTRAG_API_KEY" LIGHTRAG_BASE_URL_HCP=http://localhost:9621 LIGHTRAG_BASE_URL_PATIENT=http://localhost:9622 \
      "$APP_DIR/.venv/bin/python" "$APP_DIR/scripts/ingest_dupixent.py" --ingest "$scope"
  done
  echo "    ingest done (allow a few min for background graph extraction)"
else
  echo "    graph already has $HCP_DOCS docs — skipping ingest"
fi

echo "==> 7/8  Build frontend + point it at this domain"
cd "$APP_DIR/client"
sudo -u "$RUN_USER" npm ci --silent
sudo -u "$RUN_USER" npm run build --silent
cat > "$APP_DIR/client/dist/config.js" <<EOF
window.__BACKEND_URL__ = "https://${SITE_ADDRESS}";
window.__WS_URL__ = "wss://${SITE_ADDRESS}";
window.LIGHTRAG_URL = "";
window.LIGHTRAG_API_KEY = "";
window.__VERSION__ = "prod";
EOF
chown -R "$RUN_USER:$RUN_USER" "$APP_DIR"

echo "==> 8/8  Start backend + Caddy (auto-HTTPS)"
systemctl enable --now synthio-backend
install -d /etc/caddy
cp "$APP_DIR/deploy/Caddyfile" /etc/caddy/Caddyfile
# Pass SITE_ADDRESS to Caddy via its systemd drop-in environment
mkdir -p /etc/systemd/system/caddy.service.d
cat > /etc/systemd/system/caddy.service.d/override.conf <<EOF
[Service]
Environment=SITE_ADDRESS=${SITE_ADDRESS}
EOF
systemctl daemon-reload
systemctl enable --now caddy
systemctl restart caddy

echo ""
echo "=================================================================="
echo " ✅  Deployed.  Open:  https://${SITE_ADDRESS}"
echo "     (first load waits a few seconds for the Let's Encrypt cert)"
echo " Logs:  journalctl -u synthio-backend -f"
echo "=================================================================="
