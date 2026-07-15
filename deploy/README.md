# Deploy — Synthio Voice AI on AWS EC2 (24/7)

A single always-on EC2 box runs everything: the FastAPI voice backend, two local
LightRAG servers, and Caddy (TLS + static frontend). No serverless — the app
holds persistent WebSockets and in-memory RAG graphs.

## Specs & cost

| | |
|---|---|
| **Region** | us-west-1 (N. California) |
| **Instance** | t3.small (x86, 2 vCPU/2 GB) or **t4g.small** (ARM, ~25% cheaper) |
| **Disk** | 30 GB gp3 |
| **Domain** | dummy via **sslip.io** — `<public-ip>.sslip.io`, auto-derived; real HTTPS, no domain purchase |
| **Hosting** | ~$12–16/mo on-demand 24/7 (~$10/mo with a 1-yr Savings Plan) + ~$2.5 storage |
| **Per call** | ~$0.10–0.15 (Cartesia TTS + Deepgram STT dominate) |

Why the box is small enough for 2 GB: torch / sentence-transformers no longer
load (A2UI uses OpenAI embeddings; emotion is off), so the backend is ~450 MB.

## Steps

1. **Get AWS keys OUT of the project `.env`.** They belong in `~/.aws/credentials`
   on your laptop only — this app never calls AWS at runtime.

2. **Launch** (us-west-1): EC2 → **t3.small** (or t4g.small) → Ubuntu 24.04 →
   30 GB gp3. Security group inbound: 22 (your IP), 80, 443. Attach an Elastic IP.

3. **On the box:**
   ```bash
   sudo git clone <your-repo-url> /opt/synthio
   sudo cp /opt/synthio/deploy/.env.production.example /opt/synthio/.env
   sudo nano /opt/synthio/.env        # paste the API keys (NOT AWS keys)
   sudo chmod 600 /opt/synthio/.env
   sudo bash /opt/synthio/deploy/bootstrap.sh
   ```

4. **Done.** bootstrap prints `https://<ip>.sslip.io`. First load waits a few
   seconds for the Let's Encrypt cert. Mic works (valid HTTPS).

`bootstrap.sh` handles: system deps + Node + Caddy → backend venv/deps →
LightRAG venv + two servers → **Dupixent ingest** (~$0.05, skipped if already
populated) → frontend build pointed at the domain → systemd services →
Caddy auto-HTTPS.

## Operate

```bash
journalctl -u synthio-backend -f          # backend logs
systemctl status synthio-lightrag-hcp     # RAG server (HCP)
systemctl restart synthio-backend         # restart after a code pull
```

Concurrent-call cap is `MAX_SESSIONS` in `.env` (default 6 for 2 GB).

## Update after a code change
```bash
cd /opt/synthio && sudo git pull
sudo /opt/synthio/.venv/bin/pip install -r requirements.txt
cd client && sudo -u ubuntu npm ci && sudo -u ubuntu npm run build
sudo systemctl restart synthio-backend
```
