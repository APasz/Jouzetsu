# Raspberry Pi deployment

The deployment tools target Raspberry Pi OS (Debian 13, aarch64), with an SSD
mounted at `/srv`. The OS may remain on SD. The service listens on port **6900**.
All hostnames, administrator accounts and trusted network values come from a
local configuration; the public examples use documentation-only addresses.

## Local settings

Copy `deploy/local.example.toml` to `deploy/local.toml` and replace its example
values with your SSH alias, existing administrator account, LAN interface and
trusted IPv4 subnet. Missing or invalid settings stop deployment before any
service is stopped. The application service account is always named `jouzetsu`.

`deploy/local.toml` and `deploy/private/` are ignored by Git. Keep actual host
inventories, addresses, deployment reports and other operational records there.
The code transfer excludes both paths. During an explicit deployment, the helper
separately transfers `local.toml` over SSH with owner-only permissions so the Pi's
setup module uses the same settings. Private records are never transferred.
Do not put approval phrases or API credentials in deployment settings.

## Files and permissions

```text
/srv/projects/jouzetsu/
  app/                 source checkout and production .venv/
  config.json          production configuration
  data/                chats, characters, images and application logs
  tmp/                 application temporary files on the SSD
  .tools/              uv binaries and managed ARM Python
  .cache/              dependency/download cache
  .validation/         separate development/test environment
```

The configured administrator owns code/tooling and belongs to the `jouzetsu`
group. The service runs as the non-login `jouzetsu` account. Mutable directories
use the shared group, setgid inheritance and default ACLs, so both accounts can
edit persistent project files. `JOUZETSU_SHARED_FILES=true` makes atomic
replacements `0660`; group membership alone would not override `mkstemp`'s
owner-only file mode. `UMask=0007` covers logs and ordinary files. Reconnect
existing SSH sessions after group membership changes. Protected OS files still
require normal `sudo` administration.

The systemd service makes code and tooling read-only, hides `/home`, removes
capabilities, and grants writes to the application root for atomic configuration
replacement and its data/tmp directories. Stop the service before editing saved
configuration or chat documents externally, then restart to reload them.

## Initial installation

Prerequisites are a mounted SSD at `/srv`, working SSH key authentication and an
existing administrator with `sudo`. Configure `local.toml` first. In commands
below, replace `app-host.example` with the SSH alias from your local settings.

For the initial account/directory bootstrap, transfer the `deploy` package and
local configuration to a private temporary directory on the Pi:

```sh
ssh app-host.example 'umask 077; mkdir -p /tmp/jouzetsu-bootstrap/deploy'
rsync -a --chmod=D700,F600 --exclude=private/ --exclude=__pycache__/ deploy/ \
  app-host.example:/tmp/jouzetsu-bootstrap/deploy/
ssh -t app-host.example \
  'cd /tmp/jouzetsu-bootstrap && sudo env PYTHONDONTWRITEBYTECODE=1 python3 -m deploy.setup_pi --prepare-only'
```

From the desktop checkout, run `python3 -m scripts.deploy_pi --sync-only`.
Then, on the Pi as the deployment administrator:

```sh
cd /srv/projects/jouzetsu/app
python3 -m deploy.install_runtime
sudo env PYTHONDONTWRITEBYTECODE=1 python3 -m deploy.setup_pi
```

Remove the temporary bootstrap directory after successful installation. The
runtime installer verifies the uv release checksum and installs Python 3.14.4
plus `uv sync --frozen --no-dev`. Everything is native aarch64 and remains under
the project root. The service executes Python directly and does not resolve or
download packages at boot. Runtime installation runs unprivileged; account,
firewall and systemd installation require `sudo`.

## Browser approval and inference

Fresh installations have no approval phrase and no backend. To administer access
from the desktop, open an SSH tunnel:

```sh
ssh -N -L 16900:127.0.0.1:6900 app-host.example
```

Open **http://127.0.0.1:16900/** and set a phrase in **Access & Logs**. Enter it
in each intended LAN browser. The tunnel's actual socket peer is localhost, so
it grants administrative access. Approved LAN browsers can manage normal
settings/backends but cannot administer access or reveal the phrase. Stop the
tunnel with Ctrl+C. Spoofed forwarding headers do not grant local administration.

Add inference backends using the desktop's reachable LAN address. For example,
`http://192.0.2.10:1234/v1` for LM Studio or `http://192.0.2.10:11434` for Ollama;
replace the documentation address. `127.0.0.1` in the Pi's backend settings refers
to the Pi. Inference servers must listen beyond desktop loopback and permit Pi
connections. Browsing/editing remain available when inference is offline.

### Optional desktop Ollama configuration

The files in `deploy/desktop-ollama/` are examples for the **desktop**. Replace
their documentation addresses with the application host and inference desktop
addresses before installing:

- `20-lan.conf.example` becomes
  `/etc/systemd/system/ollama.service.d/20-lan.conf`.
- `ollama-client-firewall.service.example` becomes
  `/etc/systemd/system/ollama-client-firewall.service`.
- `ollama-client.nft.example` becomes `/etc/nftables.d/ollama-client.nft`.

The override sets `OLLAMA_HOST=0.0.0.0:11434` and requires its firewall service.
The separate `inet ollama_client` table allows desktop loopback clients and the
specified application host, rejecting other remote clients on port 11434. Other
ports/tables are untouched. Existing firewall rules can impose further limits.
Local clients continue using `http://127.0.0.1:11434`.

Validate the edited rules before loading them, then reload systemd and restart:

```sh
sudo nft --check --file /etc/nftables.d/ollama-client.nft
sudo systemctl daemon-reload
sudo systemctl restart ollama-client-firewall.service
sudo systemctl restart ollama.service
```

`OLLAMA_HOST` and service overrides follow the
[official Ollama Linux configuration instructions](https://github.com/ollama/ollama/blob/main/docs/faq.mdx).

## Service, logs and updates

```sh
sudo systemctl status jouzetsu.service
sudo systemctl restart jouzetsu.service
journalctl -u jouzetsu.service -b
```

Application logs are in `data/logs/`; process output goes to journald. Persistent
OS logs on the SSD are a separate host configuration. The service starts at
boot, restarts after crashes and allows 45 seconds for graceful shutdown and
draft flushing. Inference jobs do not resume after a process restart.

The service requires `/srv` and stops with that mount. Its mountpoint condition
prevents using SD storage underneath an unavailable SSD. The Pi can still boot
without the SSD when its fstab mount is configured with `nofail`.

From the desktop checkout:

```sh
python3 -m scripts.deploy_pi --dry-run
python3 -m scripts.deploy_pi
```

Only `jouzetsu/`, `scripts/`, `deploy/`, `tests/`, `pyproject.toml`, `uv.lock` and
`README.md` are transferred into `app/`, with the private-path exclusions above.
Rsync deletes obsolete code while protecting `.venv/`, `config.json` and `data/`.
Production mutable files reside outside that transfer root. The separately
transferred local deployment settings contain machine identifiers, not application
credentials. A dry run neither stops services nor copies settings or changes files.

An update stops at the first failed transfer/install/check. Inspect the error and
logs, correct it and rerun deployment. No automated rollback or backups are
provided. Keep independent backups of irreplaceable data.

## Network boundary and Tailscale

`JOUZETSU_PORT` in the unit enforces port **6900**. The setup module renders
`jouzetsu.nft.in` using the validated interface/subnet from `local.toml` and
installs the resulting private host configuration under `/etc/nftables.d/`.
The firewall allows loopback and the configured trusted LAN, rejecting other
traffic to this port. Other ports/tables are untouched. Jouzetsu requires this
firewall service before starting; other firewall rules may still restrict access.

Tailscale enrolment is separate. After enrolling the host, add an explicit
`iifname "tailscale0" tcp dport 6900 accept` rule before the rejection and reload
the dedicated firewall service. Restrict intended devices through tailnet access
controls. Keep HTTP on a trusted LAN or encrypted Tailscale link and avoid public
port forwarding or Funnel exposure. An exit node does not by itself publish the
application; routing, access policy and forwarding configuration govern exposure.

## Verification

Run desktop checks as documented in the main README. Native ARM tests should use
a separate validation environment:

```sh
cd /srv/projects/jouzetsu/app
export UV_CACHE_DIR=/srv/projects/jouzetsu/.cache/uv
export UV_PYTHON_INSTALL_DIR=/srv/projects/jouzetsu/.tools/python
export UV_PROJECT_ENVIRONMENT=/srv/projects/jouzetsu/.validation/.venv
export TMPDIR=/srv/projects/jouzetsu/tmp
/srv/projects/jouzetsu/.tools/bin/uv sync --frozen --group dev --python 3.14.4
/srv/projects/jouzetsu/.validation/.venv/bin/pytest -q -m 'not browser'
```

Tests use temporary application homes. Host acceptance should check real-account
cross-access, HTTP/CSRF and local-admin rules, graceful stop, crash restart, boot
startup, mount dependencies and firewall rules. Exercise the intended physical
phone/browser and real inference model separately. Keep host-specific acceptance
records under the ignored `deploy/private/` directory rather than in public docs.
