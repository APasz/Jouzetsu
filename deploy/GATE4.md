# Gate 4 deployment acceptance

The first native ARM deployment passed service, storage, permissions and network
checks. Machine identifiers and raw operational records are kept outside version
control; this report records the reusable implementation and validation results.

## Implementation

- Added a non-login `jouzetsu` service account and shared group, with normal
  editing access for the locally configured deployment administrator.
- Added opt-in shared-file permissions to the central atomic writer, preserving
  private defaults. Tests cover new/replaced files, invalid settings and failed
  permission changes without damaging prior data.
- Added SSD-local uv, Python 3.14.4 and frozen ARM dependencies, plus an independent
  validation environment.
- Added systemd boot startup, crash recovery, graceful shutdown, SSD mount
  requirements and service file/home restrictions.
- Added a dedicated firewall table, generated from local network settings.
- Added code-only rsync updates that protect application data/configuration and
  virtual environments. Host settings and private records are excluded from the
  code transfer; settings are sent separately over SSH when deploying.
- Added generic desktop Ollama listener/firewall examples and operating guidance.

## Deployment validation

These results describe the initial deployed application, before the subsequent
privacy/configuration cleanup:

| Check | Result |
| --- | --- |
| Full desktop suite | 805 passed, 2 deprecation warnings |
| Native ARM suite excluding browser tests | 687 passed, 118 deselected, 2 deprecation warnings |
| Ruff | All checks passed |
| Strict basedpyright | 0 errors, 0 warnings, 0 notes |
| Diff whitespace check | Passed |
| Real-account file access | Service and administrator could edit each other's persistent project files |
| Service restrictions | Code/tooling rejected writes; administrator SSH files inaccessible |
| Crash recovery | Process restarted automatically after SIGKILL and recovered HTTP service |
| Graceful stop with open SSE | Stream drained and closed; stop completed in 0.265 seconds |
| Boot startup | Application/firewall returned automatically; HTTP 200 and zero failed units |
| Storage and power | SSD mounts present; no throttle flags |
| Real Ollama discovery | Pi-to-desktop API and application backend checks passed; three models discovered |

LAN checks confirmed HTTP 200 for the approval page, and HTTP 403 for unapproved
events, access administration, forged localhost forwarding headers and mutations
without CSRF/approval. Existing automated browser and simulated-backend coverage
also passed. The first full suite had one model-picker scroll assertion failure;
the focused rerun and final full suite passed without changing that UI code.

## Practical checks and limits

Choose an approval phrase through the localhost SSH tunnel and approve each
intended browser. Configure the desired reachable backend/model and exercise
real generation, long streams and phone reconnection. Discovery confirms
connectivity but is not an inference-generation test. Tailnet enrolment, backup
features, automatic rollback and destructive power-loss testing remain outside
this deployment's acceptance checks.

Application durability depends on the storage stack honouring synchronisation.
Record hardware-specific limitations and host warnings privately instead of
publishing an installation inventory.
