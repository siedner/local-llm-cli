"""launchd integration for the local-llm daemon."""

from __future__ import annotations

import os
from pathlib import Path

from .constants import DAEMON_LOG_FILE, DEFAULT_HOST, DEFAULT_PORT, LAUNCHD_LABEL, LAUNCHD_PLIST, PACKAGE_ROOT
from .doctor import get_mlx_python


def write_launchd_plist(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> Path:
    """Write a user launchd plist for the daemon."""
    LAUNCHD_PLIST.parent.mkdir(parents=True, exist_ok=True)
    DAEMON_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    python = get_mlx_python()
    env_pythonpath = os.environ.get("PYTHONPATH", "")
    paths = [str(PACKAGE_ROOT)]
    if env_pythonpath:
        paths.append(env_pythonpath)
    pythonpath = ":".join(paths)

    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>{LAUNCHD_LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>{python}</string>
    <string>-m</string>
    <string>local_llm.daemon</string>
    <string>--host</string>
    <string>{host}</string>
    <string>--port</string>
    <string>{port}</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PYTHONPATH</key>
    <string>{pythonpath}</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>{DAEMON_LOG_FILE}</string>
  <key>StandardErrorPath</key>
  <string>{DAEMON_LOG_FILE}</string>
</dict>
</plist>
"""
    LAUNCHD_PLIST.write_text(plist, encoding="utf-8")
    return LAUNCHD_PLIST
