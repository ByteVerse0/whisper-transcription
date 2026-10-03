#!/usr/bin/env python3
"""
transcribe-remote.py

Sends video/audio files to the Whisper container (LXC 103) on Proxmox,
transcribes them with faster-whisper on the GPU, and downloads the .txt and
.srt files into a single "trascrizioni" folder located in the path you pass
(inside the folder, or in the folder containing the file). The folder is created
if missing; existing files are never overwritten, a " (n)" suffix is added.

Usage:
    python3 transcribe-remote.py [FILE_OR_FOLDER]

- FILE: transcribes that single file.
- FOLDER: transcribes every audio/video file in it (not recursive), in
  alphabetical order. Files that already have both .txt and .srt are skipped.
- No argument: asks for the path.

One SSH connection and one password prompt for the whole batch.
Works both from the local network and remotely via Tailscale.

Requirements: pip install paramiko
Environment variables (set at least one):
    PROXMOX_HOST_LAN        address of the Proxmox host on the local network
    PROXMOX_HOST_TAILSCALE  address of the Proxmox host on the Tailscale network
"""

import sys
import os

VENV_PYTHON = os.path.expanduser("~/.venvs/whisper/bin/python")

if os.path.isfile(VENV_PYTHON) and sys.executable != os.path.realpath(VENV_PYTHON):
    try:
        import paramiko  # noqa: F401
    except ImportError:
        os.execv(VENV_PYTHON, [VENV_PYTHON] + sys.argv)

import getpass
import shlex
import signal
import socket

try:
    import paramiko
except ImportError:
    print("This script requires paramiko.")
    print("Install it with:  pip install paramiko")
    sys.exit(1)

# Addresses of the Proxmox host, read from the environment so that no real IP is
# stored in the repository. Set at least one of the two (see README).
PROXMOX_HOST_LAN = os.environ.get("PROXMOX_HOST_LAN")
PROXMOX_HOST_TAILSCALE = os.environ.get("PROXMOX_HOST_TAILSCALE")
PROXMOX_USER = "root"
LXC_ID = "103"
REMOTE_CMD = "/usr/local/bin/transcribe"  # name of the transcription script inside the container
OUTPUT_DIR_NAME = "trascrizioni"  # created next to the original files

MEDIA_EXTENSIONS = {
    ".mkv", ".mp4", ".mov", ".avi", ".webm",
    ".m4a", ".mp3", ".wav", ".flac", ".ogg",
}


def _pick_host():
    """Try the LAN first; if it does not respond, use Tailscale."""
    candidates = [
        (host, label)
        for host, label in [(PROXMOX_HOST_LAN, "LAN"), (PROXMOX_HOST_TAILSCALE, "Tailscale")]
        if host
    ]
    if not candidates:
        print("Error: no Proxmox address configured.")
        print("Set PROXMOX_HOST_LAN and/or PROXMOX_HOST_TAILSCALE (see README).")
        sys.exit(1)
    for host, label in candidates:
        try:
            s = socket.create_connection((host, 22), timeout=2)
            s.close()
            print(f"Connecting via {label} ({host})")
            return host
        except OSError:
            continue
    print("Error: Proxmox is not reachable via LAN or Tailscale.")
    print("Check that PROXMOX_HOST_LAN / PROXMOX_HOST_TAILSCALE are correct and the host is online.")
    sys.exit(1)


def _collect_files(target):
    """Return the list of media files for a file or folder path."""
    if os.path.isfile(target):
        return [target]
    if os.path.isdir(target):
        return sorted(
            os.path.join(target, f)
            for f in os.listdir(target)
            if not f.startswith(".") and os.path.splitext(f)[1].lower() in MEDIA_EXTENSIONS
        )
    return []


def _output_dir(target):
    """One 'trascrizioni' folder: inside the pasted folder, or in the pasted file's folder."""
    parent = target if os.path.isdir(target) else os.path.dirname(os.path.abspath(target))
    return os.path.join(os.path.abspath(parent), OUTPUT_DIR_NAME)


def _already_done(file_path, out_dir):
    base = os.path.join(out_dir, os.path.splitext(os.path.basename(file_path))[0])
    return os.path.isfile(base + ".txt") and os.path.isfile(base + ".srt")


def _free_output_paths(out_dir, base):
    """Return (txt, srt) paths that do not exist yet, adding ' (n)' if needed."""
    n = 0
    while True:
        name = base if n == 0 else f"{base} ({n})"
        txt = os.path.join(out_dir, f"{name}.txt")
        srt = os.path.join(out_dir, f"{name}.srt")
        if not os.path.exists(txt) and not os.path.exists(srt):
            return txt, srt
        n += 1


def main():
    signal.signal(signal.SIGINT, lambda *_: (print("\nInterrupted."), sys.exit(1)))

    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = input("Path of the file or folder: ")
    target = os.path.expanduser(target.strip().strip("'\"").replace("\\ ", " "))

    if not os.path.exists(target):
        print(f"Error: path not found: {target}")
        sys.exit(1)

    out_dir = _output_dir(target)
    files = _collect_files(target)
    if os.path.isdir(target):
        todo = [f for f in files if not _already_done(f, out_dir)]
        if len(files) - len(todo):
            print(f"Skipping {len(files) - len(todo)} already transcribed file(s).")
        files = todo

    if not files:
        print("Nothing to transcribe.")
        return
    print(f"{len(files)} file(s) to transcribe.")

    if os.path.isdir(out_dir):
        print(f"Output folder: {out_dir} (already exists, files will be added)")
    else:
        os.makedirs(out_dir)
        print(f"Output folder: {out_dir} (created)")

    host = _pick_host()
    password = getpass.getpass(f"Password for {PROXMOX_USER}@{host}: ")

    print(f"Connecting to {host}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(host, username=PROXMOX_USER, password=password, timeout=10)
    except paramiko.AuthenticationException:
        print("Error: wrong password.")
        sys.exit(1)
    except Exception as e:
        print(f"Connection error: {e}")
        sys.exit(1)

    ssh.get_transport().set_keepalive(30)
    print("Connected.")

    failed = []
    try:
        for i, path in enumerate(files, 1):
            print(f"\n=== File {i}/{len(files)}: {os.path.basename(path)} ===")
            try:
                _process_file(ssh, path, out_dir)
            except Exception as e:
                print(f"\nError: {e}")
                failed.append(os.path.basename(path))
    finally:
        ssh.close()

    print(f"\nFinished: {len(files) - len(failed)}/{len(files)} transcribed.")
    if failed:
        print("Failed:")
        for name in failed:
            print(f"  {name}")
        sys.exit(1)


def _process_file(ssh, file_path, out_dir):
    filename = os.path.basename(file_path)
    base = os.path.splitext(filename)[0]
    size_mb = os.path.getsize(file_path) / (1024 * 1024)

    local_txt, local_srt = _free_output_paths(out_dir, base)

    q_tmp = shlex.quote(f"/tmp/{filename}")
    q_root_file = shlex.quote(f"/root/{filename}")
    q_root_txt = shlex.quote(f"/root/{base}.txt")
    q_root_srt = shlex.quote(f"/root/{base}.srt")
    q_tmp_txt = shlex.quote(f"/tmp/{base}.txt")
    q_tmp_srt = shlex.quote(f"/tmp/{base}.srt")

    try:
        print(f"[1/5] Upload: {filename} ({size_mb:.1f} MB)")
        sftp = ssh.open_sftp()
        sftp.put(file_path, f"/tmp/{filename}", callback=_progress)
        sftp.close()
        print()

        print(f"[2/5] Copying into container {LXC_ID}...")
        _run(ssh, f"pct push {LXC_ID} {q_tmp} {q_root_file}")
        _run(ssh, f"rm -f {q_tmp}")

        dur = _run(ssh, f"pct exec {LXC_ID} -- ffprobe -v quiet -show_entries format=duration -of csv=p=0 {q_root_file}")
        dur_s = float(dur.strip())
        dur_min = int(dur_s // 60)
        dur_sec = int(dur_s % 60)
        print(f"\n[3/5] Transcribing ({dur_min}m {dur_sec}s of audio/video)...\n")
        _run(ssh, f"pct exec {LXC_ID} -- {REMOTE_CMD} {q_root_file}", stream=True)

        print(f"\n[4/5] Downloading results...")
        _run(ssh, f"pct pull {LXC_ID} {q_root_txt} {q_tmp_txt}")
        _run(ssh, f"pct pull {LXC_ID} {q_root_srt} {q_tmp_srt}")

        sftp = ssh.open_sftp()
        sftp.get(f"/tmp/{base}.txt", local_txt)
        sftp.get(f"/tmp/{base}.srt", local_srt)
        sftp.close()

        print(f"[5/5] Cleaning up remote files...")
        _run(ssh, f"pct exec {LXC_ID} -- rm -f {q_root_file} {q_root_txt} {q_root_srt}")
        _run(ssh, f"rm -f {q_tmp_txt} {q_tmp_srt}")

        print(f"Done: {local_txt}")
        print(f"      {local_srt}")

    except Exception:
        print("\nCleaning up remote files...")
        _run_safe(ssh, f"pct exec {LXC_ID} -- rm -f {q_root_file} {q_root_txt} {q_root_srt}")
        _run_safe(ssh, f"rm -f {q_tmp} {q_tmp_txt} {q_tmp_srt}")
        raise


def _run(ssh, cmd, stream=False):
    _, stdout, stderr = ssh.exec_command(cmd, timeout=None)
    if stream:
        for line in stdout:
            print(f"  {line.rstrip()}")
    exit_code = stdout.channel.recv_exit_status()
    if exit_code != 0:
        err = stderr.read().decode().strip()
        raise RuntimeError(f"{cmd.split()[0]}: {err}")
    if not stream:
        return stdout.read().decode()


def _run_safe(ssh, cmd):
    try:
        _run(ssh, cmd)
    except Exception:
        pass


def _progress(transferred, total):
    pct = transferred / total * 100
    bar_len = 30
    filled = int(bar_len * transferred / total)
    bar = "=" * filled + "-" * (bar_len - filled)
    mb_done = transferred / (1024 * 1024)
    mb_total = total / (1024 * 1024)
    print(f"\r  [{bar}] {mb_done:.1f}/{mb_total:.1f} MB ({pct:.0f}%)", end="", flush=True)


if __name__ == "__main__":
    main()
