# Remote video transcription with faster-whisper on Proxmox

Transcribes video and audio recordings (for example lecture recordings in MKV) into plain text and SRT subtitles, using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) on an NVIDIA GPU inside an LXC container on a Proxmox host. A small Python client on your laptop uploads the file, runs the transcription remotely, downloads the results and cleans up, with one command. It works on the home network and, through Tailscale, from anywhere.

## How it works

```
Laptop (macOS/Linux)                    Proxmox host                     LXC container (GPU)
transcribe-remote.py  --SFTP/SSH-->     /tmp  --pct push-->              /root/<file>
                                                                          trascrivi (faster-whisper large-v3)
<folder>/trascrizioni/*.txt,*.srt  <--  /tmp  <--pct pull--              /root/<file>.txt, .srt
```

The transcription time does not depend on where you are: only the upload and download of the files do.

## Requirements

| Component | Detail |
|---|---|
| Proxmox VE host | With the NVIDIA driver installed and working (`nvidia-smi` lists the GPU) |
| GPU | NVIDIA, tested on an RTX 3060 with 12 GB of VRAM |
| Container | Debian 12 LXC, 2 cores, 8 GB RAM, 30 GB disk |
| Python packages in the container | `faster-whisper`, `nvidia-cublas-cu12`, `nvidia-cudnn-cu12` |
| Model | Whisper `large-v3`, `int8_float16` (about 2 GB of VRAM, about 3 GB of disk for the cached model) |
| Client | Python 3 and `paramiko`, SSH access to the Proxmox host |
| Optional | Tailscale on the host and on the client, for access from outside the home network |

Tested versions: faster-whisper 1.2.1, ctranslate2 4.8.2, nvidia-cublas-cu12 12.9.2.10, nvidia-cudnn-cu12 9.26.0.51, NVIDIA driver 595.80 (CUDA 13.2), ffmpeg 5.1.

Placeholders used below: `<CT_IP>` is the address of the container and `<GATEWAY_IP>` your router.

## Setup

### 1. Create the container on the Proxmox host

```bash
pct create 103 local:vztmpl/debian-12-standard_12.12-1_amd64.tar.zst \
  --hostname whisper \
  --cores 2 \
  --memory 8192 \
  --rootfs local-lvm:30 \
  --unprivileged 1 \
  --net0 name=eth0,bridge=vmbr0,ip=<CT_IP>/24,gw=<GATEWAY_IP>
```

Adjust the template name to the one returned by `pveam list local`.

### 2. Give the container access to the GPU

Read the major numbers of the NVIDIA devices on the host. They depend on the driver version:

```bash
ls -la /dev/nvidia*
```

Stop the container and append the following lines to `/etc/pve/lxc/103.conf`, replacing `195`, `511` and `236` with the numbers you found for `/dev/nvidia0`, `/dev/nvidia-uvm` and `/dev/nvidia-caps`:

```
lxc.cgroup2.devices.allow: c 195:* rwm
lxc.cgroup2.devices.allow: c 511:* rwm
lxc.cgroup2.devices.allow: c 236:* rwm
lxc.mount.entry: /dev/nvidia0 dev/nvidia0 none bind,optional,create=file
lxc.mount.entry: /dev/nvidiactl dev/nvidiactl none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-uvm dev/nvidia-uvm none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-uvm-tools dev/nvidia-uvm-tools none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-caps dev/nvidia-caps none bind,optional,create=dir
```

In an unprivileged container the device files must be accessible to "other" users on the host. Check the permissions with `ls -l /dev/nvidia*` if the GPU is not visible inside the container.

```bash
pct start 103
pct enter 103
```

Inside the container, install the NVIDIA user-space libraries with the same installer version as the host. The kernel module stays on the host, so it must be skipped:

```bash
apt update && apt install -y wget
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/595.80/NVIDIA-Linux-x86_64-595.80.run
chmod +x NVIDIA-Linux-x86_64-595.80.run
./NVIDIA-Linux-x86_64-595.80.run --no-kernel-module --no-questions --ui=none
```

```bash
nvidia-smi
```

The GPU must be listed. Use the installer version that matches your host driver.

### 3. Install the Python environment (inside the container)

```bash
apt install -y python3 python3-venv ffmpeg
python3 -m venv /root/whisper-env
```

```bash
/root/whisper-env/bin/pip install faster-whisper nvidia-cublas-cu12 nvidia-cudnn-cu12
```

### 4. Make the CUDA libraries visible

ctranslate2 looks for cuBLAS and cuDNN through `LD_LIBRARY_PATH`, but the pip packages install them inside the virtual environment. Export the path every time the environment is activated:

```bash
cat >> /root/whisper-env/bin/activate << 'EOF'

# --- LD_LIBRARY_PATH for NVIDIA cublas/cudnn ---
_NVIDIA_LIB="$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="$_NVIDIA_LIB/cublas/lib:$_NVIDIA_LIB/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset _NVIDIA_LIB
EOF
```

Adjust `python3.11` if your virtual environment uses another Python version. Then check that the model really loads on the GPU:

```bash
source /root/whisper-env/bin/activate
python3 -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cuda', compute_type='int8_float16'); print('Model loaded on GPU')"
```

The first run downloads the model (about 3 GB). Counting CUDA devices is not enough as a test: only loading a model proves that cuBLAS and cuDNN work.

### 5. Install the transcription script

From the Proxmox host, with this repository copied to it:

```bash
pct push 103 container/trascrivi /usr/local/bin/trascrivi
pct exec 103 -- chmod +x /usr/local/bin/trascrivi
pct exec 103 -- /usr/local/bin/trascrivi --help
```

### 6. Set up the client

```bash
pip3 install paramiko
```

Tell the client where the Proxmox host is. Set one or both variables, for example in your shell profile:

```bash
export PROXMOX_HOST_LAN="<proxmox-lan-address>"
export PROXMOX_HOST_TAILSCALE="<proxmox-tailscale-address>"
```

The script tries the LAN address first (TCP port 22, 2 second timeout) and falls back to the Tailscale address. To get the Tailscale address of the host, run `tailscale ip -4` on it.

## Usage

Transcribe one file, or every audio/video file in a folder (alphabetical order, files that already have both outputs are skipped):

```bash
python3 transcribe-remote.py /path/to/recording.mkv
```

```bash
python3 transcribe-remote.py /path/to/folder
```

Without an argument the script asks for the path. It asks once for the SSH password of the Proxmox user.

Results are written to a `trascrizioni` folder inside the folder you passed (or next to the file). Existing files are never overwritten: a ` (1)`, ` (2)` suffix is added instead. Remote temporary files are deleted after each file.

To transcribe by hand from the Proxmox host:

```bash
pct push 103 /tmp/video.mkv /root/video.mkv
pct exec 103 -- trascrivi /root/video.mkv
pct pull 103 /root/video.txt /tmp/video.txt
pct pull 103 /root/video.srt /tmp/video.srt
```

Language defaults to Italian; use `trascrivi FILE --lingua en` for another language.

## Settings

| Parameter | Value | Why |
|---|---|---|
| `model` | `large-v3` | Best quality available |
| `compute_type` | `int8_float16` | Int8 weights, float16 compute: about 2 GB of VRAM with good accuracy |
| `beam_size` | 5 | Good balance of speed and quality |
| `vad_filter` | `True` | Voice activity detection skips silences and reduces invented text |
| `min_silence_duration_ms` | 500 | Pauses shorter than 500 ms are not treated as silence |
| `language` | `it` | Forced language avoids detection errors on short or noisy segments |

## Performance

Measured on a 63 minute, 225 MB recording on an RTX 3060:

| Metric | Value |
|---|---|
| Processing time | 5 min 19 s |
| Speed | About 12 times faster than real time |
| VRAM used by Whisper | About 2 GB |

## Troubleshooting

| Message or symptom | Cause | Fix |
|---|---|---|
| `TypeError: expected str, bytes or os.PathLike object, not NoneType` when locating the NVIDIA libraries | `nvidia.cublas.lib` is a namespace package, so its `__file__` is `None` | Use `list(nvidia.cublas.lib.__path__)[0]` instead of `os.path.dirname(nvidia.cublas.lib.__file__)` |
| Model fails to load on GPU although CUDA devices are found | cuBLAS or cuDNN not on `LD_LIBRARY_PATH` | Apply step 4 and activate the environment through `activate` |
| `no Proxmox address configured` | Neither environment variable is set | Export `PROXMOX_HOST_LAN` and/or `PROXMOX_HOST_TAILSCALE` |
| `Proxmox is not reachable via LAN or Tailscale` | Wrong address or host offline | Check the addresses and the connection |
| Shell parse errors with `scp` or paths with spaces | Unquoted path | Wrap the path in double quotes |

## Notes

- Do not run two transcriptions at the same time: the VRAM would not be enough, especially if the GPU is shared with other services.
- The first transcription after the container is rebuilt downloads the model again. Later runs use the cache in `/root/.cache/huggingface/`.
- The script logs in with the SSH password of the Proxmox user and accepts unknown host keys automatically. Use it only on networks you trust, or through Tailscale.
- Tailscale carries SSH traffic inside an encrypted tunnel and exposes no port to the internet.

## License

MIT, see [LICENSE](LICENSE).
