#!/usr/bin/env bash
# Tests to run on the RPi during a freeze/crash/problem, to diagnose a memory leak or
# camera freeze. 

set -u
VIDEOS_DIR="${1:-}"

hr() { printf '%s\n' "----------------------------------------"; }

# 1) ffmpeg processes: expected 2 (box + corridor). Many, or zombies = leak.
hr; echo "[1] ffmpeg processes"
n=$(pgrep -x ffmpeg | wc -l)
echo "count: $n"
ps -o pid,ppid,rss,etime,stat,cmd -C ffmpeg 2>/dev/null | head -20
z=$(ps -o stat= -C ffmpeg 2>/dev/null | grep -c Z)
if   [ "$n" -gt 4 ]; then echo ">> LEAK LIKELY: $n ffmpeg (expected ~2). stop_recording isn't reaping them."
elif [ "$z" -gt 0 ]; then echo ">> ZOMBIES: $z defunct ffmpeg, not reaped."
else echo ">> ok ($n ffmpeg, $z zombies)"; fi

# 2) village process memory: RSS climbing over a session = the real leak.
hr; echo "[2] village process RSS"
pid=$(pgrep -f 'village/village/main.py' | head -1)
if [ -n "${pid:-}" ]; then
  rss_kb=$(awk '/VmRSS/{print $2}' /proc/"$pid"/status 2>/dev/null)
  echo "pid $pid  RSS: $(( ${rss_kb:-0} / 1024 )) MB"
  echo "(re-run every few min: RSS should be flat, not climbing)"
else echo ">> village process not found"; fi

# 3) memory + swap: if swap is used and growing, may cause freezes.
hr; echo "[3] memory / swap"
free -h
echo "--- swap in/out over 3s (si/so should be ~0 when healthy) ---"
vmstat 1 3 | tail -2

# 4) thermal throttle: days of load can throttle the Pi (not a leak, but lags).
hr; echo "[4] throttle / temp / camera detected"
command -v vcgencmd >/dev/null && { vcgencmd measure_temp; vcgencmd get_throttled; \
  vcgencmd get_camera; } || echo "(vcgencmd not available)"
echo "(get_throttled non-zero => throttling/undervoltage; get_camera detected=0 => link lost)"

# 5) DECISIVE when only village freezes (other apps fine): which Python thread
#    holds the GIL. Run this DURING a freeze.
hr; echo "[5] py-spy thread dump (the culprit line)"
if command -v py-spy >/dev/null && [ -n "${pid:-}" ]; then
  py-spy dump --pid "$pid" 2>&1 | head -60
else
  echo "(py-spy not installed or village not found: 'sudo pip install py-spy')"
fi

# 6) kernel camera-link errors: CSI-2 / ribbon-cable / sensor faults land here.
#    check for black frames / frozen feed
hr; echo "[6] kernel camera errors (dmesg)"
DM=$(dmesg -T 2>/dev/null || sudo dmesg -T 2>/dev/null)
if [ -n "$DM" ]; then
  echo "$DM" | grep -iE \
    'unicam|csi|rp1-cfe|cfe|imx[0-9]|Camera|corrupt|frame.*drop|drop.*frame|i2c.*(timeout|nack)|under.?voltage|over.?current' \
    | tail -30
  echo "(lines mentioning csi/unicam/imx/i2c = link or sensor fault => cable/heat/power;"
  echo " empty = no kernel camera fault logged, freeze is likely software/GIL, see [5])"
else
  echo "(dmesg empty or needs root, retry: sudo bash $0)"
fi

# 7) is the box video actually being written? encoder dead => size frozen.
#    Run DURING the freeze: stalled size == 'no video saved'.
hr; echo "[7] video file growth (encoder alive?)"
[ -z "$VIDEOS_DIR" ] && VIDEOS_DIR=$(find "$HOME" -maxdepth 6 -type d -path '*/data/videos' 2>/dev/null | head -1)
if [ -n "$VIDEOS_DIR" ]; then
  vid=$(ls -t "$VIDEOS_DIR"/*.mp4 2>/dev/null | head -1)
  if [ -n "${vid:-}" ]; then
    s1=$(stat -c%s "$vid"); sleep 2; s2=$(stat -c%s "$vid")
    echo "$vid"
    echo "size: $s1 -> $s2 bytes over 2s"
    if [ "$s2" -gt "$s1" ]; then echo ">> ok: encoder is writing frames"
    else echo ">> STALLED: file not growing, encoder dead / camera frozen (the bug)"; fi
  else echo "(no .mp4 in $VIDEOS_DIR)"; fi
else
  echo "(videos dir not found, pass it as arg: bash $0 /path/to/data/videos)"
fi
hr
