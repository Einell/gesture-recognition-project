#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
macOS 音乐播放器：播放 10 秒后自动关闭（无淡出）
依赖：ffmpeg + pynput
"""

import subprocess
import time
import os
import sys
import threading
from pynput import keyboard as pynput_keyboard

# ── 配置 ───────────────────────────────────────
MUSIC_FILE = "/Users/ein/Music/Music/music.mp3"  # 👈 请确认路径存在
AUTO_STOP_AFTER = 10.0  # 播放 10 秒后自动停止（秒）

# ── 全局变量 ───────────────────────────────────
stop_requested = threading.Event()
player_process = None
key_listener = None


# ── 工具函数 ───────────────────────────────────
def get_duration(file_path):
    """用 ffprobe 获取音频总时长（秒）"""
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path
        ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=3)
        return float(result.stdout.strip())
    except Exception:
        return None


def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def safe_terminate(proc):
    if proc and proc.poll() is None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            pass


def stop_all():
    global player_process
    stop_requested.set()
    safe_terminate(player_process)
    player_process = None
    # 保险：杀死残留 afplay
    subprocess.run(["pkill", "-KILL", "afplay"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ── 按键监听 ───────────────────────────────────
def on_press(key):
    try:
        if key == pynput_keyboard.KeyCode.from_char('q') or key == pynput_keyboard.Key.space:
            print("\n⏹ 手动停止（q/空格）")
            stop_all()
            return False
    except AttributeError:
        pass
    return True


def start_listener():
    global key_listener
    key_listener = pynput_keyboard.Listener(on_press=on_press)
    key_listener.start()


# ── 主播放逻辑 ─────────────────────────────────
def play_music():
    global player_process

    if not os.path.isfile(MUSIC_FILE):
        print(f"❌ 文件不存在：{MUSIC_FILE}")
        return

    # 获取总时长（用于显示，非必需）
    total_duration = get_duration(MUSIC_FILE)
    display_duration = format_time(total_duration) if total_duration else "??"

    print(f"🎵 正在播放：{os.path.basename(MUSIC_FILE)}")
    print(f"⏱ 总时长：{display_duration}｜自动停止：{AUTO_STOP_AFTER} 秒")
    print("ℹ️ 按 [q] 或 [空格] 可提前停止\n")

    # 启动 afplay（全音量）
    player_process = subprocess.Popen(
        ["afplay", "-v", "1.0", MUSIC_FILE],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # 启动监听
    stop_requested.clear()
    start_listener()

    start_time = time.time()
    try:
        while player_process.poll() is None and not stop_requested.is_set():
            elapsed = time.time() - start_time

            # 更新进度条
            if total_duration:
                progress = min(elapsed / total_duration, 1.0)
                bar = "█" * int(30 * progress) + "░" * (30 - int(30 * progress))
                sys.stdout.write(
                    f"\r[{bar}] {format_time(elapsed)} / {display_duration} "
                    f"({progress*100:.1f}%)"
                )
            else:
                sys.stdout.write(f"\r▶ 已播放：{elapsed:.1f} 秒")
            sys.stdout.flush()

            # 检查是否到 10 秒
            if elapsed >= AUTO_STOP_AFTER and not stop_requested.is_set():
                print(f"\n⏰ {AUTO_STOP_AFTER} 秒到！自动停止播放...")
                stop_all()
                break

            time.sleep(0.1)

        if not stop_requested.is_set():
            sys.stdout.write("\n✅ 自然播放结束\n")

    except KeyboardInterrupt:
        print("\n🛑 Ctrl+C 中断")
        stop_all()
    finally:
        if key_listener and key_listener.is_alive():
            key_listener.stop()
        stop_all()
        sys.stdout.write("\n")


# ── 入口 ───────────────────────────────────────
if __name__ == "__main__":
    play_music()