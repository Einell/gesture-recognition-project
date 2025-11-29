#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
macOS 音乐播放模块（供手势识别项目调用）
✅ 可导入：import playmusic; playmusic.play()
✅ 可独立运行：python playmusic.py
"""

import subprocess
import time
import os
import sys
import threading
from pathlib import Path

# ── 默认配置（可被参数覆盖）───────────────────────
DEFAULT_MUSIC_FILE = "/Users/ein/Music/Music/music.mp3"
DEFAULT_DURATION = 10.0  # 秒


# ── 核心工具函数 ────────────────────────────────
def get_duration(file_path):
    """用 ffprobe 获取音频时长（秒），失败返回 None"""
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(file_path)
        ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=3)
        return float(result.stdout.strip())
    except Exception:
        return None


def safe_terminate(proc):
    """安全终止子进程"""
    if proc and proc.poll() is None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            pass


def stop_all_afplay():
    """强制停止所有 afplay 进程"""
    subprocess.run(["pkill", "-KILL", "afplay"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ── 主播放函数（对外 API）────────────────────────
def play(music_file=None, duration=None, verbose=True):
    """
    播放音乐（10 秒后自动停止）

    参数:
        music_file (str | Path): 音乐文件路径，默认为 DEFAULT_MUSIC_FILE
        duration (float): 播放时长（秒），默认 10.0
        verbose (bool): 是否打印日志

    返回:
        bool: True 表示启动成功
    """
    music_file = Path(music_file or DEFAULT_MUSIC_FILE)
    duration = duration or DEFAULT_DURATION

    if not music_file.exists():
        if verbose:
            print(f"❌ 音乐文件不存在: {music_file}")
        return False

    # 后台播放逻辑（不阻塞调用线程）
    def _play_task():
        try:
            if verbose:
                total_dur = get_duration(music_file)
                display_total = f"{int(total_dur) // 60:02d}:{int(total_dur) % 60:02d}" if total_dur else "??"
                print(f"🎵 开始播放: {music_file.name} | 自动停止: {duration} 秒")

            proc = subprocess.Popen(
                ["afplay", "-v", "1.0", str(music_file)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            start_time = time.time()
            while proc.poll() is None:
                if time.time() - start_time >= duration:
                    break
                time.sleep(0.1)

            safe_terminate(proc)
            stop_all_afplay()

            if verbose:
                print("⏹ 音乐播放已停止")

        except Exception as e:
            if verbose:
                print(f"⚠️ 播放异常: {e}")

    # 启动后台线程
    thread = threading.Thread(target=_play_task, daemon=True, name="MusicPlayer")
    thread.start()
    return True


# ── 独立运行入口（保持兼容）───────────────────────
if __name__ == "__main__":
    # 支持命令行参数: python playmusic.py [file] [duration]
    import argparse

    parser = argparse.ArgumentParser(description="macOS 音乐播放器")
    parser.add_argument("file", nargs="?", default=DEFAULT_MUSIC_FILE, help="音乐文件路径")
    parser.add_argument("duration", nargs="?", type=float, default=DEFAULT_DURATION, help="播放时长（秒）")
    args = parser.parse_args()

    print("🚀 手动测试模式")
    success = play(
        music_file=args.file,
        duration=args.duration,
        verbose=True
    )
    if not success:
        sys.exit(1)