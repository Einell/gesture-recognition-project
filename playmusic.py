# macos的音乐播放模块
# 提供音乐播放、停止接口，支持自定义播放时长后自动关闭
import subprocess
import time
import threading
from pathlib import Path

_current_proc = None
_stop_playing = False

# 音乐路径
DEFAULT_MUSIC_FILE = "/Users/ein/Music/Music/music.mp3"
# 默认播放时长
DEFAULT_DURATION = 10.0


# 获取音频时长
def get_duration(file_path):
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

# 安全终止进程
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

# 强制停止所有 afplay 进程
def stop_all_afplay():
    subprocess.run(["pkill", "-KILL", "afplay"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# 停止播放
def stop_playback(verbose=True):
    global _stop_playing, _current_proc
    _stop_playing = True
    if _current_proc:
        safe_terminate(_current_proc)
        stop_all_afplay()
        _current_proc = None
        if verbose:
            print("音乐播放已停止")
    return True


# 音乐播放
def play(music_file=None, duration=None, verbose=True):

    global _current_proc, _stop_playing
    music_file = Path(music_file or DEFAULT_MUSIC_FILE)
    duration = duration or DEFAULT_DURATION

    if not music_file.exists():
        if verbose:
            print(f"音乐文件不存在!")
        return False

    # 重置停止标志
    _stop_playing = False

    # 后台播放
    def _play_task():
        global _current_proc, _stop_playing

        try:
            if verbose:
                total_dur = get_duration(music_file)
                display_total = f"{int(total_dur) // 60:02d}:{int(total_dur) % 60:02d}" if total_dur else "??"
                print(f"开始播放: {music_file.name} ,将在{duration} 秒后自动停止")
            proc = subprocess.Popen(
                ["afplay", "-v", "1.0", str(music_file)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            _current_proc = proc

            start_time = time.time()
            while proc.poll() is None and not _stop_playing:
                if time.time() - start_time >= duration:
                    break
                time.sleep(0.1)

            safe_terminate(proc)
            stop_all_afplay()
            _current_proc = None

            if verbose:
                if _stop_playing:
                    print("音乐播放已提前停止")
                else:
                    print("音乐播放已停止")

        except Exception as e:
            if verbose:
                print(f"播放异常: {e}")
            _current_proc = None

    # 启动后台线程
    thread = threading.Thread(target=_play_task, daemon=True, name="MusicPlayer")
    thread.start()
    return True
