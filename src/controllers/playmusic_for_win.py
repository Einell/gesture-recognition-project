# windows_music_player.py
# Windows 音乐播放模块
# 提供音乐播放、停止接口，支持自定义播放时长后自动关闭
import subprocess
import time
from pathlib import Path
import pygame
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# 尝试导入 Windows 音频控制
try:
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL

    HAS_PYCAW = True
except ImportError:
    HAS_PYCAW = False
    print("提示: 安装 'pip install pycaw comtypes' 以获得音量控制功能")

_current_proc = None
_stop_playing = False
_pygame_initialized = False

# 音乐路径
DEFAULT_MUSIC_FILE = '../../resources/music.mp3'
# 默认播放时长
DEFAULT_DURATION = 10.0


def init_pygame():
    """初始化 pygame 音频系统"""
    global _pygame_initialized
    if not _pygame_initialized:
        try:
            pygame.mixer.init()
            _pygame_initialized = True
            print("Pygame 音频系统初始化成功")
        except Exception as e:
            print(f"Pygame 初始化失败: {e}")
            return False
    return _pygame_initialized


def get_duration(file_path):
    """获取音频文件时长（使用多种方法）"""
    try:
        # 方法1: 使用 ffprobe
        try:
            result = subprocess.run([
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(file_path)
            ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=3)
            duration = result.stdout.strip()
            if duration:
                return float(duration)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        # 方法2: 使用 pygame（用于常见格式）
        if init_pygame():
            try:
                sound = pygame.mixer.Sound(file_path)
                return sound.get_length()
            except:
                pass


        return None
    except Exception as e:
        print(f"获取音频时长失败: {e}")
        return None


def safe_terminate(proc):
    """安全终止进程"""
    if proc and proc.poll() is None:
        try:
            # 先尝试终止，再强制结束
            proc.terminate()
            try:
                proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            pass


def stop_all_media_processes():
    """强制停止所有媒体播放进程"""
    try:
        # 停止 pygame 播放
        if _pygame_initialized:
            pygame.mixer.music.stop()

        # 终止可能的播放器进程
        processes_to_kill = ['ffplay', 'wmplayer', 'music', 'audio']
        for proc_name in processes_to_kill:
            try:
                subprocess.run(f"taskkill /f /im {proc_name}.exe",
                               shell=True,
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
            except:
                pass
    except Exception:
        pass


def set_windows_volume(volume_percent):
    """设置 Windows 系统音量"""
    if not HAS_PYCAW:
        return False

    try:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(
            IAudioEndpointVolume._iid_,
            CLSCTX_ALL,
            None
        )
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        volume.SetMasterVolumeLevelScalar(volume_percent / 100.0, None)
        return True
    except Exception as e:
        print(f"设置系统音量失败: {e}")
        return False


def stop_playback(verbose=True):
    """停止播放"""
    global _stop_playing, _current_proc
    _stop_playing = True

    # 停止 pygame 播放
    if _pygame_initialized:
        pygame.mixer.music.stop()

    # 终止子进程
    if _current_proc:
        safe_terminate(_current_proc)
        _current_proc = None

    # 停止其他可能的媒体进程
    stop_all_media_processes()

    if verbose:
        print("音乐播放已停止")
    return True


def play_with_ffplay(music_file, duration, verbose=True):
    """使用 ffplay 播放音乐"""
    global _current_proc, _stop_playing

    try:
        # 检查 ffplay 是否可用
        subprocess.run(["ffplay", "-version"],
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,
                       timeout=1)

        # 获取音频时长用于显示
        total_dur = get_duration(music_file)
        if verbose and total_dur:
            display_total = f"{int(total_dur) // 60:02d}:{int(total_dur) % 60:02d}"
            print(f"开始播放: {music_file.name} (时长: {display_total})")
            print(f"将在 {duration} 秒后自动停止")

        # 使用 ffplay 播放（自动结束后台运行）
        cmd = [
            "ffplay",
            "-nodisp",  # 不显示窗口
            "-autoexit",  # 播放完后自动退出
            "-loglevel", "quiet",  # 安静模式
            str(music_file)
        ]

        _current_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW  # Windows: 不创建控制台窗口
        )

        # 等待指定时间或进程结束
        start_time = time.time()
        while _current_proc.poll() is None and not _stop_playing:
            if time.time() - start_time >= duration:
                break
            time.sleep(0.1)

        safe_terminate(_current_proc)
        _current_proc = None

        return True

    except FileNotFoundError:
        if verbose:
            print("未找到 ffplay，请安装 FFmpeg")
        return False
    except Exception as e:
        if verbose:
            print(f"ffplay 播放失败: {e}")
        return False


def play_with_pygame(music_file, duration, verbose=True):
    """使用 pygame 播放音乐"""
    global _stop_playing

    try:
        if not init_pygame():
            return False

        # 获取音频时长
        total_dur = get_duration(music_file)
        if verbose and total_dur:
            display_total = f"{int(total_dur) // 60:02d}:{int(total_dur) % 60:02d}"
            print(f"开始播放: {music_file.name} (时长: {display_total})")
            print(f"将在 {duration} 秒后自动停止")

        # 加载并播放音乐
        pygame.mixer.music.load(str(music_file))
        pygame.mixer.music.play()

        # 等待指定时间
        start_time = time.time()
        while pygame.mixer.music.get_busy() and not _stop_playing:
            if time.time() - start_time >= duration:
                break
            time.sleep(0.1)

        pygame.mixer.music.stop()

        return True

    except Exception as e:
        if verbose:
            print(f"pygame 播放失败: {e}")
        return False


def play_with_windows(music_file, duration, verbose=True):
    """使用 Windows 默认播放器"""
    global _current_proc, _stop_playing

    try:
        # 获取音频时长
        total_dur = get_duration(music_file)
        if verbose and total_dur:
            display_total = f"{int(total_dur) // 60:02d}:{int(total_dur) % 60:02d}"
            print(f"开始播放: {music_file.name} (时长: {display_total})")
            print(f"将在 {duration} 秒后自动停止")

        # 使用 start 命令在后台播放
        cmd = f'start /MIN /B wmplayer "{music_file}" /play /close'

        _current_proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW
        )

        # 等待指定时间
        start_time = time.time()
        while not _stop_playing:
            if time.time() - start_time >= duration:
                break
            time.sleep(0.1)

        # 停止播放
        stop_all_media_processes()
        safe_terminate(_current_proc)
        _current_proc = None

        return True

    except Exception as e:
        if verbose:
            print(f"Windows 播放器播放失败: {e}")
        return False


def play(music_file=None, duration=None, verbose=True, method='auto'):
    """
    播放音乐

    参数:
    - music_file: 音乐文件路径
    - duration: 播放时长（秒）
    - verbose: 是否显示详细信息
    - method: 播放方法 ('auto', 'ffplay', 'pygame', 'windows')
    """
    global _current_proc, _stop_playing

    music_file = Path(music_file or DEFAULT_MUSIC_FILE).resolve()
    duration = duration or DEFAULT_DURATION

    if not music_file.exists():
        if verbose:
            print(f"音乐文件不存在: {music_file}")
        return False

    # 重置停止标志
    _stop_playing = False

    # 选择播放方法
    play_methods = []

    if method == 'auto':
        # 自动选择最佳播放方式
        play_methods = ['ffplay', 'pygame', 'windows']
    else:
        play_methods = [method]

    success = False
    for method_name in play_methods:
        if verbose:
            print(f"尝试使用 {method_name} 播放...")

        if method_name == 'ffplay':
            success = play_with_ffplay(music_file, duration, verbose)
        elif method_name == 'pygame':
            success = play_with_pygame(music_file, duration, verbose)
        elif method_name == 'windows':
            success = play_with_windows(music_file, duration, verbose)

        if success:
            break
        elif verbose:
            print(f"{method_name} 播放失败，尝试其他方法...")

    if not success and verbose:
        print("所有播放方法都失败了")

    # 播放结束处理
    if verbose:
        if _stop_playing:
            print("音乐播放已提前停止")
        else:
            print("音乐播放已停止")

    return success