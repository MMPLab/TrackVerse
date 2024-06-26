import re
import os
import av


def check_dirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs, exist_ok=True)


def check_video(video_path):
    try:
        av.open(video_path)
        return True
    except Exception:
        return False


def check_file(path):
    try:
        open(path).readline()
        return True
    except Exception:
        return False


class ProgressTracker:
    def __init__(self, progress_fn):
        check_dirs(os.path.dirname(progress_fn))
        self.progress_fn = progress_fn

        # Load vids already done
        try:
            self.completed = set([ln.strip() for ln in open(self.progress_fn, 'r') if len(ln.strip()) > 0])
        except FileNotFoundError:
            self.completed = set()
            open(self.progress_fn, 'w').close()

    def add(self, vid):
        self.completed.add(vid)
        open(self.progress_fn, 'a').write(vid+'\n')

    def check_completed(self, vid):
        return vid in self.completed


def ts2sec(ts):
    m = re.match('(\d+):(\d+):(\d+)\.(\d+)', str(ts))
    sec = int(m.group(1))*3600+int(m.group(2))*60+int(m.group(3))+float(m.group(4))/1000
    return sec