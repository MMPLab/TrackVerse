import os
import yt_dlp
from utils import misc as misc_utils
from enum import Enum


class STATUS(Enum):
    SUCCESS = 0
    DONE = 2
    FAIL = 1


class YoutubeDL(object):
    def __init__(self, downl_dir, push_to_euler):
        self.downl_dir = downl_dir
        misc_utils.check_dirs(self.downl_dir)
        self.downl_tracker = misc_utils.ProgressTracker(os.path.join(downl_dir, 'downloaded.txt'))
        self.push_to_euler = push_to_euler

    def download_video(self, youtube_id):
        url = f"https://www.youtube.com/watch?v={youtube_id}"
        folder = f"{self.downl_dir}/{youtube_id[:2]}"
        filename = f"{folder}/{youtube_id}.mp4"
        misc_utils.check_dirs(folder)

        # Download video
        if self.downl_tracker.check_completed(youtube_id) or misc_utils.check_video(filename):
            return STATUS.DONE, filename

        format_id = '22'  # for 720p videos with audio
        ydl_opts = {
            'outtmpl': f'{folder}/%(id)s.%(ext)s',
            'merge_output_format': 'mp4',
            'format': format_id,  # 720P
            'skip_download': False,
            'ignoreerrors': True,
            'quiet': True,
            'progress': False,
            'no_post_overwrites': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            down_result = ydl.download([url])

        self.downl_tracker.add(youtube_id)
        if down_result != 0:
            return STATUS.FAIL, None

        if self.push_to_euler:
            euler_base_folder = '/srv/home/groups/pmorgado/datasets/object_tracks_db_fixed_detic'
            euler_dest = f'{euler_base_folder}/videos_mp4/{youtube_id[:2]}/{youtube_id}.mp4'
            os.system(f'rsync {filename} euler:{euler_dest}')
            os.remove(filename)

        return STATUS.SUCCESS, filename
