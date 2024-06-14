import av
import numpy as np
from torch.utils.data import IterableDataset


class VideoContainer(object):
    def _pts2sec(self, pts):
        return pts * self.stream.time_base

    def _sec2pts(self, sec):
        return int(sec / self.stream.time_base)

    def _no2sec(self, no):
        return no / self._rate + self._ss

    def _sec2no(self, sec):
        return int(round((sec - self._ss) * self._rate))

    def _pts2no(self, pts):
        return self._sec2no(self._pts2sec(pts))

    def _no2pts(self, no):
        return self._sec2pts(self._no2sec(no))


class VideoReader(VideoContainer):
    def __init__(self, filename, rate=None, start_time=None, duration=None, frame_max_size=None, skip_nonkey_frames=False):
        self.filename = filename
        self.container = av.open(self.filename)
        self.stream = self.container.streams.video[0]
        self.stream.codec_context.thread_count = 1
        if skip_nonkey_frames:
            self.stream.codec_context.skip_frame = "NONKEY"

        # Original rate, start time and duration
        self._rate = self.stream.average_rate
        self._ss = self.stream.start_time * self.stream.time_base
        self._dur = self.stream.duration * self.stream.time_base
        self._ff = self._ss + self._dur
        self._size = (self.stream.codec_context.height, self.stream.codec_context.width)

        # Output rate, start time and duration
        self._set_output_stream(rate, start_time, duration, frame_max_size)

    def _set_output_stream(self, rate=None, start_time=None, duration=None, frame_max_size=None):
        self.rate = rate if rate is not None else self._rate
        self.start_time = start_time if start_time is not None else self._ss
        self.duration = min(duration, self._ff - self.start_time) if duration is not None else self._ff - self.start_time
        self.num_frames = len(self._find_output_frames())

        if frame_max_size is None:
            self.frame_size = self._size
        else:
            ar = self._size[0] / self._size[1]
            if ar > 1:
                self.frame_size = (int(frame_max_size), int(frame_max_size / ar))
            else:
                self.frame_size = (int(frame_max_size * ar), int(frame_max_size))

    def _find_output_frames(self):
        out_times = np.linspace(self.start_time, self.start_time + self.duration, int(self.duration * self.rate), endpoint=False)
        out_frame_no = [self._sec2no(t) for t in out_times]
        return out_frame_no

    def update_output_stream(self, rate=None, start_time=None, duration=None, frame_max_size=None):
        self._set_output_stream(rate, start_time, duration, frame_max_size)

    def read(self):
        out_frame_no = self._find_output_frames()
        self.container.seek(int(self._no2sec(out_frame_no[0]) * av.time_base))

        n_read = 0
        for frame in self.container.decode(video=0):
            if n_read == len(out_frame_no):
                break

            fno = self._pts2no(frame.pts)
            fts = self._pts2sec(frame.pts)
            if fno < out_frame_no[n_read]:
                continue

            pil_img = frame.to_image()
            if self.frame_size is not None and self.frame_size != self._size:
                pil_img = pil_img.resize(self.frame_size)

            while fno >= out_frame_no[n_read]:    # This 'while' takes care of the case where _rate < rate
                yield pil_img, fts
                n_read += 1
                if n_read == len(out_frame_no):
                    break

    def read_keyframes(self):
        stream = self.container.streams.video[0]
        stream.codec_context.skip_frame = "NONKEY"

        self.container.seek(int(self.start_time * av.time_base))
        for frame in self.container.decode(stream):
            fts = self._pts2sec(frame.pts)
            if fts > self.start_time + self.duration:
                break
            pil_img = frame.to_image()
            if self.frame_size is not None and self.frame_size != self._size:
                pil_img = pil_img.resize(self.frame_size)
            yield pil_img, fts

    def __len__(self):
        return self.num_frames


class VideoWriter(VideoContainer):
    def __init__(self, filename, rate, frame_size):
        if not filename.endswith(".mp4"):
            filename += ".mp4"
        self.container = av.open(filename, mode='w')
        self.stream = self.container.add_stream('libx264', rate=rate)
        self.stream.height = frame_size[0]
        self.stream.width = frame_size[1]
        self.stream.pix_fmt = 'yuv420p'
        self.stream.gop_size = int(rate//2)
        self.stream.codec_context.thread_count = 1

        self._rate = self.stream.average_rate
        self._size = (self.stream.codec_context.height, self.stream.codec_context.width)

    def write(self, frame, rng=(0, 255)):
        # Convert frame to uint8
        frame_np = (np.array(frame).astype(float) - rng[0]) / (rng[1] - rng[0]) * 255
        frame_np = np.clip(np.round(frame_np).astype(np.uint8), 0, 255)

        # Convert to av.VideoFrame
        frame_av = av.VideoFrame.from_ndarray(frame_np, format='rgb24')

        # Write to file
        for packet in self.stream.encode(frame_av):
            self.container.mux(packet)

    def __del__(self):
        # Flush stream
        for packet in self.stream.encode():
            self.container.mux(packet)

        # Close the file
        self.container.close()


class VideoDB(IterableDataset):
    def __init__(self, video_fn, frame_rate=None, max_dur=None, start_time=None, transform=None, skip_nonkey_frames=False):
        self.reader = VideoReader(video_fn, rate=frame_rate, start_time=start_time, duration=max_dur, skip_nonkey_frames=skip_nonkey_frames)
        self.transform = transform
        if self.transform is None:
            self.transform = lambda x: np.array(x)

    def __iter__(self):
        for frame, ts in self.reader.read():
            yield self.transform(frame), float(ts)

    def __len__(self):
        return self.reader.num_frames


if __name__ == '__main__':
    filename = 'data/hdvila-100m/video_clips/EY6jfk_PRuE/EY6jfk_PRuE.0.mp4'
    video = VideoReader(filename)

    video.update_output_stream(start_time=0.2, duration=3, rate=8, frame_max_size=360)
    video_out = VideoWriter('tmp.mp4', int(video.rate), video.frame_size)
    for frame, ts in video.read():
        video_out.write(np.array(frame).astype(np.uint8))
    # video_out.release()

    video.update_output_stream(start_time=3, rate=8)
    video_out = VideoWriter('tmp2.mp4', int(video.rate), video.frame_size)
    for frame, ts in video.read():
        video_out.write(np.array(frame).astype(np.uint8))
    video_out.release()