import subprocess, shlex
import numpy as np
import cv2

class LibcameraStream:
    def __init__(self, width, height, fps):
        self.width = width
        self.height = height
        self.frame_size = width * height * 3 // 2  # YUV420
        cmd = f"libcamera-vid --inline --nopreview --timeout 0 --width {width} --height {height} --framerate {fps} --codec yuv420 --stdout"
        self.proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, bufsize=self.frame_size)

    def read(self):
        data = self.proc.stdout.read(self.frame_size)
        if len(data) != self.frame_size:
            return False, None
        yuv = np.frombuffer(data, dtype=np.uint8).reshape((self.height * 3 // 2, self.width))
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        return True, bgr

    def release(self):
        if self.proc:
            self.proc.terminate()
