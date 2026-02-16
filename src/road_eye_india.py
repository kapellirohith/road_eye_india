import os, time, json, csv, datetime, threading, subprocess, shlex, shutil
from collections import deque
import cv2, numpy as np

try:
    from picamera2 import Picamera2
    HAVE_PICAM = True
except Exception:
    Picamera2 = None
    HAVE_PICAM = False

try:
    from tflite_runtime.interpreter import Interpreter
except Exception:
    from tensorflow.lite import Interpreter


class LibcameraStream:
    def __init__(self, width, height, fps):
        self.width = width
        self.height = height
        self.frame_size = width * height * 3 // 2  # YUV420
        bin_path = shutil.which("rpicam-vid") or shutil.which("libcamera-vid")
        if not bin_path:
            raise RuntimeError("rpicam-vid/libcamera-vid not found. Install libcamera-apps.")
        cmd = (
            f"{bin_path} --nopreview --timeout 0 "
            f"--width {width} --height {height} --framerate {fps} "
            f"--codec yuv420 -o -"
        )
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
            try:
                self.proc.wait(timeout=2)
            except Exception:
                pass


class TFLiteDetector:
    def __init__(self, model_path, labels_path, input_size, score_thresh, nms_thresh, max_det, threads=4):
        self.labels = self._load_labels(labels_path)
        self.num_classes = len(self.labels)
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.max_det = max_det

        self.interpreter = Interpreter(model_path=model_path, num_threads=threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()

        self.input_dtype = self.input_details["dtype"]
        self.input_scale, self.input_zero = self.input_details.get("quantization", (0.0, 0))
        self.output_scale, self.output_zero = self.output_details[0].get("quantization", (0.0, 0))
        self.input_h = int(self.input_details["shape"][1])
        self.input_w = int(self.input_details["shape"][2])

    def _load_labels(self, path):
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def _quantize(self, img):
        if self.input_dtype == np.float32:
            return img.astype(np.float32) / 255.0
        if self.input_dtype == np.uint8:
            return img.astype(np.uint8)
        if self.input_dtype == np.int8:
            img = img.astype(np.float32) / 255.0
            if self.input_scale > 0:
                img = img / self.input_scale + self.input_zero
            return np.clip(np.round(img), -128, 127).astype(np.int8)
        return img

    def _dequantize(self, arr):
        if arr.dtype in (np.int8, np.uint8) and self.output_scale > 0:
            return (arr.astype(np.float32) - self.output_zero) * self.output_scale
        return arr.astype(np.float32)

    def infer(self, frame_bgr, score_thresh_override=None):
        h, w = frame_bgr.shape[:2]
        thresh = self.score_thresh if score_thresh_override is None else score_thresh_override

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_w, self.input_h))
        input_data = np.expand_dims(resized, axis=0)
        input_data = self._quantize(input_data)

        self.interpreter.set_tensor(self.input_details["index"], input_data)
        self.interpreter.invoke()

        out = self.interpreter.get_tensor(self.output_details[0]["index"])
        out = self._dequantize(out)
        out = np.squeeze(out)
        if out.ndim == 3:
            out = out[0]
        if out.ndim == 2:
            if out.shape[0] == 4 + self.num_classes:
                out = out.T
            elif out.shape[1] != 4 + self.num_classes:
                return []
        else:
            return []

        boxes_list, scores_list, class_list = [], [], []
        for row in out:
            x, y, bw, bh = row[:4]
            cls_scores = row[4:]
            class_id = int(np.argmax(cls_scores))
            score = float(cls_scores[class_id])
            if score < thresh:
                continue
            if class_id < 0 or class_id >= self.num_classes:
                continue
            max_xywh = max(x, y, bw, bh)
            if max_xywh <= 1.5:
                x *= w; y *= h; bw *= w; bh *= h
            elif max_xywh <= max(self.input_w, self.input_h) * 1.5:
                x = x * w / self.input_w
                y = y * h / self.input_h
                bw = bw * w / self.input_w
                bh = bh * h / self.input_h
            x1 = int(x - bw / 2); y1 = int(y - bh / 2)
            x2 = int(x + bw / 2); y2 = int(y + bh / 2)
            x1 = max(0, min(w - 1, x1)); y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2)); y2 = max(0, min(h - 1, y2))
            bw = max(1, x2 - x1); bh = max(1, y2 - y1)
            boxes_list.append([x1, y1, bw, bh])
            scores_list.append(score)
            class_list.append(class_id)

        keep = cv2.dnn.NMSBoxes(boxes_list, scores_list, thresh, self.nms_thresh)
        detections = []
        if len(keep) > 0:
            for k in keep.flatten():
                x, y, bw, bh = boxes_list[k]
                detections.append({
                    "label": self.labels[class_list[k]],
                    "score": scores_list[k],
                    "bbox": (x, y, x + bw, y + bh)
                })
                if len(detections) >= self.max_det:
                    break
        return detections


class FrameRing:
    def __init__(self, maxlen):
        self.frames = deque(maxlen=maxlen)

    def push(self, frame):
        self.frames.append(frame.copy())

    def get(self):
        return list(self.frames)


class ClipWriter:
    def __init__(self, fps, frame_size):
        self.fps = fps
        self.frame_size = frame_size
        self.queue = deque(maxlen=2)
        self.lock = threading.Lock()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def submit(self, frames, path):
        if not frames:
            return
        with self.lock:
            if len(self.queue) >= 2:
                return
            self.queue.append((frames, path))

    def _run(self):
        while True:
            job = None
            with self.lock:
                if self.queue:
                    job = self.queue.popleft()
            if job is None:
                time.sleep(0.05)
                continue
            frames, path = job
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(path, fourcc, self.fps, self.frame_size)
            if out.isOpened():
                for f in frames:
                    out.write(f)
                out.release()


def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def ensure_log(path):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "label", "score", "distance_m", "bbox", "brightness"])


def start_audio_record(base_dir, path_cfg):
    try:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        audio_path = os.path.join(base_dir, path_cfg["full_dir"], f"audio_{ts}.wav")
        cmd = ["arecord", "-f", "cd", "-t", "wav", "-r", "16000", "-c", "1", audio_path]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return proc, audio_path
    except Exception:
        return None, None


def stop_audio_record(proc):
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except Exception:
            pass


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    with open(os.path.join(base_dir, "config.json"), "r") as f:
        cfg = json.load(f)

    cam_cfg = cfg["camera"]
    det_cfg = cfg["detector"]
    evt_cfg = cfg["events"]
    rec_cfg = cfg["recording"]
    path_cfg = cfg["paths"]
    disp_cfg = cfg["display"]
    tune_cfg = cfg["tuning"]
    roi_cfg = cfg["roi"]

    ensure_dirs(
        os.path.join(base_dir, path_cfg["logs_dir"]),
        os.path.join(base_dir, path_cfg["snapshots_dir"]),
        os.path.join(base_dir, path_cfg["clips_dir"]),
        os.path.join(base_dir, path_cfg["full_dir"])
    )
    log_path = os.path.join(base_dir, path_cfg["log_csv"])
    ensure_log(log_path)

    detector = TFLiteDetector(
        model_path=os.path.join(base_dir, det_cfg["model_path"]),
        labels_path=os.path.join(base_dir, det_cfg["labels_path"]),
        input_size=det_cfg["input_size"],
        score_thresh=det_cfg["score_thresh"],
        nms_thresh=det_cfg["nms_thresh"],
        max_det=det_cfg["max_detections"],
        threads=det_cfg.get("threads", 4)
    )

    frame_size = (cam_cfg["width"], cam_cfg["height"])
    record_fps = rec_cfg["record_fps"] or cam_cfg["fps"]

    use_picam = cam_cfg.get("use_picamera2", False) and HAVE_PICAM
    use_libcam = cam_cfg.get("use_libcamera", False)
    cap = None
    picam2 = None
    libcam = None

    if use_picam:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"format": cam_cfg["format"], "size": frame_size},
            controls={"FrameRate": cam_cfg["fps"]}
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(0.2)
    elif use_libcam:
        libcam = LibcameraStream(cam_cfg["width"], cam_cfg["height"], cam_cfg["fps"])
    else:
        cap = cv2.VideoCapture(cam_cfg["usb_index"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
        cap.set(cv2.CAP_PROP_FPS, cam_cfg["fps"])

    ring = FrameRing(maxlen=int(rec_cfg["clip_pre_s"] * cam_cfg["fps"]))
    clip_writer = ClipWriter(record_fps, frame_size)

    full_out = None
    if rec_cfg["record_full"]:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        full_path = os.path.join(base_dir, path_cfg["full_dir"], f"full_{ts}.avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        full_out = cv2.VideoWriter(full_path, fourcc, record_fps, frame_size)

    audio_proc = None
    if rec_cfg.get("record_audio", False):
        audio_proc, _ = start_audio_record(base_dir, path_cfg)

    infer_every = max(1, det_cfg["infer_every"])
    last_dets = []
    frame_idx = 0
    fps = 0.0
    last_event_time = 0.0
    event_streak = 0
    collecting_post = 0
    clip_frames = []
    clip_path = ""

    while True:
        t0 = time.perf_counter()

        if use_picam:
            rgb = picam2.capture_array()
            if rgb is None:
                continue
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif use_libcam:
            ok, frame = libcam.read()
            if not ok:
                continue
        else:
            ok, frame = cap.read()
            if not ok:
                continue

        h, w = frame.shape[:2]
        brightness = int(frame.mean())
        score_boost = tune_cfg["night_score_boost"] if brightness < tune_cfg["night_brightness"] else 0.0
        score_thresh = det_cfg["score_thresh"] + score_boost

        if frame_idx % infer_every == 0:
            dets = detector.infer(frame, score_thresh_override=score_thresh)
            last_dets = dets
        else:
            dets = last_dets

        warning = False
        event_candidates = []

        for d in dets:
            label = d["label"]
            score = d["score"]
            x1, y1, x2, y2 = d["bbox"]
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            cy = y1 + bh // 2

            in_road = cy >= int(roi_cfg["road_y_start"] * h)

            if label in evt_cfg["road_only_classes"] and not in_road:
                continue

            dist = None
            if label in evt_cfg["distance_classes"]:
                dist = round(evt_cfg["distance_factor"] / max(1, bh), 1)

            close = dist is not None and dist < evt_cfg["warn_distance_m"]
            color = (0, 0, 255) if close else (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {score:.2f}"
            if dist is not None:
                text += f" {dist}m"
            cv2.putText(frame, text, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if close:
                warning = True

            if label in evt_cfg["event_classes"] and score >= evt_cfg["event_score_thresh"]:
                if label in evt_cfg["road_only_classes"]:
                    event_candidates.append(d)
                elif close or in_road:
                    event_candidates.append(d)

        if warning:
            cv2.putText(frame, "WARNING: CLOSE", (w // 2 - 110, 40),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 3)

        ring.push(frame)

        now = time.time()
        if event_candidates:
            event_streak += 1
        else:
            event_streak = 0

        if event_streak >= evt_cfg["confirm_frames"] and (now - last_event_time) > evt_cfg["cooldown_s"]:
            event = max(event_candidates, key=lambda x: x["score"])
            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            label = event["label"]

            snap_path = os.path.join(base_dir, path_cfg["snapshots_dir"], f"{ts}_{label}.jpg")
            cv2.imwrite(snap_path, frame)

            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([ts, label, f"{event['score']:.3f}", "", event["bbox"], brightness])

            if rec_cfg["save_clips"]:
                clip_frames = ring.get() + [frame.copy()]
                collecting_post = int(rec_cfg["clip_post_s"] * cam_cfg["fps"])
                clip_path = os.path.join(base_dir, path_cfg["clips_dir"], f"{ts}_{label}.avi")

            last_event_time = now
            event_streak = 0

        if collecting_post > 0:
            clip_frames.append(frame.copy())
            collecting_post -= 1
            if collecting_post == 0:
                clip_writer.submit(clip_frames, clip_path)

        dt = time.perf_counter() - t0
        inst_fps = 1.0 / dt if dt > 0 else 0.0
        fps = inst_fps if fps == 0 else (fps * 0.9 + inst_fps * 0.1)
        cv2.putText(frame, f"FPS {fps:.1f} | B {brightness}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if full_out:
            full_out.write(frame)

        if disp_cfg.get("enabled", False) and os.environ.get("DISPLAY"):
            dw, dh = disp_cfg.get("size", [640, 480])
            cv2.imshow("ROAD_EYE INDIA", cv2.resize(frame, (dw, dh)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    if full_out:
        full_out.release()
    if audio_proc:
        stop_audio_record(audio_proc)
    if picam2:
        picam2.stop()
    if libcam:
        libcam.release()
    if cap:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
