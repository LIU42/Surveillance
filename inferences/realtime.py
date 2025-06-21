import threading
import collections
import cv2
import time
import toml
import numpy as np
import inferences.engines as engines


configs = toml.load('inferences/configs/config.toml')

segment_length = configs['segment-length']
history_length = configs['history-length']

capture_interval = configs['capture-interval']
prepare_interval = configs['prepare-interval']
predict_interval = configs['predict-interval']


def execute_task_in_seconds(task, args=None, target_seconds=0):
    start_seconds = time.perf_counter()

    if args is None:
        task_execution_result = task()
    else:
        task_execution_result = task(args)

    finish_seconds = time.perf_counter()

    delta_seconds = finish_seconds - start_seconds
    delay_seconds = target_seconds - delta_seconds

    if delay_seconds > 0:
        time.sleep(delay_seconds)

    return task_execution_result


class RealtimeInferenceSession:
    def __init__(self, source):
        self.capture = cv2.VideoCapture(source)

        self.segment_queue = collections.deque(maxlen=segment_length)
        self.feature_queue = collections.deque(maxlen=history_length)

        self.capture_running = True
        self.prepare_running = True
        self.predict_running = True

        self.current_frame = None
        self.current_score = None

        self.current_lock = threading.Lock()
        self.segment_lock = threading.Lock()

        self.capture_thread = threading.Thread(target=self.capture_process)
        self.prepare_thread = threading.Thread(target=self.prepare_process)
        self.predict_thread = threading.Thread(target=self.predict_process)

        self.capture_thread.start()
        self.prepare_thread.start()
        self.predict_thread.start()

    def __del__(self):
        self.release()

    def capture_task(self):
        read_success, captured_frame = self.capture.read()

        if read_success:
            with self.current_lock:
                self.current_frame = captured_frame

    def capture_process(self):
        while self.capture_running:
            execute_task_in_seconds(self.capture_task, target_seconds=capture_interval)

    def prepare_task(self):
        with self.current_lock:
            current_frame = self.current_frame

        if current_frame is not None:
            with self.segment_lock:
                self.segment_queue.append(engines.frame_preprocess(current_frame))

    def prepare_process(self):
        while self.prepare_running:
            execute_task_in_seconds(self.prepare_task, target_seconds=prepare_interval)

    def load_segment_frames(self):
        if not len(self.segment_queue) == segment_length:
            current_segment_frames = None
        else:
            current_segment_frames = self.segment_queue.copy()
            self.segment_queue.clear()

        return current_segment_frames

    def predict_task(self):
        with self.segment_lock:
            segment_frames = self.load_segment_frames()

        if segment_frames is not None:
            self.feature_queue.append(engines.extract_segment_features(engines.segment_preprocess(segment_frames)))

            features = np.stack(self.feature_queue, axis=0)
            features = engines.features_preprocess(features)

            realtime_scores = engines.detection_by_features(features)

            with self.current_lock:
                self.current_score = realtime_scores[-1]

    def predict_process(self):
        while self.predict_running:
            execute_task_in_seconds(self.predict_task, target_seconds=predict_interval)

    def get_result(self):
        with self.current_lock:
            result_frame = self.current_frame
            result_score = self.current_score

        if result_frame is None:
            return None

        if result_score is None:
            return None

        return engines.draw_detection_result(result_frame, result_score)

    def release(self):
        self.capture_running = False
        self.prepare_running = False
        self.predict_running = False

        self.capture_thread.join()
        self.prepare_thread.join()
        self.predict_thread.join()

        self.capture.release()
