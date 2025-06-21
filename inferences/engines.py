import onnxruntime as ort
import cv2
import toml
import numpy as np


configs = toml.load('inferences/configs/config.toml')

detection_session = ort.InferenceSession(configs['detection-model-path'], providers=configs['providers'])
extraction_session = ort.InferenceSession(configs['extraction-model-path'], providers=configs['providers'])

width = configs['segment-width']
height = configs['segment-height']
length = configs['segment-length']

x1 = configs['crop-x1']
x2 = configs['crop-x2']
y1 = configs['crop-y1']
y2 = configs['crop-y2']

std = configs['normalization-std']
mean = configs['normalization-mean']

smoothing_weight = np.ones(configs['smoothing-window']) / configs['smoothing-window']


def normalize(inputs):
    return (inputs - mean) / std


def frame_preprocess(frame):
    preprocessed = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    preprocessed = preprocessed[y1:y2, x1:x2]

    return cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)


def load_next_segment(capture):
    segment_frames = []

    while len(segment_frames) < length:
        read_success, captured_frame = capture.read()

        if read_success:
            segment_frames.append(frame_preprocess(captured_frame))
        else:
            return False, None

    return True, segment_preprocess(segment_frames)


def segment_preprocess(frames):
    preprocessed = np.stack(frames, axis=0).transpose((3, 0, 1, 2))

    if configs['precision'] == 'fp16':
        preprocessed = normalize(preprocessed).astype(np.float16)
    else:
        preprocessed = normalize(preprocessed).astype(np.float32)

    return np.expand_dims(preprocessed, axis=0)


def extract_segment_features(segment):
    extraction_outputs = extraction_session.run(['outputs'], {'inputs': segment})
    extraction_outputs = extraction_outputs[0]

    return np.squeeze(extraction_outputs, axis=0)


def extract_video_features(video_path):
    features = []
    capture = cv2.VideoCapture(video_path)

    while capture.isOpened():
        load_success, preprocessed_segment = load_next_segment(capture)

        if load_success:
            features.append(extract_segment_features(preprocessed_segment))
        else:
            capture.release()

    return np.stack(features, axis=0)


def features_preprocess(features):
    if configs['precision'] == 'fp16':
        preprocessed = features.astype(np.float16)
    else:
        preprocessed = features.astype(np.float32)

    return np.expand_dims(preprocessed, axis=0)


def sigmoid(inputs):
    return 1 / (1 + np.exp(-inputs))


def detection_by_features(features):
    detection_outputs = detection_session.run(['outputs'], {'inputs': features})
    detection_outputs = detection_outputs[0]

    return sigmoid(np.squeeze(detection_outputs, axis=0))


def score_smoothing(scores):
    return np.convolve(scores, smoothing_weight, mode='same').round(decimals=2)


def expand_scores(scores):
    return np.array(scores).repeat(length, axis=0)


def detection_by_video(video_path):
    features = extract_video_features(video_path)
    features = features_preprocess(features)

    return score_smoothing(detection_by_features(features))


def anomaly_prompt_enhancement(frame, prompt):
    return cv2.putText(frame, prompt, (120, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 215), thickness=2)


def anomaly_border_enhancement(frame, border):
    x2 = frame.shape[1]
    y2 = frame.shape[0]

    return cv2.rectangle(frame, (0, 0), (x2, y2), (0, 0, 215), thickness=border)


def draw_detection_result(frame, score):
    if score > configs['anomaly-threshold']:
        frame = anomaly_prompt_enhancement(frame, configs['anomaly-prompt'])
        frame = anomaly_border_enhancement(frame, configs['anomaly-border'])

        return cv2.putText(frame, f'{score:.2f}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 215), thickness=2)
    else:
        return cv2.putText(frame, f'{score:.2f}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 215, 0), thickness=2)
