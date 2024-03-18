import mediapipe as mp

modelPath = '/Users/bamlakdeju/Desktop/ML/faceAnonymizer/blaze_face_short_range.tflite'
imagePath = './data/face.jpeg'

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

imageOptions = FaceDetectorOptions(
    base_options = BaseOptions(model_asset_path=modelPath),
    running_mode = VisionRunningMode.IMAGE
)

videoOptions = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=modelPath),
    running_mode=VisionRunningMode.VIDEO
)

liveOptions = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=modelPath),
    running_mode=VisionRunningMode.LIVE_STREAM,
)