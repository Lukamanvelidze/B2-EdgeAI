import cv2
from ultralytics import YOLO

model = YOLO('global_model.pt')
video_path = '/home/luka/Desktop/AIC_2023_Track2/AIC23_Track2_NL_Retrieval/data/validation/S02/c007/vdo.avi'

# Set delay (milliseconds) for slowing down video
# For example, 50 ms delay ≈ 20 fps playback speed
delay = 50

try:
    results = model.predict(
        source=video_path,
        show=True,
        imgsz=416,
        conf=0.25,
        stream=True,
        verbose=True
    )

    for result in results:
        # result.show() is internally called with show=True, so frame is displayed

        # Wait for 'delay' ms and check if 'q' is pressed to quit
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            print("Quitting...")
            break

except Exception as e:
    print(f"Built-in display failed: {e}")
    print("Try the manual method instead")
