import cv2
from ultralytics import YOLO

model = YOLO('global_model.pt')
model.model.names = {
    0: "bike",
    1: "bus",
    2: "caravan",
    3: "coupe",
    4: "crossover",
    5: "hatchback",
    6: "jeep",
    7: "mpv",
    8: "pickup-truck",
    9: "sedan",
    10: "suv",
    11: "taxi",
    12: "truck",
    13: "van",
    14: "vehicle",
    15: "wagon"
}
print(model.names)


video_path = '/home/luka/Desktop/AIC_2023_Track2/AIC23_Track2_NL_Retrieval/data/validation/S02/c007/vdo.avi'

# Set delay (milliseconds) for slowing down video
# For example, 50 ms delay ≈ 20 fps playback speed
delay = 50
print(model.model.model[-1].nc)  # Should print 16


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
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            print("Quitting...")
            break

except Exception as e:
    print(f"Built-in display failed: {e}")
    print("Try the manual method instead")
