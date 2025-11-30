from ultralytics import YOLO
import cv2
import sys
import platform

model = YOLO('best.onnx') 

cap = cv2.VideoCapture(0)

# Diagnostics: print environment helpful for macOS camera permission troubleshooting
print(f"Python executable: {sys.executable}")
print(f"Platform: {platform.platform()}")
print(f"OpenCV version: {cv2.__version__}")

if not cap.isOpened():
    print("ERROR: OpenCV VideoCapture failed to open the camera.")
    print("This often means macOS blocked camera access for the app (Terminal/VSCode/python).")
    print("Mac fix: System Settings -> Privacy & Security -> Camera -> allow Terminal or your IDE (VS Code).")
    print("You can reset camera permissions with: tccutil reset Camera")
    print("Then re-run this script and accept the camera permission prompt.")
    sys.exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break  

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    gray_image_3d = cv2.merge([gray_image, gray_image, gray_image]) 
    
    results = model(gray_image_3d)
    print("===> ",results)
    result = results[0]

    try:
        annotated_frame = result.plot()
    except AttributeError:
        print("Error: plot() method not available for results.")
        break
    
    cv2.imshow('YOLO Inference', annotated_frame)
    
    if cv2.waitKey(1) == 27: 
        break

cap.release()
cv2.destroyAllWindows()
