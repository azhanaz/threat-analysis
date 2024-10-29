import cv2
import torch
from torchvision import models, transforms
from PIL import Image
from queue import Queue
import threading
import time

# Load a lighter model, e.g., MobileNetV2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
model.eval()
model.to(device)

# Define the transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Function to detect objects in a frame
def detect_objects(frame):
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor_image = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(tensor_image)
    
    boxes, labels, scores = predictions[0]['boxes'].cpu().numpy(), predictions[0]['labels'].cpu().numpy(), predictions[0]['scores'].cpu().numpy()
    return boxes, labels, scores

# Function to capture frames and put them in the queue
def capture_frames(queue):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Lower resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    frame_skip = 2
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        queue.put(frame)

    cap.release()

# Function to process frames from the queue
def process_frames(queue):
    while True:
        if not queue.empty():
            frame = queue.get()
            start_time = time.time()
            boxes, labels, scores = detect_objects(frame)
            end_time = time.time()
            print(f"Processing time: {end_time - start_time:.2f} seconds")

            # Draw the bounding boxes on the frame
            for box, label, score in zip(boxes, labels, scores):
                if score > 0.5:
                    x0, y0, x1, y1 = box
                    cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
                    cv2.putText(frame, f"Score: {score:.2f}", (int(x0), int(y0) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow('Live Object Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

# Main function to start threads
def main():
    frame_queue = Queue()

    capture_thread = threading.Thread(target=capture_frames, args=(frame_queue,))
    process_thread = threading.Thread(target=process_frames, args=(frame_queue,))

    capture_thread.start()
    process_thread.start()

    capture_thread.join()
    process_thread.join()

if __name__ == "__main__":
    main()
