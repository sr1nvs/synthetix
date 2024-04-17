from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')

results = model('dataset/images/train/00a7ef03-00000000.jpg')

for result in results:
    masks = result.masks  
    probs = result.probs  
    result.show() 
    result.save(filename='result.jpg')

