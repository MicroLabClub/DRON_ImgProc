import sys
import cv2
from ultralytics import YOLO
from PIL import Image
image = cv2.imread(sys.argv[1])
model = YOLO('flower_detect.pt')
results = model(image)
for r in results:
    image_array = r.plot()
    image = Image.fromarray(image_array[..., ::-1]) 
flowers=dict()
for result in results:
    boxes = result.boxes.cpu().numpy()
    for box in boxes:      
        if result.names[int(box.cls[0])] not in flowers:
            flowers[result.names[int(box.cls[0])]]=1
        else:
            flowers[result.names[int(box.cls[0])]]=flowers[result.names[int(box.cls[0])]]+1 
message = 'In image are '
if (flowers=={}):
    message+='no flowers'
for flower in flowers:
    message=message+str(flowers[flower])+' '+flower+', '
message=message[:-2]
message+='.'
print(message)
sys.stdout.flush()