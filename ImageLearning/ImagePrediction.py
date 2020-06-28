
import sys
sys.path.insert(0, '/Users/hjl/Desktop/FR/Models')

import os

import ImageNet4 as net
import torch
from torchvision import transforms

import face_recognition
from PIL import Image, ImageDraw, ImageFont


img_input_path = '/Users/hjl/Desktop/FR/IMG_Test.jpg'

model_parameters_dir = '/Users/hjl/Desktop/FR/Models/ImageNet4.pt' 

# Load Image
image = face_recognition.load_image_file(img_input_path)

# Get Original Image in PIL
pil_image = Image.fromarray(image)

# Prepare Drawing
draw = ImageDraw.Draw(pil_image)

# Find Face(s)
face_locations = face_recognition.face_locations(image)

print('- Found {} face(s) in: '.format(len(face_locations)) + img_input_path)

if len(face_locations) == 0:
    sys.exit()


labelDic = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 
            4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Load FER Model
model = net.ImageNet()
model.eval()

if os.path.exists(model_parameters_dir):
    model.load_state_dict(torch.load(model_parameters_dir, map_location=torch.device('cpu')))
    print('- Model Loaded: ' + model_parameters_dir)
else:
    print('- No Model Parameters File')
    sys.exit()
    
# Run Each Face
for face_location in face_locations:

    # Location of each face in this image
    top, right, bottom, left = face_location
    margin = 50     # keep more face margin space
    top -= margin
    right += margin
    bottom += margin
    left -= margin

    # Access the actual face:
    face_image = image[top:bottom, left:right]
    face_pil_image = Image.fromarray(face_image)
    
    # Transform for Model
    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize([48, 48]),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
        ])
    
    face_image_tensor = tf(face_pil_image)
    
    ## Run FER Model
    y = model(face_image_tensor[None])
    prediction = torch.argmax(y, dim=1)
    
    Emotion = labelDic[int(prediction)]
    # print(Emotion)
    
    # Draw a box around the face
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

    # Draw a label with a emotion below the face
    text_font = ImageFont.truetype('/Library/Fonts/Microsoft/Arial.ttf', 50)
    text_width, text_height = text_font.getsize(Emotion)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left + 6, bottom - text_height - 5), Emotion, fill=(255, 255, 255, 255), font=text_font)

    #face_pil_image.show()

del draw

pil_image.show()





