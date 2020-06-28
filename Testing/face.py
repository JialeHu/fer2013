
import face_recognition
import cv2
from PIL import Image, ImageDraw


image = face_recognition.load_image_file('/Users/hkh/Desktop/data/Training/Angry/00000000.jpg')
face_mark = face_recognition.face_landmarks(image)
print("Found {} face(s) in this photograph.".format(len(face_mark)))

encoding = face_recognition.face_encodings(image)
print(encodingt)

# Create a PIL imagedraw object so we can draw on the picture
pil_image = Image.fromarray(image)
d = ImageDraw.Draw(pil_image)

for face_landmarks in face_mark:

    # Print the location of each facial feature in this image
	for facial_feature in face_landmarks.keys():
		print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))

    # Let's trace out each facial feature in the image with a line!
	for facial_feature in face_landmarks.keys():
		d.line(face_landmarks[facial_feature], width=1)

# Show the picture
pil_image.show()




