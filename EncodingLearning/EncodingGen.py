
import os
import time
import face_recognition

def WriteEncoding(Expression,inPath,toPath):

	print('Expression: ' + Expression)

	inputPath = inPath + Expression
	path, dirs, files = next(os.walk(inputPath))
	numImage = len(files)

	# print(inputPath + "/" + files[0])
	print('# Images: ' + str(numImage))

	csvName = Expression + '.csv'
	csvFile = open(toPath + csvName, 'w')
	csvFile.write("Filename,Encodings\n")

	print('Save CSV to: ' + toPath + csvName)

	emptyCount = 0

	print('...')

	for element in files:

		imageFile = inputPath + '/' + element

		image = face_recognition.load_image_file(imageFile)
		encoding = face_recognition.face_encodings(image)

		if encoding == []:
			emptyCount += 1
			csvFile.write(element + ',' + '\n')
		else:
			data = ' '.join(str(x) for x in encoding[0])
			csvFile.write(element + ',' + data + '\n')

	print('# Empty Encoding: ' + str(emptyCount) + '\n')
	csvFile.close()


if __name__ == '__main__':

## Gen Train Encodings
	"""
	if not os.path.exists("/Users/hkh/Desktop/data/Encodings_train"):
		os.makedirs("/Users/hkh/Desktop/data/Encodings_train")
	destinationPath = "/Users/hkh/Desktop/data/Encodings_train/"

	inputPath = '/Users/hkh/Desktop/data/Training/'

	WriteEncoding('Angry', inputPath, destinationPath)
	WriteEncoding('Disgust', inputPath, destinationPath)
	WriteEncoding('Fear', inputPath, destinationPath)
	WriteEncoding('Happy', inputPath, destinationPath)
	WriteEncoding('Neutral', inputPath, destinationPath)
	WriteEncoding('Sad', inputPath, destinationPath)
	WriteEncoding('Surprise', inputPath, destinationPath)
	"""

## Gen Test Encodings

	if not os.path.exists("/Users/hkh/Desktop/data/Encodings_privatetest"):
		os.makedirs("/Users/hkh/Desktop/data/Encodings_privatetest")
	destinationPath = "/Users/hkh/Desktop/data/Encodings_privatetest/"

	inputPath = '/Users/hkh/Desktop/data/PrivateTest/'

	WriteEncoding('Angry', inputPath, destinationPath)
	WriteEncoding('Disgust', inputPath, destinationPath)
	WriteEncoding('Fear', inputPath, destinationPath)
	WriteEncoding('Happy', inputPath, destinationPath)
	WriteEncoding('Neutral', inputPath, destinationPath)
	WriteEncoding('Sad', inputPath, destinationPath)
	WriteEncoding('Surprise', inputPath, destinationPath)



