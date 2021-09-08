import cv2
from matplotlib import pyplot as plt
import numpy as np
import csv
import boto3
import re
from CameraCapture import capture

AWS_DEFAULT_REGION = 'us-west-2'

with open('student02.csv', 'r') as input:
    next(input)
    reader = csv.reader(input)
    for line in reader:
        access_key_id = line[2]
        secret_access_key = line[3]

client = boto3.client('rekognition',
                      region_name=AWS_DEFAULT_REGION,
                      aws_access_key_id=access_key_id,
                      aws_secret_access_key=secret_access_key)

def detect_face():
    photo1 = 'saved_img.jpg'
    img1 = cv2.imread(photo1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    height_shape1 = img1.shape[0]
    width_shape1 = img1.shape[1]

    plt.imshow(img1)
    plt.show()

    with open(photo1, 'rb') as source_image1:
        source_bytes = source_image1.read()

    capture()
    # Target Image
    photo2 = 'saved_img.jpg'
    img2 = cv2.imread(photo2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    height_shape2 = img2.shape[0]
    width_shape2 = img2.shape[1]

    plt.imshow(img2)
    plt.show()

    with open(photo2, 'rb') as source_image2:
        target_bytes = source_image2.read()

    # Detect using AWS
    response = client.compare_faces(SourceImage={'Bytes': source_bytes},
                                    TargetImage={'Bytes': target_bytes},
                                    # MinConfidence = 95,
                                    )

    # Search Faces in Target_Image
    for person in response['FaceMatches']:
        print(str(person['Similarity']))
        width = int(person['Face']['BoundingBox']['Width'] * width_shape2)
        height = int(person['Face']['BoundingBox']['Height'] * height_shape2)
        left = int(person['Face']['BoundingBox']['Left'] * width_shape2)
        top = int(person['Face']['BoundingBox']['Top'] * height_shape2)

        img2 = cv2.rectangle(img2, (left, top), (left + width, top + height), (0, 255, 0), 2)

    # Plotting
    resized1 = cv2.resize(img1, (1000, 1000), interpolation=cv2.INTER_CUBIC)
    resized2 = cv2.resize(img2, (1000, 1000), interpolation=cv2.INTER_CUBIC)

    plt.subplot(1, 2, 1), plt.imshow(resized1)
    plt.title('SourceImage'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(resized2)
    plt.title('TargetImage'), plt.xticks([]), plt.yticks([])
    plt.show()