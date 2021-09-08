import cv2
from matplotlib import pyplot as plt
import numpy as np
import csv
import boto3
import re

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

def detect_text():
    photo = 'saved_img.jpg'
    img = cv2.imread(photo)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.show()

    with open(photo, 'rb') as source_image:
        source_bytes = source_image.read()

    response = client.detect_text(Image={'Bytes': source_bytes})

    # print(response)
    text_list = []
    for label in response['TextDetections']:
        print("Text : " + str(label['DetectedText']))
        text_list.append(str(label['DetectedText']))

    def remove(string):
        return string.replace(" ", "")

    for text in text_list:
        text = remove(text)
        verified_nomor = re.split("/^([A-Z]{1,3})(\s|-)*([1-9][0-9]{0,3})(\s|-)*([A-Z]{0,3}|[1-9][0-9]{1,2})$/i", text)
        if verified_nomor:
            print("Plat Nomor terdeteksi")
            break

    plat_nomor = verified_nomor[0]
    plat_nomor