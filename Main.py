from CropImage import cropImage
from CameraCapture import capture
from DetectText import detect_text
from DetectFace import detect_face

def main():
  
  capture()
  cropImage()
  detect_text()
  detect_face()
  
main()