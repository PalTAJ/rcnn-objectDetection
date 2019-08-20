# rcnn-objectDetection v0.01



in this project im using  faster_rcnn_inception model with region proposal network (RPN)for generating region proposals.

requirements and packages :

in order to run this repostory please refer to the link below and download Tensorflow’s Object Detection API
https://github.com/tensorflow/models


you may run the following commands to get the nesscary packages:
       pip install protobuf
       pip install pillow
       pip install lxml
       pip install Cython
       pip install jupyter
       pip install matplotlib
       pip install pandas
       pip install opencv-python 
       pip install tensorflow


run the following commands from the models-master\research directory:
  python setup.py build
  python setup.py install
  
Testing the API:
go to object_detection directory and enter the following command:
jupyter notebook object_detection_tutorial.ipynb

after installing the nessarcy packages move everything in this repostory to models/research/object_detection
remember to modify main.py for images paths that you wish to test the model on
then you may run the file main.py by opening terminal and typing:
python main.py

to include color_detection you will need to get my color_detection model from here :
https://github.com/PalTAJ/knn-colorDetection

then get everything in it to models/research/object_detection directory.
next open main.py (for the object detection) and enter uncomment the return_coordinate function call.
next go to color_detection directory open main.py and add the following code for single image color detection option :
#in my case i only classifiy 3 objects.
images = []
for file in glob.glob("/images outputed from objection detection model path/*.jpg"):
    images.append(file)
if len(images) > 0:
    test_image = images[0]
    source_image = cv2.imread(test_image)
if len(images) > 1:
    test_image1 = images[1]
    source_image1 = cv2.imread(test_image1)

if len(images) > 2:
    test_image2 = images[2]
    source_image2 = cv2.imread(test_image2)

## Single Image Testing
# test_image = 'greentest.jpg'
# source_image = cv2.imread(test_image)
def click(test_image,source_image,n):
    featureExtractor.main(2,test_image) # extract feature for our test image
    prediction = knn_classifier.main('features.data', 'test.data')
    cv2.putText(source_image, prediction,(100, 85),cv2.FONT_HERSHEY_PLAIN , 3,200, 	thickness = 3)
    cv2.imwrite(str(n)+".jpg", source_image )

    cv2.imshow('color ', source_image)
    cv2.waitKey(0)

click(test_image,source_image,1)
click(test_image1,source_image1,2)
click(test_image2,source_image2,3)

then save it and run main.py by typing :
python main.py


if you decided to train your own model please refer to tensorflow documentation :D


