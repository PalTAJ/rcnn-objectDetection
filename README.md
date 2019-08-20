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
next go to color_detection directory open main.py and add the following code (from link below) for single image color detection option :
#in my case i only classifiy 3 objects.
https://github.com/PalTAJ/rcnn-objectDetection/blob/master/changes_colord.py

then save it and run main.py by typing :
python main.py


if you decided to train your own model please refer to tensorflow documentation :D


