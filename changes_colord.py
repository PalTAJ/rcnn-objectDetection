images = []
for file in glob.glob("/saved_segmented_from OD model path/*.jpg"):
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
