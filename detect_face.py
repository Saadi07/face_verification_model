import cv2
import os

# In this file I'm collecting images from users to create dataset of images and storing it in images folder.
# Further I'm collecting 30 images of each person and storing it in their respective class folder.

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam.
cap = cv2.VideoCapture(0)

# Taking name as input to name the image and create a class directory
user_name = input('\n enter user name end press <return> ==>  ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")

parent_dir = "/home/saadi09/fInal_face_verification/images/"

# Creating directory of user class
path = os.path.join(parent_dir, user_name)
os.mkdir(path)
print(path)

count = 0
while True:
    # Read the frame
    _, img = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1
        os.path.join(parent_dir, user_name)
        cv2.imwrite(path + "/" + str(user_name) + '.' +
                    str(count) + ".jpg", gray[y:y + h, x:x + w])

    # Display
    cv2.imshow('img', img)

    k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30:  # Take 30 face sample and stop video
        break

# Release the VideoCapture object
cap.release()
