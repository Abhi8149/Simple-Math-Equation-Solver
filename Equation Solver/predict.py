import cv2
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
import streamlit as st 

label = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '+', 11: '÷', 12: '*', 13: '-'}

model=load_model('cnn_model.keras')

def predict(image_path):
    image=cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #converts black to white and white to black
    #means convert number to white and background to black as countour can be captured easily in black background
    _,binary_image=cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    contours,_=cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #geeting the (x,y,w,h) (x,y)-top left corner of contour (w,h)- width and height of image
    #Here its an list of contour of all the digits or operators present in the image
    bounding_boxes=[]
    filtered_contours=[]
    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour)
        if w*h>=100:
            bounding_boxes.append((x,y,w,h))
            filtered_contours.append(contour)
            
    # bounding_boxes=[cv2.boundingRect(contour) for contour in contours]

    #sorting all the contours based on there x-coordinate
    sorted_indices = sorted(range(len(bounding_boxes)), key=lambda i: bounding_boxes[i][0])

    #arranging the contour from left ro right so that it can be readible and solvable
    sorted_contours=[filtered_contours[i] for i in sorted_indices]

    #capturing each digit individually
    rois=[]
    #adding padding around the image so that it can be captured carefully
    padding=25

    for contour in sorted_contours:
        x,y,w,h=cv2.boundingRect(contour)
        x_start=max(0, x-padding)
        y_start=max(0, y-padding)
        x_end=min(image.shape[1], x+w+padding)
        y_end=min(image.shape[0], y+h+padding)

        roi=image[y_start:y_end, x_start:x_end]
        roi=cv2.resize(roi, (32,32))
        rois.append(roi)

    rois=np.array(rois)
    rois=rois/255.0
    rois=np.expand_dims(rois, axis=-1)
    predictions=model.predict(rois)
    predicted_labels=np.argmax(predictions, axis=1)
    st.write('Here is the prediction labels', predicted_labels)
    #converts the grayscale image to BGR color 
    image_color=cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for i, contour in enumerate(sorted_contours):
        x,y,w,h=cv2.boundingRect(contour)
        label_one=label[predicted_labels[i]]
        cv2.rectangle(image_color, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(image_color, label_one, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
    
    st.image(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
    equation=''.join(label[predicted_labels[i]] for  i in range(len(predicted_labels)))  
    safe_equation = equation.replace('÷', '/')
    print("Equation is:",equation)
    return equation, eval(safe_equation)