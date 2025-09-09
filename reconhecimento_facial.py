import cv2
import numpy as np
import face_recognition
import os

images = 'images'

imgkayky = face_recognition.load_image_file(f'{images}/kayky1.jpg')
imgkayky = cv2.cvtColor(imgkayky, cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file(f'{images}/kayky2.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

imgpedro1 = face_recognition.load_image_file(f'{images}/pedro1.jpg')
imgpedro1 = cv2.cvtColor(imgpedro1, cv2.COLOR_BGR2RGB)

imgpedro2 = face_recognition.load_image_file(f'{images}/pedro2.jpg')
imgpedro2 = cv2.cvtColor(imgpedro2, cv2.COLOR_BGR2RGB)


faceLoc = face_recognition.face_locations(imgkayky)[0]
encodekayky = face_recognition.face_encodings(imgkayky)[0]
cv2.rectangle(imgkayky, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (0, 0, 255), 2)

faceTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceTest[3], faceTest[0]), (faceTest[1], faceTest[2]), (0, 0, 255), 2)

facepedro1 = face_recognition.face_locations(imgpedro1)[0]
encodepedro1 = face_recognition.face_encodings(imgpedro1)[0]
cv2.rectangle(imgpedro1, (facepedro1[3], facepedro1[0]), (facepedro1[1], facepedro1[2]), (0, 0, 255), 2)

facepedro2 = face_recognition.face_locations(imgpedro2)[0]
encodepedro2 = face_recognition.face_encodings(imgpedro2)[0]
cv2.rectangle(imgpedro2, (facepedro2[3], facepedro2[0]), (facepedro2[1], facepedro2[2]), (0, 0, 255), 2)

results = face_recognition.compare_faces([encodekayky], encodeTest)
distance = face_recognition.face_distance([encodekayky], encodeTest)

resultspedropedro = face_recognition.compare_faces([encodepedro1], encodepedro2)
distancepedropedro = face_recognition.face_distance([encodepedro1], encodepedro2)

resultspedrokayky = face_recognition.compare_faces([encodepedro1], encodekayky)
distancepedrokayky = face_recognition.face_distance([encodepedro1], encodekayky)

print('Encodekayky', encodekayky)
print('Encodepedro', encodepedro1)

print('kayky com kayky', results)
print('kayky com kayky', distance)

print('pedro com pedro', resultspedropedro)
print('pedro com pedro', distancepedropedro)

print('pedro com kayky', resultspedrokayky)
print('pedro com kayky', distancepedrokayky)

cv2.imshow('kayky', imgkayky)
cv2.imshow('test', imgTest)

cv2.imshow('pedro with beard', imgpedro1)
cv2.imshow('pedro without beard', imgpedro2)


cv2.waitKey(0)
