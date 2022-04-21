import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('./checkpoints/handWrittenCNN.h5')
model.summary()

src = cv2.imread("nums.png", cv2.IMREAD_COLOR)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

ret, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

binary = cv2.bitwise_not(gray, binary)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours :
    # rect[0] : 직사각형의 왼쪽 상단 점의 x좌표
    # rect[1] : 직사각형의 왼쪽 상단 점의 y좌표
    # rect[2] : 직사각형의 가로 길이
    # rect[3] : 직사각형의 세로 길이
    #cv2.drawContours(src, [contour], 0, (0, 255, 255), 2)

    rect = cv2.boundingRect(contour)

    roi = binary[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]].copy()

    # 예측 정확도 향상을 위해 resize 할 때 숫자가 찌그러지지 않도록
    # 정사각형 모양으로 빈화면을 채워준다
    if (rect[2] < rect[3]):
        plus = (rect[3] - rect[2]) // 2
        roi = cv2.copyMakeBorder(roi, 0, 0, plus, plus, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    elif (rect[2] > rect[3]):
        plus = (rect[2] - rect[3]) // 2
        roi = cv2.copyMakeBorder(roi, plus, plus, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    roi = cv2.resize(roi, (28, 28))
    roi = roi.reshape(-1, 28, 28, 1)

    predicted = model.predict(roi)
    classNum = predicted.argmax(axis=-1)[0]

    leftTop = (rect[0], rect[1])
    rightTop = (rect[0] + rect[2], rect[1])
    leftDown = (rect[0], rect[1] + rect[3])
    rightDown = (rect[0] + rect[2], rect[1] + rect[3])

    text = f'{classNum}'
    cv2.putText(src, text, leftTop, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.line(src, leftTop, rightTop, color=(255, 0, 0), thickness=1)
    cv2.line(src, leftTop, leftDown, color=(255, 0, 0), thickness=1)
    cv2.line(src, rightDown, rightTop, color=(255, 0, 0), thickness=1)
    cv2.line(src, rightDown, leftDown, color=(255, 0, 0), thickness=1)


plt.imshow(src); plt.show()

