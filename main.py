import learning
import matplotlib.pyplot as plt

print('Processing your photo.')
# image_input = input('Enter the path of your photo: ')
image_input = './predict2.jpg'
im = learning.cv2.imread(f'{image_input}')
bbox, label, conf = learning.cv.detect_common_objects(im)
output_image = learning.draw_bbox(im, bbox, label, conf)
plt.title('Object detection')
plt.imshow(output_image)
plt.show()

greenCoeff = 0.1
brownCoeff = 0.2
whiteCoeff = 0.3

totalPrice = 0

for bottle in bbox:
    img = im[bottle[1]:bottle[3], bottle[0]:bottle[2]]

    pred = learning.predict_glass(img)
    if pred[0] == 'GreenGlass':
        print('Green glass detected, ~' + str(int(round(pred[1] * 100))) + '%')
        totalPrice += greenCoeff
    if pred[0] == 'BrownGlass':
        print('Brown glass detected, ~' + str(int(round(pred[1] * 100))) + '%')
        totalPrice += brownCoeff
    if pred[0] == 'WhiteGlass':
        print('White glass detected, ~' + str(int(round(pred[1] * 100))) + '%')
        totalPrice += whiteCoeff

totalPrice = int(totalPrice * 10) / 10
print('Total price is: ' + str(totalPrice))

print('Reading user\'s unique account info from QR code.')
inputImage = learning.cv2.imread('./qr_test.png')  # key = 73058


# Display barcode and QR code location
def display(im, bbox):
    n = len(bbox)
    for j in range(n):
        learning.cv2.line(im, tuple(bbox[j][0]), tuple(bbox[(j + 1) % n][0]), (255, 0, 0), 3)


qrDecoder = learning.cv2.QRCodeDetector()

# Detect and decode the qr code
data, bbox, rectifiedImage = qrDecoder.detectAndDecode(inputImage)
if len(data) > 0:
    print("Decoded key from QR code: {}".format(data))

data = '05110'

r = learning.httpx.get('https://baltoshackos.herokuapp.com/Home/GetPasswordTransaction/key?='
                       + data)
input_str = r.text
temp = list(filter(lambda s: s.isdigit(), input_str))
code = ''  # string of 5 digits
for i in temp:
    code += str(i)

print('Sending payment to user (' + str(totalPrice) + ').')
r = learning.httpx.get('https://baltoshackos.herokuapp.com/Home/SetUserMoney/AccId?=' +
                       code + '&value=' + str(totalPrice))

print('Status: ' + str(r))
print('User\'s balance: ' + r.text)
