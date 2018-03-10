import datetime
import math
import keras
from keras.models import Model
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input

int_map = {0: '0', 1: '1', 2: '10', 3: '100', 4: '101', 5: '102', 6: '103', 7: '104', 8: '105', 9: '106', 10: '107', 11: '108', 12: '109', 13: '11', 14: '110', 15: '111', 16: '112', 17: '113', 18: '114', 19: '115', 20: '116', 21: '117', 22: '118', 23: '119', 24: '12', 25: '120', 26: '121', 27: '122', 28: '123', 29: '124', 30: '125', 31: '126', 32: '127', 33: '128', 34: '129', 35: '13', 36: '130', 37: '131', 38: '14', 39: '15', 40: '16', 41: '17', 42: '18', 43: '19', 44: '2', 45: '20', 46: '21', 47: '22', 48: '23', 49: '24', 50: '25', 51: '26', 52: '27', 53: '28', 54: '29', 55: '3', 56: '30', 57: '31', 58: '32', 59: '33', 60: '34', 61: '35', 62: '36', 63: '37', 64: '38', 65: '39', 66: '4', 67: '40', 68: '41', 69: '42', 70: '43', 71: '44', 72: '45', 73: '46', 74: '47', 75: '48', 76: '49', 77: '5', 78: '50', 79: '51', 80: '52', 81: '53', 82: '54', 83: '55', 84: '56', 85: '57', 86: '58', 87: '59', 88: '6', 89: '60', 90: '61', 91: '62', 92: '63', 93: '64', 94: '65', 95: '66', 96: '67', 97: '68', 98: '69', 99: '7', 100: '70', 101: '71', 102: '72', 103: '73', 104: '74', 105: '75', 106: '76', 107: '77', 108: '78', 109: '79', 110: '8', 111: '80', 112: '81', 113: '82', 114: '83', 115: '84', 116: '85', 117: '86', 118: '87', 119: '88', 120: '89', 121: '9', 122: '90', 123: '91', 124: '92', 125: '93', 126: '94', 127: '95', 128: '96', 129: '97', 130: '98', 131: '99'}

TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
#SAVED_MODEL1 = "/home/users/nus/e0146089/super_iv3.h5"
#SAVED_MODEL2 = "/home/users/nus/e0146089/train_log/v2/food_v2a-20171012_162315.0.80515.h5"
SAVED_MODEL1 = "/home/users/nus/e0146089/rc_models/food_v2-20171018_212218.0.82729.f.h5"
SAVED_MODEL2 = "/home/users/nus/e0146089/rc_models/food_v2a-20171019_192233.0.82394.f.h5"

OUTPUT = "/home/users/nus/e0146089/predict_siv3_%s.csv"%TIMESTAMP
# test images in transferred_test/unknown
PREDICT_DIR = "/home/users/nus/e0146089/transferred_test"
#PREDICT_DIR = "/home/users/nus/e0146089/predict_test"
#BATCH_SIZE = 128
BATCH_SIZE =32

probabilities = {}
labels = {}

model1 = keras.models.load_model(SAVED_MODEL1)
print("loading %s"%SAVED_MODEL1)
predict_datagen = image.ImageDataGenerator(preprocessing_function=preprocess_input)
predict_generator = predict_datagen.flow_from_directory( PREDICT_DIR, target_size=(299,299), batch_size=BATCH_SIZE, class_mode=None, shuffle=False)
filenames1 = predict_generator.filenames
probabilities1 = model1.predict_generator(predict_generator, int(math.ceil(len(filenames1)/float(BATCH_SIZE))))

for i in range(len(probabilities1)):
    f = filenames1[i][filenames1[i].rindex("/")+1:]
    probabilities[f]=probabilities1[i]

model2 = keras.models.load_model(SAVED_MODEL2)
print("loading %s"%SAVED_MODEL2)
predict_datagen = image.ImageDataGenerator(preprocessing_function=preprocess_input)
predict_generator = predict_datagen.flow_from_directory( PREDICT_DIR, target_size=(299,299), batch_size=BATCH_SIZE, class_mode=None, shuffle=False)
filenames2 = predict_generator.filenames
probabilities2 = model2.predict_generator(predict_generator, int(math.ceil(len(filenames2)/float(BATCH_SIZE))))

for i in range(len(probabilities2)):
    f = filenames2[i][filenames2[i].rindex("/")+1:]
    probabilities[f]=probabilities[f]+probabilities2[i]

for f in probabilities:
    category = int_map[probabilities[f].argmax()]
    index = int(f.split(".")[0])
    labels[index]=(f,category)

f = open(OUTPUT,"w")
f.write("image_name,category\n")
keys = labels.keys()
keys = sorted(keys)
for i in keys:
    (name,category)=labels[i]
    f.write("%s,%s\n"%(name,category))
f.close()
