import os
import numpy as np
import pandas as pd
import datetime

DIR="/media/cdrom/predictions"
TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = "/media/cdrom/predictions/predict_%s.csv"%TIMESTAMP
#output_file = "/media/cdrom/predictions/predict_irn7_20171108_130016.csv"
prediction_pandas_irn = []
prediction_pandas_irn.append("raw_pred_irv2_40.csv")
prediction_pandas_irn.append("raw_pred_irv2_all_20.csv")
prediction_pandas_iv3 = []
prediction_pandas_iv3.append("raw_pred_incep_40.csv")
prediction_files_irn = []
prediction_files_iv3 = []
#prediction_files.append(("predict_irn7_20171108_130016_n.txt","predict_irn7_20171108_130016_p.txt"))
prediction_files_irn.append(("predict_irn6_20171108_132412_n.txt","predict_irn6_20171108_132412_p.txt"))
prediction_files_irn.append(("predict_irn5_20171108_134955_n.txt","predict_irn5_20171108_134955_p.txt"))
prediction_files_irn.append(("predict_irn4-20171029_001258_n.txt","predict_irn4-20171029_001258_p.txt"))
prediction_files_irn.append(("predict_irn3_20171101_085810_n.txt","predict_irn3_20171101_085810_p.txt"))
prediction_files_irn.append(("predict_irn_20171028_132554_n.txt","predict_irn_20171028_132554_p.txt"))
prediction_files_irn.append(("predict_irn2_20171028_233102_n.txt","predict_irn2_20171028_233102_p.txt"))
prediction_files_iv3.append(("predict_d05_20171028_140447_n.txt","predict_d05_20171028_140447_p.txt"))
prediction_files_iv3.append(("predict_v4_20171106_142635_n.txt","predict_v4_20171106_142635_p.txt"))
prediction_files_iv3.append(("predict_v3_20171026_200930_n.txt","predict_v3_20171026_200930_p.txt"))
#prediction_files.append(("predict_rn50-2_20171024_180619_n.txt","predict_rn50-2_20171024_180619_p.txt"))

INT_MAP = {0: '0', 1: '1', 2: '10', 3: '100', 4: '101', 5: '102', 6: '103', 7: '104', 8: '105', 9: '106', 10: '107', 11: '108', 12: '109', 13: '11', 14: '110', 15: '111', 16: '112', 17: '113', 18: '114', 19: '115', 20: '116', 21: '117', 22: '118', 23: '119', 24: '12', 25: '120', 26: '121', 27: '122', 28: '123', 29: '124', 30: '125', 31: '126', 32: '127', 33: '128', 34: '129', 35: '13', 36: '130', 37: '131', 38: '14', 39: '15', 40: '16', 41: '17', 42: '18', 43: '19', 44: '2', 45: '20', 46: '21', 47: '22', 48: '23', 49: '24', 50: '25', 51: '26', 52: '27', 53: '28', 54: '29', 55: '3', 56: '30', 57: '31', 58: '32', 59: '33', 60: '34', 61: '35', 62: '36', 63: '37', 64: '38', 65: '39', 66: '4', 67: '40', 68: '41', 69: '42', 70: '43', 71: '44', 72: '45', 73: '46', 74: '47', 75: '48', 76: '49', 77: '5', 78: '50', 79: '51', 80: '52', 81: '53', 82: '54', 83: '55', 84: '56', 85: '57', 86: '58', 87: '59', 88: '6', 89: '60', 90: '61', 91: '62', 92: '63', 93: '64', 94: '65', 95: '66', 96: '67', 97: '68', 98: '69', 99: '7', 100: '70', 101: '71', 102: '72', 103: '73', 104: '74', 105: '75', 106: '76', 107: '77', 108: '78', 109: '79', 110: '8', 111: '80', 112: '81', 113: '82', 114: '83', 115: '84', 116: '85', 117: '86', 118: '87', 119: '88', 120: '89', 121: '9', 122: '90', 123: '91', 124: '92', 125: '93', 126: '94', 127: '95', 128: '96', 129: '97', 130: '98', 131: '99'}

def load_prediction(filenames_txt,probabilities_txt):
    print "loading %s"%probabilities_txt
    n = np.loadtxt(os.path.join(DIR,filenames_txt),np.str)
    p = np.loadtxt(os.path.join(DIR,probabilities_txt),np.float64)
    return (n,p)

def load_prediction_pandas(pandas_csv):
    print "loading %s"%pandas_csv
    csv = pd.read_csv(os.path.join(DIR,pandas_csv),index_col=0)
    n = csv.index
    p = csv.values
    return (n,p)

predictions_irn = []
predictions_iv3 = []
for i in prediction_files_irn:
   (n,p)=load_prediction(i[0],i[1])
   predictions_irn.append((n,p))
for i in prediction_files_iv3:
   (n,p)=load_prediction(i[0],i[1])
   predictions_iv3.append((n,p))
for i in prediction_pandas_irn:
   (n,p)=load_prediction_pandas(i)
   predictions_irn.append((n,p))
for i in prediction_pandas_iv3:
   (n,p)=load_prediction_pandas(i)
   predictions_iv3.append((n,p))


probabilities_irn = {}
probabilities_iv3 = {}
for (n,p) in predictions_irn:
    for i in range(len(n)):
        filename = n[i]
        if "/" in filename: filename=filename[filename.rindex("/")+1:]
        if filename not in probabilities_irn:probabilities_irn[filename]=p[i]
        else:
           probabilities_irn[filename]=probabilities_irn[filename]+p[i]
for (n,p) in predictions_iv3:
    for i in range(len(n)):
        filename = n[i]
        if "/" in filename: filename=filename[filename.rindex("/")+1:]
        if filename not in probabilities_iv3:probabilities_iv3[filename]=p[i]
        else:
           probabilities_iv3[filename]=probabilities_iv3[filename]+p[i]


labels = {}
for filename in probabilities_irn:
    index = int(filename.split(".")[0])
    p_irn = probabilities_irn[filename]/float(len(probabilities_irn))
    p_iv3 = probabilities_iv3[filename]/float(len(probabilities_iv3))
    p = (p_irn*0.6)+(p_iv3*0.4)
    category = INT_MAP[p.argmax()]
    labels[index]=(filename,category)

print "wriitng output: %s"%output_file
f = open(output_file,"w")
f.write("image_name,category\n")
keys = labels.keys()
keys = sorted(keys)
for i in keys:
    (name,category)=labels[i]
    f.write("%s,%s\n"%(name,category))
f.close()


