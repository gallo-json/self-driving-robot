import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("train_dir", help="Train  dir", type=str)
parser.add_argument("test_dir", help="Test  dir", type=str)
parser.add_argument('--colab', help='Make the txt files for google colab', action='store_true')
args = parser.parse_args()

print(args.colab)

if args.colab:
    train_txt = 'train-colab.txt'
    test_txt = 'test-colab.txt'
else:
    train_txt = 'train.txt'
    test_txt = 'test.txt'

train_images = []

for train_directory, class_folders, files in os.walk(args.train_dir):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            if args.colab:
                train_images.append('/content/drive/MyDrive/darknet/' + os.path.join(train_directory, file))
            else:
                train_images.append(os.path.join(train_directory, file))

with open(train_txt, "w") as outfile:
    for image in train_images:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()


test_images = []

for test_directory, class_folders, files in os.walk(args.test_dir):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            if args.colab:
                test_images.append('/content/drive/MyDrive/darknet/' + os.path.join(test_directory, file))
            else:
                test_images.append(os.path.join(test_directory, file))
    
with open(test_txt, "w") as outfile:
    for image in test_images:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()
