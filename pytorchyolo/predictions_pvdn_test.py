import cv2
from pytorchyolo import detect, models
import glob, os

# Load the YOLO model
# model = models.load_model(
#   "config/yolov3-custom.cfg",
#   "<PATH_TO_YOUR_WEIGHTS_FOLDER>/yolov3.weights")
def run():
    model = models.load_model("config/yolov3-custom.cfg","checkpoints/yolov3_ckpt_14.pth")


    all_images = glob.glob("/private_shared/Projects/PyTorch-YOLOv3/data/custom/images/*.png")
    print(all_images[0])
    train_valid_images = []
    valid_images = []
    with open('data/custom/train.txt') as f:
        train_valid_images.extend(f.readlines())

    with open('data/custom/valid.txt') as f:
        train_valid_images.extend(f.readlines())
    # images = glob.glob("vasp_images/*.png")

    print(train_valid_images[0])
    train_valid_image_names = []
    for image in train_valid_images:
        image_name = str(str(image).split("/")[-1])
        train_valid_image_names.append(image_name)

    print(train_valid_image_names[0])
    test_images = []
    for image in all_images:
        image_name = str(str(image).split("/")[-1])
        if any(image_name in x  for x in train_valid_image_names):
            continue
        test_images.append(image)

    print(len(train_valid_images))
    print(len(test_images))

    color = [
        (0, 255, 0),
        (0, 0, 255),
        (255, 0, 0),
    ]

    for image in test_images:
        image_filename = str(image).split("/")[-1]
        print(image_filename)
        # Load the image as a numpy array
        img = cv2.imread(image)

        # Convert OpenCV bgr to rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Runs the YOLO model on the image
        boxes = detect.detect_image(model, img)

        for box in boxes:
            x1, y1, x2, y2, conf, cl = box

            img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color[int(cl)], 2)

        if(len(boxes)):
            # print(os.getcwd())
            output_image_path = os.path.join(os.getcwd(), "output_pvdn_test_images", image_filename)
            cv2.imwrite(output_image_path, img)
            

        # print(boxes)


if __name__ == "__main__":
    run()