import argparse
from pytorchyolo.models import load_model
from pytorchyolo.utils.utils import to_cpu, load_classes, print_environment_info, provide_determinism, worker_seed_set
from pytorchyolo.utils.parse_config import parse_data_config
import cv2
from pytorchyolo import detect, models
import glob, os

IMAGE_PARENT_DIR = os.getcwd()
IMAGE_FOLDER_NAME = "vasp_images"
IMAGE_DIR = os.path.join(IMAGE_PARENT_DIR, IMAGE_FOLDER_NAME)
# print(IMAGE_DIR)
OUTPUT_DIR = os.path.join(IMAGE_PARENT_DIR, "output_" + IMAGE_FOLDER_NAME)
if not os.path.exists(OUTPUT_DIR):
   os.makedirs(OUTPUT_DIR)

def run():
    print_environment_info()
    parser = argparse.ArgumentParser(description="Predicts and classifies vehicle lights in images present in a folder.")
    parser.add_argument("-m", "--model", type=str, default="config/yolov3.cfg", help="Path to model definition file (.cfg)")
    parser.add_argument("--pretrained_weights", type=str, help="Path to checkpoint file (.weights or .pth). Starts training from checkpoint model")
    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    model = load_model(args.model, args.pretrained_weights)


    all_test_images = glob.glob(str(IMAGE_DIR) + "/*.png")
    all_test_images.extend(glob.glob(str(IMAGE_DIR) + "/*.jpg"))

    # print(all_test_images)

    color = [
        (0, 255, 0),
        (0, 0, 255),
        (255, 0, 0),
    ]

    for image in all_test_images:
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
            cv2.imwrite(os.path.join(OUTPUT_DIR, image_filename), img)
            

        # print(boxes)


if __name__ == "__main__":
    run()