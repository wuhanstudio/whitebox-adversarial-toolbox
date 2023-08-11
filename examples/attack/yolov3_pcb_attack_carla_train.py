import cv2
import datetime
import numpy as np

from what.models.detection.datasets.coco import COCO_CLASS_NAMES
from what.models.detection.utils.box_utils import draw_bounding_boxes
from what.models.detection.yolo.utils.yolo_utils import yolo_process_output, yolov3_anchors, yolov3_tiny_anchors

from what.attacks.detection.yolo.PCB import PCBAttack
from what.utils.resize import bilinear_resize

from what.cli.model import *
from what.utils.file import get_file

import carla

import queue
import random

# Part 1

client = carla.Client('localhost', 2000)
world  = client.get_world()

# Set up the simulator in synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True # Enables synchronous mode
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

# Get the world spectator
spectator = world.get_spectator() 

# Get the map spawn points
spawn_points = world.get_map().get_spawn_points()

# spawn vehicle
bp_lib = world.get_blueprint_library()
vehicle_bp =bp_lib.find('vehicle.lincoln.mkz_2020')
vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

# spawn camera
camera_bp = bp_lib.find('sensor.camera.rgb')
camera_init_trans = carla.Transform(carla.Location(z=2))
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

# Create a queue to store and retrieve the sensor data
image_queue = queue.Queue()
camera.listen(image_queue.put)

for i in range(20):
    vehicle_bp = bp_lib.filter('vehicle')

    # Exclude bicycle
    car_bp = [bp for bp in vehicle_bp if int(bp.get_attribute('number_of_wheels')) == 4]
    npc = world.try_spawn_actor(random.choice(car_bp), random.choice(spawn_points))

    if npc:
        npc.set_autopilot(True)

# Part 2

def clear():
    settings = world.get_settings()
    settings.synchronous_mode = False # Disables synchronous mode
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    camera.stop()

    for npc in world.get_actors().filter('*vehicle*'):
        if npc:
            npc.destroy()

    print("Vehicles Destroyed.")

# Target Model
what_yolov3_model_list = what_model_list[0:4]

# Check what_model_list for all supported models
index = 3

# Download the model first if not exists
WHAT_YOLOV3_MODEL_FILE = what_yolov3_model_list[index][WHAT_MODEL_FILE_INDEX]
WHAT_YOLOV3_MODEL_URL  = what_yolov3_model_list[index][WHAT_MODEL_URL_INDEX]
WHAT_YOLOV3_MODEL_HASH = what_yolov3_model_list[index][WHAT_MODEL_HASH_INDEX]

if not os.path.isfile(os.path.join(WHAT_MODEL_PATH, WHAT_YOLOV3_MODEL_FILE)):
    get_file(WHAT_YOLOV3_MODEL_FILE,
             WHAT_MODEL_PATH,
             WHAT_YOLOV3_MODEL_URL,
             WHAT_YOLOV3_MODEL_HASH)

if __name__ == '__main__':

    classes = COCO_CLASS_NAMES

    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Adversarial Attack
    model_path = os.path.join(WHAT_MODEL_PATH, what_yolov3_model_list[index][WHAT_MODEL_FILE_INDEX])
    attack = PCBAttack(model_path, "multi_untargeted", classes, decay=0.99)
    attack.fixed = False

    last_outs = None
    last_boxes = None
    last_probs = None

    vehicle.set_autopilot(True)

    while(True): 
        try:
            world.tick()

            # Move the spectator to the top of the vehicle 
            transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4,z=50)), carla.Rotation(yaw=-180, pitch=-90)) 
            spectator.set_transform(transform) 

            # Retrieve and reshape the image
            image = image_queue.get()

            truth_img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
            origin_cv_image = truth_img[:, :, :-1]

            # For YOLO, the input pixel values are normalized to [0, 1]
            input_cv_image = cv2.resize(origin_cv_image, (416, 416))
            input_cv_image = np.array(input_cv_image).astype(np.float32) / 255.0

            # Image preprocessing
            input_cv_image = cv2.cvtColor(input_cv_image, cv2.COLOR_BGR2RGB)

            # Yolo inference
            input_cv_image, outs = attack.attack(input_cv_image)

            if last_outs is not None:
                res_list = []
                for out, last_out in zip(outs, last_outs):
                    out = out.reshape((-1, 5 + len(classes)))
                    last_out = last_out.reshape((-1, 5 + len(classes)))

                    res = np.mean(out[:, 4] - last_out[:, 4])
                    res_list.append(res)
            else:
                last_outs = outs

            boxes, labels, probs = yolo_process_output(outs, yolov3_tiny_anchors, len(classes))

            # Draw bounding boxes
            out_img = cv2.cvtColor(origin_cv_image, cv2.COLOR_RGB2BGR)
            out_img = out_img.astype(np.float32) / 255.0
            height, width, _ = out_img.shape
            noise = attack.noise

            # Resize the noise to the same shape as the input image
            # noise_r = bilinear_resize(noise[:, :, 0], height, width)
            # noise_g = bilinear_resize(noise[:, :, 1], height, width)
            # noise_b = bilinear_resize(noise[:, :, 2], height, width)
            # noise = np.dstack((noise_r, noise_g, noise_b))

            # out_img = out_img + noise
            out_img = np.clip(out_img, 0.0, 1.0)

            out_img = (out_img * 255.0).astype(np.uint8)

            out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
            out_img = draw_bounding_boxes(out_img, boxes, labels, classes, probs);

            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", out_img)

            if (cv2.waitKey(1) & 0xFF == ord('q')):
                clear()
                break

        except KeyboardInterrupt as e:
            clear()
            break

    cv2.destroyAllWindows()

    np.save('noise.npy', attack.noise)
