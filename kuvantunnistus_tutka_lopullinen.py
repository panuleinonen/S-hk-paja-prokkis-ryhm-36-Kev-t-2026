
#Necessary imports
import cv2
from picamera2 import Picamera2
import numpy as np
import RPi.GPIO as GPIO
import time
import threading
from RpiMotorLib import RpiMotorLib

DIR_PIN_Y  = 19
DIR_PIN_X  = 20
STEP_PIN_X = 21
STEP_PIN_Y = 26
HALF_STEP = 17

#HC-SR04 pins
TRIG_PIN = 22
ECHO_PIN = 23

DEADZONE   = 40    #Doesn't make corrections if the error is tjhis or smaller
MAX_STEPS  = 5     #Max pulse per motor per correction to avoid overshoot
STEP_DELAY = 0.0005 #Time between steps.
SWEEP_DELAY = 0.005 #Time between ultrasonic pulses
DETECTION_DISTANCE = 40 #in centimeter

GPIO.setmode(GPIO.BCM) #GPIO.BCM = non-board specific, GPIO.BOARD = Board specific.


#Sets pins using a list rather than an unclean block of GPIO.setup(pin, GPIO.I/O) lines.

for pin in (DIR_PIN_X, STEP_PIN_X, DIR_PIN_Y, STEP_PIN_Y, HALF_STEP):
    GPIO.setup(pin, GPIO.OUT)

GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)

GPIO.output(HALF_STEP, GPIO.HIGH) #Sets MS2 (M1 on DRV8825) to high triggering 1/4 stepping. (I Won't update the variable name...)

#A4988Nema(dir_pin, step_pin, (MS1, MS2, MS3), "motor_type")

motor_x = RpiMotorLib.A4988Nema(DIR_PIN_X, STEP_PIN_X, (14, 15, 18), "DRV8825")
motor_y = RpiMotorLib.A4988Nema(DIR_PIN_Y, STEP_PIN_Y, (14, 15, 18), "DRV8825")

# Common variables and individual threads.
error_lock = threading.Lock() #Can not be acquired by a thread when set.
motor_x_lock = threading.Lock() #Prevents motor thread and scanning thread from interfering
face_detected = threading.Event()
object_in_range = threading.Event() #Set by scan_thread, gates inference_thread
latest_error = {"x": 0, "y": 0}
new_detection = threading.Event()
stop_event = threading.Event()

'''Neural net that we used for our project can be found here: https://github.com/sr6033/face-detection-with-OpenCV-and-DNN/tree/master 
    It's the res10_300x300_ssd_...'''

net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
picam2.configure(config)
picam2.start()

#Function used for calculating distance of object detected by HC-SR04
#Kopsattu Joonaksen tutkapyorii.py skriptistä

def get_distance_cm():
    GPIO.output(TRIG_PIN, False)
    time.sleep(0.02)
    GPIO.output(TRIG_PIN, True)
    time.sleep(0.00001)
    GPIO.output(TRIG_PIN, False)

    timeout_start = time.time()
    while GPIO.input(ECHO_PIN) == 0:
        if time.time() - timeout_start > 0.04:
            return None

    pulse_start = time.time()
    while GPIO.input(ECHO_PIN) == 1:
        if time.time() - pulse_start > 0.04:
            return None

    pulse_end = time.time()
    return (pulse_end - pulse_start) * 34300 / 2


'''Neural networks inference loop: detects face, 
determines error and makes sure nothing happens if there's no face'''

def inference_thread():
    while not stop_event.is_set():
        object_in_range.wait()  #Waits for scan_thread to set the variable before continuing.
        if stop_event.is_set():
            break

        frame = picam2.capture_array()
        h, w = frame.shape[:2]  #480 [height], 640 [width]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)
        )
        net.setInput(blob)
        detections = net.forward()

        '''Using the neural nets confidence variable, 
        chooses which target to follow if there's more than one'''

        best_conf = 0
        best_box  = None
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > 0.5 and conf > best_conf:
                best_conf = conf
                best_box  = detections[0, 0, i, 3:7]

        if best_box is not None:
            box = best_box * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            cx = round((x1 + x2) / 2)
            cy = round((y1 + y2) / 2)

            ex = 320 - cx   # positive: face is left of center
            ey = 240 - cy   # positive: face is above centerline

            with error_lock:
                latest_error["x"] = ex
                latest_error["y"] = ey
            new_detection.set()  #Sets
            face_detected.set()  #Sets face_detected and causes scan_thread to pause

            '''Draws dot on screen for debugginh purposes'''

            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(
                frame, f"ex={ex} ey={ey}",
                (x1, max(y1 - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
        else:

            with error_lock:
                latest_error["x"] = 0
                latest_error["y"] = 0

            face_detected.clear() #Unsets face_detected, freeing scan_thread.

        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()


'''Stepper motor control code. 
   Driven in another thread to avoid blocking problems prevelant in previous iterations.'''

def motor_thread():
    while not stop_event.is_set():
        new_detection.wait()
        new_detection.clear()

        with error_lock:
            ex = latest_error["x"]
            ey = latest_error["y"]

        print(f"error_x={ex}, error_y={ey}")

        # X-axis motor
        if abs(ex) > DEADZONE:
            steps = min(int(abs(ex) / 60), MAX_STEPS)
            #motor_x.motor_go(direction, "Step-type, e.g. Full, 1/2, 1/4.", nmbr of steps, STEP_DELAY, False, 0.0)
            clockwise_x = ex > 0
            with motor_x_lock:
                motor_x.motor_go(clockwise_x, "1/4",
                steps, STEP_DELAY, False, 0.0)

        # Y-axis motor
        if abs(ey) > DEADZONE:
            steps = min(int(abs(ey) / 60), MAX_STEPS)
            clockwise_y = ey < 0
            motor_y.motor_go(clockwise_y, "1/4", steps, 
                             STEP_DELAY, False, 0.0)

        time.sleep(0.005)  #Sleeps thread for a moment to save resources

def scan_thread():
    while not stop_event.is_set():
        #Blocks sweep if face is detected.
        if face_detected.is_set() or object_in_range.is_set():
            time.sleep(0.05)
            continue

        distance = get_distance_cm()

        if distance is not None and distance <= DETECTION_DISTANCE:
            if not object_in_range.is_set():
                print(f"detection {distance:.1f} cm")
                object_in_range.set()
        else:
            if object_in_range.is_set():
                print("No object, resuming scan")
                object_in_range.clear()
                face_detected.clear()

            with motor_x_lock:
                motor_x.motor_go(True, "1/4", 1, STEP_DELAY, False, 0.0)

        time.sleep(SWEEP_DELAY)


#Master thread. As the name suggests, starts threads.

print("Kaynnissa -- paina q Lopettaaksesi")

t_inference = threading.Thread(target=inference_thread, daemon=True)
t_motors    = threading.Thread(target=motor_thread,    daemon=True)
t_scan      = threading.Thread(target=scan_thread, daemon=True)
try:
    t_inference.start()
    t_motors.start()
    t_scan.start()

    #stop_event is the press of q on a keyboard
    stop_event.wait()

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    GPIO.cleanup()
    print("Lopetettu.")
