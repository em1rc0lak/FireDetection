import cv2
import numpy as np  
  
ADD_MOTION_AND_FLICKER = True

def r1(y, cb):
    return (y >= cb)


def r2(cr, cb):
    return (cr >= cb)


def r3(y, cb, cr, y_hat, cb_hat, cr_hat):
    mask = (y >= y_hat) & (cb <= cb_hat) & (cr >= cr_hat)
    return mask


def r4(cb, cr, tau):
    return (np.abs(cb - cr) >= tau)


def f1(cr):
    coeffs = (-2.62e-10, 3.27e-07, -1.75e-04, 5.16e-02,
              -9.10, -5.60e04)
    return coeffs[0]*(cr**7) + coeffs[1]*(cr**6) + coeffs[2]*(cr**5) + coeffs[3]*(cr**4) +\
                coeffs[4]*(cr**3) + coeffs[5]*(cr) + 1.40e06  


def f2(cr):
    coeffs = (-6.77e-08, 5.50e-05, -1.76e-02, 2.78, -2.15e02)
    return coeffs[0]*(cr**5) + coeffs[1]*(cr**4) + coeffs[2]*(cr**3) + \
        coeffs[3]*(cr**2) + coeffs[4]*(cr) + 6.62e03


def f3(cr):
    coeffs = (1.80e-04, -1.02e-01, 21.66, -2.05e03)
    return coeffs[0]*(cr**4) + coeffs[0]*(cr**3) + \
        coeffs[0]*(cr**2) + coeffs[0]*(cr) + 7.29e04


def r5(cb, cr):
    mask = (cb >= f1(cr)) & (cb <= f3(cr)) & (cb <= f2(cr))
    return mask.astype(np.uint8)

def get_color_mask(y, cr, cb):
    y_hat, cb_hat, cr_hat = float(y.mean()), float(cb.mean()), float(cr.mean())
    rule1 = r1(y, cb)
    rule2 = r2(cr, cb)
    rule3 = r3(y, cb, cr, y_hat, cb_hat, cr_hat)
    rule4 = r4(cb, cr, 62.0)
    rule5 = r5(cb, cr)
    return(rule1 & rule2 & rule3 & rule4 & rule5).astype(np.uint8)

def get_flicker_mask(Y, avg_Y, alpha=0.1, flicker_thresh=8):

    if avg_Y is None:
        avg_Y = Y.copy()
        return np.zeros_like(Y, dtype=np.uint8), avg_Y

    cv2.accumulateWeighted(Y, avg_Y, alpha)
    flicker = cv2.absdiff(Y, avg_Y) #take the difference from averag if high the pixel flickers
    _, flicker_mask = cv2.threshold(flicker, flicker_thresh, 255, cv2.THRESH_BINARY)

    return flicker_mask.astype(np.uint8), avg_Y


def get_motion_mask(Y, prev_Y, motion_threshold = 8):

    if prev_Y is None:
        prev_Y = Y.copy()
        return np.zeros_like(Y, dtype=np.uint8), prev_Y

    diff = cv2.absdiff(Y.astype(np.uint8), prev_Y.astype(np.uint8))
    prev_Y = Y.copy()

    _, motion_mask = cv2.threshold(diff, motion_threshold, 255, cv2.THRESH_BINARY)

    return motion_mask.astype(np.uint8), prev_Y

def draw_fire_rectangle(image, mask):

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return
        
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)


def main():

    for i in range(1,4):
        video_path = "fire_videos/" + str(i) + ".mp4"
        cap = cv2.VideoCapture(video_path)

        prev_Y = None
        avg_Y = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if (i == 1):
                frame = cv2.resize(frame, None, fx = 2, fy = 2)

            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb.astype(np.float64))
            
            color_mask = get_color_mask(y, cr, cb)
            combined = color_mask

            if ADD_MOTION_AND_FLICKER:
                motion_mask, prev_Y = get_motion_mask(y, prev_Y)
                flicker_mask, avg_Y = get_flicker_mask(y, avg_Y)
                combined = cv2.bitwise_and(color_mask, cv2.bitwise_or(motion_mask, flicker_mask))
            

            fire_detected = combined.any()
            draw_fire_rectangle(frame, combined)

            label = "Fire detected" if fire_detected else "No fire"
            color = (0, 0, 255) if fire_detected else (0, 255, 0)
            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.imshow("Video", frame)

            key = cv2.waitKey(1) & 0xFF  
            #press q to watch the next video
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

main()
