import numpy as np
import cv2 as cv
import os

# === Video Frame Reader ===
def read_video(video_path):
    """
    Generator function to read and resize frames from a video.

    Args:
        video_path (str): Path to the input video.

    Yields:
        frame (ndarray): Resized video frame (640x480).
    """
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.resize(frame, (640, 480))
        yield frame


# === Canny Edge Detector with Preprocessing ===
def canny_edge_detector(image):
    """
    Applies preprocessing (grayscale, blur, gamma correction, brightness & contrast adjustments)
    followed by Canny edge detection.

    Args:
        image (ndarray): Input BGR image.

    Returns:
        edges (ndarray): Binary image with detected edges.
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) 
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # Manual gamma correction to enhance brightness details
    gamma = 2.4
    normalized = blurred / 255.0
    gamma_corrected = np.power(normalized, gamma)
    gamma_corrected_img = np.uint8(255 * gamma_corrected)

    # Manual brightness adjustment
    brightness_value = -50  # Darken the image
    brightness_img = np.clip(gamma_corrected_img.astype(np.int16) + brightness_value, 0, 255).astype(np.uint8)

    # Manual contrast adjustment
    contrast_factor = 1.5
    mean = np.mean(brightness_img)
    contrast_img = np.clip((brightness_img - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)

    # Canny edge detection
    edges = cv.Canny(contrast_img, 70, 170)
    return edges


# === Region of Interest Masking ===
def region_of_interest(image):
    """
    Masks the input image to keep only the trapezoidal region where lanes usually appear.

    Args:
        image (ndarray): Input grayscale or binary image.

    Returns:
        masked_image (ndarray): Image masked to show only region of interest.
    """
    height = image.shape[0]
    width = image.shape[1]
    
    # Define ROI as a trapezoid
    bottom_left = (int(0.01 * width), height)
    top_left = (int(0.3 * width), int(0.5 * height))
    top_right = (int(0.7 * width), int(0.5 * height))
    bottom_right = (int(1 * width), height)

    # Define polygon and apply mask
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    mask = np.zeros_like(image)
    mask_color = 255
    cv.fillPoly(mask, vertices, mask_color)
    cv.imshow('Mask', mask)
    masked_image = cv.bitwise_and(image, mask)
    return masked_image


# === Line Detection using Hough Transform ===
def detect_lines(image):
    """
    Detects lines using the Probabilistic Hough Transform.

    Args:
        image (ndarray): Edge-detected image.

    Returns:
        lines (list): List of detected lines [[x1, y1, x2, y2], ...].
    """
    rho = 1
    theta = np.pi / 180
    threshold = 50
    min_line_length = 60
    max_line_gap = 100

    lines = cv.HoughLinesP(
        image,
        rho,
        theta,
        threshold,
        np.array([]),
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    return lines if lines is not None else []


# === Fit Lane Lines Using Linear Regression ===
def fit_lane_lines(lines, image_height):
    """
    Fits straight lane lines from detected segments using linear regression.

    Args:
        lines (list): Detected lines from Hough Transform.
        image_height (int): Height of the image for y-range of lanes.

    Returns:
        left_line, right_line (list): Each line as [x1, y1, x2, y2] or None.
    """
    left_points = []
    right_points = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Avoid divide-by-zero
        if abs(slope) < 0.3:
            continue  # Ignore nearly horizontal lines
        if slope < 0:
            left_points += [(x1, y1), (x2, y2)]
        else:
            right_points += [(x1, y1), (x2, y2)]

    def make_line(points):
        if len(points) < 2:
            return None
        x_coords, y_coords = zip(*points)
        poly = np.polyfit(y_coords, x_coords, 1)  # x = m*y + b
        y1 = image_height
        y2 = int(image_height * 0.6)
        x1 = int(np.polyval(poly, y1))
        x2 = int(np.polyval(poly, y2))
        return [x1, y1, x2, y2]

    left_line = make_line(left_points)
    right_line = make_line(right_points)

    return left_line, right_line


# === Draw Lines on Image ===
def draw_lines(image, lines, color=(0, 255, 0), thickness=4):
    """
    Draws lines on a copy of the input image.

    Args:
        image (ndarray): Original image.
        lines (list): List of lines as [x1, y1, x2, y2].
        color (tuple): BGR color of the lines.
        thickness (int): Thickness of the lines.

    Returns:
        image_with_lines (ndarray): Image with lines drawn.
    """
    image_with_lines = image.copy()
    
    if lines is not None:
        for line in lines:
            if line is not None and len(line) == 4:
                x1, y1, x2, y2 = line
                cv.line(image_with_lines, (x1, y1), (x2, y2), color, thickness)

    return image_with_lines


# === Obstacle Detection using HSV Filtering and Contours ===
def detect_obstacles(frame):
    """
    Detects obstacles in a video frame using color-based HSV filtering and contour analysis.

    Args:
        frame (ndarray): Input BGR image.

    Modifies:
        frame (ndarray): Draws rectangles around detected obstacles.
        
    Returns:
        obstacle_boxes (list): List of bounding boxes around detected obstacles.
    """
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Define color range for obstacle detection
    lower_color = np.array([0, 45, 50])
    upper_color = np.array([15, 255, 255])
    mask = cv.inRange(hsv, lower_color, upper_color)
    cv.imshow('Obstacle Mask', mask)

    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    obstacle_boxes = []
    for contour in contours:
        if cv.contourArea(contour) > 150:  # Area threshold for filtering noise
            x, y, w, h = cv.boundingRect(contour)
            obstacle_boxes.append((x, y, w, h))
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv.putText(frame, "Obstacle", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
    return obstacle_boxes


# === Determine Vehicle Direction from Lane Lines ===
def determine_direction(left_line, right_line, frame):
    """
    Estimates vehicle direction (left, right, straight) based on lane position.

    Args:
        left_line (list): Coordinates of the left lane line.
        right_line (list): Coordinates of the right lane line.
        frame (ndarray): Image on which direction is drawn.

    Modifies:
        frame (ndarray): Draws the direction text on the image.
    """
    direction = "Undetermined"

    if left_line is not None and right_line is not None:
        left_x = left_line[0]
        right_x = right_line[0]
        lane_center = (left_x + right_x) // 2
        frame_center = frame.shape[1] // 2
        offset = lane_center - frame_center

        threshold = 30

        if offset < -threshold:
            direction = "Turn Left"
        elif offset > threshold:
            direction = "Turn Right"
        else:
            direction = "Go Straight"

    elif left_line is not None:
        direction = "Turn Left (Only Left Lane Visible)"
    elif right_line is not None:
        direction = "Turn Right (Only Right Lane Visible)"
    else:
        direction = "No Lanes Detected"

    cv.putText(frame, f"Direction: {direction}", (30, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)


# === Determine if the Vehicle should Stop or Move ===
def determine_vehicle_action(left_fit, right_fit, obstacles):
    """
    Determines whether the vehicle should stop or move.
    Conditions:
      - STOP if no lanes are detected
      - STOP if obstacle(s) detected ahead
      - MOVE otherwise
    """
    lanes_detected = left_fit is not None or right_fit is not None
    if not lanes_detected:
        return "STOP, No Lanes"
    if len(obstacles) > 100:
        return "STOP, Obstacle Ahead"
    return "MOVE"



# === Main Function to Process Video ===
def main():
    folder_path = r"E:\6th Semester\DIP\Project\DIP Project Videos"
    for filename in os.listdir(folder_path):
        if filename.endswith('.mp4'):
            file_path = os.path.join(folder_path, filename)
            print(f"🎥 Reading {filename}...")
            for frame in read_video(file_path):
                edges = canny_edge_detector(frame)
                roi = region_of_interest(edges)
                lines = detect_lines(roi)
                
                left_line, right_line = fit_lane_lines(lines, frame.shape[0])
                lines = [left_line, right_line]
                output_img = draw_lines(frame, lines)
                
                obstacles = detect_obstacles(output_img)
                determine_direction(left_line, right_line, output_img)
                
                action = determine_vehicle_action(left_line, right_line, obstacles)
                color = (0, 0, 255) if (action == "STOP, No Lanes" or action == "STOP, Obstacle Ahead") else (0, 255, 0)
                cv.putText(output_img, f"Action: {action}", (30, 100),
                           cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                cv.imshow("Lanes", output_img)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    cv.destroyAllWindows()
                    break
                
                
main()


