import carla
import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from collections import deque

# Connect to CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Spawn a vehicle
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('model3')[0]  # Example vehicle: Tesla Model 3
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# Set up the front camera
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '800')
camera_bp.set_attribute('image_size_y', '600')
camera_bp.set_attribute('fov', '110')
camera_transform = carla.Transform(carla.Location(x=2.0, z=1.4))  # Position at the front
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# Create output directory
os.makedirs('output', exist_ok=True)

# Set up OpenCV video writer
video_writer = cv2.VideoWriter('output/front_camera_view.avi', 
                               cv2.VideoWriter_fourcc(*'XVID'), 
                               20, (800, 600))  # Frame size same as camera resolution

# Set up speed data for real-time graph
speed_data = deque(maxlen=300)  # Store speed data for 30 seconds at 10Hz

# Define a callback to save video frames
def save_video_frame(image):
    # Convert the image to a numpy array
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # BGRA format

    # Convert BGRA to BGR for OpenCV
    frame = array[:, :, :3].copy()

    # Get vehicle velocity
    velocity = vehicle.get_velocity()
    speed_kmh = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5
    speed_text = f"Speed: {speed_kmh:.2f} km/h"

    # Add speed to speed data
    speed_data.append(speed_kmh)

    # Put the speed text on the frame
    cv2.putText(frame, speed_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Write the frame to the video
    video_writer.write(frame)

# Start listening to the camera
camera.listen(lambda image: save_video_frame(image))

# Apply control to the vehicle to make it move
vehicle.apply_control(carla.VehicleControl(throttle=0.5))

# Run the simulation for a while to capture video
try:
    time.sleep(10)  # Record for 10 seconds
finally:
    # Clean up
    camera.stop()
    vehicle.apply_control(carla.VehicleControl(throttle=0.0))  # Stop the vehicle
    vehicle.destroy()
    camera.destroy()
    video_writer.release()

# Generate and save the final speed graph
plt.figure(figsize=(10, 4))
plt.plot(list(speed_data), color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Speed (km/h)')
plt.title('Vehicle Speed Over Time')
plt.grid(True)
plt.tight_layout()
plt.savefig('output/speed_graph.png')
plt.close()

print("Video saved as 'output/front_camera_view.avi'")
print("Speed graph saved as 'output/speed_graph.png'")
