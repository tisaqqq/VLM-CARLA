import carla
import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from collections import deque
import random

# Connect to CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Spawn a vehicle
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('model3')[0]  # Example vehicle: Tesla Model 3
spawn_point = world.get_map().get_spawn_points()[0]
spawn_point = random.choice(world.get_map().get_spawn_points())
vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
if vehicle is None:
    raise RuntimeError('Failed to spawn vehicle due to a collision. Try changing the spawn point.')

# Spawn pedestrians at random locations
pedestrian_blueprints = blueprint_library.filter('walker.pedestrian.*')
walker_controller_bp = blueprint_library.find('controller.ai.walker')
crossing_points = [location for location in world.get_map().get_spawn_points() if location.location.x < 50 and location.location.y < 50]  # Crossing points for pedestrians
pedestrians = []

for i in range(20):  # Spawn 20 pedestrians
    pedestrian_bp = random.choice(pedestrian_blueprints)
    spawn_point = random.choice(crossing_points)
    pedestrian = world.try_spawn_actor(pedestrian_bp, spawn_point)
    if pedestrian is not None:
        walker_controller = world.spawn_actor(walker_controller_bp, carla.Transform(), attach_to=pedestrian)
        walker_controller.start()
        walker_controller.go_to_location(random.choice(crossing_points).location)
        walker_controller.set_max_speed(1.5)  # Walking speed
        pedestrians.append((pedestrian, walker_controller))

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

# Define a drive loop with acceleration and deceleration behavior
def drive_loop(vehicle, duration=60):
    start_time = time.time()
    while time.time() - start_time < duration:
        elapsed_time = time.time() - start_time

        # Accelerate for the first 10 seconds
        if elapsed_time < 10:
            throttle = 0.5 * (elapsed_time / 10)
            brake = 0.0
        # Maintain speed for the next 20 seconds
        elif elapsed_time < 30:
            throttle = 0.5
            brake = 0.0
        # Decelerate for the next 10 seconds
        elif elapsed_time < 40:
            throttle = 0.5 * (1 - (elapsed_time - 30) / 10)
            brake = 0.0
        # Brake to slow down for the next 10 seconds
        elif elapsed_time < 50:
            throttle = 0.0
            brake = 0.3 * ((elapsed_time - 40) / 10)
        # Accelerate again for the last 10 seconds to complete the loop
        else:
            throttle = 0.5 * ((elapsed_time - 50) / 10)
            brake = 0.0

        # Apply control to the vehicle
        vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake))
        time.sleep(0.1)  # Control frequency of 10 Hz

# Run the drive loop
try:
    drive_loop(vehicle, duration=60)  # Drive for 60 seconds in a loop
finally:
    # Clean up
    camera.stop()
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))  # Stop the vehicle
    vehicle.destroy()
    camera.destroy()
    video_writer.release()

    # Destroy pedestrians
    for pedestrian, walker_controller in pedestrians:
        walker_controller.stop()
        pedestrian.destroy()
        walker_controller.destroy()

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
