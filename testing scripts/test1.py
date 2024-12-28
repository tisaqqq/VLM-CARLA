import carla
import cv2
import numpy as np
import os
import time
import random
import matplotlib.pyplot as plt

# Connect to the client and retrieve the world object
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Retrieve the spectator object
spectator = world.get_spectator()

# Get the location and rotation of the spectator through its transform
transform = spectator.get_transform()
location = transform.location
rotation = transform.rotation

# Set the spectator with an empty transform
spectator.set_transform(carla.Transform())

# Add NPCs
# Get the blueprint library and filter for the vehicle blueprints
vehicle_blueprints = world.get_blueprint_library().filter('*vehicle*')
# Get the map's spawn points
spawn_points = world.get_map().get_spawn_points()

# Spawn 50 vehicles randomly distributed throughout the map
for i in range(0, 50):
    world.try_spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))

# Spawn ego vehicle
ego_vehicle = world.try_spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))

# Add sensors
# Create a transform to place the camera on top of the vehicle
camera_init_trans = carla.Transform(carla.Location(x=1.5, z=2.4))

# We create the camera through a blueprint that defines its properties
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '800')
camera_bp.set_attribute('image_size_y', '600')
camera_bp.set_attribute('fov', '90')

# We spawn the camera and attach it to our ego vehicle
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)

# Set up output directory and video writer
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

video_path = os.path.join(output_dir, "ego_vehicle_camera.avi")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (800, 600))

# Speed data collection
speed_data = []
time_data = []
start_time = time.time()

# Function to process and display the camera image using OpenCV
def process_img(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # BGRA format
    img_bgr = array[:, :, :3]  # Convert BGRA to BGR
    cv2.imshow("Ego Vehicle Camera", img_bgr)

    # Write the frame to the video file
    video_writer.write(img_bgr)

    # End the process when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        raise KeyboardInterrupt

# Start camera with callback to process the image
camera.listen(lambda image: process_img(image))

# Enable autopilot for all vehicles
for vehicle in world.get_actors().filter('*vehicle*'):
    vehicle.set_autopilot(True)

try:
    while True:
        # Record the ego vehicle's speed
        if ego_vehicle is not None:
            velocity = ego_vehicle.get_velocity()
            speed = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5 * 3.6  # Convert m/s to km/h
            current_time = time.time() - start_time
            speed_data.append(speed)
            time_data.append(current_time)

            # Update the spectator to follow the ego vehicle
            transform = ego_vehicle.get_transform()
            spectator.set_transform(carla.Transform(transform.location + carla.Location(z=20), carla.Rotation(pitch=-90)))
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down.")
finally:
    # Clean up
    camera.stop()
    video_writer.release()
    if ego_vehicle is not None:
        ego_vehicle.destroy()
    for actor in world.get_actors().filter('*vehicle*'):
        if actor is not None:
            actor.destroy()
    cv2.destroyAllWindows()

    # Plot the speed graph
    plt.figure()
    plt.plot(time_data, speed_data, label='Speed (km/h)', color='b')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (km/h)')
    plt.title('Ego Vehicle Speed Over Time')
    plt.legend()
    plt.grid(True)

    # Save the graph to the output directory
    graph_path = os.path.join(output_dir, "ego_vehicle_speed.png")
    plt.savefig(graph_path)
    plt.show()
