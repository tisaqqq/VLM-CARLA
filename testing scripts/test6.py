import carla
import cv2
import numpy as np
import os
import time
import random
import matplotlib.pyplot as plt

# Connect to the client and retrieve the world object
client = carla.Client('localhost', 2000)
world = client.get_world()

client.load_world('Town05')

blueprint_library = world.get_blueprint_library()

# Retrieve the spectator object
spectator = world.get_spectator()
transform = spectator.get_transform()
location = transform.location
rotation = transform.rotation
spectator.set_transform(carla.Transform())



# Add NPCs
# Get the blueprint library and filter for the vehicle blueprints
vehicle_blueprints = world.get_blueprint_library().filter('*vehicle*')
# Get the map's spawn points
spawn_points = world.get_map().get_spawn_points()

# Spawn 50 vehicles randomly distributed throughout the map
for i in range(0, 50):
    world.try_spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))

ego_vehicle = world.try_spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))

# Add sensors
# Create a transform to place the camera on top of the vehicle
camera_init_trans = carla.Transform(carla.Location(z=20), carla.Rotation(pitch=-90))

# We create the camera through a blueprint that defines its properties
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')

# We spawn the camera and attach it to our ego vehicle
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)

# save data to out folderr as a series o PNG image files
camera.listen(lambda image: image.save_to_disk('out1/%06d.png' % image.frame))


#set the vehicle in motion with traffic manager
for vehicle in world.get_actors().filter('*vehicle*'):
    vehicle.set_autopilot(True)
    vehicle.set_simulate_physics(True)


#add walkers
walker_blueprints = blueprint_library.filter('walker.pedestrian.*')
walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
walker_list = []

for _ in range(10):  # Adjust the number of pedestrians as needed
    spawn_point = random.choice(spawn_points)
    walker_bp = random.choice(walker_blueprints)
    
    # Spawn the walker
    walker = world.try_spawn_actor(walker_bp, spawn_point)
    if walker is not None:
        # Spawn the controller and attach it to the walker
        walker_controller = world.spawn_actor(walker_controller_bp, carla.Transform(), attach_to=walker)
        
        # Set the walker to autopilot mode
        walker_controller.start()
        
        # Add walker and controller to list for cleanup
        walker_list.append((walker, walker_controller))


try:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        raise KeyboardInterrupt
    else:
        while True:
        # Record the ego vehicle's speed
            if ego_vehicle is not None:
                transform = ego_vehicle.get_transform()
                spectator.set_transform(carla.Transform(transform.location + carla.Location(z=20), carla.Rotation(pitch=-90)))
            for walker,walker_controller in walker_list:
                walker_controller.set_max_speed(1 + random.random())  # Speed between 1 and 2 m/s
                
                # Add walker and controller to list for cleanup
                walker_list.append((walker, walker_controller))
            time.sleep(1)
except KeyboardInterrupt: 
    print("Shutting down.")
finally:
    # Clean up
    camera.stop()
    #video_writer.release()
    if ego_vehicle is not None:
        ego_vehicle.destroy()
    for actor in world.get_actors().filter('*vehicle*'):
        if actor is not None:
            actor.destroy()
    cv2.destroyAllWindows()