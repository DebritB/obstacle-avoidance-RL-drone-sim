import pybullet as p
import pybullet_data
import time
import numpy as np
import matplotlib.pyplot as plt
import os

class ContinuousActionDroneEnv1:
    def __init__(self, use_gui=True):
        # Initialize PyBullet with GUI or DIRECT mode
        if use_gui:
            p.connect(p.GUI)
            print("PyBullet GUI connected.")
        else:
            p.connect(p.DIRECT)
            print("PyBullet in DIRECT mode.")

        # Define goal position and initial parameters
        self.goal_position = [14, 0, 1]  # Target position for the drone
        self.target_altitude = 1  # Target altitude for the drone

        # Load environment
        self.init_environment()

        # Set maximum force for movement and proportional gain for altitude
        self.max_force = 50000       # Max force for continuous actions
        self.kp_altitude = 6000      # Proportional gain for altitude control

        # Set high angular damping to minimize rotational effects
        p.changeDynamics(self.drone_id, -1, angularDamping=1)
        self.previous_x_position = 0

    def init_environment(self):
        """Initialize PyBullet and set up the drone, obstacles, boundaries, and goal marker."""
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)  # Enable gravity

        # Load ground plane
        p.loadURDF("plane.urdf")

        # Load the drone at the starting position
        drone_start_pos = [0, 0, 1]
        drone_start_orientation = [0, 0, 0, 1]  # No rotation
        self.drone_id = p.loadURDF("cf2.urdf", drone_start_pos, drone_start_orientation)
        print(f"Drone loaded at position: {drone_start_pos}")

        # Create boundary walls
        self.create_physical_boundary()

        # Create obstacles
        self.create_obstacle1([6, 1.13, 1])
        self.create_obstacle2([6, -5.13, 1])

        # Create goal marker at the goal position
        self.create_visual_marker(self.goal_position, [1, 0, 0, 1], size=0.5)

    def reset(self):
        """Reset the environment and drone position."""
        # Reset the drone's position and orientation
        p.resetBasePositionAndOrientation(self.drone_id, [0, 0, 1], [0, 0, 0, 1])
        p.resetBaseVelocity(self.drone_id, [0, 0, 0], [0, 0, 0])        
        self.no_progress_steps=0
        self.n_steps=0
        self.previous_x_position = 0
        return self.get_observation()

    def get_observation(self):
        """Get the current position and distance to the goal as an observation."""
        pos, _ = p.getBasePositionAndOrientation(self.drone_id)
        
        # Calculate distance to goal
        distance_to_goal = np.linalg.norm(np.array(self.goal_position) - np.array(pos))

        # Observation array with position and distance to goal
        observation = np.array(pos + (distance_to_goal,))
        return observation

    def step(self, action, previous_distance_to_goal):
        """Apply differential scaling with more force near the +x-axis and less near the yz-plane."""
        # Normalize the action vector
        action = np.array(action)
        if np.linalg.norm(action) != 0:
            action = action / np.linalg.norm(action)  # Normalize to unit vector

        # Calculate the angle relative to the +x-axis and use cosine for smooth scaling
        angle_from_x = np.arccos(np.clip(action[0], -1.0, 1.0))  # Angle between action vector and +x direction
        # Adjust the force based on the angle
        if angle_from_x <= np.pi / 2:
            force = self.max_force * action  # Full force
        else:
            force = (2/3) * self.max_force * action

        # Apply the computed force vector
        p.applyExternalForce(self.drone_id, -1, force.tolist(), [0, 0, 0], p.WORLD_FRAME)

        # Altitude control: Adjust upward force to stabilize altitude if action does not provide enough lift
        pos, _ = p.getBasePositionAndOrientation(self.drone_id)
        altitude_error = self.target_altitude - pos[2]
        upward_force = self.kp_altitude * altitude_error
        p.applyExternalForce(self.drone_id, -1, [0, 0, upward_force], [0, 0, 0], p.WORLD_FRAME)

        # Reset orientation to prevent any unwanted tilting or rotation
        p.resetBasePositionAndOrientation(self.drone_id, pos, [0, 0, 0, 1])

        # Step the simulation
        p.stepSimulation()

        # Get new observation
        obs = self.get_observation()

        # Calculate reward and check if done
        reward, done = self.calculate_reward_and_done(obs, previous_distance_to_goal, self.previous_x_position)

        # Update `previous_distance_to_goal` with the current distance for the next step
        current_distance_to_goal = obs[-1]

        # Update previous_x_position with the current x-coordinate
        self.previous_x_position = pos[0]

        return obs, reward, done, current_distance_to_goal


    def calculate_reward_and_done(self, obs, previous_distance_to_goal, previous_x_position=None):

        distance_to_goal = obs[-1]
        current_x_position, _ = p.getBasePositionAndOrientation(self.drone_id)

        current_position, _ = p.getBasePositionAndOrientation(self.drone_id)
        
        # 1. Distance-Based Reward
        if current_x_position[0] > 6.001:
            if (previous_distance_to_goal - distance_to_goal)>0:
                distance_reward = (previous_distance_to_goal - distance_to_goal) * 500000
            else:
                distance_reward = (previous_distance_to_goal - distance_to_goal) * 100
        else:
           distance_reward = 0
        # 2. Goal Achievement Reward
        goal_reward = 0
        done = False
        if distance_to_goal < 1.0:
            goal_reward = 30000000000
            done = True

        # 3. Collision Penalty
        collision = len(p.getContactPoints(bodyA=self.drone_id)) > 0
        collision_penalty = -100 if collision and current_x_position[0] > 6.001 else -50 if collision else 0
        if collision:
            done = True

        # 4. Step Penalty
        step_penalty = -1 if current_x_position[0] <= 6.001 else -10

        # 5. X-Movement Reward (add reward for moving in +x direction)
        x_movement_reward = 0
        if current_x_position[0] <= 6.001:
            if previous_x_position is not None and current_x_position[0] > previous_x_position:
                x_movement_reward = 2
        
        directional_reward = 0
        if current_x_position[0] > 6.001:  # Only apply after crossing the wall
            # Calculate direction to the goal
            direction_to_goal = np.array(self.goal_position) - np.array(current_position)
            direction_to_goal = direction_to_goal / np.linalg.norm(direction_to_goal)  # Normalize

            # Calculate movement direction based on position change
            movement_direction = np.array(current_position) - np.array([self.previous_x_position, 0, 0])
            if np.linalg.norm(movement_direction) != 0:
                movement_direction = movement_direction / np.linalg.norm(movement_direction)

                # Calculate alignment reward based on the dot product
                alignment = np.dot(movement_direction, direction_to_goal)
                scale_factor = 1 + (9 * (min(max(current_x_position[0], 6), 13.5) - 6) / (13.5 - 6))
                directional_reward = alignment * 100 * scale_factor

        # 6. Stuck Termination
        if self.no_progress_steps >= 3000:
            done = True

        reward = distance_reward + goal_reward + collision_penalty + step_penalty + x_movement_reward + directional_reward

        
        if current_x_position[0] > 6.001:
            reward += 500

        print(current_x_position[0])
        return reward, done




    def create_physical_boundary(self):
        """Create a physical 3D boundary box with six walls enclosing the area."""
        wall_thickness = 0.1
        wall_height = 3
        wall_length_x = 16
        wall_length_y = 10
        z_position = 1.5

        boundary_walls = [
            {"position": [-1, 0, z_position], "size": [wall_thickness, wall_length_y / 2, wall_height / 2]},
            {"position": [15, 0, z_position], "size": [wall_thickness, wall_length_y / 2, wall_height / 2]},
            {"position": [7, 5, z_position], "size": [wall_length_x / 2, wall_thickness, wall_height / 2]},
            {"position": [7, -5, z_position], "size": [wall_length_x / 2, wall_thickness, wall_height / 2]},
            {"position": [7, 0, wall_height], "size": [wall_length_x / 2, wall_length_y / 2, wall_thickness]},
            {"position": [7, 0, 0], "size": [wall_length_x / 2, wall_length_y / 2, wall_thickness]},
        ]

        for wall in boundary_walls:
            position = wall["position"]
            size = wall["size"]
            wall_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
            wall_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=[0, 0, 1, 0.3])
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_shape,
                              baseVisualShapeIndex=wall_visual,
                              basePosition=position)
            print(f"Boundary wall created at {position} with size {size}")

    def create_obstacle1(self, position):
            """Create a semi-transparent obstacle (wall) in the drone's path."""
            wall_orientation = p.getQuaternionFromEuler([0, 0, np.pi / 2])
            
            obstacle_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[4, 0.0001, 3])
            obstacle_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[4, 0.0001, 3], rgbaColor=[0.8, 0.2, 0.2, 0.7])
            
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=obstacle_shape,
                            baseVisualShapeIndex=obstacle_visual,
                            basePosition=position, baseOrientation=wall_orientation)
            print(f"Obstacle created at {position}")

    def create_obstacle2(self, position):
        """Create a semi-transparent obstacle (wall) in the drone's path."""
        wall_orientation = p.getQuaternionFromEuler([0, 0, np.pi / 2])
        
        obstacle_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.0001, 3])
        obstacle_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.0001, 3], rgbaColor=[0.8, 0.2, 0.2, 0.7])
        
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=obstacle_shape,
                          baseVisualShapeIndex=obstacle_visual,
                          basePosition=position, baseOrientation=wall_orientation)
        print(f"Obstacle created at {position}")

    def create_visual_marker(self, position, color, size):
        """Create a visual marker at the goal position."""
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=size, rgbaColor=color)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=position)
        print(f"Goal marker created at {position}")

class ContinuousActionDroneEnv2:
    def __init__(self, use_gui=True):
        # Initialize PyBullet with GUI or DIRECT mode
        if use_gui:
            p.connect(p.GUI)
            print("PyBullet GUI connected.")
        else:
            p.connect(p.DIRECT)
            print("PyBullet in DIRECT mode.")

        # Define goal position and initial parameters
        self.goal_position = [14, 0, 1]  # Target position for the drone
        self.target_altitude = 1  # Target altitude for the drone

        # Load environment
        self.init_environment()

        # Set maximum force for movement and proportional gain for altitude
        self.max_force = 50000       # Max force for continuous actions
        self.kp_altitude = 6000      # Proportional gain for altitude control

        # Set high angular damping to minimize rotational effects
        p.changeDynamics(self.drone_id, -1, angularDamping=1)
        self.previous_x_position = 0

    def init_environment(self):
        """Initialize PyBullet and set up the drone, obstacles, boundaries, and goal marker."""
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)  # Enable gravity

        # Load ground plane
        p.loadURDF("plane.urdf")

        # Load the drone at the starting position
        drone_start_pos = [0, 0, 1]
        drone_start_orientation = [0, 0, 0, 1]  # No rotation
        self.drone_id = p.loadURDF("cf2.urdf", drone_start_pos, drone_start_orientation)
        print(f"Drone loaded at position: {drone_start_pos}")

        # Create boundary walls
        self.create_physical_boundary()

        # Create obstacles
        #self.create_obstacle1([6, 6.13, 1])
        self.create_obstacle2([6, -1.13, 1])

        # Create goal marker at the goal position
        self.create_visual_marker(self.goal_position, [1, 0, 0, 1], size=0.5)

    def reset(self):
        """Reset the environment and drone position."""
        # Reset the drone's position and orientation
        p.resetBasePositionAndOrientation(self.drone_id, [0, 0, 1], [0, 0, 0, 1])
        p.resetBaseVelocity(self.drone_id, [0, 0, 0], [0, 0, 0])        
        self.no_progress_steps=0
        self.n_steps=0
        self.previous_x_position = 0
        return self.get_observation()

    def get_observation(self):
        """Get the current position and distance to the goal as an observation."""
        pos, _ = p.getBasePositionAndOrientation(self.drone_id)
        
        # Calculate distance to goal
        distance_to_goal = np.linalg.norm(np.array(self.goal_position) - np.array(pos))

        # Observation array with position and distance to goal
        observation = np.array(pos + (distance_to_goal,))
        return observation

    def step(self, action, previous_distance_to_goal):
        """Apply differential scaling with more force near the +x-axis and less near the yz-plane."""
        # Normalize the action vector
        action = np.array(action)
        if np.linalg.norm(action) != 0:
            action = action / np.linalg.norm(action)  # Normalize to unit vector

        # Calculate the angle relative to the +x-axis and use cosine for smooth scaling
        angle_from_x = np.arccos(np.clip(action[0], -1.0, 1.0))  # Angle between action vector and +x direction
        # Adjust the force based on the angle
        if angle_from_x <= np.pi / 2:
            force = self.max_force * action  # Full force
        else:
            force = (2/3) * self.max_force * action

        # Apply the computed force vector
        p.applyExternalForce(self.drone_id, -1, force.tolist(), [0, 0, 0], p.WORLD_FRAME)

        # Altitude control: Adjust upward force to stabilize altitude if action does not provide enough lift
        pos, _ = p.getBasePositionAndOrientation(self.drone_id)
        altitude_error = self.target_altitude - pos[2]
        upward_force = self.kp_altitude * altitude_error
        p.applyExternalForce(self.drone_id, -1, [0, 0, upward_force], [0, 0, 0], p.WORLD_FRAME)

        # Reset orientation to prevent any unwanted tilting or rotation
        p.resetBasePositionAndOrientation(self.drone_id, pos, [0, 0, 0, 1])

        # Step the simulation
        p.stepSimulation()

        # Get new observation
        obs = self.get_observation()

        # Calculate reward and check if done
        reward, done = self.calculate_reward_and_done(obs, previous_distance_to_goal, self.previous_x_position)

        # Update `previous_distance_to_goal` with the current distance for the next step
        current_distance_to_goal = obs[-1]

        # Update previous_x_position with the current x-coordinate
        self.previous_x_position = pos[0]

        return obs, reward, done, current_distance_to_goal


    def calculate_reward_and_done(self, obs, previous_distance_to_goal, previous_x_position=None):

        distance_to_goal = obs[-1]
        current_x_position, _ = p.getBasePositionAndOrientation(self.drone_id)

        current_position, _ = p.getBasePositionAndOrientation(self.drone_id)
        
        # 1. Distance-Based Reward
        if current_x_position[0] > 6.001:
            if (previous_distance_to_goal - distance_to_goal)>0:
                distance_reward = (previous_distance_to_goal - distance_to_goal) * 500000
            else:
                distance_reward = (previous_distance_to_goal - distance_to_goal) * 100
        else:
           distance_reward = 0
        # 2. Goal Achievement Reward
        goal_reward = 0
        done = False
        if distance_to_goal < 1.0:
            goal_reward = 30000000000
            done = True

        # 3. Collision Penalty
        collision = len(p.getContactPoints(bodyA=self.drone_id)) > 0
        collision_penalty = -100 if collision and current_x_position[0] > 6.001 else -50 if collision else 0
        if collision:
            done = True

        # 4. Step Penalty
        step_penalty = -1 if current_x_position[0] <= 6.001 else -10

        # 5. X-Movement Reward (add reward for moving in +x direction)
        x_movement_reward = 0
        if current_x_position[0] <= 6.001:
            if previous_x_position is not None and current_x_position[0] > previous_x_position:
                x_movement_reward = 2
        
        directional_reward = 0
        if current_x_position[0] > 6.001:  # Only apply after crossing the wall
            # Calculate direction to the goal
            direction_to_goal = np.array(self.goal_position) - np.array(current_position)
            direction_to_goal = direction_to_goal / np.linalg.norm(direction_to_goal)  # Normalize

            # Calculate movement direction based on position change
            movement_direction = np.array(current_position) - np.array([self.previous_x_position, 0, 0])
            if np.linalg.norm(movement_direction) != 0:
                movement_direction = movement_direction / np.linalg.norm(movement_direction)

                # Calculate alignment reward based on the dot product
                alignment = np.dot(movement_direction, direction_to_goal)
                scale_factor = 1 + (9 * (min(max(current_x_position[0], 6), 13.5) - 6) / (13.5 - 6))
                directional_reward = alignment * 100 * scale_factor

        # 6. Stuck Termination
        if self.no_progress_steps >= 3000:
            done = True

        reward = distance_reward + goal_reward + collision_penalty + step_penalty + x_movement_reward +directional_reward

        
        if current_x_position[0] > 6.001:
            reward += 500

        print(current_x_position[0])
        return reward, done




    def create_physical_boundary(self):
        """Create a physical 3D boundary box with six walls enclosing the area."""
        wall_thickness = 0.1
        wall_height = 3
        wall_length_x = 16
        wall_length_y = 10
        z_position = 1.5

        boundary_walls = [
            {"position": [-1, 0, z_position], "size": [wall_thickness, wall_length_y / 2, wall_height / 2]},
            {"position": [15, 0, z_position], "size": [wall_thickness, wall_length_y / 2, wall_height / 2]},
            {"position": [7, 5, z_position], "size": [wall_length_x / 2, wall_thickness, wall_height / 2]},
            {"position": [7, -5, z_position], "size": [wall_length_x / 2, wall_thickness, wall_height / 2]},
            {"position": [7, 0, wall_height], "size": [wall_length_x / 2, wall_length_y / 2, wall_thickness]},
            {"position": [7, 0, 0], "size": [wall_length_x / 2, wall_length_y / 2, wall_thickness]},
        ]

        for wall in boundary_walls:
            position = wall["position"]
            size = wall["size"]
            wall_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
            wall_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=[0, 0, 1, 0.3])
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_shape,
                              baseVisualShapeIndex=wall_visual,
                              basePosition=position)
            print(f"Boundary wall created at {position} with size {size}")

    def create_obstacle1(self, position):
            """Create a semi-transparent obstacle (wall) in the drone's path."""
            wall_orientation = p.getQuaternionFromEuler([0, 0, np.pi / 2])
            
            obstacle_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.0001, 3])
            obstacle_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.0001, 3], rgbaColor=[0.8, 0.2, 0.2, 0.7])
            
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=obstacle_shape,
                            baseVisualShapeIndex=obstacle_visual,
                            basePosition=position, baseOrientation=wall_orientation)
            print(f"Obstacle created at {position}")

    def create_obstacle2(self, position):
        """Create a semi-transparent obstacle (wall) in the drone's path."""
        wall_orientation = p.getQuaternionFromEuler([0, 0, np.pi / 2])
        
        obstacle_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[4, 0.0001, 3])
        obstacle_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[4, 0.0001, 3], rgbaColor=[0.8, 0.2, 0.2, 0.7])
        
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=obstacle_shape,
                          baseVisualShapeIndex=obstacle_visual,
                          basePosition=position, baseOrientation=wall_orientation)
        print(f"Obstacle created at {position}")

    def create_visual_marker(self, position, color, size):
        """Create a visual marker at the goal position."""
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=size, rgbaColor=color)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=position)
        print(f"Goal marker created at {position}")