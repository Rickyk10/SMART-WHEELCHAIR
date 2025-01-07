import cv2
import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop

# Define the room locations with their coordinates and names
rooms = {
    "Bedroom": {"coords": (350, 700), "bbox": [(300, 650), (400, 750)]},  # (start, end) points for the room
    "Living Room": {"coords": (200, 400), "bbox": [(150, 350), (250, 450)]},
    "Kitchen": {"coords": (600, 200), "bbox": [(550, 150), (650, 250)]},
    "Bathroom": {"coords": (700, 500), "bbox": [(650, 450), (750, 550)]}
}

def load_image(image_path):
    """Load the image and handle errors."""
    floor_plan = cv2.imread(image_path)
    if floor_plan is None or floor_plan.size == 0:
        raise FileNotFoundError("Error: Unable to load the image. Check the file path.")
    return floor_plan

def preprocess_image(floor_plan):
    """Convert the image into a binary map."""
    gray_image = cv2.cvtColor(floor_plan, cv2.COLOR_BGR2GRAY)
    _, binary_map = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
    binary_map = cv2.bitwise_not(binary_map)  # Invert: walls -> 255, free space -> 0
    return binary_map

def astar(binary_map, start, goal):
    """A* algorithm to find the shortest path."""
    rows, cols = binary_map.shape
    visited = np.zeros((rows, cols), dtype=bool)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Straight movements only
    queue = []
    heappush(queue, (0, start, []))  # (cost, current position, path)

    while queue:
        cost, current, path = heappop(queue)
        x, y = current

        if visited[x, y]:
            continue
        visited[x, y] = True

        path = path + [current]
        if current == goal:
            return path

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny] and binary_map[nx, ny] == 0:
                heappush(queue, (cost + 1, (nx, ny), path))

    return None  # No path found

def visualize_path(floor_plan, path):
    """Visualize the path on the floor plan."""
    path_image = floor_plan.copy()

    for (x, y) in path:
        cv2.circle(path_image, (y, x), 2, (0, 255, 0), -1)  # Draw the path

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(path_image, cv2.COLOR_BGR2RGB))
    plt.title("Path Simulation")
    plt.axis("off")
    plt.show()

def simulate_movement(binary_map, path):
    """Simulate movement along the path with faster updates."""
    plt.figure(figsize=(10, 10))
    plt.title("Wheelchair Movement Simulation")
    plt.imshow(binary_map, cmap="gray")

    # Plot the path in chunks for faster updates
    step = 5  # Number of points to plot at once
    for i in range(0, len(path), step):
        chunk = path[i:i+step]
        for pos in chunk:
            plt.scatter(pos[1], pos[0], c='red', s=10)
        plt.pause(0.0001)  # Further reduced pause for faster simulation

    plt.show()
    print("Simulation complete!")

def main():
    # Path to the uploaded image
    image_path = "C:/Users/saisa/OneDrive/Desktop/walls.jpg"

    try:
        # Load and preprocess image
        floor_plan = load_image(image_path)
        binary_map = preprocess_image(floor_plan)

        # User input to select destination room
        print("Rooms available: Bedroom, Living Room, Kitchen, Bathroom")
        destination_room = input("Enter the destination room: ").strip()

        if destination_room not in rooms:
            print("Invalid room name.")
            return

        # Define start and goal based on user input
        start = rooms["Bedroom"]["coords"]  # Starting from the Bedroom
        goal = rooms[destination_room]["coords"]  # Destination based on user input

        # Pathfinding using A*
        path = astar(binary_map, start, goal)
        if path is None:
            print("No valid path found. Check start and goal positions.")
            return

        # Visualize the result
        visualize_path(floor_plan, path)

        # Simulate the wheelchair's movement
        simulate_movement(binary_map, path)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
