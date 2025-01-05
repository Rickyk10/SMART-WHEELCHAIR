import cv2
import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop

def load_image(image_path):
    """Load the image and handle errors."""
    floor_plan = cv2.imread('/Users/ricky/Downloads/walls.jpg')
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
    """A* algorithm to find the shortest path with straight-line preference."""
    rows, cols = binary_map.shape
    visited = np.zeros((rows, cols), dtype=bool)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Straight movements only
    diagonal_directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    queue = []
    heappush(queue, (0, start, [], None))  # (cost, current position, path, previous direction)

    while queue:
        cost, current, path, prev_direction = heappop(queue)
        x, y = current

        if visited[x, y]:
            continue
        visited[x, y] = True

        path = path + [current]
        if current == goal:
            return path

        for dx, dy in directions + diagonal_directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny] and binary_map[nx, ny] == 0:
                # Add a penalty for changes in direction
                direction = (dx, dy)
                direction_penalty = 0 if prev_direction == direction else 1

                # Avoid sticking too close to walls
                wall_penalty = np.sum(binary_map[max(0, nx-1):min(rows, nx+2), max(0, ny-1):min(cols, ny+2)] == 255)

                new_cost = cost + 1 + direction_penalty * 0.5 + wall_penalty * 0.1
                heappush(queue, (new_cost, (nx, ny), path, direction))

    return None  # No path found

def visualize_path(floor_plan, binary_map, path):
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
    """Simulate movement along the path."""
    plt.figure(figsize=(10, 10))
    plt.title("Wheelchair Movement Simulation")
    plt.imshow(binary_map, cmap="gray")

    for pos in path:
        plt.scatter(pos[1], pos[0], c='red', s=10)
        plt.pause(0.001)  # Way faster movement simulation

    plt.show()
    print("Simulation complete!")

# Main function
def main():
    # Path to the uploaded image
    image_path = "/mnt/data/walls.jpg"

    try:
        # Load and preprocess image
        floor_plan = load_image(image_path)
        binary_map = preprocess_image(floor_plan)

        # Define valid start and goal coordinates (row, col)
        start = (500, 150)  # Approximate position of Utility Closet
        goal = (350, 700)   # Approximate position of Bedroom 2

        # Pathfinding using A*
        path = astar(binary_map, start, goal)
        if path is None:
            print("No valid path found. Check start and goal positions.")
            return

        # Visualize the result
        visualize_path(floor_plan, binary_map, path)

        # Simulate the wheelchair's movement
        simulate_movement(binary_map, path)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
