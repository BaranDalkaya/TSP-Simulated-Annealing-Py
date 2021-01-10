import numpy as np
import numpy.random as rand
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import copy
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
map = mpimg.imread("map.png")

with open('capitals.json', 'r') as capitals_file:
    capitals = json.load(capitals_file)

capitals_list = list(capitals.items())
capitals_list = [(c[0], tuple(c[1])) for c in capitals_list]

def coord(path):
    """Strip the city name from the element of the path list and return
    a the xt coordinates  for the city. For example,
        "Atlanta", (585.6, 376.8) -> (585.6, 376.8)"""
    _, coord = path
    return coord

def coords(path):
    """Strip the city name from each element of the path list and return
    a list of tuples containing only pairs of xy coordinates for the
    cities. For example,
        [("Atlanta", (585.6, 376.8)), ...] -> [(585.6, 376.8), ...]
    """
    _, coords = zip(*path)
    return coords

def show_path(path_, starting_city, w=35, h=15):
    """Plot a TSP path overlaid on a map of the US States & their capitals."""
    path=coords(path_)
    x, y = list(zip(*path))

    _, (x0, y0) = starting_city

    plt.imshow(map)
    plt.plot(x0, y0, 'y*', markersize=15)  # y* = yellow star for starting point
    plt.plot(x + x[:1], y + y[:1])  # include the starting point at the end of path
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches([w, h])

def Temp(t, a=0.99, T_o=10**8):
    """Calculate the temperature of the system"""
    return (a**t) * T_o

def random_path(list_of_cities = capitals_list, a = 8):
    """Generate a random path from capitals list file. Chooses 8 cities by default."""
    return random.sample(list_of_cities, a)

def pair_wise_ex(path):
    """Swap the locations of 2 randomly chosen cities in our path list"""
    my_path = path.copy()
    index1 = rand.randint(0, 8)
    index2 = rand.randint(0, 8)
    city_swap1 = my_path[index1]
    city_swap2 = my_path[index2]
    my_path[index1] = city_swap2
    my_path[index2] = city_swap1

    return my_path

def total_distance(path):
    """Calculates the total distance of the path"""
    distance = 0
    tour_position = 0
    path_coords = coords(path)
    while tour_position < len(path_coords):
        if tour_position != len(path_coords) - 1:
            from_city = path_coords[tour_position]
            to_city = path_coords[tour_position + 1]
        else:
            from_city = path_coords[tour_position]
            to_city = path_coords[0]

        from_x, from_y = from_city
        to_x, to_y = to_city

        distance += ((to_x - from_x)**2 + (to_y - from_y)**2) ** 0.5
        tour_position += 1
    return distance


def sim_an(time, print_local = "false"):
    """Simulates annealing looping t number of times"""

    # Initialise our cities to visit, second parameter can be changed for different number of cities
    path = random_path(capitals_list, len(capitals_list))

    t = 0
    distances = []
    temperatures =[]
    best_path = 0

    # Loop algo until condition is met
    while t < time:

        # Calculate temperate with pre-defined alpha and initial temp
        temp = Temp(t, a = 0.85, T_o = 1000)

        newpath = pair_wise_ex(path)

        # Calculate total distance of old and new path
        d_old = total_distance(path)
        d_new = total_distance(newpath)

        # Calculate acceptance probability.
        Prob = 1 if d_old > d_new else np.exp( (d_old - d_new) / temp )

        # Check if we will accept the new path
        if Prob > rand.rand():
            path = newpath

        # Current best solution
        d_best = total_distance(path)
        path_best = path

        if print_local == "true":
            print("{0}: {1}".format(t+1, d_best))

        distances.append(d_best)
        temperatures.append(temp)
        t += 1

    return distances, temperatures, path_best


# Run simulated annealing function
dist, temps, path_best = sim_an(200, "true")


# Plot path on the map of USA and the distance as a function of temperature
plt.subplot(1, 2, 1)
show_path(path_best, path_best[0])

plt.subplot(1, 2, 2)
plt.plot(temps, dist)
plt.xscale("log")
plt.yscale("log")


# Simulate annealing process 20 times find a optimum result
count = 0
best_distances = []
best_paths = []

while count < 20:
    dist, temps, path_best = sim_an(200)
    best_distances.append(dist[-1])
    best_paths.append(path_best)
    count += 1

index_sol = best_distances.index(min(best_distances))
print("Best distance: {}".format(best_distances[index_sol]))
print("Path of solution:")
for city in best_paths[index_sol]:
    a, b = city
    print(a)
