import sys
from sklearn.cluster import KMeans
import numpy as np
from heapq import *


points = {
    'T': 50,
    'H': 70,
    '_': 100,
    'X': 120,
    '+': 150,
    '*': 200,
    '~': 800,
    '#': float('inf')
}


def read(file_name):
    file = open(file_name, "r")
    N, M, C, R = (int(x) for x in file.readline().split())
    customers = []
    for i in range(C):
        x, y, reward = (int(x) for x in file.readline().split())
        customers.append((x, y, reward))
    terrain = []
    for i in range(M):
        terrain.append([])
        row = file.readline()
        for c in row[:-1]:
            terrain[i].append(c)
    for i in range(M):
        for j in range(N):
            terrain[i][j] = points[terrain[i][j]]
    return N, M, C, R, customers, terrain


def write(file_name, total):
    file = open(file_name, "w")
    for el in total:
        row = ""
        row += str(el['x']) + " " + str(el['y']) + " "
        row += "".join(el['path']) + "\n"
        file.write(row)


def heuristic(a, b, terrain):
    # return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2
    #print(type(terrain[b[0]][b[1]]))
    #print(b)
    # print(np.array(terrain).shape)
    return terrain[b[1]][b[0]]


def astar(array, start, goal, terrain):
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    start = tuple(start)
    goal = tuple(goal)

    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal, terrain)}
    oheap = []

    heappush(oheap, (fscore[start], start))

    while oheap:

        current = heappop(oheap)[1]

        if current == goal:
            # print(gscore[current] + fscore[current])
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j

            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:
                    if array[neighbor[0]][neighbor[1]] == float('inf'):
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue

            tentative_g_score = gscore[tuple(current)] + heuristic(current, neighbor, terrain)

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal, terrain)
                heappush(oheap, (fscore[neighbor], neighbor))

    return False


def main():
    file_name = sys.argv[1]
    N, M, C, R, customers, terrain = read(file_name)

    customers_keras = []
    for i in range(len(customers)):
        customers_keras.append([customers[i][0], customers[i][1]])
    X = np.array(customers_keras)

    kmeans = KMeans(n_clusters=R, random_state=0).fit(X)
    centroids = np.clip(np.floor(kmeans.cluster_centers_).astype(int), 0, max(N, M))
    print(centroids)

    clusters = []
    for i in range(R):
        actual_centroid = [centroids[i][0], centroids[i][1]]
        clusters.append([])
        clusters[-1] = [actual_centroid]
    #print(clusters)

    temp = list(zip(customers_keras, kmeans.labels_))
    for (coords, label) in temp:
        ctrid = clusters[label][0]
        if terrain[ctrid[1]][ctrid[0]] != float('inf'):
            clusters[label].append(coords)
    #print(clusters)

    #print(terrain)
    # solution = astar(np.array(terrain), clusters[0][1], clusters[0][0], terrain)
    # print(solution)
    # solutions = []
    total = []
    for c in clusters:
        office = c[0]
        hqs = c[1:]
        for hq in hqs:
            #print(hq, office)
            sol = astar(np.transpose(np.array(terrain)), hq, office, terrain)
            # sol = sol[1:]
            if sol != False:
                sol.append(tuple(hq))

                # print(sol)

                sol_dir = []
                guard = False
                while len(sol) > 1:
                    start, end = sol[0], sol[1]
                    if terrain[end[1]][end[0]] == float('inf'):
                        guard = True
                        break

                    if (start[0] - end[0]) == -1:
                        sol_dir.append('R')
                    elif (end[0] - start[0]) == -1:
                        sol_dir.append('L')
                    elif (start[1] - end[1]) == -1:
                        sol_dir.append('D')
                    elif (end[1] - start[1]) == -1:
                        sol_dir.append('U')
                    sol = sol[1:]
                if guard:
                    break

                if len(sol_dir) > 0:
                    total.append({
                        'x': office[0],
                        'y': office[1],
                        'path': sol_dir
                    })

            # print(sol_dir)
            # solutions.append(sol_dir)
            #print(len(solutions), '\n')

    print(total)
    write(file_name.split(".")[0]+"_OUT.txt", total)
    #print(np.round(kmeans.cluster_centers_).astype(int))
    #print(kmeans.labels_)



if __name__ == "__main__":
    main()
