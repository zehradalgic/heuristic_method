import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from heapq import heappop, heappush

# Grid boyutlarını ve engelleri içeren bir grid oluştur
grid_size = 50
grid = np.zeros((grid_size, grid_size))
np.random.seed(42)
num_obstacles = 400
for _ in range(num_obstacles):
    x = np.random.randint(0, grid_size)
    y = np.random.randint(0, grid_size)
    grid[x, y] = 1

start = (0, 0)  # Başlangıç noktası
goal = (grid_size - 1, grid_size - 1)  # Hedef noktası
grid[start] = 0
grid[goal] = 0

# A* algoritması için fonksiyonlar
def heuristic(a, b):
    """Manhattan mesafesi (heuristik)"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_search(grid, start, goal):
    """A* algoritması ile yol bulma"""
    rows, cols = grid.shape
    open_set = []
    heappush(open_set, (0 + heuristic(start, goal), 0, start))  # (f_score, g_score, node)

    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current_g, current = heappop(open_set)

        # Hedefe ulaşıldığında yolu geri izleyerek döndür
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # Ters çevirip geri döndür

        # Komşuları kontrol et
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4 yönlü hareket
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:  # Grid sınırları içinde mi?
                if grid[neighbor] == 1:  # Engel mi?
                    continue
                tentative_g_score = current_g + 1  # Her hareketin maliyeti 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], tentative_g_score, neighbor))

    return None  # Yol bulunamadı

# A* algoritmasını çalıştır ve yolu bul
path = a_star_search(grid, start, goal)

# Hareketli simülasyon için animasyon
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(grid, cmap="gray_r")  # Gridin siyah-beyaz görselleştirmesi
ax.scatter(start[1], start[0], color="green", label="Start")  # Başlangıç
ax.scatter(goal[1], goal[0], color="red", label="Goal")       # Hedef
path_line, = ax.plot([], [], color="blue", linewidth=2, label="Path")  # Yolu çizecek çizgi
ax.legend()

def update(frame):
    """Animasyon için her adımda yolu güncelle"""
    if frame < len(path):
        current_path = path[:frame + 1]
        path_x, path_y = zip(*current_path)
        path_line.set_data(path_y, path_x)  # Çizgiyi güncelle
    return path_line,

ani = animation.FuncAnimation(fig, update, frames=len(path), interval=100, blit=True)

plt.title("A* Pathfinding Animation")
plt.show()