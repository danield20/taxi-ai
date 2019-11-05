import sys
import copy
from heapq import heappop, heappush

#* Initial configurations
height = 0
width = 0
fuel = 0
start_x = 0
start_y = 0
K = 0
clients = []
grid = []

#* Possible moves
N = 1
S = 2
E = 3
V = 4
P = 5
D = 6

#* State members state = (x, y, fule, client, rem_clients, grid)
x_idx = 0
y_idx = 1
fuel_idx = 2
client = 3
rem_clients = 4
venit = 5
grid_idx = 6

def make_move(state, move):
    if (move == N):
        state[x_idx] -= 1
        state[fuel_idx] -= 1
    elif (move == S):
        state[x_idx] += 1
        state[fuel_idx] -=1
    elif (move == E):
        state[y_idx] += 1
        state[fuel_idx] -=1
    elif (move == V):
        state[y_idx] -= 1
        state[fuel_idx] -=1
    elif (move == P):
        pick_up_client(state)
        state[rem_clients] -= 1
    elif (move == D):
        state[venit] += get_client_pay(state[client])
        state[client] = -1

def get_client_pay(client):
    return clients[client][4]

def pick_up_client(state):
    x = state[x_idx]
    y = state[y_idx]
    grid = state[grid_idx]

    state[client] = grid[x][y][0]
    grid[x][y][0] = -1

def get_pos_moves(state):
    x = state[x_idx]
    y = state[y_idx]
    grid = state[grid_idx]
    moves = copy.deepcopy(grid[x][y][1])

    if (state[client] != -1):
        _, _, dx, dy, _ = clients[state[client]]
        if x == dx and y == dy:
            moves.append(D)
    else:
        if grid[x][y][0] != -1:
            moves.append(P)

    return moves

def is_final(state):
    if state[rem_clients] == 0 and state[client] == -1:
        return True
    if state[fuel_idx] == 0:
        return True

    max_from_clients = sum([c for _, _, _, _, c in clients])

    if max(max_from_clients - state[fuel_idx] - state[venit], 0) == 0:
        return True

    return False

def h1(state):
    max_from_clients = sum([c for _, _, _, _, c in clients])

    return max(max_from_clients - state[fuel_idx] - state[venit], 0)

def h2(state):
    max_from_clients = sum([c for _, _, _, _, c in clients])

    return max_from_clients - state[venit]

def reconstruct_road(final_state, road):
    actions = []
    cur_tuple = tuple(final_state[:-1])

    while road[cur_tuple] != None:
        prev_state, prev_move = road[cur_tuple]
        actions.append(prev_move)
        cur_tuple = tuple(prev_state[:-1])

    actions.reverse()

    return actions

def breadth_first_search():
    s0 = [start_x, start_y, fuel, -1, len(clients), 0, copy.deepcopy(grid)]
    open = [s0]
    act = {}
    visited = {}
    act[tuple(s0[:-1])] = None

    while open != []:
        current = open.pop(0)
        visited[tuple(current[:2])] = current[2:-1]

        if is_final(current):
            return (current, act)
        else:
            possible_moves = get_pos_moves(current)
            for move in possible_moves:
                next_state = copy.deepcopy(current)
                make_move(next_state, move)

                if tuple(next_state[:2]) in visited:
                    f, c, r, v = visited[tuple(next_state[:2])]
                    if c == next_state[client] and r == next_state[rem_clients] \
                        and f >= next_state[fuel_idx] and v <= next_state[venit]:
                        continue

                act[tuple(next_state[:-1])] = (copy.deepcopy(current), move)
                open.append(next_state)

    return False

def uniform_cost_search():
    s0 = [start_x, start_y, fuel, -1, len(clients), 0, copy.deepcopy(grid)]
    open = []
    heappush(open, (0, s0)) # tuplu de cost real, nod
    act = {}
    visited = {}
    act[tuple(s0[:-1])] = None

    while open != []:
        (c, current) = heappop(open)

        visited[tuple(current[:2])] = current[2:-1]

        if is_final(current):
            return (current, act)
        else:
            possible_moves = get_pos_moves(current)
            for move in possible_moves:
                next_state = copy.deepcopy(current)
                make_move(next_state, move)

                if tuple(next_state[:2]) in visited:
                    f, c, r, v = visited[tuple(next_state[:2])]
                    if c == next_state[client] and r == next_state[rem_clients] \
                        and f >= next_state[fuel_idx] and v <= next_state[venit]:
                        continue

                act[tuple(next_state[:-1])] = (copy.deepcopy(current), move)

                if move != D or move != P:
                    heappush(open, (c + 1, next_state))
                else:
                    heappush(open, (c, next_state))

    return False

def depth_first_search():
    s0 = [start_x, start_y, fuel, -1, len(clients), 0, copy.deepcopy(grid)]
    open = [s0]
    act = {}
    visited = {}
    act[tuple(s0[:-1])] = None

    while open != []:
        current = open.pop(0)
        visited[tuple(current[:2])] = current[2:-1]

        if is_final(current):
            return (current, act)
        else:
            possible_moves = get_pos_moves(current)
            for move in possible_moves:
                next_state = copy.deepcopy(current)
                make_move(next_state, move)

                if tuple(next_state[:2]) in visited:
                    f, c, r, v = visited[tuple(next_state[:2])]
                    if c == next_state[client] and r == next_state[rem_clients] \
                        and f >= next_state[fuel_idx] and v <= next_state[venit]:
                        continue

                act[tuple(next_state[:-1])] = (copy.deepcopy(current), move)
                open.insert(0, next_state)

    return False

def depth_limited_search(max_depth):
    s0 = [start_x, start_y, fuel, -1, len(clients), 0, copy.deepcopy(grid)]
    open = [s0]
    depth = [0]
    act = {}
    visited = {}
    act[tuple(s0[:-1])] = None

    while open != []:
        current = open.pop(0)
        current_depth = depth.pop(0)
        visited[tuple(current[:2])] = current[2:-1]

        if (current_depth == max_depth):
            continue

        if is_final(current):
            return (current, act)
        else:
            possible_moves = get_pos_moves(current)
            for move in possible_moves:
                next_state = copy.deepcopy(current)
                make_move(next_state, move)

                if tuple(next_state[:2]) in visited:
                    f, c, r, v = visited[tuple(next_state[:2])]
                    if c == next_state[client] and r == next_state[rem_clients] \
                        and f >= next_state[fuel_idx] and v <= next_state[venit]:
                        continue

                act[tuple(next_state[:-1])] = (copy.deepcopy(current), move)
                open.insert(0, next_state)
                depth.insert(0, current_depth + 1)

    return False

def iterative_deepening_search():
    depth = 0
    while True:
        if depth_limited_search(depth) != False:
            return (depth_limited_search(depth), depth)
        depth += 1

def greedy_best_first_search(e):
    s0 = [start_x, start_y, fuel, -1, len(clients), 0, copy.deepcopy(grid)]
    open = []
    heappush(open, (e(s0), s0)) # tuplu de cost euristic, nod
    act = {}
    visited = {}
    act[tuple(s0[:-1])] = None

    while open != []:
        (c, current) = heappop(open)

        visited[tuple(current[:2])] = current[2:-1]

        if is_final(current):
            return (current, act)
        else:
            possible_moves = get_pos_moves(current)
            for move in possible_moves:
                next_state = copy.deepcopy(current)
                make_move(next_state, move)

                if tuple(next_state[:2]) in visited:
                    f, c, r, v = visited[tuple(next_state[:2])]
                    if c == next_state[client] and r == next_state[rem_clients] \
                        and f >= next_state[fuel_idx] and v <= next_state[venit]:
                        continue

                act[tuple(next_state[:-1])] = (copy.deepcopy(current), move)

                heappush(open, (e(next_state), next_state))

def a_star(e):
    s0 = [start_x, start_y, fuel, -1, len(clients), 0, copy.deepcopy(grid)]
    open = []
    heappush(open, (0 + e(s0), s0)) # tuplu de cost euristic, nod
    act = {}
    act[tuple(s0[:-1])] = None
    discovered = { tuple(s0[:-1]): (None, 0)}

    while open != []:
        (_, current) = heappop(open)
        g = discovered.get(tuple(current[:-1]))[1]

        if is_final(current):
            return (current, act)
        else:
            possible_moves = get_pos_moves(current)
            for move in possible_moves:
                next_state = copy.deepcopy(current)
                make_move(next_state, move)
                if (discovered.get(tuple(next_state[:-1])) == None):
                    heappush(open, (g + 1 + e(next_state), next_state))
                    discovered[tuple(next_state[:-1])] = (current, g + 1)
                    act[tuple(next_state[:-1])] = (copy.deepcopy(current), move)
                else:
                    prev_distance = discovered[tuple(next_state[:-1])][1]
                    if (g + 1 < prev_distance):
                        heappush(open, (g + 1 + e(next_state), next_state))
                        discovered[tuple(next_state[-1])] = (current, g + 1)
                        act[tuple(next_state[:-1])] = (copy.deepcopy(current), move)

def hill_climbing_search(e):
    s0 = [start_x, start_y, fuel, -1, len(clients), 0, copy.deepcopy(grid)]
    current = s0
    current_cost = (0 - e(current), 0) # g(s) + h(s), g(s), h(s) = -e(s)
    act = {}
    act[tuple(s0[:-1])] = None
    done = False
    while done == False:
        maxim, g = current_cost
        next_state = None
        possible_moves = get_pos_moves(current)
        for move in possible_moves:
            next_s = copy.deepcopy(current)
            make_move(next_s, move)
            cost = g + 2 - e(next_s)
            if cost > maxim:
                next_maxim_cost = (cost, g+1)
                next_state = next_s
                best_move = move
        if next_state == None:
            done = True
        else:
            act[tuple(next_state[:-1])] = (copy.deepcopy(current), best_move)
            current = next_state
            current_cost = next_maxim_cost
    return(current, act)


def get_start_pos_clients():
    return [(x,y) for (x,y,_,_,_) in clients]

def get_end_pos_clients():
    return [(dx, dy) for (_,_,dx,dy,_) in clients]

def read_input(file):
    global height, width, fuel, start_x, start_y

    f = open(file, "r")
    lines = f.readlines()

    height, width, fuel = [int(i) for i in lines[0].split(" ")]
    start_x, start_y = [int(i) for i in lines[1].split(" ")]
    K = int(lines[2])

    for i in range(3, 3 + K):
        t = tuple([int(i) for i in lines[i].split(" ")])
        clients.append(t)

    base_idx = K + 4
    for i in range(height):
        current_line = lines[base_idx + i][:-1].split(" ")
        curr_line_cells = []

        for idx, elem in enumerate(current_line):
            next_elem = current_line[idx + 1]
            pos_moves = []
            curr_cell = []

            if i == 0:
                pos_moves.append(S)
            elif i == height - 1:
                pos_moves.append(N)
            else:
                pos_moves.extend([N, S])

            if next_elem == ':':
                pos_moves.append(E)

            if elem == ':':
                pos_moves.append(V)

            start_list = get_start_pos_clients()

            if (i, idx) in start_list:
                curr_cell.append(start_list.index((i,idx)))
            else:
                curr_cell.append(-1)

            curr_cell.append(pos_moves)

            curr_line_cells.append(curr_cell)

            if idx == len(current_line) - 2:
                break

        grid.append(curr_line_cells)

def print_solution(state, road):
    print("Venit: " + str(state[venit] + state[fuel_idx]))
    actions = reconstruct_road(state, road)
    final_road = "["
    for idx, i in enumerate(actions):
        if i == N:
            final_road += "N"
        if i == S:
            final_road += "S"
        if i == E:
            final_road += "E"
        if i == V:
            final_road += "V"
        if i == P:
            final_road += "P"
        if i == D:
            final_road += "D"
        if idx != len(actions) - 1:
            final_road += ", "
    final_road += "]"
    print("Road: " + final_road)

def main():
    if len(sys.argv) == 1:
        print("No input file given.")

    read_input(sys.argv[1])

    # # BFS optimized solution
    # print("BFS")
    # final_state_bfs, road = breadth_first_search()
    # print_solution(final_state_bfs, road)
    # print("")

    # # Uniform cost search solution
    # print("UCS")
    # final_state_ucs, road = uniform_cost_search()
    # print_solution(final_state_ucs, road)
    # print("")

    # # Depth first search
    # print("DFS")
    # final_state_dfs, road = depth_first_search()
    # print_solution(final_state_dfs, road)
    # print("")

    # # Depth limited search
    # print("DLS")
    # max_depth = 10
    # if depth_limited_search(max_depth) != False:
    #     final_state_dls, road = depth_limited_search(max_depth)
    #     print_solution(final_state_dls, road)
    #     print("")
    # else:
    #     print("Solution not found for depth limited search for given depth\n")

    # # Iterative deepening
    # print("ID")
    # (final_state_id, road), dept = iterative_deepening_search()
    # print_solution(final_state_id, road)
    # print("Found for depth " + str(dept))
    # print("")

    # Greedy bfs optimized solution
    print("GBFS")
    final_state_gbfs, road = greedy_best_first_search(h1)
    print_solution(final_state_gbfs, road)
    print("")

    # A*
    print("A*")
    final_state_a_star, road = a_star(h1)
    print_solution(final_state_a_star, road)
    print("")

    # Hill climb search
    print("HCS")
    final_state_hcs, road = hill_climbing_search(h1)
    print_solution(final_state_hcs, road)
    print("")

if __name__== "__main__":
    main()