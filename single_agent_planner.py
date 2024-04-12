import heapq
import itertools


import heapq


def move(loc, dir):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1]


def move_joint_state(locs, dir):

    new_locs = []
    for i in range(len(locs)):
        new_locs.append((locs[i][0] + dir[i][0], locs[i][1] + dir[i][1]))
    
    return new_locs

def generate_motions_recursive(num_agents):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
    
    joint_state_motions = list(itertools.product(directions, repeat=num_agents))


    return joint_state_motions


def is_valid_motion(old_loc, new_loc):
    ##############################
    # Task 1.3/1.4: Check if a move from old_loc to new_loc is valid
    # Check if two agents are in the same location (vertex collision)
    #  
    # for loc in new_loc:
    #     if new_loc.count(loc) > 1:
    #         return False
    if len(set(new_loc)) != len(new_loc):
        return False

    # Check edge collision
    #  
    # edges = []
    # for i in range(len(new_loc)):
    #     edges.append((old_loc[i], new_loc[i]))
    # for edge in edges:
    #     if edge[-1] in edges:
    #         return False
    for i in range(len(new_loc)):
        for j in range(len(new_loc)):
            if i == j:
                continue
            if new_loc[i] == old_loc[j] and new_loc[j] == old_loc[i]:
                return False

    return True


def compute_heuristics(my_map, goal):
    # Use Dijkstra to build a shortest-path tree rooted at the goal location
    open_list = []
    closed_list = dict()
    root = {'loc': goal, 'cost': 0}
    heapq.heappush(open_list, (root['cost'], goal, root))
    closed_list[goal] = root
    while len(open_list) > 0:
        (cost, loc, curr) = heapq.heappop(open_list)
        for dir in range(5):
            child_loc = move(loc, dir)
            child_cost = cost + 1
            if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
               or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
               continue
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            child = {'loc': child_loc, 'cost': child_cost}
            if child_loc in closed_list:
                existing_node = closed_list[child_loc]
                if existing_node['cost'] > child_cost:
                    closed_list[child_loc] = child
                    # open_list.delete((existing_node['cost'], existing_node['loc'], existing_node))
                    heapq.heappush(open_list, (child_cost, child_loc, child))
            else:
                closed_list[child_loc] = child
                heapq.heappush(open_list, (child_cost, child_loc, child))

    # build the heuristics table
    h_values = dict()
    for loc, node in closed_list.items():
        h_values[loc] = node['cost']
    return h_values


def build_constraint_table(constraints, agent):
    tab = {}
    max_timestep=0
    for constraint in constraints:
        if(constraint['agent'] == agent):
            timestep = constraint['timestep']
            if timestep not in tab:
                tab[timestep] = []
            tab[timestep].append(constraint)
            if max_timestep < timestep:
                max_timestep = timestep 

    return tab, max_timestep
def get_location(path, time):
    if time < 0:
        return path[0]
    elif time < len(path):
        return path[time]
    else:
        return path[-1]  # wait at the goal location

def get_sum_of_cost(paths):
    rst = 0
    if paths is None:
        return -1
    for path in paths:
        rst += len(path) - 1
    return rst

def is_constrained(curr_loc, next_loc, next_time, constraint_table):
    if next_time in constraint_table:
        constraints = constraint_table[next_time]
        for constraint in constraints:
            locs = constraint['loc'] 
            if len(locs) == 1:
                if next_loc == locs[0]:
                    return True
            else:
                if {curr_loc, next_loc} == set(locs): 
                    return True
    return False


def get_path(goal_node):
    path = []
    curr = goal_node
    prune = True
    while curr is not None:
        if(len(path) == 0):
            path.append(curr['loc'])
        else:
            if(prune):
                if(path[0] != curr['loc']):
                    path.append(curr['loc'])
                    prune = False
            else:
                path.append(curr['loc'])
        curr = curr['parent']
    path.reverse()
    return path


def get_path2(locs, goal_node):
    # def get_path(goal_node):
    path = []
    curr = goal_node
    while curr is not None:
        path.append(curr["loc"])
        curr = curr["parent"]
    path.reverse()
    return path

def push_node(open_list, node):
    # print(node['g_val'] + node['h_val'], node['h_val'], node['loc'],"push")
    heapq.heappush(open_list, (int(node['g_val'] + node['h_val']), node['h_val'], node['loc'], node))

def pop_node(open_list):
    _, _, _, curr = heapq.heappop(open_list)
    return curr

def compare_nodes(n1, n2):
    """Return true is n1 is better than n2."""
    return n1['g_val'] + n1['h_val'] < n2['g_val'] + n2['h_val']

def in_map(map, loc):
    if loc[0] >= len(map) or loc[1] >= len(map[0]) or min(loc) < 0:
        return False
    else:
        return True

def all_in_map(map, locs):
    for loc in locs:
        if not in_map(map, loc):
            return False
    return True

def a_star(my_map, start_loc, goal_loc, h_values, agent, constraints):
    """ my_map      - binary obstacle map
        start_loc   - start position
        goal_loc    - goal position
        agent       - the agent that is being re-planned
        constraints - constraints defining where robot should or cannot go at each timestep
    """

    ##############################
    # Task 1.2: Extend the A* search to search in the space-time domain
    #           rather than space domain, only.

    open_list = []
    closed_list = dict()
    h_value = h_values[start_loc]
    constraint_table, max_timestep = build_constraint_table(constraints, agent)

    upper_t2 = 0
    for row in my_map:
        upper_t2 += len(row) - sum(row)
    upper_t1 = agent*upper_t2

    if max_timestep < upper_t2:
        upper_t2 = max_timestep

    root = {'loc': start_loc, 'timestep' : 0, 'g_val': 0, 'h_val': h_value, 'parent': None}

    push_node(open_list, root)
    closed_list[(root['loc'], root['timestep'])] = root
    while len(open_list) > 0:
        curr = pop_node(open_list)
        if curr['loc'] == goal_loc:
            if upper_t2 < max_timestep:
                constrat = False
                for c_t in range(curr['timestep'], max_timestep):
                    if is_constrained(curr['loc'], curr['loc'], c_t, constraint_table):
                        constrat = True
                
                if not constrat:
                    return get_path(curr)
            else:
                if curr['loc'] == goal_loc and curr['timestep'] >= upper_t2:
                    return get_path(curr)

        if curr['timestep'] > upper_t1 and upper_t1 != 0:
            continue
        for dir in range(5):
            child_loc = move(curr['loc'], dir)

            if not in_map(my_map, child_loc):
                continue
            
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            
            if is_constrained(curr['loc'], child_loc, curr['timestep'] + 1, constraint_table):
                continue
            
            
            child = {'loc': child_loc,
                        'timestep' : curr['timestep'] + 1,
                        'g_val': curr['g_val'] + 1,
                        'h_val': h_values[child_loc],
                        'parent': curr}

            if (child['loc'], child['timestep']) in closed_list:
                existing_node = closed_list[(child['loc'], child['timestep'])]
                if compare_nodes(child, existing_node):
                    closed_list[(child['loc'], child['timestep'])] = child
                    push_node(open_list, child)
            else:
                closed_list[(child['loc'], child['timestep'])] = child
                push_node(open_list, child)
        
    ##############################
    return None  # Failed to find solutions

def joint_state_a_star(my_map, starts, goals, h_values, num_agents):
    """ my_map      - binary obstacle map
        start_loc   - start positions
        goal_loc    - goal positions
        num_agent   - total number of agents in fleet
    """

    open_list = []
    closed_list = dict()
    earliest_goal_timestep = 0
    h_value = 0
     ##############################
    # Task 1.1: Iterate through starts and use list of h_values to calculate total h_value for root node
    #
    #  
    for i in range(num_agents):
        h_value += h_values[i][starts[i][0], starts[i][1]]
    
    root = {'loc': starts, 'g_val': 0, 'timestep' : 0, 'h_val': h_value, 'parent': None,"time_step":0 }
    push_node(open_list, root)
    temp=list(root['loc'])
    temp.append(root['time_step'])
    closed_list[tuple(root['loc'])] = root

     ##############################
    # Task 1.1:  Generate set of all possible motions in joint state space
    #
    #  
    directions = generate_motions_recursive(num_agents)
    while len(open_list) > 0:
        curr = pop_node(open_list)
        if curr['loc'] == goals:
            return get_path(curr)

        for dir in directions:
            
            ##############################
            # Task 1.1:  Update position of each agent
            #
            #  
            child_loc = move_joint_state(curr['loc'], dir)
            
            if not all_in_map(my_map, child_loc):
                continue
             ##############################
            # Task 1.1:  Check if any agent is in an obstacle
            #
            valid_move = True
            #  
            for loc in child_loc:
                if my_map[loc[0]][loc[1]]:
                    valid_move = False
                    
            
            if not valid_move:
                continue

             ##############################
            # Task 1.1:   check for collisions
            #
            #  
            if not is_valid_motion(curr['loc'],child_loc):
                continue
            
             ##############################
            # Task 1.1:  Calculate heuristic value
            #
            #  
            psuegod = 0
            var=0
            res=[]
            pppath=[]
            res = get_path2(child_loc,curr)
            for i in range(num_agents):
                pppath.append([res[j][i] for j in range(len(res))])
            # print(len(pppath),"pppath",pppath)
            for i in range(num_agents):
                pppath[i].append(child_loc[i])
            for i in range(len(pppath)):
                prune=0
                for j in range(len(pppath[i])):
                    if (pppath[i][len(pppath[i])-1-j] == goals[i]):
                        prune=prune+1
                    # print(pppath[i][len(pppath[i])-1-j],goals[i],prune)
                    if (pppath[i][len(pppath[i])-1-j] != goals[i]):
                        break
                # print(prune,"*****")
                pppath[i]=pppath[i][:len(pppath[i])-prune+1]
            psuegod=get_sum_of_cost(pppath)
            # intint(psuegod,"psuegod",curr['g_val'])
            h_value = 0
            for i in range(num_agents):
                h_value += h_values[i][child_loc[i]]
            # Create child node
            # print(psuegod,"psuegod",curr['g_val'],h_value)
            child = {'loc': child_loc,
                    'g_val': psuegod,
                    'h_val': h_value,
                    'parent': curr,
                    'time_step': curr['time_step'] + 1}
            temp=list(child['loc'])
            temp.append(child['time_step'])
            temp=tuple(temp)
            if tuple(child['loc']) in closed_list:
                existing_node = closed_list[tuple(child['loc'])]
                if compare_nodes(child, existing_node):
                    closed_list[tuple(child['loc'])]=child
                    push_node(open_list, child)
            else:
                closed_list[tuple(child['loc'])] = child
                push_node(open_list, child)

    return None  # Failed to find solutions