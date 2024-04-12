import time as timer
import heapq
import random
import numpy as np
import copy
from single_agent_planner import compute_heuristics, get_location, get_sum_of_cost
from single_agent_planner import a_star


def detect_first_collision_for_path_pair(path1, path2):
    ##############################
    # Task 2.1: Return the first collision that occurs between two robot paths (or None if there is no collision)
    #           There are two types of collisions: vertex collision and edge collision.
    #           A vertex collision occurs if both robots occupy the same location at the same timestep
    #           An edge collision occurs if the robots swap their location at the same timestep.
    #           You should use "get_location(path, t)" to get the location of a robot at time t.
    total_time=max(len(path1), len(path2))
    dectcols = dict()

    for i in range(total_time):
        lta = get_location(path1, i)
        ltb = get_location(path2, i)
        if (lta == ltb):
            dectcols['loc'] = [lta]
            dectcols['timestep'] = i
            return dectcols
        if (i+1) < total_time:
            loc_t1_a = get_location(path1, i+1)
            loc_t1_b = get_location(path2, i+1)
            if (loc_t1_a == ltb) and (loc_t1_b == lta):
                dectcols['loc'] = [lta, loc_t1_a]
                dectcols['timestep'] = i + 1
                return dectcols

    return None


def detect_collisions_among_all_paths(paths):
    ##############################
    # Task 2.1: Return a list of first collisions between all robot pairs.
    #           A collision can be represented as dictionary that contains the id of the two robots, the vertex or edge
    #           causing the collision, and the timestep at which the collision occurred.
    #           You should use your detect_collision function to find a collision between two robots.

    cols=[]
    for i in range(len(paths)):
        for j in np.arange(i+1,len(paths)):
            col = detect_first_collision_for_path_pair(paths[i],paths[j])
            if col is None:
                continue
            cols.append({'a1': i, 'a2': j, 'loc': col['loc'], 'timestep': col['timestep']})

    return cols


def standard_splitting(collision):
    ##############################
    # Task 2.2: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint prevents the first agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the second agent to be at the
    #                            specified location at the specified timestep.
    #           Edge collision: the first constraint prevents the first agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the second agent to traverse the
    #                          specified edge at the specified timestep

    cons=[]
    if len(collision['loc']) == 2:
        cons.append({'agent': collision['a1'], 'loc': collision['loc'], 'timestep': collision['timestep']})
        cons.append({'agent': collision['a2'], 'loc': collision['loc'][::-1], 'timestep': collision['timestep']})
    else:
        cons.append({'agent': collision['a1'], 'loc': collision['loc'], 'timestep': collision['timestep']})
        cons.append({'agent': collision['a2'], 'loc': collision['loc'], 'timestep': collision['timestep']})

    return cons



class CBSSolver(object):
    """The high-level search of CBS."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        # print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        # print("Expand node {}".format(id))
        self.num_of_expanded += 1
        return node

    def find_solution(self):
        """ Finds paths for all agents from their start locations to their goal locations

        """

        self.start_time = timer.time()

        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths
        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}
        for i in range(self.num_of_agents):  # Find initial path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions_among_all_paths(root['paths'])
        self.push_node(root)

        # Task 2.1: Testing
        print(root['collisions'])

        # Task 2.2: Testing
        for collision in root['collisions']:
            print(standard_splitting(collision))


        while len(self.open_list) > 0:
            curr_node = self.pop_node()
            self.num_of_expanded += 1

            curr_collisions = detect_collisions_among_all_paths(curr_node['paths'])

            if len(curr_collisions) == 0:
                self.print_results(curr_node)
                return curr_node['paths']
            
            if timer.time() - self.start_time > 100:
                print("!!!!!!!!!!!!!!!!!!1Timed out!!!!!!!!!!!!!!!!")
                self.print_results(curr_node)
                return curr_node['paths']

            constraints = standard_splitting(curr_collisions[0])

            for constraint in constraints:
                next_node = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}

                temp = copy.deepcopy(curr_node['constraints'])
                temp.append(constraint)
                next_node['constraints'] = temp

                temp = copy.deepcopy(curr_node['paths'])
                next_node['paths'] = temp

                a_i = constraint['agent']

                path_i = a_star(self.my_map, self.starts[a_i], self.goals[a_i], self.heuristics[a_i], a_i, next_node['constraints'])

                if path_i is not None:
                    next_node['paths'][a_i] = path_i
                    
                    next_node_collisions = detect_collisions_among_all_paths(next_node['paths'])
                    next_node['collisions'] = next_node_collisions

                    next_node['cost'] = get_sum_of_cost(next_node['paths'])
                    self.push_node(next_node)
                    self.num_of_generated += 1

        raise BaseException('No solutions found')

        ##############################
        # Task 2.3: High-Level Search
        #           Repeat the following as long as the open list is not empty:
        #             1. Get the next node from the open list (you can use self.pop_node()
        #             2. If this node has no collision, return solution
        #             3. Otherwise, choose the first collision and convert to a list of constraints (using your
        #                standard_splitting function). Add a new child node to your open list for each constraint
        #           Ensure to create a copy of any objects that your child nodes might inherit

        # These are just to print debug output - can be modified once you implement the high-level search
        self.print_results(root)
        return root['paths']


    def print_results(self, node):
        print("\n Found a solution! \n")
        CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))
