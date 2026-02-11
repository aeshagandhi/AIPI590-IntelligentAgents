import heapq


def is_applicable(state, action):
    """
    Check if action's preconditions are satisfied.
    Copy your implementation from Challenge 1!
    """
    precond = action['preconditions']
    return precond.issubset(state)


def apply_action(state, action):
    """
    Apply action to get new state.
    Copy your implementation from Challenge 1!
    """
    return action['add_list'].union(state.difference(action['delete_list']))


def goal_satisfied(state, goal):
    """
    Check if all goal facts are in state.
    Copy your implementation from Challenge 2!
    """
    return goal.issubset(state)
    pass


def get_applicable_actions(state, actions):
    """
    Get all actions applicable in state.
    Copy your implementation from Challenge 2!
    """
    moves = []
    for action in actions:
        if is_applicable(state, action):
            moves.append(action)
    return moves


# ============================================================
# Implement the goal-count heuristic
# ============================================================

def goal_count_heuristic(state, goal):
    """
    Estimate distance to goal by counting unsatisfied goal facts.
    
    Args:
        state: Current state (set of fluents)
        goal: Goal condition (set of fluents)
    
    Returns:
        int: Number of goal facts NOT in the current state
    
    Example:
        state = {"On(A,B)", "OnTable(B)"}
        goal = {"On(A,B)", "On(B,C)", "OnTable(C)"}
        # On(A,B) is satisfied, but On(B,C) and OnTable(C) are not
        # Heuristic value = 2
    """
    return len(goal - state)
        


# ============================================================
#  forward_search 
# ============================================================

def forward_search(initial_state, goal, actions):
    """
    BFS forward search - copy from Challenge 2 for comparison!
    """
    explored = 0
    queue = []
    
    visited = {frozenset(initial_state)}
    queue.append((initial_state, []))
    while len(queue) != 0:
        state, plan = queue.pop(0)
        explored += 1
        if goal_satisfied(state, goal):
            return (plan, explored)
        applicable_actions = get_applicable_actions(state, actions)
        for action in applicable_actions:
            new_state = apply_action(state, action)
            if frozenset(new_state) not in visited:
                visited.add(frozenset(new_state))
                queue.append((new_state, plan + [action['name']]))
    
    return None, explored  # No plan found


# ============================================================
# Implement the heuristic search algorithm
# ============================================================

def heuristic_search(initial_state, goal, actions):
    """
    Find a plan using A*-like search with goal-count heuristic.
    
    Args:
        initial_state: Starting state (set of fluents)
        goal: Goal condition (set of fluents)
        actions: List of all possible action dictionaries
    
    Returns:
        tuple: (plan, explored_count)
    
    Algorithm:
        1. Initialize priority queue with (f, counter, g, state, plan)
           where f = g + h, g = 0, h = heuristic(initial_state)
        2. Initialize visited set
        3. While queue not empty:
           a. Pop state with lowest f value
           b. Skip if already visited (we might add duplicates)
           c. Mark as visited, increment explored
           d. If goal satisfied: return plan
           e. For each applicable action:
              - Compute successor state
              - If not visited: compute f = g+1 + h(successor), add to queue
        4. Return None if no plan found
    
    Note: We use a counter as tie-breaker since sets aren't comparable.
    """
    explored = 0
    counter = 0  # Tie-breaker for priority queue
    h_initial = goal_count_heuristic(initial_state, goal) 
    pq = []
    visited = set()
    
    heapq.heappush(pq, (h_initial, counter, 0, initial_state, []))
    while len(pq) != 0:
        item = heapq.heappop(pq)
        g = item[2]
        state = item[3]
        plan = item[4]
        if frozenset(state) not in visited:
            # continue
            visited.add(frozenset(state))
            explored += 1
            if goal_satisfied(state, goal):
                return (plan, explored)
            for action in get_applicable_actions(state, actions):
                new_state = apply_action(state, action)
                if frozenset(new_state) not in visited:
                    h = goal_count_heuristic(new_state, goal)
                    new_f = (g + 1) + h
                    updated_plan = plan + [action['name']]

                    
                    heapq.heappush(pq, (new_f, counter, g+1, new_state, updated_plan))
                    counter += 1
    return None, explored  # No plan found


def make_library_actions(books, locations, connected_pairs):
    """
    Create all ground actions for:
    Move(l1,l2), PickUp(b,l), Deliver(b)
    """
    actions = []

    # define move actions
    for (l1, l2) in connected_pairs:
        actions.append({
            "name": f"Move({l1},{l2})",
            "preconditions": {f"AtR({l1})"},
            "add_list": {f"AtR({l2})"},
            "delete_list": {f"AtR({l1})"},
        })

    # define pickup actions
    for b in books:
        for l in locations:
            actions.append({
                "name": f"PickUp({b},{l})",
                "preconditions": {f"AtR({l})", f"AtB({b},{l})", "HandEmpty"},
                "add_list": {f"Holding({b})"},
                "delete_list": {f"AtB({b},{l})", "HandEmpty"},
            })

    # define deliver actions
    for b in books:
        actions.append({
            "name": f"Deliver({b})",
            "preconditions": {"AtR(Desk)", f"Holding({b})"},
            "add_list": {f"Delivered({b})", "HandEmpty"},
            "delete_list": {f"Holding({b})"},
        })

    return actions



if __name__ == "__main__":
    # run an example instance of the library retrieval robot
    
    books = ["B1", "B2"]
    locations = ["Desk", "StacksA", "StacksB", "Returns"]

    # stores locations that are connected to each other, not part of state itself
    connected_pairs = [
        ("Desk", "StacksA"), ("StacksA", "Desk"),
        ("Desk", "StacksB"), ("StacksB", "Desk"),
        ("Desk", "Returns"), ("Returns", "Desk"),
        ("StacksA", "StacksB"), ("StacksB", "StacksA"),
    ]

    actions = make_library_actions(books, locations, connected_pairs)

    # robot at desk not holding anything, with books at stacksA and returns locations
    # goal is to pick up books and deliver to desk
    initial_state = {
        "AtR(Desk)",
        "HandEmpty",
        "AtB(B1,StacksA)",
        "AtB(B2,Returns)",
    }

    goal = {
        "Delivered(B1)",
        "Delivered(B2)",
    }

    # run BFS
    bfs_plan, bfs_explored = forward_search(set(initial_state), set(goal), actions)

    print("BFS")
    print("Plan:", bfs_plan if bfs_plan is not None else "no plan found")
    print("Plan length:", len(bfs_plan) if bfs_plan is not None else 0)
    print("States explored:", bfs_explored)

    # run A*
    astar_plan, astar_explored = heuristic_search(set(initial_state), set(goal), actions)

    print("\n A* (goal-count heuristic)")
    print("Plan:", astar_plan if astar_plan is not None else "no plan found")
    print("Plan length:", len(astar_plan) if astar_plan is not None else 0)
    print("States explored:", astar_explored)
