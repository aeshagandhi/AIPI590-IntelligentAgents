import heapq

def is_applicable(state, action):
    """
    Check if action's preconditions are satisfied.
    Copy your implementation from Challenge 1!
    """
    # Return True if ALL preconditions are in state
    precond = action['preconditions']
    return precond.issubset(state)


def apply_action(state, action):
    """
    Apply action to get new state.
    Copy your implementation from Challenge 1!
    """
    # Return (state - delete_list) | add_list
    return action['add_list'].union(state.difference(action['delete_list']))


def goal_satisfied(state, goal):
    """
    Check if all goal facts are in state.
    Copy your implementation from Challenge 2!
    """
    # Return True if ALL goal facts are in state
    return goal.issubset(state)
    pass


def get_applicable_actions(state, actions):
    """
    Get all actions applicable in state.
    Copy your implementation from Challenge 2!
    """
    # Return list of actions whose preconditions are satisfied
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
    # Count how many goal facts are missing from state
    return len(goal - state)
        


# ============================================================
# BFS Forward Search
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
# Heuristic search algorithm
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
    
    # Calculate initial heuristic
    h_initial = goal_count_heuristic(initial_state, goal)  # Replace with: goal_count_heuristic(initial_state, goal)
    
    pq = []
    
    # Visited set for states we've fully processed
    visited = set()
    
    heapq.heappush(pq, (h_initial, counter, 0, initial_state, []))
    while len(pq) != 0:
        item = heapq.heappop(pq)
        g = item[2]
        state = item[3]
        plan = item[4]
        # state = frozenset(state)
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
