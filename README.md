# Assignment 1: Classical Planning
## Author: Aesha Gandhi ##

## Part 1: Domain Selection & Description

### Domain: Library Book Retrieval Robot

#### Real-World Scenario
Modern libraries often employ automated retrieval robots to collect requested books from different sections and deliver them to the front desk for pickup. This system improves efficiency in large library facilities where books may be stored across multiple floors or sections, reducing the time people wait for their requested materials.

#### Entities/Objects
- **Robot**: R (the autonomous retrieval agent)
- **Books**: B1, B2, B3, ... (physical library materials)
- **Locations/Sections**: 
  - StacksA (general fiction section)
  - StacksB (non-fiction section)
  - Returns (returned books area)
  - Desk (front desk/pickup location)

#### Agent Goal
The robot's goal is to retrieve a set of requested books from different library sections and deliver each of them to the front desk. To accomplish this, the robot must:
1. Navigate between different library locations
2. Pick up books from their current locations
3. Transport books to the desk
4. Deliver books one at a time due to carrying capacity constraints

#### Why Planning is Required
A simple reflex agent would be insufficient for this domain for several reasons:

1. **Limited Carrying Capacity**: The robot can only carry one book at a time, requiring strategic decisions about which book to retrieve first and when to return to the desk.

2. **Sequential Dependencies**: The robot must complete a delivery (return to desk and drop off) before it can pick up another book, creating dependencies between actions.

3. **Multi-Goal Achievement**: With multiple books to deliver, the robot must reason about the optimal order of retrievals to minimize total travel distance and time.

4. **State-Dependent Actions**: Actions like picking up a book are only valid in certain states (robot at correct location, hand empty), requiring forward planning rather than reactive behavior.

5. **Path Planning**: The robot must navigate between multiple locations in the correct sequence, which requires considering future states rather than just reacting to the current situation.

A reflex agent acting only on current perceptions might try actions that are invalid (such as trying to pick up a second book while already holding one) or take highly inefficient routes (or making unnecessary trips). Classical planning allows the robot to take a complete, valid path of actions that efficiently achieves all delivery goals.

---

## Part 2: STRIPS Formalization

### Predicates (Fluents)

#### Robot Location
- **AtR(l)**: Robot is currently at location l
  - l ∈ {Desk, StacksA, StacksB, Returns}

#### Book Location/Status
- **AtB(b, l)**: Book b is at location l (only true when book is not being carried)
  - b ∈ {B1, B2, B3, ...}
  - l ∈ {Desk, StacksA, StacksB, Returns}
- **Holding(b)**: Robot is currently carrying book b
- **Delivered(b)**: Book b has been successfully delivered to the desk

#### Robot Capacity
- **HandEmpty**: Robot is not carrying any book (hand/gripper is empty)

---

### Action Schemas

#### Action 1: Move
**Purpose**: Navigate the robot from one location to another

- **Action**: Move(l1, l2)
- **Parameters**: 
  - l1: source location
  - l2: destination location
- **Preconditions**: 
  - AtR(l1) - robot is at the source location
  - Connected(l1, l2) - there is a valid path between locations
- **Add Effects**: 
  - AtR(l2) - robot is now at destination
- **Delete Effects**: 
  - AtR(l1) - robot is no longer at source

#### Action 2: PickUp
**Purpose**: Pick up a book from its current location

- **Action**: PickUp(b, l)
- **Parameters**: 
  - b: book to pick up
  - l: location where book is located
- **Preconditions**: 
  - AtR(l) - robot is at the book's location
  - AtB(b, l) - book is at that location
  - HandEmpty - robot is not carrying anything
- **Add Effects**: 
  - Holding(b) - robot is now carrying the book
- **Delete Effects**: 
  - AtB(b, l) - book is no longer at that location
  - HandEmpty - robot's hand is no longer empty

#### Action 3: Deliver
**Purpose**: Deliver a book to the front desk

- **Action**: Deliver(b)
- **Parameters**: 
  - b: book to deliver
- **Preconditions**: 
  - AtR(Desk) - robot is at the desk
  - Holding(b) - robot is carrying the book
- **Add Effects**: 
  - Delivered(b) - book is marked as delivered
  - HandEmpty - robot's hand is now empty
- **Delete Effects**: 
  - Holding(b) - robot is no longer carrying the book

---

### Example Problem Instance

#### Initial State
The robot is starting its shift at the front desk with empty hands. Two books have been requested:
- Book B1 is located in StacksA (general fiction section)
- Book B2 is located in Returns (returned books area)

```
initial_state = {
    "AtR(Desk)",
    "HandEmpty",
    "AtB(B1, StacksA)",
    "AtB(B2, Returns)"
}
```

#### Goal State
Both requested books must be delivered to the front desk for patron pickup.

```
goal = {
    "Delivered(B1)",
    "Delivered(B2)"
}
```

#### Example Solution Plan
One valid plan to achieve the goal:

1. Move(Desk, StacksA)
2. PickUp(B1, StacksA)
3. Move(StacksA, Desk)
4. Deliver(B1)
5. Move(Desk, Returns)
6. PickUp(B2, Returns)
7. Move(Returns, Desk)
8. Deliver(B2)

This plan successfully retrieves both books and delivers them to the desk, respecting the constraint that the robot can only carry one book at a time.


## Part 3: Implementation & Results

### Implementation Overview

I implemented the planning system in Python using STRIPS-based forward search with A* search and a simple heuristic.

**Goal-Count Heuristic**: This heuristic counts how many goal conditions aren't satisfied yet in the current state. For example, if the goal is to deliver 2 books but neither has been delivered yet, the heuristic returns 2. This helps guide the search toward states that are closer to the goal.

**A* Search**: The search uses a priority queue that orders states by `f(n) = g(n) + h(n)`, where g(n) is how many actions we've taken so far and h(n) is the heuristic estimate of how many more we need. This helps find optimal plans more efficiently than basic breadth-first search.

### Experimental Results

Using the example problem instance (retrieve B1 from StacksA and B2 from Returns):

#### A* Search with Goal-Count Heuristic

```
Plan: ['Move(Desk,StacksA)', 'PickUp(B1,StacksA)', 'Move(StacksA,Desk)', 
       'Deliver(B1)', 'Move(Desk,Returns)', 'PickUp(B2,Returns)', 
       'Move(Returns,Desk)', 'Deliver(B2)']
Plan length: 8
States explored: 26
```

#### Analysis

The A* search found an optimal 8-step plan while only exploring 26 states. The heuristic really helps by:

1. Focusing on states that get us closer to the goal (more deliveries completed)
2. Avoiding wasting time exploring irrelevant paths (like going to locations that don't have the books we need)
3. Still finding the optimal solution

The plan shows the robot doing two complete retrieval cycles:
- First cycle: Go to StacksA → Pick up B1 → Go back to Desk → Deliver B1
- Second cycle: Go to Returns → Pick up B2 → Go back to Desk → Deliver B2

This shows how the heuristic makes search way more efficient compared to uninformed methods, especially as problems get bigger with more books and locations.
