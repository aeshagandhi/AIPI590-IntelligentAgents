# scheduling logic

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import List, Optional

@dataclass
class Task:
    name: str
    deadline: datetime
    duration_minutes: int
    priority: str = "medium" # low, medium, high
    metadata: dict = field(default_factory=dict)
    
    
    def to_dict(self) -> dict:
        data = asdict(self)
        data["deadline"] = self.deadline.isoformat()
        return data
    
@dataclass
class TimeSlot:
    start: datetime
    end: datetime
    
    @property
    def duration_minutes(self) -> int:
        # compute duration of a time slot 
        return int((self.end - self.start).total_seconds() / 60)
    
    def to_dict(self) -> dict:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "duration_minutes": self.duration_minutes
        }
    
        
@dataclass
class ScheduleBlock:
    task_name: str
    start: datetime 
    end: datetime 
    minutes_scheduled: int
    
    def to_dict(self) -> dict:
        return {
            "task_name": self.task_name,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "minutes_scheduled": self.minutes_scheduled
        }

@dataclass
class SchedulingResult:
    scheduled_blocks: List[ScheduleBlock]
    unscheduled_tasks: List[dict]
    remaining_slots: List[TimeSlot]

    def to_dict(self) -> dict:
        return {
            "scheduled_blocks": [block.to_dict() for block in self.scheduled_blocks],
            "unscheduled_tasks": self.unscheduled_tasks,
            "remaining_slots": [slot.to_dict() for slot in self.remaining_slots],
        }

PRIORITY_RANK = {
    "high": 0,
    "medium": 1,
    "low": 2
}

def sort_tasks_for_scheduling(tasks: List[Task]) -> List[Task]:
    # sorts tasks by earliest deadline, then highest priorty, then quickest to complete
    return sorted(tasks, key = lambda t: (t.deadline, PRIORITY_RANK.get(t.priority.lower(), 1), t.duration_minutes))

def merge_slots(slots: List[TimeSlot]) -> List[TimeSlot]:
    # merge overlapping or adjacent timeslots
    if not slots:
        return []
    
    sorted_slots = sorted(slots, key = lambda s: s.start)
    # start with earliest time slot
    merged = [sorted_slots[0]]
    for current in sorted_slots[1:]:
        last = merged[-1]
        if current.start <= last.end:
            merged[-1] = TimeSlot(start = last.start, end = max(last.end, current.end),)
        else:
            merged.append(current)
    return merged
            
def allocate_from_slot(slot: TimeSlot, minutes_needed: int) -> tuple[Optional[ScheduleBlock], Optional[TimeSlot], int]:
    # from the start of a slot, allocates minutes needed and returns the schedule block, remaining slot, and minutes actually allocated
    available = slot.duration_minutes
    if available <= 0 or minutes_needed <= 0:
        return None, slot, 0

    allocated_minutes = min(available, minutes_needed)
    allocated_end = slot.start + timedelta(minutes=allocated_minutes)

    remaining_slot = None
    if allocated_end < slot.end:
        remaining_slot = TimeSlot(start=allocated_end, end=slot.end)

    temp_block = ScheduleBlock(
        task_name="",
        start=slot.start,
        end=allocated_end,
        minutes_scheduled=allocated_minutes,
    )
    return temp_block, remaining_slot, allocated_minutes


def build_schedule(tasks: List[Task], slots: List[TimeSlot]) -> SchedulingResult: 
    # schedules tasks by the earliest deadline
    # can split a task across multiple slots if needed
    # use slot times that occur before the task deadline
    
    tasks_sorted = sort_tasks_for_scheduling(tasks)
    remaining_slots = merge_slots(slots)
    
    scheduled_blocks: List[ScheduleBlock] = []
    unscheduled_tasks: List[dict] = []
    
    # go task by task in order of priority and fill available time slots until the task is completed or we run out of time before its deadline 
    for task in tasks_sorted:
        minutes_left = task.duration_minutes
        updated_slots: List[TimeSlot] = []
        
        for slot in remaining_slots:
            # if the slot starts after the task deadline, skip it 
            if slot.start >= task.deadline:
                updated_slots.append(slot)
                continue
                
            usable_end = min(slot.end, task.deadline)
            usable_slot = TimeSlot(start = slot.start, end=usable_end)
            
            if usable_slot.duration_minutes <= 0:
                updated_slots.append(slot)
                continue
            
            if minutes_left > 0:
                temp_block, remaining_usable_slot, allocated = allocate_from_slot(usable_slot, minutes_left)
                if allocated > 0 and temp_block is not None:
                    block = ScheduleBlock(task_name = task.name, start = temp_block.start, end = temp_block.end, minutes_scheduled = allocated)
                    scheduled_blocks.append(block)
                    # decrement minutes left and update remaining slots
                    minutes_left -= allocated
                    
                    # if original slot has leftover time after deadline preserve that remaining time
                    if usable_end < slot.end:
                        updated_slots.append(TimeSlot(start =usable_end, end = slot.end))
                    # if there is leftover time after allocation, preserve the slot too
                    if remaining_usable_slot is not None:
                        updated_slots.append(remaining_usable_slot)
                else:
                    updated_slots.append(slot)
                    
            remaining_slots = merge_slots(updated_slots)
            
            if minutes_left > 0:
                # new task from remaining mins and add to unscheduled tasks
                temp_task = {
                    "task_name": task.name,
                    "deadline": task.deadline.isoformat(),
                    "requested_duration_minutes": task.duration_minutes,
                    "unscheduled_minutes": minutes_left,
                    "priority": task.priority
                }
                unscheduled_tasks.append(temp_task)
    scheduled_blocks.sort(key = lambda block: block.start)
    return SchedulingResult(scheduled_blocks=scheduled_blocks, unscheduled_tasks = unscheduled_tasks, remaining_slots = remaining_slots)


def validate_schedule(
    scheduled_blocks: List[ScheduleBlock],
    tasks: List[Task],
) -> dict:
    """
    Basic validation:
    - no overlapping scheduled blocks
    - task time before deadline
    - track how much of each task got scheduled
    """
    issues: List[str] = []

    blocks = sorted(scheduled_blocks, key=lambda b: b.start)
    for i in range(1, len(blocks)):
        if blocks[i].start < blocks[i - 1].end:
            issues.append(
                f"Overlap detected between '{blocks[i - 1].task_name}' and '{blocks[i].task_name}'."
            )

    task_lookup = {task.name: task for task in tasks}
    scheduled_minutes_by_task: dict[str, int] = {}

    for block in blocks:
        scheduled_minutes_by_task[block.task_name] = (
            scheduled_minutes_by_task.get(block.task_name, 0) + block.minutes_scheduled
        )

        task = task_lookup.get(block.task_name)
        if task is not None and block.end > task.deadline:
            issues.append(
                f"Task '{block.task_name}' has scheduled time after its deadline."
            )

    completion = []
    for task in tasks:
        scheduled = scheduled_minutes_by_task.get(task.name, 0)
        completion.append(
            {
                "task_name": task.name,
                "scheduled_minutes": scheduled,
                "requested_minutes": task.duration_minutes,
                "fully_scheduled": scheduled >= task.duration_minutes,
            }
        )

    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "completion": completion,
    }