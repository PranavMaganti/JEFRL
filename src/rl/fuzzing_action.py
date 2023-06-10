from enum import IntEnum


class FuzzingAction(IntEnum):
    REPLACE = 0
    ADD = 1
    REMOVE = 2
    MODIFY = 3
    MOVE_UP = 4
    MOVE_DOWN = 5
    MOVE_LEFT = 6
    MOVE_RIGHT = 7
    END = 8

    def __str__(self):
        match self:
            case FuzzingAction.REPLACE:
                return "Replace"
            case FuzzingAction.ADD:
                return "Add"
            case FuzzingAction.REMOVE:
                return "Remove"
            case FuzzingAction.MODIFY:
                return "Modify"
            case FuzzingAction.MOVE_UP:
                return "Move Up"
            case FuzzingAction.MOVE_DOWN:
                return "Move Down"
            case FuzzingAction.MOVE_LEFT:
                return "Move Left"
            case FuzzingAction.MOVE_RIGHT:
                return "Move Right"
            case FuzzingAction.END:
                return "End"
