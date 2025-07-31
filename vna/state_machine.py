from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
from vna.vna_experiment import VNAExperiment

class State(ABC):
    """
    Abstract base class for a state in the state machine.
    """
    def __init__(self):
        self._state_machine: Optional[StateMachine] = None

    @property
    def state_machine(self) -> StateMachine:
        return self._state_machine

    @state_machine.setter
    def state_machine(self, state_machine: StateMachine) -> None:
        self._state_machine = state_machine

    @abstractmethod
    def execute(self, experiment_data: VNAExperiment) -> None:
        """
        Execute the state's logic.
        """
        pass

class StateMachine:
    """
    Manages the states and transitions of the VNA workflow.
    """
    def __init__(self, initial_state: State, experiment_data: VNAExperiment):
        self._current_state: Optional[State] = None
        self._experiment_data = experiment_data
        self.transition_to(initial_state)

    def transition_to(self, new_state: State):
        """
        Transitions to a new state.
        """
        print(f"Transitioning to {new_state.__class__.__name__}")
        self._current_state = new_state
        self._current_state.state_machine = self

    def run(self):
        """
        Runs the state machine until there are no more states.
        """
        while self._current_state:
            self._current_state.execute(self._experiment_data)
