import pacman
import time
import curses
import signal
import sys
from pprint import pprint

DRAW_EVERY = 1
SLEEP_TIME = 0  # This can be overwritten by __init__
DISPLAY_MOVES = False
QUIET = False  # Supresses output


class NullGraphics:
    def initialize(self, state, is_blue=False):
        pass

    def update(self, state):
        pass

    def pause(self):
        time.sleep(SLEEP_TIME)

    def draw(self, state):
        print(state)

    def finish(self):
        pass


class PacmanGraphics:
    def __init__(self, speed=None):
        if speed != None:
            global SLEEP_TIME
            SLEEP_TIME = speed

    def initialize(self, state, is_blue=False):
        self.draw(state)
        self.pause()
        self.turn = 0
        self.agent_counter = 0

    def update(self, state):
        num_agents = len(state.agent_states)
        self.agent_counter = (self.agent_counter + 1) % num_agents
        if self.agent_counter == 0:
            self.turn += 1
            if DISPLAY_MOVES:
                ghosts = [pacman.nearest_point(
                    state.get_ghost_position(i)) for i in range(1, num_agents)]
                print(("%4d) P: %-8s" % (self.turn, str(pacman.nearest_point(state.get_pacman_position()))),
                      '| Score: %-5d' % state.score, '| Ghosts:', ghosts))
            if self.turn % DRAW_EVERY == 0:
                self.draw(state)
                self.pause()
        if state._win or state._lose:
            self.draw(state)

    def pause(self):
        time.sleep(SLEEP_TIME)

    def draw(self, state):
        print(state)

    def finish(self):
        pass


CURSE_WINDOW = None


class CurseDisplay:
    def __init__(self, speed=None):
        if speed != None:
            global SLEEP_TIME
            SLEEP_TIME = speed

    def initialize(self, state, is_blue=False):
        global CURSE_WINDOW
        self.stdscr = curses.initscr()
        CURSE_WINDOW = self.stdscr
        curses.cbreak()
        self.stdscr.keypad(1)
        self.stdscr.nodelay(1)
        self.draw(state)
        self.turn = 0
        self.agent_counter = 0

        def handler(signal, frame):
            curses.nocbreak()
            self.stdscr.keypad(0)
            curses.echo()
            curses.endwin()
        signal.signal(signal.SIGINT, handler)

    def update(self, state):
        num_agents = len(state.agent_states)
        self.agent_counter = (self.agent_counter + 1) % num_agents
        if self.agent_counter == 0:
            self.turn += 1
            if DISPLAY_MOVES:
                ghosts = [pacman.nearest_point(
                    state.get_ghost_position(i)) for i in range(1, num_agents)]
                print(("%4d) P: %-8s" % (self.turn, str(pacman.nearest_point(state.get_pacman_position()))),
                      '| Score: %-5d' % state.score, '| Ghosts:', ghosts))
            if self.turn % DRAW_EVERY == 0:
                self.draw(state)
                self.pause()
        if state._win or state._lose:
            self.draw(state)

    def pause(self):
        time.sleep(SLEEP_TIME)
        key = self.stdscr.getch()
        if key == ord('q'):
            self.finish()
            sys.exit()

    def draw(self, state):
        self.stdscr.addstr(0, 10, "Hit 'Ctrl+c' /q  to quit")
        self.stdscr.addstr(1, 0, str(state))
        self.stdscr.refresh()

    def finish(self):
        curses.nocbreak()
        self.stdscr.keypad(0)
        curses.echo()
        curses.endwin()
        # pass
