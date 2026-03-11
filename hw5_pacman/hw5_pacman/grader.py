#!/usr/bin/env python3

import os
import text_display
import layout
import traceback
import time
import math
import pacman
import random
import warnings
import sys
import multi_agents_solution
from ghost_agents import RandomGhost, DirectionalGhost
from game import Agent
import grader_util

grader = grader_util.Grader()
submission = grader.load('submission')

FINAL_GRADE = True
# random seed at the beginning of each question for more fairness in grading...
SEED = 'testing'
BIG_NEGATIVE = -10000


text_display.SLEEP_TIME = 0
text_display.DRAW_EVERY = 1000
thismodule = sys.modules[__name__]


############################################################
# check python version


if not (sys.version_info[0] == 3 and sys.version_info[1] == 12):
    warnings.warn(
        f"Note that you are not using python 3.12. Your code may not work in gradescope.")


def run(layname, pac, ghosts, n_games=1, name='games'):
    """
    Runs a few games and outputs their statistics.
    """
    if grader.fatal_error:
        return {'time': 65536, 'wins': 0, 'games': None, 'scores': [0]*n_games, 'timeouts': n_games}

    starttime = time.time()
    lay = layout.get_layout(layname, 3)
    disp = text_display.NullGraphics()

    print(('*** Running %s on' % name, layname, '%d time(s).' % n_games))
    games = pacman.run_games(lay, pac, ghosts, disp,
                             n_games, False, catch_exceptions=False)
    print(('*** Finished running %s on' % name, layname,
          'after %d seconds.' % (time.time() - starttime)))

    stats = {'time': time.time() - starttime, 'wins': [g.state.is_win() for g in games].count(True), 'games': games, 'scores': [
        g.state.get_score() for g in games], 'timeouts': [g.agent_timeout for g in games].count(True)}
    print(('*** Won %d out of %d games. Average score: %f ***' %
          (stats['wins'], len(games), sum(stats['scores']) * 1.0 / len(games))))

    return stats



def comparison_checking(their_pac, our_pac_options, agent_name):
    """
    Skeleton used for question 2, 3 and 4...
    Takes in their Pacman agent, wraps it in ours, and assigns points.
    """
    print('Running our grader (hidden from you)...')
    random.seed(SEED)
    off_by_one = False
    partial_ply_bug = False
    total_suboptimal = 0
    timeout = False


    return timeout, off_by_one, partial_ply_bug, total_suboptimal


def test0(agent_name):
    stats = {}
    if agent_name == 'alphabeta':
        stats = run('small_classic', submission.AlphaBetaAgent(depth=2), [DirectionalGhost(
            i + 1) for i in range(2)], name='%s (depth %d)' % ('alphabeta', 2))
    elif agent_name == 'minimax':
        stats = run('small_classic', submission.MinimaxAgent(depth=2), [DirectionalGhost(
            i + 1) for i in range(2)], name='%s (depth %d)' % ('minimax', 2))
    else:
        stats = run('small_classic', submission.ExpectimaxAgent(depth=2), [DirectionalGhost(
            i + 1) for i in range(2)], name='%s (depth %d)' % ('expectimax', 2))
    if stats['timeouts'] > 0:
        grader.fail('Your ' + agent_name +
                    ' agent timed out on small_classic.  No autograder feedback will be provided.')
        return
    grader.assign_full_credit()


game_play = {}


def test1(agent_name, basic=False):
    if agent_name not in game_play and not grader.fatal_error:
        if agent_name == 'minimax':
            game_play[agent_name] = comparison_checking(
                submission.MinimaxAgent(depth=2), {}, agent_name)
        elif agent_name == 'alphabeta':
            game_play[agent_name] = comparison_checking(
                submission.AlphaBetaAgent(depth=2), {agent_name: 'True'}, agent_name)
        elif agent_name == 'expectimax':
            game_play[agent_name] = comparison_checking(
                submission.ExpectimaxAgent(depth=2), {agent_name: 'True'}, agent_name)
        else:
            raise Exception("Unexpected agent name: " + agent_name)

    timeout, off_by_one, partial_ply_bug, total_suboptimal = game_play[agent_name]
    if timeout:
        grader.fail('Your ' + agent_name +
                    ' agent timed out on small_classic.  No autograder feedback will be provided.')
        return

    if not basic and off_by_one:
        grader.fail('Depth off by 1')
    grader.assign_full_credit()


def test2(agent_name, basic=False):
    if agent_name not in game_play and not grader.fatal_error:
        if agent_name == 'minimax':
            game_play[agent_name] = comparison_checking(
                submission.MinimaxAgent(depth=2), {}, agent_name)
        elif agent_name == 'alphabeta':
            game_play[agent_name] = comparison_checking(
                submission.AlphaBetaAgent(depth=2), {agent_name: 'True'}, agent_name)
        elif agent_name == 'expectimax':
            game_play[agent_name] = comparison_checking(
                submission.ExpectimaxAgent(depth=2), {agent_name: 'True'}, agent_name)
        else:
            raise Exception("Unexpected agent name: " + agent_name)

    timeout, off_by_one, partial_ply_bug, total_suboptimal = game_play[agent_name]
    if timeout:
        grader.fail('Your ' + agent_name +
                    ' agent timed out on small_classic.  No autograder feedback will be provided.')
        return
    if not basic and partial_ply_bug:
        grader.fail('Incomplete final search ply bug')
    grader.assign_full_credit()


def test3(agent_name, basic=False):
    if agent_name not in game_play and not grader.fatal_error:
        if agent_name == 'minimax':
            game_play[agent_name] = comparison_checking(
                submission.MinimaxAgent(depth=2), {}, agent_name)
        elif agent_name == 'alphabeta':
            game_play[agent_name] = comparison_checking(
                submission.AlphaBetaAgent(depth=2), {agent_name: 'True'}, agent_name)
        elif agent_name == 'expectimax':
            game_play[agent_name] = comparison_checking(
                submission.ExpectimaxAgent(depth=2), {agent_name: 'True'}, agent_name)
        else:
            raise Exception("Unexpected agent name: " + agent_name)

    timeout, off_by_one, partial_ply_bug, total_suboptimal = game_play[agent_name]
    if timeout:
        grader.fail('Your '+agent_name +
                    ' agent timed out on small_classic.  No autograder feedback will be provided.')
        return
    if not basic and total_suboptimal > 0:
        grader.fail('Suboptimal moves: ' + str(total_suboptimal))
    grader.assign_full_credit()


max_seconds = 10

grader.add_manual_part(
    '1a', 5, description='Recurrence for multi-agent minimiax')

# 1b
grader.add_basic_part('1b-0-basic', lambda: test0('minimax'), 4, max_seconds=max_seconds,
                      description='Tests minimax for timeout on small_classic.')
grader.add_basic_part('1b-1-basic', lambda: test1('minimax', True), 0,
                      max_seconds=max_seconds, description='Tests minimax for timeout on hidden test 1.')
grader.add_basic_part('1b-2-basic', lambda: test2('minimax', True), 0,
                      max_seconds=max_seconds, description='Tests minimax for timeout on hidden test 2.')
grader.add_basic_part('1b-3-basic', lambda: test3('minimax', True), 0,
                      max_seconds=max_seconds, description='Tests minimax for timeout on hidden test 3.')

grader.add_hidden_part('1b-1-hidden', lambda: test1('minimax'), 2, max_seconds=max_seconds,
                       description='Tests minimax for off by one bug on small_classic.')
grader.add_hidden_part('1b-2-hidden', lambda: test2('minimax'), 2, max_seconds=max_seconds,
                       description='Tests minimax for search depth bug on small_classic.')
grader.add_hidden_part('1b-3-hidden', lambda: test3('minimax'), 2, max_seconds=max_seconds,
                       description='Tests minimax for suboptimal moves on small_classic.')

# 2a
grader.add_basic_part('2a-0-basic', lambda: test0('alphabeta'), 4, max_seconds=max_seconds,
                      description='Tests alphabeta for timeout on small_classic.')
grader.add_basic_part('2a-1-basic', lambda: test1('alphabeta', True), 0,
                      max_seconds=max_seconds, description='Tests alphabeta for timeout on hidden test 1.')
grader.add_basic_part('2a-2-basic', lambda: test2('alphabeta', True), 0,
                      max_seconds=max_seconds, description='Tests alphabeta for timeout on hidden test 2.')
grader.add_basic_part('2a-3-basic', lambda: test3('alphabeta', True), 0,
                      max_seconds=max_seconds, description='Tests alphabeta for timeout on hidden test 3.')

grader.add_hidden_part('2a-1-hidden', lambda: test1('alphabeta'), 2, max_seconds=max_seconds,
                       description='Tests alphabeta for off by one bug on small_classic.')
grader.add_hidden_part('2a-2-hidden', lambda: test2('alphabeta'), 2, max_seconds=max_seconds,
                       description='Tests alphabeta for search depth bug on small_classic.')
grader.add_hidden_part('2a-3-hidden', lambda: test3('alphabeta'), 2, max_seconds=max_seconds,
                       description='Tests alphabeta for suboptimal moves on small_classic.')

grader.add_manual_part(
    '3a', 5, description='Recurrence for multi-agent expectimax')

# 3b
grader.add_basic_part('3b-0-basic', lambda: test0('expectimax'), 4, max_seconds=max_seconds,
                      description='Tests expectimax for timeout on small_classic.')
grader.add_basic_part('3b-1-basic', lambda: test1('expectimax', True), 0,
                      max_seconds=max_seconds, description='Tests expectimax for timeout on hidden test 1.')
grader.add_basic_part('3b-2-basic', lambda: test2('expectimax', True), 0,
                      max_seconds=max_seconds, description='Tests expectimax for timeout on hidden test 2.')
grader.add_basic_part('3b-3-basic', lambda: test3('expectimax', True), 0,
                      max_seconds=max_seconds, description='Tests expectimax for timeout on hidden test 3.')

grader.add_hidden_part('3b-1-hidden', lambda: test1('expectimax'), 2, max_seconds=max_seconds,
                       description='Tests expectimax for off by one bug on small_classic.')
grader.add_hidden_part('3b-2-hidden', lambda: test2('expectimax'), 2, max_seconds=max_seconds,
                       description='Tests expectimax for search depth bug on small_classic.')
grader.add_hidden_part('3b-3-hidden', lambda: test3('expectimax'), 2, max_seconds=max_seconds,
                       description='Tests expectimax for suboptimal moves on small_classic.')

############################################################
# Problem 4: evaluation function


def average(list):
    sum = 0.0
    count = 0.0
    for item in list:
        if item is not None:
            sum += item
            count += 1.0
    return 0 if count == 0 else sum / count


def runq4():
    """
    Runs their expectimax agent a few times and checks for victory!
    """
    random.seed(SEED)
    n_games = 20

    print(('Running your agent %d times to compute the average score...' % n_games))
    print(('The timeout message (if any) is obtained by running the game once, rather than %d times' % n_games))
    params = '-l small_classic -p ExpectimaxAgent -a eval_fn=better -q -n %d -c' % n_games
    games = pacman.run_games(**pacman.read_command(params.split(' ')))
    timeouts = [game.agent_timeout for game in games].count(True)
    wins = [game.state.is_win() for game in games].count(True)
    average_win_score = 0
    if wins >= n_games / 2:
        average_win_score = average(
            [game.state.get_score() if game.state.is_win() else None for game in games])
    print(('Average score of winning games: %d \n' % average_win_score))
    return timeouts, wins, average_win_score


def testq4():
    timeouts, wins, average_win_score = 1024, 0, 0

    if not grader.fatal_error:
        timeouts, wins, average_win_score = runq4()

    if timeouts > 0:
        grader.fail(
            'Agent timed out on small_classic with better_evaluation_function. No autograder feedback will be provided.')
        return
    if wins == 0:
        grader.fail('Your better evaluation function never won any games.')
        return
    for score in range(1300, 1700, 100):
        if average_win_score >= score:
            grader.add_points(1)

    grader.set_side({'score': average_win_score})


# EXTRA CREDIT

# grader.add_manual_part('4a', 4, extra_credit=True, description='Points for placing in the top 3 (1st place: 4, 2nd place: 3, 3rd place: 1)')
# grader.add_basic_part('4a-1-basic', lambda : testq4(), 4, max_seconds=max_seconds, extra_credit=True, description='1 extra credit point per 100 point increase above 1200.')
# grader.add_manual_part('4b', 1, extra_credit=True, description='Description of your evaluation function.')

grader.add_manual_part(
    '5a', 2, description='Description of why minimax and expectimax agents differ.')
grader.add_manual_part(
    '5b', 1, description='Suggested change to default state evaluation function.')
grader.add_manual_part(
    '5c', 2, description='Another concrete example of an AI misalignment problem.')

grader.grade()
