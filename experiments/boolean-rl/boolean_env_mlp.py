import torch
import sympy
from sympy.logic.boolalg import BooleanFunction, And, Or, Not, Equivalent, Implies, Xor
import numpy as np
import random
from multiprocessing import Process, Queue
import time

def _apply_rule_wrapper(rule, expr, queue):
    try:
        result = rule(expr)
        queue.put(result)
    except Exception as e:
        queue.put(e)

def _simplify_logic(expr):
    return sympy.simplify_logic(expr)

def _simplify_logic_dnf(expr):
    return sympy.simplify_logic(expr, form='dnf')

def _simplify_logic_cnf(expr):
    return sympy.simplify_logic(expr, form='cnf')

def _to_anf(expr):
    return sympy.logic.boolalg.to_anf(expr)

def _to_cnf(expr):
    return sympy.logic.boolalg.to_cnf(expr)

def _to_dnf(expr):
    return sympy.logic.boolalg.to_dnf(expr)

def _to_nnf(expr):
    return sympy.logic.boolalg.to_nnf(expr)

class BooleanSimplificationEnv:
    def __init__(self, max_expression_depth, max_literals, max_steps):
        self.max_expression_depth = max_expression_depth
        self.max_literals = max_literals
        self.max_steps = max_steps

        self.literals = [sympy.Symbol(chr(ord('A') + i)) for i in range(max_literals)]
        self.current_expression = None
        self.initial_complexity = 0
        self.known_best_complexity = 0
        self.steps_taken = 0
        self.action_space_size = len(self._get_available_rules())
        self.history = []
        self.reset()

    def _generate_random_expr(self, depth):
        if depth == 0 or random.random() < 0.3:
            return random.choice(self.literals)
        else:
            op_types = [And, Or, Not, Equivalent, Implies, Xor]
            op_select = random.choice(op_types)

            if op_select == Not:
                return op_select(self._generate_random_expr(depth - 1))
            else:
                arg1 = self._generate_random_expr(depth - 1)
                arg2 = self._generate_random_expr(depth - 1)
                return op_select(arg1, arg2)

    def _get_complexity(self, expr):
        if isinstance(expr, sympy.Symbol):
            return 1
        elif isinstance(expr, BooleanFunction):
            complexity = 1
            for arg in expr.args:
                complexity += self._get_complexity(arg)
            return complexity
        else:
            return 0

    def reset(self, max_retries=10):
        for _ in range(max_retries):
            self.current_expression = self._generate_random_expr(self.max_expression_depth)
            if isinstance(self.current_expression, sympy.Symbol) or not self.current_expression.args:
                continue

            self.initial_complexity = self._get_complexity(self.current_expression)

            q = Queue()
            p = Process(target=_apply_rule_wrapper, args=(_simplify_logic, self.current_expression, q))
            p.start()
            p.join(5) # Reduced timeout for faster retries

            if p.is_alive():
                p.terminate()
                p.join()
                continue

            result = q.get()
            if isinstance(result, Exception):
                continue

            self.known_best_complexity = self._get_complexity(result)

            if self.initial_complexity > self.known_best_complexity:
                self.steps_taken = 0
                self.history = [self.current_expression]
                return self._get_state()

        # Fallback to a simple, valid expression if all retries fail
        self.current_expression = self.literals[0] & self.literals[1]
        self.initial_complexity = self._get_complexity(self.current_expression)
        self.known_best_complexity = self._get_complexity(sympy.simplify_logic(self.current_expression))
        self.steps_taken = 0
        self.history = [self.current_expression]
        return self._get_state()

    def _get_state(self):
        count_literals = len(self.current_expression.atoms(sympy.Symbol))
        count_and = len(self.current_expression.atoms(And))
        count_or = len(self.current_expression.atoms(Or))
        count_not = len(self.current_expression.atoms(Not))
        count_equivalent = len(self.current_expression.atoms(Equivalent))
        count_implies = len(self.current_expression.atoms(Implies))
        count_xor = len(self.current_expression.atoms(Xor))

        def get_depth(expr):
            if not hasattr(expr, 'args') or not expr.args:
                return 0
            return 1 + max(get_depth(arg) for arg in expr.args) if expr.args else 0

        depth = get_depth(self.current_expression)
        current_complexity = self._get_complexity(self.current_expression)

        state = np.array([count_literals, count_and, count_or, count_not, count_equivalent, count_implies, count_xor, depth, current_complexity])
        return state

    def get_state_size(self):
        return len(self._get_state())

    def get_action_size(self):
        return len(self._get_available_rules())

    def _get_available_rules(self):
        return [
            _simplify_logic,
            _simplify_logic_dnf,
            _simplify_logic_cnf,
            _to_anf,
            _to_cnf,
            _to_dnf,
            _to_nnf,
        ]

    def step(self, action):
        self.steps_taken += 1

        rules = self._get_available_rules()
        if not (0 <= action < len(rules)):
            return self._get_state(), -10.0, True, {}

        old_complexity = self._get_complexity(self.current_expression)

        q = Queue()
        p = Process(target=_apply_rule_wrapper, args=(rules[action], self.current_expression, q))
        p.start()
        p.join(10)

        if p.is_alive():
            p.terminate()
            p.join()
            print("--- SymPy operation timed out ---")
            return self._get_state(), -5.0, True, {}

        result = q.get()
        if isinstance(result, Exception):
            print(f"--- SymPy operation failed with error: {result} ---")
            return self._get_state(), -5.0, True, {}

        self.current_expression = result
        self.history.append(self.current_expression)
        new_complexity = self._get_complexity(self.current_expression)

        reward = 0.0
        done = False

        complexity_reduction = old_complexity - new_complexity

        if new_complexity < self.known_best_complexity:
            # Significant reward for finding a new best simplification
            reward += 100.0
            self.known_best_complexity = new_complexity
            done = True
        elif new_complexity == self.known_best_complexity:
            # Smaller reward for reaching the known best
            reward += 50.0
            done = True
        elif complexity_reduction > 0:
            # Reward proportional to the reduction in complexity
            reward += 10.0 * complexity_reduction
        else:
            # Penalty for increasing or not changing complexity
            reward -= 2.0

        # Small penalty for each step to encourage shorter solutions
        reward -= 1.0

        if self.steps_taken >= self.max_steps:
            done = True
            if new_complexity > self.known_best_complexity:
                # Larger penalty for failing to simplify
                reward -= 20.0

        return self._get_state(), reward, done, {'history': self.history}
        
    

