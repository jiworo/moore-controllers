import argparse
from collections import defaultdict
from itertools import combinations, product
import json
from pathlib import Path
import time
from typing import Callable

import z3

from molehill.constraints.constraint import Constraint


class CustomConstraint(Constraint):
    def __init__(self):
        super().__init__()
        self.mdp_states, self.init_state = None, None
        self.actions = None
        self.aut_states, self.MaybeQ = None, None
        self.gamma, self.delta = None, None
        self.order = None
        self.nr_atoms, self.has_predicates, self.predicates = None, None, None
        self.out_degree, self.has_out_degree = None, None
        self.var_names, self.sgn, self.leq, self.geq, self.true = None, "rvc", "<=", ">=", 'T'  # Just for display
        self.t_pass = None

    def register_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--order", type=int, help="|Q|", required=True)
        parser.add_argument("--nr-atoms", type=int, help="Maximum predicate size", default=-1)
        parser.add_argument("--out-degree", type=int, help="Maximum out-degree", default=-1)

    def set_args(self, args: argparse.Namespace) -> None:
        self.order = args.order
        self.nr_atoms, self.has_predicates = args.nr_atoms, args.nr_atoms >= 1
        self.out_degree, self.has_out_degree = args.out_degree, args.out_degree >= 1

    def build_constraint(
            self,
            function: z3.FuncDeclRef,
            variables: list[z3.BitVecRef],
            variables_in_ranges: Callable[[list[z3.BitVecRef]], z3.ExprRef],
            **args
    ) -> z3.ExprRef:
        constrain_gamma, constrain_delta, constrain_predicates = [], [], []

        quotient = args["quotient"]
        pomdp = quotient.pomdp
        transition_matrix = pomdp.transition_matrix
        fixed_idxs = set()

        # linus
        row_to_assignment: list[list[tuple[int, int]]] = quotient.coloring.getChoiceToAssignment()
        state_to_var: dict[int, z3.BitVecRef] = {}

        for state_idx in range(transition_matrix.nr_columns):
            if "init" in pomdp.labels_state(state_idx):
                init_idx = state_idx

            row_idxs = transition_matrix.get_rows_for_group(state_idx)
            if len(row_idxs) == 1:
                fixed_idxs.add(state_idx)
                continue

            var_idx = row_to_assignment[row_idxs[0]][0][0]
            for row_idx in row_idxs:
                assert len(row_to_assignment[row_idx]) == 1
                assert row_to_assignment[row_idx][0][0] == var_idx

            state_to_var[state_idx] = variables[var_idx]
        # sunil

        S, self.mdp_states = z3.EnumSort('S', [f"s{state_idx}" for state_idx in range(pomdp.nr_states)])
        self.init_state = self.mdp_states[init_idx]
        Q, self.aut_states = z3.EnumSort('Q', [f"q{i}" for i in range(self.order)])

        rgf = [z3.Int(f"g_{i}") for i in range(pomdp.nr_states)]  # q_{g_i} = gamma(s_i)
        maxima = [z3.Int(f"m_{i}") for i in range(pomdp.nr_states)]
        constrain_gamma.append(
            z3.And(maxima[0] == rgf[0], rgf[0] == 0)  # m_0 = g_0 = 0
        )
        constrain_gamma.extend(
            z3.And(
                0 <= rgf[k], rgf[k] < self.order,
                maxima[k] == z3.If(maxima[k - 1] >= rgf[k], maxima[k - 1], rgf[k]),  # m_k = max(m_{k-1}, g_k)
                rgf[k] <= (maxima[k - 1] + 1)  # g_k <= m_{k-1} + 1
            )
            for k in range(1, pomdp.nr_states)
        )
        self.gamma = z3.Function("gamma", S, Q)  # Questionable but well
        constrain_gamma.extend(
            (rgf[i] == qi) == (self.gamma(s) == self.aut_states[qi])
            for i, s in enumerate(self.mdp_states) for qi in range(self.order)
        )

        act_labels = sorted(lab for lab in pomdp.choice_labeling.get_labels() if lab != "__no_label__")
        Act, self.actions = z3.EnumSort("Act", act_labels)
        label_to_act: dict[str, Act] = dict(zip(act_labels, self.actions))

        self.MaybeQ = z3.Datatype("MaybeQ")
        self.MaybeQ.declare("Just", ("fromJust", Q))
        self.MaybeQ.declare("Nothing")
        self.MaybeQ = self.MaybeQ.create()
        self.delta = z3.Function("delta", Q, Act, S, self.MaybeQ)
        supp_trans: dict[S, dict[Act, list[S]]] = {}

        for state_idx in range(transition_matrix.nr_columns):
            q_pred = self.gamma(s_pred := self.mdp_states[state_idx])

            if state_idx in fixed_idxs:
                for act_idx, row_idx in enumerate(transition_matrix.get_rows_for_group(state_idx)):
                    act_label = pomdp.choice_labeling.get_labels_of_choice(row_idx).pop()
                    successors = []

                    if act_label != "__no_label__":  # Fixed case
                        for entry in transition_matrix.get_row(row_idx):
                            q_suc = self.gamma(s_suc := self.mdp_states[entry.column])
                            constrain_delta.append(
                                self.delta(q_pred, label_to_act[act_label], s_suc) == self.MaybeQ.Just(q_suc)
                            )
                            successors.append(s_suc)

                        supp_trans.setdefault(s_pred, {label_to_act[act_label]: successors})
                        constrain_delta.extend(
                            self.delta(q_pred, label_to_act[lab], s_obs) == self.MaybeQ.Nothing
                            for lab, s_obs in product(act_labels, self.mdp_states) if lab != act_label
                        )
                    else:  # Unreachable case
                        supp_trans.setdefault(s_pred, {})
                        constrain_gamma.append(self.gamma(s_pred) == self.aut_states[0])
                continue

            var = state_to_var[state_idx]  # Hole case
            act_domain = set()
            row_idxs = transition_matrix.get_rows_for_group(state_idx)

            for act_idx, row_idx in enumerate(row_idxs):
                act_domain.add(act_label := pomdp.choice_labeling.get_labels_of_choice(row_idx).pop())
                act = label_to_act[act_label]
                successors = []

                for entry in transition_matrix.get_row(row_idx):
                    successors.append(s_suc := self.mdp_states[entry.column])
                    q_suc = self.gamma(s_suc)
                    constrain_delta.append(
                        z3.If(
                            var == z3.BitVecVal(act_idx, var.size()),
                            self.delta(q_pred, act, s_suc) == self.MaybeQ.Just(q_suc),
                            z3.And(self.delta(q_pred, act, s_obs) == self.MaybeQ.Nothing for s_obs in self.mdp_states)
                        )
                    )

                supp_trans.setdefault(s_pred, {}).setdefault(act, []).extend(successors)

            constrain_delta.extend(
                self.delta(q_pred, label_to_act[lab], s_obs) == self.MaybeQ.Nothing
                for lab, s_obs in product(act_labels, self.mdp_states) if lab not in act_domain
            )

        supp_obs = z3.Function("supp_obs", Q, Act, z3.SetSort(S))  # { s' in supp(P(s,a,⋅)) ∧ gamma(s) = q }
        for q_pred in self.aut_states:
            for act in self.actions:
                for s_obs in self.mdp_states:
                    suc_candidates = [
                        self.gamma(s_pred) == q_pred for s_pred in self.mdp_states
                        if act in supp_trans[s_pred] and s_obs in supp_trans[s_pred][act]
                    ]
                    constrain_delta.append(
                        z3.IsMember(s_obs, supp_obs(q_pred, act)) == z3.Or(suc_candidates)
                    )
                    constrain_delta.append(  # Handle illegal observations
                        z3.Implies(
                            z3.Not(z3.IsMember(s_obs, supp_obs(q_pred, act))),
                            self.delta(q_pred, act, s_obs) == self.MaybeQ.Nothing
                        )
                    )

        if self.has_predicates or self.has_out_degree:
            supp_delta = z3.Function("supp_delta", Q, Act, z3.SetSort(Q))  # Helper for { q' in supp(delta(q,⋅,⋅)) }
            constrain_delta.extend(
                z3.IsMember(q_suc, supp_delta(q_pred, act)) == z3.Or(
                    self.delta(q_pred, act, s_obs) == self.MaybeQ.Just(q_suc) for s_obs in self.mdp_states
                )
                for q_pred, act, q_suc in product(self.aut_states, self.actions, self.aut_states)
            )

        if self.has_predicates:
            RelOp, (LEQ, GEQ, T) = z3.EnumSort("RelOp", [self.leq, self.geq, self.true])

            def predicate(trans: str) -> (
                    Callable[[list[z3.IntNumRef]], z3.BoolRef],
                    list[RelOp], list[z3.ArithRef], list[z3.ArithRef]
            ):
                rel_ops = [z3.Const(f"{self.sgn[0]}-{trans}-{n}", RelOp) for n in range(self.nr_atoms)]
                val_idxs = [z3.Int(f"{self.sgn[1]}-{trans}-{n}") for n in range(self.nr_atoms)]
                constants = [z3.Int(f"{self.sgn[2]}-{trans}-{n}") for n in range(self.nr_atoms)]

                def template(state_vals: list[z3.ArithRef]) -> z3.BoolRef:
                    atoms = [
                        z3.If(
                            rel_ops[i] == LEQ, z3.Select(state_vals, val_idxs[i]) <= constants[i], z3.If(
                                rel_ops[i] == GEQ, z3.Select(state_vals, val_idxs[i]) >= constants[i], True
                            )
                        )
                        for i in range(self.nr_atoms)
                    ]
                    return z3.And(*atoms)

                return template, rel_ops, val_idxs, constants

            state_to_vals: dict[z3.DatatypeRef, z3.ArrayRef] = {}
            var_domain = set()

            for state in pomdp.states:
                var_to_val = json.loads(str(pomdp.state_valuations.get_json(state.id)))
                vals = z3.K(z3.IntSort(), z3.IntVal(-1))  # Questionable

                for idx, val in enumerate(var_to_val.values()):
                    var_domain.add(val)
                    vals = z3.Store(vals, idx, val)

                state_to_vals[self.mdp_states[state.id]] = vals

            self.var_names = list(var_to_val.keys())  # This Python behavior is useful for once
            var_domain = list(sorted(var_domain))

            self.predicates: dict[tuple[Q, Act, Q], tuple[Callable, list, list, list]] = {
                (q_pred, act, q_suc): predicate(f"{q_pred}-{act}-{q_suc}")
                for q_pred, act, q_suc in product(self.aut_states, self.actions, self.aut_states)
            }
            constrain_predicates.extend(
                z3.And(var_idx >= 0, var_idx < len(self.var_names))
                for *_, var_idxs, _ in self.predicates.values() for var_idx in var_idxs
            )
            constrain_predicates.extend(  # Share state variable domain
                z3.Or(const == val for val in var_domain) for *_, consts in self.predicates.values() for const in consts
            )
            constrain_predicates.extend(
                z3.Implies(
                    self.delta(q_pred, act, s_obs) != self.MaybeQ.Nothing,
                    eval_(state_to_vals[s_obs]) == (self.delta(q_pred, act, s_obs) == self.MaybeQ.Just(q_suc))
                )
                for ((q_pred, act, q_suc), (eval_, *_)), s_obs in product(self.predicates.items(), self.mdp_states)
            )
            constrain_predicates.extend(  # Unique atoms, avoid, e.g., x >= 0 /\ x >= 1
                z3.And(
                    z3.Implies(
                        val_idxs[i] == val_idxs[j],
                        z3.Or(rel_ops[i] == T, rel_ops[j] == T, rel_ops[i] != rel_ops[j])
                    ),
                    z3.Implies(z3.And(val_idxs[i] == val_idxs[j], rel_ops[i] == LEQ), constants[i] > constants[j]),
                    z3.Implies(z3.And(val_idxs[i] == val_idxs[j], rel_ops[i] == GEQ), constants[i] < constants[j])
                )
                for *_, rel_ops, val_idxs, constants in self.predicates.values()
                for i, j in combinations(range(self.nr_atoms), 2)
            )
            constrain_predicates.extend(
                z3.And(  # Fix arbitrary value choices
                    z3.Implies(rel_ops[i] == T, z3.And(val_idxs[i] == 0, constants[i] == var_domain[0])),
                    z3.Implies(  # No need for predicates if out-degree < 2
                        z3.And(
                            z3.PbEq([(z3.IsMember(q, supp_delta(q_pred, act)), 1) for q in self.aut_states], 1),
                            z3.IsMember(q_suc, supp_delta(q_pred, act))
                        ),
                        rel_ops[i] == T
                    )
                )
                for (q_pred, act, q_suc), (*_, rel_ops, val_idxs, constants) in self.predicates.items()
                for i in range(self.nr_atoms)
            )
            constrain_predicates.extend(
                z3.Implies(rel_ops[i - 1] == T, rel_ops[i] == T)  # [r!=T, T] equiv [T, r!=T]
                for _, (*_, rel_ops, _, _) in self.predicates.items() for i in range(1, self.nr_atoms)
            )

        if self.has_out_degree:
            constrain_delta.extend(
                z3.PbLe(
                    [(z3.IsMember(q_suc, supp_delta(q_pred, act)), 1) for q_suc in self.aut_states], self.out_degree
                )
                for q_pred, act in product(self.aut_states, self.actions)
            )

        self.t_pass = time.time()

        return z3.And(
            function(*variables),
            variables_in_ranges(variables),
            *constrain_gamma,
            *constrain_delta,
            *constrain_predicates,
        )

    def show_result(self, model: z3.ModelRef, solver: z3.Solver, **args) -> None:
        t_receive = time.time()

        z3.set_option(max_args=10_000_000, max_lines=1_000_000, max_depth=10_000_000, max_visited=1_000_000)
        print(model)

        trans_to_lab: dict[tuple[Q, Act, Q], list | str] = defaultdict(list)
        policy: dict[Q, Act] = defaultdict()

        for q_pred, act, s_obs in product(self.aut_states, self.actions, self.mdp_states):
            q_suc = model.eval(self.gamma(s_obs))

            if model.eval(self.delta(q_pred, act, s_obs)).eq(self.MaybeQ.Just(q_suc)):
                policy[q_pred] = act

                if not self.has_predicates:
                    trans_to_lab[(q_pred, act, q_suc)].append(s_obs)
                elif (q_pred, act, q_suc) not in trans_to_lab:
                    trans_to_lab[(q_pred, act, q_suc)] = " /\\ ".join(
                        f"{self.var_names[model.eval(v).as_long()]} {r_val} {model.eval(c)}"
                        for r, v, c in zip(*self.predicates[(q_pred, act, q_suc)][1:])
                        if (r_val := model.eval(r).decl().name()) != self.true
                    )

        trans_to_lab = {
            trans: sorted(lab, key=lambda s: int(s.decl().name()[1:])) if isinstance(lab, list) else lab
            for trans, lab in trans_to_lab.items()
        }

        path = Path.cwd() / "out.dot"
        print("Writing dotfile to", path)

        with open(path, 'w') as graph:
            graph.write(
                f"// {Path.cwd().name} where q_init={model.eval(self.gamma(self.init_state))}\n"
                f"// ord={self.order}, n_atoms={self.nr_atoms}, out-degree={self.out_degree}\n"
                f"// {t_receive - self.t_pass} seconds\n"
                "digraph finite_state_machine {\n"
                "\trankdir=LR;\n"
            )
            for (q_pred, act, q_suc), lab in trans_to_lab.items():
                graph.write(f'\t"{q_pred}/{act}" -> "{q_suc}/{policy[q_suc]}" [label="{lab}"]\n')

            graph.write("}\n")
