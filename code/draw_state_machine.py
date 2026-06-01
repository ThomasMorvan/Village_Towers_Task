import re
from collections import defaultdict, deque
from typing import Literal


class State:
    def __init__(self, name, timer="", outputs=None, transitions=None):
        self.name = name
        self.timer = timer
        self.outputs = outputs or []
        self.transitions = transitions or {}

    def __repr__(self):
        return (f"State(name={self.name!r}, timer={self.timer!r}, "
                f"outputs={self.outputs!r}, transitions={self.transitions!r})")


class DrawStateMachine:
    def __new__(cls, filepath: str, title: str = None,
                arrow_h: int = 3, h_gap: int = 4,
                skip_side: Literal["nested", "right",
                                   "alternate"] = "alternate",
                verbose: bool = False):
        obj = object.__new__(cls)
        obj.filepath = filepath
        obj.title = title
        obj.arrow_h = max(3, arrow_h)
        obj.h_gap = h_gap
        obj.skip_side = skip_side
        obj.verbose = verbose
        obj.states = []
        obj.all_vals = {}
        return obj.render()

    def render(self):
        try:
            self.states, self.all_vals = self._extract_states()
            levels, by_level, state_map = self._build_graph()
            text = self._draw(levels, by_level, state_map)
            return text
        except Exception as e:
            print(f"Error rendering state machine: {e}")
            return ""

    # methods to extract states
    def _extract_states(self):
        with open(self.filepath) as f:
            task_str = f.read()

        # First check if looks like a task file with create_trial
        # and add_state calls
        if "def create_trial" not in task_str or "add_state" not in task_str:
            print(f"DrawStateMachine Warning: file {self.filepath} "
                  f"does not appear to be a valid task file.")
            return [], {}

        ############################################
        # Get all variables from the text
        direct_vals = defaultdict(set)
        chains = defaultdict(set)
        all_vars = defaultdict(set)

        # self.X = "..." or self.X = '...'
        pattern_1 = r"self\.(\w+)\s*=\s*['\"]([^'\"]+)['\"]"
        for m in re.finditer(pattern_1, task_str):
            direct_vals[f"self.{m.group(1)}"].add(m.group(2))
            if self.verbose:
                print(f"Pattern 1 match: {m.group(0)} --> "
                      f"self.{m.group(1)} = '{m.group(2)}'")

        # self.X = 123 or self.X = 3.14
        pattern_2 = r"self\.(\w+)\s*=\s*(-?\.?\d[\d.]*)\s*(?:#.*)?$"
        for m in re.finditer(pattern_2, task_str, re.MULTILINE):
            raw = m.group(2)
            try:
                direct_vals[f"self.{m.group(1)}"].add(
                    int(raw) if '.' not in raw else float(raw))
                if self.verbose:
                    print(f"Pattern 2 match: {m.group(0)} --> "
                          f"self.{m.group(1)} = {raw}")
            except ValueError:
                pass

        # self.settings.X = 123 or self.settings.X = 3.14
        pattern_3 = r"self\.settings\.(\w+)\s*=\s*(-?\.?\d[\d.]*)\s*(?:#.*)?$"
        for m in re.finditer(pattern_3, task_str, re.MULTILINE):
            raw = m.group(2)
            try:
                direct_vals[f"self.settings.{m.group(1)}"].add(
                    int(raw) if '.' not in raw else float(raw))
                if self.verbose:
                    print(f"Pattern 3 match: {m.group(0)} --> "
                          f"self.settings.{m.group(1)} = {raw}")
            except ValueError:
                pass

        # self.X = self.Y (chain)
        pattern_4 = r"self\.(\w+)\s*=\s*(self\.\w+)\s*(?:#.*)?$"
        for m in re.finditer(pattern_4, task_str, re.MULTILINE):
            chains[f"self.{m.group(1)}"].add(m.group(2))
            if self.verbose:
                print(f"Pattern 4 match: {m.group(0)} --> "
                      f"self.{m.group(1)} = {m.group(2)}")

        for k, vals in direct_vals.items():
            all_vars[k].update(vals)
        for k, refs in chains.items():
            for ref in refs:
                all_vars[k].update(direct_vals.get(ref, set()))

        if self.verbose:
            print(f"Collected variable values: {all_vars}")

        #####################################################
        # get add_state(...) calls and extract state info
        states = []
        for m in re.finditer(r'\badd_state\s*\(', task_str):
            # balance ( and ) to get full call
            depth = 0
            _pos = m.end() - 1
            for i in range(_pos, len(task_str)):
                if task_str[i] == '(':
                    depth += 1
                elif task_str[i] == ')':
                    depth -= 1
                    if depth == 0:
                        body = task_str[_pos + 1:i]
                        break
            else:  # Yay! A for ... else!
                body = task_str[_pos + 1:]

            if self.verbose:
                print(f"Found add_state call: {body}")

            # Get current state arguments as k:v pairs
            state_args = {}
            for part in self._split_depth(body):
                part = part.strip()
                if not part or '=' not in part:
                    continue
                key, _, val = part.partition('=')
                key = key.strip()
                if re.match(r'^\w+$', key):
                    state_args[key] = val.strip()

            if self.verbose:
                print(f"State args: {state_args}")

            if 'state_name' not in state_args:
                continue

            # state name
            name = state_args['state_name'].strip("'\"")
            if self.verbose:
                print(f" --> State name: {name}")

            # get state timer
            timer = ""
            if 'state_timer' in state_args:
                raw = state_args['state_timer']
                if raw in all_vars and all_vars[raw]:
                    timer = "/".join(str(v) for
                                     v in sorted(all_vars[raw], key=str))
                else:
                    timer = self._remove_stuff(raw)
                if self.verbose:
                    print(f" --> State timer: {timer}")

            # get output actions
            output_actions = []
            if 'output_actions' in state_args:
                oa = state_args['output_actions'].strip()
                if oa.startswith('['):
                    for item in self._split_depth(oa[1:-1]
                                                  if oa.endswith(']')
                                                  else oa[1:]):
                        item = item.strip()
                        if item.startswith('*'):
                            output_actions.append('*cues')
                        elif item:
                            output_actions.append(self._remove_stuff(item))
                if self.verbose:
                    print(f" --> Output actions: {output_actions}")

            # get state change conditions
            state_change_conds = {}
            if 'state_change_conditions' in state_args:
                cond = state_args['state_change_conditions'].strip()
                if cond.startswith('{'):
                    inner = cond[1:-1] if cond.endswith('}') else cond[1:]
                    for item in self._split_depth(inner):
                        item = item.strip()
                        if not item:
                            continue
                        colon = item.find(':')
                        if colon < 0:
                            continue
                        k = self._remove_stuff(item[:colon].strip())
                        v = item[colon + 1:].strip()
                        if ((v.startswith('"') and v.endswith('"')) or
                                (v.startswith("'") and v.endswith("'"))):
                            state_change_conds[k] = [v[1:-1]]
                        else:
                            possible = all_vars.get(v)
                            if possible:
                                state_change_conds[k] = list(possible)
                            else:
                                state_change_conds[k] = [
                                    f"[{self._remove_stuff(v)}]"]
                if self.verbose:
                    print(f" --> State change conds: {state_change_conds}")

            state = State(name, timer, output_actions, state_change_conds)
            states.append(state)

            if self.verbose:
                print(f"Added state: {state}")
                print("-" * 60)

        return states, all_vars

    @staticmethod
    def _split_depth(text, sep=','):
        """Split text based on sep,
        e.g. "a=1, b=[2,3], c={'x': 4}" --> ["a=1", "b=[2,3]", "c={'x': 4}"]
        """
        parts, current, depth = [], [], 0
        for ch in text:
            if ch in '([{':
                depth += 1
            elif ch in ')]}':
                depth -= 1
            if ch == sep and depth == 0:
                parts.append(''.join(current))
                current = []
            else:
                current.append(ch)
        if current:
            parts.append(''.join(current))
        return parts

    @staticmethod
    def _remove_stuff(s):
        s = s.replace("self.settings.", "")
        s = s.replace("self.", "")
        s = s.replace("Output.", "")
        s = s.replace("Event.", "")
        return s

    # build state graph
    def _build_graph(self):

        # Assign levels to states based on longest path from root
        levels = {}
        if self.states:
            # Adjency: for each state, which states can it transition to?
            adj = defaultdict(set)
            all_nodes = set()
            for s in self.states:
                all_nodes.add(s.name)
                for targets in s.transitions.values():
                    for t in targets:
                        adj[s.name].add(t)
                        all_nodes.add(t)
            if self.verbose:
                print(f"Adjacency list: {dict(adj)}")

            # Detect edges that create cycles using DFS (Depth-First Search).
            # mark nodes as we visit them:
            # unvisited (0), on the stack (1), and done (2).
            # If we encounter an edge to a node on the stack (1), it means
            # we're trying to visit a node that's already on the stack,
            # which indicates a cycle (back edge).
            visits = {}   # 0 (unvisited), 1 (on stack), 2 (done)
            back_edges = set()
            for start in all_nodes:
                if visits.get(start, 0) != 0:
                    continue
                visits[start] = 1
                stack = [(start, iter(list(adj.get(start, []))))]
                while stack:
                    u, children = stack[-1]
                    try:
                        v = next(children)
                        cv = visits.get(v, 0)
                        if cv == 1:
                            back_edges.add((u, v))
                        elif cv == 0:
                            visits[v] = 1
                            stack.append((v, iter(list(adj.get(v, [])))))
                    except StopIteration:
                        visits[u] = 2
                        stack.pop()
            if self.verbose:
                print(f"Back edges (cycles): {back_edges}")

            # remove edges that point back up the graph, which cause cycles
            # and build Directed Acyclic Graph without back-edges
            dag = defaultdict(set)
            for u, vs in adj.items():
                for v in vs:
                    if (u, v) not in back_edges:
                        dag[u].add(v)
            if self.verbose:
                print(f"Directed Acyclic Graph: {dict(dag)}")

            # Topological sort using Kahn algorithm to find
            # longest path levels from root
            dag_nodes = set(dag.keys())
            for vs in dag.values():
                dag_nodes.update(vs)
            in_deg = defaultdict(int)
            for vs in dag.values():
                for v in vs:
                    in_deg[v] += 1

            queue = deque(n for n in dag_nodes if in_deg[n] == 0)
            dist = defaultdict(int)
            while queue:
                u = queue.popleft()
                for v in dag[u]:
                    dist[v] = max(dist[v], dist[u] + 1)
                    in_deg[v] -= 1
                    if in_deg[v] == 0:
                        queue.append(v)
            if self.verbose:
                print(f"Longest path levels: {dict(dist)}")

            # Any state unreachable from root gets placed at max level + 1
            # but should not happen
            max_lvl = max(dist.values(), default=0)
            for s in self.states:
                if s.name not in dist:
                    dist[s.name] = max_lvl + 1

            levels = dict(dist)

        if self.verbose:
            print(f"Final state levels: {levels}")

        by_level = defaultdict(list)
        known = {s.name for s in self.states}
        for s in self.states:
            by_level[levels[s.name]].append(s.name)
        for s in self.states:
            for targets in s.transitions.values():
                for t in targets:
                    if t not in known:
                        lvl = levels.get(t, max(levels.values(),
                                                default=0) + 1)
                        if t not in [n for ns in by_level.values()
                                     for n in ns]:
                            by_level[lvl].append(t)

        state_map = {s.name: s for s in self.states}

        if self.verbose:
            print(f"States by level: {dict(by_level)}")

        return levels, by_level, state_map

    # Methods to draw
    @staticmethod
    def _box(text, timer=""):
        box_contents = [text] if text else []
        if timer and timer not in ("0", ""):
            box_contents.append(f"t={timer}")
        inner = max(len(box_content) for box_content in box_contents) + 2
        top = "┏" + "━" * inner + "┓"
        bot = "┗" + "━" * inner + "┛"
        lines = [top] + \
                ["┃" + box_content.center(inner) + "┃"
                    for box_content in box_contents] + \
                [bot]
        return lines, len(top)

    @staticmethod
    def _remove_stuff_2(text):
        return (text.replace("Port1In", "L-poke")
                    .replace("Port3In", "R-poke")
                    .replace("Port2In", "M-poke")
                    .replace("Port1Out", "L-out")
                    .replace("Port3Out", "R-out")
                    .replace("Port2Out", "M-out"))

    def _draw(self, levels, by_level, state_map):
        """Draw the state machine as text based on the levels and state info."""

        # Pre-draw boxes to get their widths for layout calculations.
        box_lines = {}
        box_widths = {}
        for name in (n for ns in by_level.values() for n in ns):
            s = state_map.get(name, {})
            lines, w = self._box(name, s.timer if hasattr(s, "timer") else "")
            box_lines[name] = lines
            box_widths[name] = w

        def lvl_w(lvl):
            ns = by_level[lvl]
            return sum(box_widths[n] for n in ns) + self.h_gap * (len(ns) - 1)

        # define pos of each box
        max_boxes_w = max(lvl_w(lvl) for lvl in by_level)
        cx = max_boxes_w // 2 + 4  # 4 chars left margin

        x_c = {}   # box center x
        y_t = {}   # box top y
        cur_y = 2 if self.title else 0
        for lvl in sorted(by_level.keys()):
            names = by_level[lvl]
            cur_x = cx - lvl_w(lvl) // 2
            for name in names:
                bw = box_widths[name]
                x_c[name] = cur_x + bw // 2
                y_t[name] = cur_y
                cur_x += bw + self.h_gap
            cur_y += max(len(box_lines[n]) for n in names) + self.arrow_h
        total_h = cur_y + 4

        # Aggregate edges combine multiple events
        adj_agg = defaultdict(list)
        skip_agg = defaultdict(list)

        for s in self.states:
            for event, targets in s.transitions.items():
                label = self._remove_stuff_2(event)
                for nxt in targets:
                    label_t = label
                    if len(targets) > 1:
                        label_t = label + "→" + \
                                  nxt.replace("_", " ").split()[0]
                    nl = levels.get(nxt, levels.get(s.name, 0))
                    if nl == levels.get(s.name, 0) + 1:
                        adj_agg[(s.name, nxt)].append(label_t)
                    elif nl > levels.get(s.name, 0) + 1 and nxt in x_c:
                        skip_agg[(s.name, nxt)].append(label_t)

        def _combine_labels(lbls):
            unique = list(dict.fromkeys(lbls))
            if len(unique) == 1:
                return unique[0]
            parts = [lbl.rsplit("→", 1) for lbl in unique]
            if all(len(p) == 2 and p[1] == parts[0][1] for p in parts):
                return "/".join(p[0] for p in parts)
            return "/".join(unique)

        adj_edges = [(s, n, _combine_labels(lbls), lbls)
                     for (s, n), lbls in adj_agg.items()]
        skip_edges = [(s, n, _combine_labels(lbls))
                      for (s, n), lbls in skip_agg.items()]

        if self.verbose:
            print(f"Adjacent edges: {adj_edges}")
            print(f"Skip edges: {skip_edges}")

        # Process skip edges
        max_label = max((len(label) for _, _, label in skip_edges), default=0)
        side_step = max_label + 4

        if self.skip_side == "nested":
            skip_edges.sort(key=lambda e:
                            levels.get(e[1], 0) - levels.get(e[0], 0))
            skip_info = [("right", i) for i in range(len(skip_edges))]
        elif self.skip_side == "alternate":
            skip_info = [("right" if i % 2 == 0 else "left", i // 2)
                         for i in range(len(skip_edges))]
        elif self.skip_side == "right":
            skip_info = [("right", i) for i in range(len(skip_edges))]
        else:
            raise ValueError(f"Invalid skip_side: {self.skip_side}")

        n_right = sum(1 for s, _ in skip_info if s == "right")
        n_left = sum(1 for s, _ in skip_info if s == "left")

        right_x = max(x_c[n] - box_widths[n]//2 + box_widths[n] - 1
                      for n in x_c)
        left_x = min(x_c[n] - box_widths[n]//2 for n in x_c)
        right_base = right_x + 9
        left_base = left_x - 9 - (n_left - 1) * side_step if n_left else 0

        # Widen left margin if left-side columns are needed
        left_margin_needed = max(4, left_x - left_base + 2) if n_left else 4
        if left_margin_needed > (cx - max_boxes_w // 2):
            shift = left_margin_needed - (cx - max_boxes_w // 2)
            cx += shift
            for n in x_c:
                x_c[n] += shift
            right_x += shift
            left_x += shift
            right_base += shift
            left_base += shift

        right_span = n_right * side_step + 4 if n_right else 4
        canvas_w = right_base + right_span

        # utils to put chars on grid
        def put(x, y, ch):
            if 0 <= y < total_h and 0 <= x < canvas_w:
                grid[y][x] = ch

        def put_s(x, y, s, force=False):
            for i, c in enumerate(s):
                if 0 <= x + i < canvas_w:
                    if force or grid[y][x + i] == ' ':
                        grid[y][x + i] = c

        def vbar(x, y1, y2):
            for y in range(min(y1, y2), max(y1, y2) + 1):
                put(x, y, '│')

        def hbar(x1, y, x2):
            for x in range(min(x1, x2), max(x1, x2) + 1):
                ch = grid[y][x]
                if ch == '│':
                    put(x, y, '┼')
                elif ch == ' ':
                    put(x, y, '─')

        def skip_side_x(side, col_idx):
            return (right_base + col_idx * side_step if side == "right"
                    else left_base + col_idx * side_step)

        # Initialize canvas
        grid = [[' '] * canvas_w for _ in range(total_h)]

        if self.title:
            put_s(cx - len(self.title) // 2, 0, self.title, force=True)

        # Boxes
        for name, lines in box_lines.items():
            if self.verbose:
                print(f"Placing box for {name} "
                      f"at x={x_c[name]}, y={y_t[name]}")
            xl = x_c[name] - box_widths[name] // 2
            for i, line in enumerate(lines):
                put_s(xl, y_t[name] + i, line, force=True)

        # if self.verbose:
        #     print("Canvas after drawing boxes:")
        #     for row in grid:
        #         print(''.join(row))

        # skip edges vertical
        for (side, col_idx), (src, nxt, _) in zip(skip_info, skip_edges):
            sx_col = skip_side_x(side, col_idx)
            src_mid_y = y_t[src] + len(box_lines[src]) // 2
            tgt_mid_y = y_t[nxt] + len(box_lines[nxt]) // 2
            vbar(sx_col, src_mid_y + 1, tgt_mid_y - 1)

        # normal transitions arrows grouped by level
        by_sl = defaultdict(list)
        for e in adj_edges:
            by_sl[levels.get(e[0], 0)].append(e)

        for sl, edges in by_sl.items():
            srcs = list(dict.fromkeys(e[0] for e in edges))
            nxts = list(dict.fromkeys(e[1] for e in edges))

            if len(srcs) == 1 and len(nxts) == 1:
                # 1 to 1 straight arrow with label in the middle
                src, nxt, lbl, _ = edges[0]
                sx = x_c[src]
                sy_b = y_t[src] + len(box_lines[src])
                tx = x_c[nxt]
                ny_t = y_t[nxt]
                if sx == tx:
                    vbar(sx, sy_b, ny_t - 2)
                    put(sx, ny_t - 1, 'v')
                    put_s(sx + 1, (sy_b + ny_t) // 2, lbl)
                else:
                    mid_y = (sy_b + ny_t) // 2
                    vbar(sx, sy_b, mid_y - 1)
                    put(sx, mid_y, '└' if sx < tx else '┘')
                    hbar(min(sx, tx) + 1, mid_y, max(sx, tx) - 1)
                    put(tx, mid_y, '┐' if sx < tx else '┌')
                    vbar(tx, mid_y + 1, ny_t - 2)
                    put(tx, ny_t - 1, 'v')
                    put_s((sx + tx) // 2 - len(lbl) // 2, mid_y, lbl)

            elif len(srcs) == 1 and len(nxts) > 1:
                # 1 to x fork: vertical, then horizontal with branches
                src = srcs[0]
                sx = x_c[src]
                sy_b = y_t[src] + len(box_lines[src])
                fork_y = sy_b + 1
                vbar(sx, sy_b, fork_y)
                xs = sorted(x_c[n] for n in nxts)
                hbar(xs[0], fork_y, xs[-1])
                put(xs[0], fork_y, '┌')
                put(xs[-1], fork_y, '┐')
                put(sx, fork_y, '┴')
                all_same_lbl = len(set(lbl for _, _, lbl, _ in edges)) == 1
                for branch_i, (_, nxt, lbl, raw_lbls) in enumerate(
                        sorted(edges, key=lambda e: x_c[e[1]])):
                    tx = x_c[nxt]
                    ny_t = y_t[nxt]
                    if fork_y + 1 <= ny_t - 2:
                        vbar(tx, fork_y + 1, ny_t - 2)
                    put(tx, ny_t - 1, 'v')
                    if all_same_lbl and "→" not in lbl:
                        _evts = list(dict.fromkeys(lbl.rsplit("→", 1)[0]
                                                   if "→" in lbl else lbl
                                                   for lbl in raw_lbls))
                        # ugly and maybe not generalizable
                        _POKE_SIDE = {"L-poke": "L", "R-poke": "R"}
                        _OPP = {"L": "R", "R": "L"}
                        if set(_evts) <= set(_POKE_SIDE):
                            opp = branch_i % 2 == 1
                            pairs = [f"{_POKE_SIDE[e]}:{_OPP[_POKE_SIDE[e]]
                                                        if opp
                                                        else _POKE_SIDE[e]}"
                                     for e in _evts]
                            fork_lbl = ", ".join(pairs)
                        else:
                            fork_lbl = "/".join(("*" + e
                                                 if i == branch_i % len(_evts)
                                                 else e)
                                                for i, e in enumerate(_evts))
                    elif "→" not in lbl:
                        fork_lbl = f"{lbl}→{nxt}"
                    else:
                        fork_lbl = lbl
                    put_s(tx - len(fork_lbl) // 2, fork_y - 1, fork_lbl,
                          force=True)

            elif len(srcs) > 1 and len(nxts) == 1:
                # merge from x to 1: horizontal with branches, then vertical
                nxt = nxts[0]
                tx = x_c[nxt]
                ny_t = y_t[nxt]
                merge_y = ny_t - 2
                xs = sorted(x_c[s] for s in srcs)
                hbar(xs[0], merge_y, xs[-1])
                put(tx, merge_y, '┬')
                for src, _, lbl, __ in edges:
                    sx = x_c[src]
                    sy_b = y_t[src] + len(box_lines[src])
                    if sx == xs[0]:
                        put(sx, merge_y, '└')
                    elif sx == xs[-1]:
                        put(sx, merge_y, '┘')
                    else:
                        put(sx, merge_y, '┴')
                    vbar(sx, sy_b, merge_y - 1)
                    label_y = (sy_b + merge_y) // 2
                    put_s(sx + 1 if sx <= tx else sx - len(lbl) - 1,
                          label_y, lbl)
                if merge_y + 1 <= ny_t - 2:
                    vbar(tx, merge_y + 1, ny_t - 2)
                put(tx, ny_t - 1, 'v')

            else:
                # general case: multiple sources and targets? just do
                # vertical from each source, then horizontal to each target
                for src, nxt, lbl, _ in edges:
                    sx = x_c[src]
                    sy_b = y_t[src] + len(box_lines[src])
                    tx = x_c[nxt]
                    ny_t = y_t[nxt]
                    mid_y = (sy_b + ny_t) // 2
                    vbar(sx, sy_b, mid_y - 1)
                    put(sx, mid_y, '└' if sx < tx else '┘')
                    hbar(min(sx, tx) + 1, mid_y, max(sx, tx) - 1)
                    put(tx, mid_y, '┐' if sx < tx else '┌')
                    vbar(tx, mid_y + 1, ny_t - 2)
                    put(tx, ny_t - 1, 'v')
                    put_s(min(sx, tx) + 1, mid_y, lbl)

        # if self.verbose:
        #     print("Canvas after drawing arrows:")
        #     for row in grid:
        #         print(''.join(row))

        # skip edges
        for (side, col_idx), (src, nxt, lbl) in zip(skip_info, skip_edges):
            side_x = skip_side_x(side, col_idx)
            src_mid_y = y_t[src] + len(box_lines[src]) // 2
            tgt_mid_y = y_t[nxt] + len(box_lines[nxt]) // 2
            left_x_src = x_c[src] - box_widths[src] // 2
            right_x_src = x_c[src] - box_widths[src] // 2 + box_widths[src] - 1
            left_x_nxt = x_c[nxt] - box_widths[nxt] // 2
            right_x_nxt = x_c[nxt] - box_widths[nxt] // 2 + box_widths[nxt] - 1

            lbl_str = f"[{lbl}]"
            if side == "right":
                # exit right
                hbar(right_x_src + 1, src_mid_y, side_x - 1)
                put(side_x, src_mid_y, '┐')
                mid_x = (right_x_src + 1 + side_x) // 2
                put_s(mid_x - len(lbl_str) // 2, src_mid_y - 1, lbl_str)
                # enter right
                put(side_x, tgt_mid_y, '┘')
                hbar(right_x_nxt + 2, tgt_mid_y, side_x - 1)
                put(right_x_nxt + 1, tgt_mid_y, '<')
            else:
                # exit left
                hbar(side_x + 1, src_mid_y, left_x_src - 1)
                put(side_x, src_mid_y, '┌')
                mid_x = (side_x + 1 + left_x_src - 1) // 2
                put_s(mid_x - len(lbl_str) // 2, src_mid_y - 1, lbl_str)
                # enterleft
                put(side_x, tgt_mid_y, '└')
                hbar(side_x + 1, tgt_mid_y, left_x_nxt - 2)
                put(left_x_nxt - 1, tgt_mid_y, '>')

        # final render grid
        out_lines = [''.join(row).rstrip() for row in grid]
        while out_lines and not out_lines[-1].strip():
            out_lines.pop()
        final = "\n".join(out_lines)

        if self.verbose:
            print(final)

        return final


if __name__ == "__main__":
    path = "code/tower_task.py"
    print(DrawStateMachine(path))
    print(DrawStateMachine("code/test_cate.py"))

    # In task file, do:

    # from path.to.draw_state_machine import DrawStateMachine as DSM

    # class MyTask(...):
    #  def __init__(self, ...):
    #    ...
    #    self.info = (f"This is my task, \n"
    #                 f"Here is the state machine\n{DSM(__file__)}")
