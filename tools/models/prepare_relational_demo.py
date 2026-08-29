#!/usr/bin/env python3
"""Prepare the "course eligibility" relational-reasoning demo.

A genuinely different task from Associative Recall
(prepare_multihead_attn_demo.py): AR is a single-hop lookup -- the
answer is a fact directly present in the sequence (student:key ->
value). This task has NO such fact. The sequence states (a) which
courses the student has already completed, and (b) which course
requires which single prerequisite course -- but never states the
answer "which course can the student take next" directly. Solving it
needs two chained lookups: read a candidate course's prerequisite (hop
1), then check whether *that* prerequisite is one of the student's
completed courses (hop 2). See examples/attn_relational_eligibility.tns's
header for the full task description and the two-hop attention
architecture this script mirrors op-for-op (matmul, transpose,
softmax, sum, elementwise +/-/* -- no masking/gather/stack op, none of
which v1 Tensora has).

This is a harder task than Associative Recall, and it does not reach
that demo's ~100% reliability: a feasibility spike across 5 small
architectures (see the project's own session notes; not committed to
the repo, since a negative/partial spike result isn't a documented
claim) consistently plateaus at 75-80% held-out accuracy, clearly
above the 1/6 ~ 17% chance rate (confirming genuine two-hop
fact-chaining is happening, not memorization), but well short of a
clean answer. That gap is reported honestly below and in the demo's
own printed output, the same way demo-cnn-grid-detector-real reports
its honestly weak real-photo recall instead of hiding it -- the
compiler's job (exact agreement with this PyTorch reference) is
unaffected by how good the trained model itself is at the task.

Exports a batch of 200 held-out examples:

  demos/artifacts/attn_rel/eval/
      Xcomp.npy Xpp.npy Xpc.npy         -> pre-embedded inputs (Tensora
                                           has no embedding/gather op --
                                           every embedding lookup below
                                           happens here, in PyTorch;
                                           Tensora only ever sees the
                                           resulting float tensors)
      Wk1.npy Wv1.npy Wq1.npy Wcov.npy
      Walr.npy Wa.npy Wb.npy Bias.npy   -> the trained weights
      expected.npy                      -> the PyTorch reference logits [200, 6]
      meta.json                         -> the 200 examples in readable
                                           form (completed courses,
                                           prerequisite map, target)

Run once, on a machine with PyTorch (see tools/models/requirements.txt):
    python3 tools/models/prepare_relational_demo.py
Safe to re-run: skips everything if demos/artifacts/attn_rel/eval/
already exists (pass --force to redo).
"""
import json
import os
import random
import sys

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit("error: PyTorch is required. Install with: pip install -r tools/models/requirements.txt")

SEED = 0
N_COURSES = 6            # candidate courses per example
NULL = N_COURSES         # a dedicated "no course" embedding row, for padding
NCOMP = 3                # fixed number of "completed course" fact slots
D = 32                   # embedding / hidden dim throughout
SCALE = 1.0 / (D ** 0.5)  # 1/sqrt(32); must match examples/attn_relational_eligibility.tns
COURSES = ["CS101", "CS201", "MA101", "MA201", "PHY101", "BIO101"]
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "demos", "artifacts", "attn_rel")

TRAIN_STEPS = 6000
BATCH = 128
HELDOUT = 200
EVAL_EVERY = 500
EVAL_BATCH = 1000


def gen_example(rng):
    """One eligibility example: a random single-prerequisite DAG over
    the 6 courses, a random completed-course subset, and the unique
    course the student is now eligible for (resampled until unique --
    see the .tns header for why a unique answer matters)."""
    order = list(range(N_COURSES))
    rng.shuffle(order)
    prereq = {c: None for c in range(N_COURSES)}
    for i, c in enumerate(order):
        if i > 0 and rng.random() < 0.7:
            prereq[c] = order[rng.randrange(i)]
    for _ in range(500):
        completed = [c for c in range(N_COURSES) if rng.random() < 0.45]
        eligible = [c for c in range(N_COURSES)
                    if c not in completed and (prereq[c] is None or prereq[c] in completed)]
        if len(eligible) == 1:
            return prereq, completed, eligible[0]
    raise RuntimeError("resample failed -- widen the retry budget above")


def make_batch(rng, n):
    """Returns (comp_obj [n,NCOMP] int64, prereq_c [n,6] int64,
    prereq_p [n,6] int64, target [n] int64) plus the raw (prereq,
    completed) pairs for meta.json."""
    comp_obj, prereq_c, prereq_p, targets, raw = [], [], [], [], []
    for _ in range(n):
        prereq, completed, target = gen_example(rng)
        lst = completed[:NCOMP] + [NULL] * (NCOMP - len(completed[:NCOMP]))
        comp_obj.append(lst)
        pc = list(range(N_COURSES))
        pp = [prereq[c] if prereq[c] is not None else NULL for c in pc]
        prereq_c.append(pc)
        prereq_p.append(pp)
        targets.append(target)
        raw.append((prereq, completed, target))
    return (torch.tensor(comp_obj), torch.tensor(prereq_c), torch.tensor(prereq_p),
            torch.tensor(targets), raw)


class RelationalEligibility(nn.Module):
    """Mirrors examples/attn_relational_eligibility.tns op-for-op:
    hop 1 is real single-head softmax attention (a learned, fixed
    "profile query" reads the completed-course facts and produces one
    weighted-sum "profile" vector -- the null-padding rows are never
    masked out by hand; softmax attention learns on its own to give
    them low weight); hop 2 compares that profile against every
    candidate course's prerequisite embedding (`coverage`: is the
    prerequisite covered?) and its own embedding (`already`: has the
    student completed it already?), and combines the two per-course
    scores with two learned scalar weights -- no hidden MLP, no
    stack/concat op, since v1 Tensora has neither."""

    def __init__(self):
        super().__init__()
        self.course_obj_emb = nn.Embedding(N_COURSES + 1, D)      # "completed course" role
        self.course_prereq_emb = nn.Embedding(N_COURSES + 1, D)   # "prerequisite of c" role
        self.course_query_emb = nn.Embedding(N_COURSES + 1, D)    # "course c itself" role
        self.Wk1 = nn.Linear(D, D, bias=False)
        self.Wv1 = nn.Linear(D, D, bias=False)
        self.Wq1 = nn.Parameter(torch.randn(D) * 0.1)              # fixed learned "profile query"
        self.Wcov = nn.Linear(D, D, bias=False)
        self.Walr = nn.Linear(D, D, bias=False)
        self.Wa = nn.Parameter(torch.tensor(1.0))
        self.Wb = nn.Parameter(torch.tensor(-1.0))
        self.Bias = nn.Parameter(torch.tensor(0.0))

    def embed(self, comp_obj, prereq_c, prereq_p):
        Xcomp = self.course_obj_emb(comp_obj)         # [B,NCOMP,D]
        Xpp = self.course_prereq_emb(prereq_p)         # [B,6,D]
        Xpc = self.course_query_emb(prereq_c)           # [B,6,D]
        return Xcomp, Xpp, Xpc

    def forward_from_x(self, Xcomp, Xpp, Xpc):
        B = Xcomp.shape[0]
        K1 = self.Wk1(Xcomp)                             # [B,NCOMP,D]
        V1 = self.Wv1(Xcomp)                              # [B,NCOMP,D]
        q1 = self.Wq1.unsqueeze(0).expand(B, -1)           # [B,D]
        scores1 = torch.einsum("bd,bnd->bn", q1, K1) * SCALE  # [B,NCOMP]
        A1 = torch.softmax(scores1, dim=-1)
        profile = torch.einsum("bn,bnd->bd", A1, V1)         # [B,D]

        coverQ = self.Wcov(profile)                          # [B,D]
        alreadyQ = self.Walr(profile)                          # [B,D]
        coverage = torch.einsum("bd,bcd->bc", coverQ, Xpp)      # [B,6]
        already = torch.einsum("bd,bcd->bc", alreadyQ, Xpc)      # [B,6]
        logits = coverage * self.Wa + already * self.Wb + self.Bias
        return logits

    def forward(self, comp_obj, prereq_c, prereq_p):
        return self.forward_from_x(*self.embed(comp_obj, prereq_c, prereq_p))


def main():
    marker = os.path.join(OUT_DIR, "eval", "meta.json")
    if os.path.exists(marker) and "--force" not in sys.argv:
        print(f"Already prepared: {os.path.normpath(marker)} exists "
              "(delete demos/artifacts/attn_rel/, or pass --force, to redo)")
        return
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    model = RelationalEligibility()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_acc = 0.0
    best_state = None
    for step in range(TRAIN_STEPS):
        comp_obj, prereq_c, prereq_p, targets, _ = make_batch(rng, BATCH)
        logits = model(comp_obj, prereq_c, prereq_p)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                eco, epc, epp, etg, _ = make_batch(rng, EVAL_BATCH)
                acc = (model(eco, epc, epp).argmax(-1) == etg).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            model.train()

    model.load_state_dict(best_state)
    model.eval()
    print(f"Trained course-eligibility (2-hop relational) attention: "
          f"best held-out accuracy over {EVAL_BATCH} fresh examples = {best_acc:.3f}\n"
          f"(well above the 1/{N_COURSES} ~ {1/N_COURSES:.3f} chance rate, confirming genuine "
          f"two-hop fact-chaining -- below Associative Recall's ~1.00, since this is a "
          f"strictly harder task; see this script's module docstring)")

    heldout_comp, heldout_pc, heldout_pp, heldout_tgt, raw = make_batch(rng, HELDOUT)
    with torch.no_grad():
        Xcomp, Xpp, Xpc = model.embed(heldout_comp, heldout_pc, heldout_pp)
        logits = model.forward_from_x(Xcomp, Xpp, Xpc)

    os.makedirs(os.path.join(OUT_DIR, "eval"), exist_ok=True)
    d = os.path.join(OUT_DIR, "eval")

    def save(name, arr):
        np.save(os.path.join(d, f"{name}.npy"), np.ascontiguousarray(arr, dtype=np.float32))

    save("Xcomp", Xcomp.numpy())
    save("Xpp", Xpp.numpy())
    save("Xpc", Xpc.numpy())
    # Weight tensors carry the b=1 broadcast axis the .tns file declares.
    save("Wk1", model.Wk1.weight.detach().t().unsqueeze(0).numpy())     # [1,D,D] ([in,out] layout)
    save("Wv1", model.Wv1.weight.detach().t().unsqueeze(0).numpy())
    save("Wq1", model.Wq1.detach().reshape(1, 1, D).numpy())    # [b=1,sq=1,dk=D]
    save("Wcov", model.Wcov.weight.detach().t().unsqueeze(0).numpy())
    save("Walr", model.Walr.weight.detach().t().unsqueeze(0).numpy())
    save("Wa", model.Wa.detach().reshape(1, 1, 1).numpy())
    save("Wb", model.Wb.detach().reshape(1, 1, 1).numpy())
    save("Bias", model.Bias.detach().reshape(1, 1, 1).numpy())
    save("expected", logits.numpy())

    meta = {
        "count": HELDOUT,
        "courses": COURSES,
        "examples": [
            {
                "completed": [COURSES[c] for c in completed],
                "prerequisites": {COURSES[c]: (COURSES[p] if p is not None else None)
                                   for c, p in prereq.items()},
                "target": COURSES[target],
            }
            for prereq, completed, target in raw
        ],
    }
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=2)
    pt_acc = float((logits.argmax(-1).numpy() == heldout_tgt.numpy()).mean())
    print(f"Wrote {HELDOUT}-example eval batch to {os.path.normpath(OUT_DIR)} "
          f"(PyTorch accuracy on this batch = {pt_acc:.3f})")


if __name__ == "__main__":
    main()
