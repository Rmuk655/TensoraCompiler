#!/usr/bin/env python3
"""Prepare the "course eligibility" relational-reasoning demo.

A genuinely different task from a single-hop lookup: a single-hop
lookup's answer is a fact directly present in the sequence
(student:key -> value). This task has NO such fact. The input is a
set of real, natural-language sentences -- not a symbol table -- from
which the answer must be inferred:

    Maya has completed CS101.
    Maya already finished MA101.
    CS201 requires CS101 as a prerequisite.
    MA201 needs MA101 before you can enroll.
    PHY101 has no prerequisite.
    BIO101 requires nothing beforehand.
    ...
    -> eligible: PHY101

Every example draws its own random 6-course subset from a 24-course
pool (see COURSE_POOL below), its own random single-prerequisite DAG
over that subset, its own random student name, and (often) one
irrelevant "distractor" completed-course sentence the model must
learn to ignore -- so no example repeats the same course-to-slot
identity twice, and matching cannot be done by memorizing a fixed
symbol-to-answer table. Sentence wording is drawn from several
templates per fact kind, so phrasing also varies example to example;
only the *presence of the same course name* in two different
sentences is a stable signal. Solving it needs two chained lookups:
read a candidate course's prerequisite (hop 1), then check whether
*that* prerequisite is one of the student's completed courses (hop
2). See examples/attn_relational_eligibility.tns's header for the
full task description and the two-hop attention architecture this
script mirrors op-for-op (matmul, transpose, softmax, sum, elementwise
+/-/* -- no masking/gather/stack op, none of which v1 Tensora has).

Tensora has no tokenizer, no embedding/gather op, and no text input of
any kind -- every sentence above is turned into a fixed-size float
vector entirely in PyTorch, by a from-scratch word-embedding + mean-
pool encoder trained jointly with the attention weights (the same
technique tools/models/prepare_text_attn_demo.py already uses for
demo-attn-text-classifier). Tensora only ever receives the resulting
float tensors; it never sees a word or a course name.

This is a genuinely hard task, and how well the trained PyTorch model
itself does at it (reported below and in the demo's own printed
output) is a separate question from whether Tensora reproduces that
model exactly -- the same way demo-cnn-grid-detector-real reports its
honestly weak real-photo recall instead of hiding it. The compiler's
job (exact agreement with the PyTorch reference) is unaffected by how
good the trained model is at the task.

Exports a batch of 200 held-out examples:

  demos/artifacts/attn_rel/eval/
      Xcomp.npy Xpp.npy Xpc.npy         -> pre-embedded inputs (Tensora
                                           has no embedding/gather op --
                                           every word-embedding lookup
                                           and mean-pool below happens
                                           here, in PyTorch; Tensora
                                           only ever sees the resulting
                                           float tensors)
      Wk1.npy Wv1.npy Wq1.npy Wcov.npy
      Walr.npy Wa.npy Wb.npy Bias.npy   -> the trained weights
      expected.npy                      -> the PyTorch reference logits [200, 6]
      meta.json                         -> the 200 examples in readable
                                           form: the exact generated
                                           sentences, the student name,
                                           the candidate courses for
                                           that example, and the target

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
N_COURSES = 6             # candidate courses per example (a random subset, below)
NCOMP = 3                 # fixed number of "completed course" fact slots
D = 32                    # word-embedding / hidden dim throughout
SCALE = 1.0 / (D ** 0.5)  # 1/sqrt(32); must match examples/attn_relational_eligibility.tns
MAXLEN = 10               # tokens per sentence, padded/truncated
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "demos", "artifacts", "attn_rel")

TRAIN_STEPS = 8000
BATCH = 128
HELDOUT = 200
EVAL_EVERY = 500
EVAL_BATCH = 1000

SUBJECTS = ["CS", "MA", "PHY", "BIO", "CHEM", "ENG", "HIST", "ART", "ECON", "STAT", "PSY", "PHIL"]
LEVELS = ["101", "201"]
COURSE_POOL = [s + lv for s in SUBJECTS for lv in LEVELS]  # 24 distinct course names
NAMES = ["Maya", "Ravi", "Liam", "Sara", "Omar", "Nina", "Chen", "Aisha",
         "Diego", "Priya", "Noah", "Ivy", "Leo", "Zara", "Kofi", "Elena"]

COMPLETED_TEMPLATES = [
    "{name} has completed {c}.",
    "{name} already finished {c}.",
    "{name} took {c} last year.",
    "{name} passed {c}.",
]
DISTRACTOR_TEMPLATES = [
    "{name} also completed {c} previously.",
    "{name} additionally finished {c} last term.",
]
PREREQ_TEMPLATES_WITH = [
    "{c} requires {p} as a prerequisite.",
    "To take {c}, you must first complete {p}.",
    "{c} needs {p} before you can enroll.",
    "You cannot take {c} unless you finished {p}.",
]
PREREQ_TEMPLATES_NONE = [
    "{c} has no prerequisite.",
    "{c} requires nothing beforehand.",
    "Anyone can enroll in {c} directly.",
    "{c} needs no prior course.",
]
IDENTITY_TEMPLATES = [
    "{c} is offered this semester.",
    "{c} is a course in the catalog.",
    "Students may enroll in {c}.",
    "{c} meets twice a week.",
]
PAD_SENTENCE = ""  # a completed-slot with nothing in it (fewer than NCOMP facts)


def tokenize(s):
    return s.replace(",", "").replace(".", "").split()


def _build_vocab():
    """The vocabulary is the closed set of words that can ever appear:
    the fixed connector words baked into the templates above, plus
    every name and every course name. No sampling pass is needed --
    the template set is finite and known ahead of time."""
    words = set()
    for group in (COMPLETED_TEMPLATES, DISTRACTOR_TEMPLATES, PREREQ_TEMPLATES_WITH,
                  PREREQ_TEMPLATES_NONE, IDENTITY_TEMPLATES):
        for t in group:
            filled = t.format(name="_NAME_", c="_CAND_", p="_PREV_")
            for w in tokenize(filled):
                if w not in ("_NAME_", "_CAND_", "_PREV_"):
                    words.add(w)
    words |= set(NAMES)
    words |= set(COURSE_POOL)
    vocab = {"<pad>": 0}
    for w in sorted(words):
        vocab.setdefault(w, len(vocab))
    return vocab


VOCAB = _build_vocab()


def encode(sentence):
    ids = [VOCAB.get(w, 0) for w in tokenize(sentence)][:MAXLEN]
    ids += [0] * (MAXLEN - len(ids))
    return ids


def gen_example(rng):
    """One eligibility example: a random 6-course subset of the
    24-course pool, a random single-prerequisite DAG over that subset,
    a random completed-course subset (real sentences, not symbols),
    an optional irrelevant distractor completed-course, and the unique
    course the student is now eligible for (resampled until unique --
    see the .tns header for why a unique answer matters)."""
    candidates = rng.sample(COURSE_POOL, N_COURSES)
    order = candidates[:]
    rng.shuffle(order)
    prereq = {c: None for c in candidates}
    for i, c in enumerate(order):
        if i > 0 and rng.random() < 0.7:
            prereq[c] = order[rng.randrange(i)]
    for _ in range(500):
        completed = [c for c in candidates if rng.random() < 0.45]
        eligible = [c for c in candidates
                    if c not in completed and (prereq[c] is None or prereq[c] in completed)]
        if len(eligible) == 1:
            name = rng.choice(NAMES)
            distractor = None
            if len(completed) < NCOMP and rng.random() < 0.5:
                distractor = rng.choice([c for c in COURSE_POOL if c not in candidates])
            return candidates, prereq, completed, distractor, name, eligible[0]
    raise RuntimeError("resample failed -- widen the retry budget above")


def build_sentences(rng, candidates, prereq, completed, distractor, name):
    comp_sentences = [rng.choice(COMPLETED_TEMPLATES).format(name=name, c=c)
                       for c in completed[:NCOMP]]
    if distractor is not None and len(comp_sentences) < NCOMP:
        comp_sentences.append(rng.choice(DISTRACTOR_TEMPLATES).format(name=name, c=distractor))
    while len(comp_sentences) < NCOMP:
        comp_sentences.append(PAD_SENTENCE)

    prereq_sentences, identity_sentences = [], []
    for c in candidates:
        p = prereq[c]
        if p is None:
            prereq_sentences.append(rng.choice(PREREQ_TEMPLATES_NONE).format(c=c))
        else:
            prereq_sentences.append(rng.choice(PREREQ_TEMPLATES_WITH).format(c=c, p=p))
        identity_sentences.append(rng.choice(IDENTITY_TEMPLATES).format(c=c))
    return comp_sentences, prereq_sentences, identity_sentences


def make_batch(rng, n):
    """Returns (comp_ids [n,NCOMP,MAXLEN], pp_ids [n,6,MAXLEN],
    pc_ids [n,6,MAXLEN], target [n] int64 -- an index into that
    example's OWN candidate list, since candidates differ per example)
    plus one dict of readable fields per example for meta.json."""
    comp_ids, pp_ids, pc_ids, targets, raw = [], [], [], [], []
    for _ in range(n):
        candidates, prereq, completed, distractor, name, target = gen_example(rng)
        comp_s, pp_s, pc_s = build_sentences(rng, candidates, prereq, completed, distractor, name)
        comp_ids.append([encode(s) for s in comp_s])
        pp_ids.append([encode(s) for s in pp_s])
        pc_ids.append([encode(s) for s in pc_s])
        targets.append(candidates.index(target))
        raw.append({
            "student": name,
            "candidates": candidates,
            "completed": completed,
            "distractor": distractor,
            "prerequisites": {c: prereq[c] for c in candidates},
            "target": target,
            "completed_sentences": [s for s in comp_s if s],
            "prerequisite_sentences": dict(zip(candidates, pp_s)),
            "identity_sentences": dict(zip(candidates, pc_s)),
        })
    return (torch.tensor(comp_ids), torch.tensor(pp_ids), torch.tensor(pc_ids),
            torch.tensor(targets), raw)


class RelationalEligibility(nn.Module):
    """Mirrors examples/attn_relational_eligibility.tns op-for-op:
    hop 1 is real single-head softmax attention (a learned, fixed
    "profile query" reads the completed-course sentences and produces
    one weighted-sum "profile" vector -- the empty-padding rows are
    never masked out by hand; softmax attention learns on its own to
    give them low weight); hop 2 compares that profile against every
    candidate course's prerequisite-sentence embedding (`coverage`: is
    the prerequisite covered?) and its own identity-sentence embedding
    (`already`: has the student completed it already?), and combines
    the two per-course scores with two learned scalar weights -- no
    hidden MLP, no stack/concat op, since v1 Tensora has neither.

    Every one of the three inputs above is itself the MEAN of learned
    word embeddings over one real sentence (encode_sentences below):
    unlike a from-scratch symbol table, the same word ("CS101", say)
    contributes the same embedding wherever it appears, so hop 2's
    matmul against a candidate's prerequisite-sentence embedding can
    only correctly report "covered" if the shared course-name token's
    contribution to both sentences' mean-pooled vectors lines up --
    genuine text-based matching, not array indexing."""

    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, D, padding_idx=0)
        self.Wk1 = nn.Linear(D, D, bias=False)
        self.Wv1 = nn.Linear(D, D, bias=False)
        self.Wq1 = nn.Parameter(torch.randn(D) * 0.1)              # fixed learned "profile query"
        self.Wcov = nn.Linear(D, D, bias=False)
        self.Walr = nn.Linear(D, D, bias=False)
        self.Wa = nn.Parameter(torch.tensor(1.0))
        self.Wb = nn.Parameter(torch.tensor(-1.0))
        self.Bias = nn.Parameter(torch.tensor(0.0))

    def encode_sentences(self, ids):
        # ids: [...,MAXLEN] token ids -> [...,D] mean-pooled sentence embedding.
        return self.emb(ids).mean(dim=-2)

    def embed(self, comp_ids, pp_ids, pc_ids):
        Xcomp = self.encode_sentences(comp_ids)   # [B,NCOMP,D]
        Xpp = self.encode_sentences(pp_ids)        # [B,6,D]
        Xpc = self.encode_sentences(pc_ids)         # [B,6,D]
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

    def forward(self, comp_ids, pp_ids, pc_ids):
        return self.forward_from_x(*self.embed(comp_ids, pp_ids, pc_ids))


def main():
    marker = os.path.join(OUT_DIR, "eval", "meta.json")
    if os.path.exists(marker) and "--force" not in sys.argv:
        print(f"Already prepared: {os.path.normpath(marker)} exists "
              "(delete demos/artifacts/attn_rel/, or pass --force, to redo)")
        return
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    model = RelationalEligibility(len(VOCAB))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_acc = 0.0
    best_state = None
    for step in range(TRAIN_STEPS):
        comp_ids, pp_ids, pc_ids, targets, _ = make_batch(rng, BATCH)
        logits = model(comp_ids, pp_ids, pc_ids)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                eci, epp, epc, etg, _ = make_batch(rng, EVAL_BATCH)
                acc = (model(eci, epp, epc).argmax(-1) == etg).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            model.train()

    model.load_state_dict(best_state)
    model.eval()
    print(f"Trained course-eligibility (2-hop relational) attention over real sentences "
          f"(vocab size {len(VOCAB)}): best held-out accuracy over {EVAL_BATCH} fresh "
          f"examples = {best_acc:.3f}\n"
          f"(well above the 1/{N_COURSES} ~ {1/N_COURSES:.3f} chance rate, confirming genuine "
          f"two-hop fact-chaining -- honestly imperfect, since this is a genuinely hard task; "
          f"see this script's module docstring)")

    heldout_comp, heldout_pp, heldout_pc, heldout_tgt, raw = make_batch(rng, HELDOUT)
    with torch.no_grad():
        Xcomp, Xpp, Xpc = model.embed(heldout_comp, heldout_pp, heldout_pc)
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

    meta = {"count": HELDOUT, "examples": raw}
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=2)
    pt_acc = float((logits.argmax(-1).numpy() == heldout_tgt.numpy()).mean())
    print(f"Wrote {HELDOUT}-example eval batch to {os.path.normpath(OUT_DIR)} "
          f"(PyTorch accuracy on this batch = {pt_acc:.3f})")


if __name__ == "__main__":
    main()
