#!/usr/bin/env python
"""Simulated active-learning runs that use SimplyAL's own training loop.

Each run mirrors one SimplyAL project. The ActiveLearningManager from
backend/main.py trains an episode exactly as the web app does (train_episode,
which also builds the preview page) and then selects the next batch as
/continue-from-evaluation does. Labels of acquired images are revealed from a
fully labeled CSV (simulated annotator). After every episode the model is also
scored on a stratified test set that is never available for labeling, and the
server-side waits are timed.

Outputs in --out (runs append, so several calls can share one folder):
  results.csv        one row per episode
  summary.csv        test accuracy/F1 at budget checkpoints, area under the
                     learning curve, latency at the first and last episode
                     (mean and SD over seeds)
  plateau.csv        first budget after which the mean episode-to-episode gain
                     stays below --plateau-delta points for --plateau-k episodes
  curves_*.png/.pdf  learning curves (mean +/- SD over seeds)
  env.json           hardware and library versions

Copy this file to SimplyAL/backend/ and run, for example:
  python backend/simulate_al.py --csv data/fundus.csv --label-column annotation \\
      --models resnet18,resnet50,dinov2 --strategies least_confidence,random \\
      --seeds 0,1,2 --out runs/main

Smoke test without data or downloads:
  python backend/simulate_al.py --synthetic 64 --no-pretrained --models resnet18 \\
      --strategies least_confidence,random --seeds 0 --seed-size 16 \\
      --episode-size 8 --epochs 1 --out /tmp/smoke
"""
import argparse
import contextlib
import io
import json
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, f1_score

ERROR_MARKERS = ("Falling back to random sampling", "Error computing uncertainty",
                 "Error in get_next_batch")
RUN_KEYS = ["model", "strategy", "seed", "lr", "episode_size", "batch_size", "epochs"]
TIMED = {"train_epoch": "train_s", "validate_model_with_metrics": "validation_s",
         "get_evaluation_batch": "preview_s"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path, help="CSV with an image path and a label for every image")
    src.add_argument("--synthetic", type=int, metavar="N", help="use N synthetic images (smoke test)")
    p.add_argument("--path-column", default="file_path")
    p.add_argument("--label-column", default="annotation")
    p.add_argument("--models", default="resnet18,resnet50,dinov2")
    p.add_argument("--strategies", default="least_confidence,random",
                   help="any of least_confidence, margin, entropy, diversity, random")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--test-frac", type=float, default=0.2, help="stratified held-out test fraction")
    p.add_argument("--seed-size", type=int, default=16, help="labeled images at Episode 0")
    p.add_argument("--episode-size", type=int, default=16, help="images acquired per episode")
    p.add_argument("--batch-size", type=int, default=None,
                   help="training mini-batch (default: the episode size, as in the app)")
    p.add_argument("--epochs", type=int, default=5, help="training epochs per episode")
    p.add_argument("--lr", type=float, default=1e-5, help="the app default")
    p.add_argument("--val-split", type=float, default=0.2, help="in-app validation ratio")
    p.add_argument("--max-frac", type=float, default=0.5,
                   help="stop once this fraction of seed + pool images is labeled (0-1, exclusive)")
    p.add_argument("--budgets", default="3, 6, 9, 12, 15, 18, 21, 24, 27, 30, ", help="percent-labeled checkpoints for summary.csv")
    p.add_argument("--plateau-delta", type=float, default=1.0, help="percentage points")
    p.add_argument("--plateau-k", type=int, default=3, help="consecutive episodes")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--backend-dir", type=Path, default=Path(__file__).resolve().parent,
                   help="folder containing SimplyAL's main.py")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--no-pretrained", action="store_true",
                   help="random weights and no downloads (smoke tests only)")
    p.add_argument("--allow-unfixed", action="store_true",
                   help="run margin sampling even if the v1.0.1 defect is present")
    p.add_argument("--keep-checkpoints", action="store_true")
    p.add_argument("--verbose", action="store_true", help="show SimplyAL's own log output")
    args = p.parse_args()
    if not 0 < args.max_frac < 1:
        p.error("--max-frac must be between 0 and 1")
    return args


@contextlib.contextmanager
def captured(verbose):
    """Capture SimplyAL's prints; echo them in --verbose mode."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf
    if verbose:
        sys.stdout.write(buf.getvalue())


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def disable_pretrained(simplyal):
    """Smoke-test mode: random weights so nothing is downloaded."""
    import timm
    from torchvision.models import resnet18, resnet50
    simplyal.models.resnet18 = lambda pretrained=True, **kw: resnet18(weights=None)
    simplyal.models.resnet50 = lambda pretrained=True, **kw: resnet50(weights=None)
    simplyal.create_dinov2_model = lambda num_classes=2: timm.create_model(
        "vit_small_patch14_dinov2", pretrained=False, num_classes=num_classes, img_size=224)


def load_data(args, manager_cls):
    """Return (app_tensors, test_tensors, labels, class_names)."""
    if args.synthetic:
        gen = torch.Generator().manual_seed(0)
        xs, ys = [], []
        for i in range(args.synthetic):
            y = i % 2
            x = torch.randn(3, 224, 224, generator=gen) * 0.5
            x[:, :112, :] += 0.8 if y else -0.8
            xs.append(x)
            ys.append(y)
        return xs, xs, np.array(ys), ["class_0", "class_1"]

    df = pd.read_csv(args.csv)
    missing = [c for c in (args.path_column, args.label_column) if c not in df.columns]
    if missing:
        sys.exit(f"{args.csv} lacks column(s) {missing}; found {list(df.columns)}")
    df = df.dropna(subset=[args.path_column, args.label_column])
    names = df[args.label_column].astype(str).str.strip()
    classes = sorted(names.unique())
    labels = names.map({c: i for i, c in enumerate(classes)}).to_numpy()
    reference = manager_cls()          # SimplyAL's own transforms
    torch.manual_seed(0)               # fixes the random flip applied at load time
    app, test = [], []
    for n, raw in enumerate(df[args.path_column].astype(str), 1):
        path = Path(raw) if Path(raw).is_absolute() else args.csv.parent / raw
        img = Image.open(path).convert("RGB")
        app.append(reference.transform(img))       # as stored by the app's CSV upload
        test.append(reference.val_transform(img))  # deterministic, as in inference.py
        if n % 200 == 0:
            print(f"loaded {n}/{len(df)} images", flush=True)
    print(f"loaded {len(app)} images; classes {classes}", flush=True)
    return app, test, labels, classes


def make_split(labels, seed, test_frac, seed_size):
    """Stratified test set, stratified seed set, and the remaining pool."""
    rng = np.random.RandomState(seed)
    test, rest = [], []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_frac))
        test += idx[:n_test].tolist()
        rest += idx[n_test:].tolist()
    rest = np.array(rest)
    rng.shuffle(rest)
    seed_set = []
    for c in np.unique(labels):
        members = rest[labels[rest] == c]
        seed_set += members[: int(round(seed_size * len(members) / len(rest)))].tolist()
    chosen = set(seed_set)
    pool = [int(i) for i in rest if int(i) not in chosen]
    return sorted(test), seed_set, pool


class PhaseTimer:
    """Adds up wall-clock time spent in selected ActiveLearningManager methods."""

    def __init__(self, manager, device):
        self.sync = device.type == "cuda"
        self.totals = {}
        for method, key in TIMED.items():
            self._wrap(manager, method, key)

    def _wrap(self, manager, method, key):
        original = getattr(manager, method)

        def timed(*args, **kwargs):
            if self.sync:
                torch.cuda.synchronize()
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                if self.sync:
                    torch.cuda.synchronize()
                self.totals[key] = self.totals.get(key, 0.0) + time.perf_counter() - start

        setattr(manager, method, timed)

    def pop(self):
        totals, self.totals = self.totals, {}
        return {key: totals.get(key, 0.0) for key in TIMED.values()}


def evaluate(model, tensors, idx, labels, device, batch_size, n_classes):
    """Accuracy, weighted F1, macro F1 and balanced accuracy on the test images."""
    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(idx), batch_size):
            x = torch.stack([tensors[i] for i in idx[s:s + batch_size]]).to(device)
            preds.append(model(x).argmax(dim=1).cpu())
    pred, true = torch.cat(preds).numpy(), labels[idx]
    classes = list(range(n_classes))
    return (float((pred == true).mean()),
            float(f1_score(true, pred, labels=classes, average="weighted", zero_division=0)),
            float(f1_score(true, pred, labels=classes, average="macro", zero_division=0)),
            float(balanced_accuracy_score(true, pred)))


def rare_classes(labels, n=2):
    """The n smallest classes of the whole dataset (for counting rare-class labels)."""
    values, counts = np.unique(labels, return_counts=True)
    if len(values) <= 2 or counts.max() <= 1.5 * counts.min():
        return set()  # balanced or binary: not tracked
    return set(values[np.argsort(counts, kind="stable")[:n]].tolist())


def check_margin(simplyal, allow):
    probe = simplyal.ActiveLearningManager()
    with captured(False):
        ambiguous = probe._compute_uncertainty(torch.tensor([[0.55, 0.45]]), "margin")
        clear = probe._compute_uncertainty(torch.tensor([[0.95, 0.05]]), "margin")
    if ambiguous > clear:
        return
    msg = ("margin sampling in this SimplyAL version does not rank ambiguous images first "
           f"(0.55/0.45 -> {ambiguous}, 0.95/0.05 -> {clear}); apply the v1.1 fixes "
           "or pass --allow-unfixed")
    if not allow:
        sys.exit(msg)
    print("WARNING:", msg, flush=True)


def remove_checkpoints(folder):
    if folder and Path(folder).exists():
        for pattern in ("*.pt", "*.pth"):
            for f in Path(folder).rglob(pattern):
                with contextlib.suppress(OSError):
                    f.unlink()


def run_one(simplyal, args, data, split, key, device, write_row):
    app, test, labels, classes = data
    test_idx, seed_idx, pool_idx = split
    set_seed(key["seed"])
    picker = np.random.RandomState(10_000 + key["seed"])
    m = simplyal.ActiveLearningManager()
    m.device = device
    with captured(args.verbose):
        m.initialize_project(
            f"sim_{key['model']}_{key['strategy']}_s{key['seed']}", key["model"], len(classes),
            config={"learning_rate": key["lr"], "sampling_strategy": key["strategy"],
                    "batch_size": key["batch_size"], "epochs": key["epochs"]})
    m.config["val_split"] = args.val_split
    m.config["batch_size"] = key["episode_size"]  # what /get-batch stores
    for i in seed_idx:
        m.labeled_data[i] = (app[i], int(labels[i]))
    for i in pool_idx:
        m.unlabeled_data[i] = app[i]
    universe = len(seed_idx) + len(pool_idx)
    rare = rare_classes(labels)
    timer = PhaseTimer(m, device)
    tag = f"{key['model']} | {key['strategy']} | seed {key['seed']}"
    try:
        while m.unlabeled_data:
            start = time.perf_counter()
            with captured(args.verbose):
                result = m.train_episode(key["epochs"], key["batch_size"], key["lr"])
            wait1 = time.perf_counter() - start
            phases = timer.pop()
            n_labeled = len(m.labeled_data) + len(m.validation_data)
            done = n_labeled / universe >= args.max_frac
            batch_ids, wait2, selection_error = [], float("nan"), False
            if not done:
                k = min(key["episode_size"], len(m.unlabeled_data))
                start = time.perf_counter()
                if key["strategy"] == "random":
                    batch_ids = picker.choice(list(m.unlabeled_data), size=k, replace=False).tolist()
                else:
                    with captured(args.verbose) as log:
                        batch = m.get_next_batch(key["strategy"], k)
                    batch_ids = [b["image_id"] for b in batch]
                    selection_error = any(s in log.getvalue() for s in ERROR_MARKERS)
                wait2 = time.perf_counter() - start
                if selection_error:
                    print(f"WARNING [{tag}]: SimplyAL reported a selection error; this batch was "
                          "not chosen by the requested strategy", flush=True)
            acc, f1, macro_f1, bal_acc = evaluate(m.model, test, test_idx, labels, device,
                                                  args.eval_batch_size, len(classes))
            known = list(m.labeled_data) + list(m.validation_data)
            n_rare = int(sum(int(labels[i]) in rare for i in known)) if rare else np.nan
            metrics = result["metrics"]
            row = dict(key, episode=metrics["episode"], n_labeled=n_labeled,
                       pct_labeled=100.0 * n_labeled / universe,
                       n_train=len(m.labeled_data), n_val=len(m.validation_data),
                       test_acc=acc, test_f1=f1, test_macro_f1=macro_f1, test_bal_acc=bal_acc,
                       n_rare_labeled=n_rare,
                       app_val_acc=metrics["best_val_acc"] / 100.0,
                       app_val_f1=metrics["f1_score"] / 100.0,
                       wait1_s=wait1, wait2_s=wait2, selection_error=selection_error, **phases)
            write_row(row)
            print(f"[{tag}] episode {row['episode']}: {row['pct_labeled']:.1f}% labeled, "
                  f"test acc {acc:.3f}, F1 {f1:.3f}, macro F1 {macro_f1:.3f}, "
                  + (f"rare labeled {n_rare}, " if rare else "")
                  + f"wait 1 {wait1:.1f} s, wait 2 {wait2:.1f} s",
                  flush=True)
            if done:
                break
            with captured(args.verbose):
                for i in batch_ids:
                    m.submit_label(int(i), int(labels[i]))
            if not args.keep_checkpoints:
                remove_checkpoints(m.output_dir)
    finally:
        if not args.keep_checkpoints and m.output_dir:
            shutil.rmtree(m.output_dir, ignore_errors=True)
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()


def drop_partial(results, key):
    """Remove rows of an unfinished run so a restart does not duplicate them."""
    if not results.exists():
        return
    df = pd.read_csv(results)
    mask = np.ones(len(df), dtype=bool)
    for k, v in key.items():
        col = df[k]
        mask &= (col.astype(str) == v).to_numpy() if isinstance(v, str) else \
            np.isclose(col.astype(float), float(v))
    if mask.any():
        df[~mask].to_csv(results, index=False)


def summarize(out_dir, budgets):
    df = pd.read_csv(out_dir / "results.csv")
    group_cols = [c for c in RUN_KEYS if c != "seed"]
    rows = []
    for values, group in df.groupby(group_cols):
        per_seed = []
        for _, s in group.groupby("seed"):
            s = s.sort_values("episode")
            rec = {}
            for b in budgets:
                hit = s[s.pct_labeled >= b - 1e-9]
                rec[f"acc_at_{b}"] = hit.test_acc.iloc[0] if len(hit) else np.nan
                rec[f"f1_at_{b}"] = hit.test_f1.iloc[0] if len(hit) else np.nan
                for col, name in (("test_macro_f1", "macro_f1"), ("test_bal_acc", "bal_acc"),
                                  ("n_rare_labeled", "rare")):
                    if col in s.columns:
                        rec[f"{name}_at_{b}"] = hit[col].iloc[0] if len(hit) else np.nan
            x, y = s.pct_labeled.to_numpy(), s.test_acc.to_numpy()
            rec["alc_acc"] = (float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2) / (x[-1] - x[0]))
                              if len(x) > 1 else np.nan)
            if "test_macro_f1" in s.columns and len(x) > 1:
                ym = s.test_macro_f1.to_numpy()
                rec["alc_macro_f1"] = float(np.sum(np.diff(x) * (ym[1:] + ym[:-1]) / 2) / (x[-1] - x[0]))
            selected = s[s.wait2_s.notna()]
            last = selected.iloc[-1] if len(selected) else s.iloc[-1]
            for name, r in (("first", s.iloc[0]), ("last", last)):
                rec[f"{name}_n_labeled"] = r.n_labeled
                for col in ("train_s", "validation_s", "preview_s", "wait1_s", "wait2_s"):
                    rec[f"{name}_{col}"] = r[col]
            per_seed.append(rec)
        ps = pd.DataFrame(per_seed)
        row = dict(zip(group_cols, values), n_seeds=len(ps))
        for col in ps.columns:
            row[f"{col}_mean"] = ps[col].mean()
            row[f"{col}_sd"] = ps[col].std(ddof=1) if len(ps) > 1 else np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)


def plateau(out_dir, delta, k):
    """Plateau as defined in Section 3.3, averaged over backbones and seeds."""
    df = pd.read_csv(out_dir / "results.csv")
    group_cols = ["strategy", "lr", "episode_size", "batch_size", "epochs"]
    rows = []
    for values, group in df.groupby(group_cols):
        curve = group.groupby("episode")[["pct_labeled", "test_acc", "test_f1"]].mean().sort_index()
        gain = curve[["test_acc", "test_f1"]].diff() * 100
        flat = ((gain.test_acc < delta) & (gain.test_f1 < delta)).to_numpy()
        point = np.nan
        for i in range(1, len(curve) - k + 1):
            if flat[i:i + k].all():
                point = curve.pct_labeled.iloc[i - 1]
                break
        rows.append(dict(zip(group_cols, values), models=",".join(sorted(group.model.unique())),
                         seeds=group.seed.nunique(), plateau_pct_labeled=point))
    pd.DataFrame(rows).to_csv(out_dir / "plateau.csv", index=False)


def plot_curves(out_dir, setting):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_csv(out_dir / "results.csv")
    for col, value in setting.items():
        df = df[np.isclose(df[col].astype(float), float(value))]
    if df.empty:
        return
    plateau_x = None
    if (out_dir / "plateau.csv").exists():
        p = pd.read_csv(out_dir / "plateau.csv")
        for col, value in setting.items():
            p = p[np.isclose(p[col].astype(float), float(value))]
        p = p[p.strategy == "least_confidence"]
        if len(p) and pd.notna(p.plateau_pct_labeled.iloc[0]):
            plateau_x = float(p.plateau_pct_labeled.iloc[0])
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {m: palette[i % len(palette)] for i, m in enumerate(sorted(df.model.unique()))}
    styles = {"least_confidence": "-", "random": "--"}
    second = (("test_macro_f1", "Macro F1-score")
              if "test_macro_f1" in df.columns and df.test_macro_f1.notna().any()
              and df.get("n_rare_labeled", pd.Series(dtype=float)).notna().any()
              else ("test_f1", "Weighted F1-score"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, label in zip(axes, ("test_acc", second[0]), ("Test accuracy", second[1])):
        for (model, strategy), g in df.groupby(["model", "strategy"]):
            c = g.groupby("episode").agg(x=("pct_labeled", "mean"), m=(metric, "mean"), s=(metric, "std"))
            sd = c.s.fillna(0)
            ax.plot(c.x, c.m, styles.get(strategy, ":"), color=colors[model], label=f"{model}, {strategy}")
            ax.fill_between(c.x, c.m - sd, c.m + sd, color=colors[model], alpha=0.15)
        if plateau_x is not None:
            ax.axvline(plateau_x, color="red", linestyle="--", linewidth=1, label="Plateau (least confidence)")
        ax.set_xlabel("Non-test images labeled (%)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    stem = "curves_lr{lr:g}_ep{episode_size}_bs{batch_size}_e{epochs}".format(**setting)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=200)
    plt.close(fig)


def env_info(device, n_images):
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(), "cpu_count": os.cpu_count(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "device": str(device), "n_images": n_images}
    for name in ("torchvision", "timm", "sklearn", "numpy", "pandas"):
        with contextlib.suppress(Exception):
            info[name] = __import__(name).__version__
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info["gpu"] = props.name
        info["gpu_memory_gb"] = round(props.total_memory / 1e9, 1)
    with contextlib.suppress(Exception):
        info["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    with contextlib.suppress(Exception):
        for line in open("/proc/cpuinfo"):
            if line.lower().startswith("model name"):
                info["cpu_model"] = line.split(":", 1)[1].strip()
                break
    with contextlib.suppress(Exception):
        for line in open("/etc/os-release"):
            if line.startswith("PRETTY_NAME="):
                info["os"] = line.split("=", 1)[1].strip().strip('"')
                break
    return info


def cli():
    args = parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.csv:
        args.csv = args.csv.resolve()
    sys.path.insert(0, str(args.backend_dir.resolve()))
    work = out / "work"
    work.mkdir(exist_ok=True)
    os.chdir(work)  # SimplyAL writes output/<project>/ relative to the working directory
    with captured(args.verbose):
        import main as simplyal  # SimplyAL backend module
    if args.no_pretrained:
        disable_pretrained(simplyal)
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    models = [s.strip() for s in args.models.split(",") if s.strip()]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    if "margin" in strategies:
        check_margin(simplyal, args.allow_unfixed)

    data = load_data(args, simplyal.ActiveLearningManager)
    (out / "env.json").write_text(json.dumps(env_info(device, len(data[2])), indent=2))
    results, done_file = out / "results.csv", out / "runs_done.jsonl"
    done = set(done_file.read_text().splitlines()) if done_file.exists() else set()

    def write_row(row):
        pd.DataFrame([row]).to_csv(results, mode="a", header=not results.exists(), index=False)

    batch_size = args.batch_size or args.episode_size
    for seed in seeds:
        split = make_split(data[2], seed, args.test_frac, args.seed_size)
        for model_name in models:
            for strategy in strategies:
                key = dict(model=model_name, strategy=strategy, seed=seed, lr=args.lr,
                           episode_size=args.episode_size, batch_size=batch_size, epochs=args.epochs)
                key_text = json.dumps(key, sort_keys=True)
                if key_text in done:
                    print(f"skipping finished run {key_text}", flush=True)
                    continue
                drop_partial(results, key)
                run_one(simplyal, args, data, split, key, device, write_row)
                with open(done_file, "a") as fh:
                    fh.write(key_text + "\n")
    if results.exists():
        summarize(out, budgets)
        plateau(out, args.plateau_delta, args.plateau_k)
        plot_curves(out, dict(lr=args.lr, episode_size=args.episode_size,
                              batch_size=batch_size, epochs=args.epochs))
        print(f"wrote results.csv, summary.csv, plateau.csv, env.json and curves to {out}", flush=True)


if __name__ == "__main__":
    cli()
