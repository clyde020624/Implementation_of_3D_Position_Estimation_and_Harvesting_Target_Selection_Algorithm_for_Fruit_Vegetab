"""Create the GT-mask absolute nearest-vs-Full trade-off scatter.

The plot follows the original ``tradeoff_scatter.png`` format while using the
validated GT-mask median-depth, tie-aware disagreement population. Existing
research files and the original scatter are not modified.
"""

import csv

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from mask_depth_tradeoff import analyze as reconstruct_tradeoff


INPUT_CSV = "mask_depth_tradeoff_results.csv"
OUTPUT_PNG = "figure2_mask_depth_absolute_tradeoff.png"
OUTPUT_PDF = "figure2_mask_depth_absolute_tradeoff.pdf"

DEMO_SEQUENCE = "402"
DEMO_FRAME = "1600938539283987"

EXPECTED_USABLE = 364
EXPECTED_DISAGREEMENTS = 118
EXPECTED_MEDIAN_DEPTH = 11.5
EXPECTED_MEDIAN_VISIBILITY = 0.1834
EXPECTED_VISIBILITY_HIGHER = 113

NUMERIC_FIELDS_TO_COMPARE = (
    "nearest_min_mask_depth",
    "full_mask_depth",
    "nearest_set_max_visibility",
    "full_visibility",
    "delta_depth_mm",
    "delta_visibility",
)


def load_csv_rows():
    with open(INPUT_CSV, newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    required = {
        "sequence_id",
        "frame_id",
        "nearest_min_mask_depth",
        "full_mask_depth",
        "nearest_set_max_visibility",
        "full_visibility",
        "delta_depth_mm",
        "delta_visibility",
    }
    if not rows:
        raise RuntimeError(f"No rows found in {INPUT_CSV}")
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(f"Missing CSV columns: {sorted(missing)}")
    return rows


def validate_reconstruction(csv_rows):
    reconstructed_rows, usable_frames, _ = reconstruct_tradeoff()

    csv_by_key = {
        (row["sequence_id"], row["frame_id"]): row
        for row in csv_rows
    }
    reconstructed_by_key = {
        (row["sequence_id"], row["frame_id"]): row
        for row in reconstructed_rows
    }

    problems = []
    if usable_frames != EXPECTED_USABLE:
        problems.append(
            f"usable frame mismatch: {usable_frames} != {EXPECTED_USABLE}"
        )
    if len(reconstructed_rows) != EXPECTED_DISAGREEMENTS:
        problems.append(
            "reconstructed disagreement mismatch: "
            f"{len(reconstructed_rows)} != {EXPECTED_DISAGREEMENTS}"
        )
    if set(csv_by_key) != set(reconstructed_by_key):
        csv_only = sorted(set(csv_by_key) - set(reconstructed_by_key))
        reconstruction_only = sorted(
            set(reconstructed_by_key) - set(csv_by_key)
        )
        problems.append(f"CSV-only frames: {csv_only[:5]}")
        problems.append(
            f"reconstruction-only frames: {reconstruction_only[:5]}"
        )

    for key in sorted(set(csv_by_key) & set(reconstructed_by_key)):
        csv_row = csv_by_key[key]
        reconstructed = reconstructed_by_key[key]
        for field in NUMERIC_FIELDS_TO_COMPARE:
            csv_value = float(csv_row[field])
            reconstructed_value = float(reconstructed[field])
            if abs(csv_value - reconstructed_value) > 1e-9:
                problems.append(
                    f"{key} {field}: CSV={csv_value}, "
                    f"reconstructed={reconstructed_value}"
                )
                if len(problems) >= 10:
                    break
        if len(problems) >= 10:
            break

    if problems:
        print("[ERROR] CSV and GT-mask reconstruction do not agree:")
        for problem in problems:
            print(f"  - {problem}")
        raise RuntimeError("Aborting before figure generation")

    return usable_frames


def extract_plot_data(rows):
    nearest_depth = np.asarray(
        [float(row["nearest_min_mask_depth"]) for row in rows],
        dtype=float,
    )
    full_depth = np.asarray(
        [float(row["full_mask_depth"]) for row in rows],
        dtype=float,
    )
    nearest_visibility = np.asarray(
        [float(row["nearest_set_max_visibility"]) for row in rows],
        dtype=float,
    )
    full_visibility = np.asarray(
        [float(row["full_visibility"]) for row in rows],
        dtype=float,
    )
    demo_indices = [
        index
        for index, row in enumerate(rows)
        if row["sequence_id"] == DEMO_SEQUENCE
        and row["frame_id"] == DEMO_FRAME
    ]
    return (
        nearest_depth,
        full_depth,
        nearest_visibility,
        full_visibility,
        demo_indices,
    )


def validate_statistics(
    rows,
    nearest_depth,
    full_depth,
    nearest_visibility,
    full_visibility,
    demo_indices,
):
    delta_depth = full_depth - nearest_depth
    delta_visibility = full_visibility - nearest_visibility

    median_depth = float(np.median(delta_depth))
    median_visibility = float(np.median(delta_visibility))
    visibility_higher = int(np.count_nonzero(delta_visibility > 0))
    positive_depth = int(np.count_nonzero(delta_depth > 0))

    problems = []
    if len(rows) != EXPECTED_DISAGREEMENTS:
        problems.append(
            f"disagreement rows: {len(rows)} != {EXPECTED_DISAGREEMENTS}"
        )
    if abs(median_depth - EXPECTED_MEDIAN_DEPTH) > 1e-9:
        problems.append(
            f"median delta depth: {median_depth} != {EXPECTED_MEDIAN_DEPTH}"
        )
    if abs(median_visibility - EXPECTED_MEDIAN_VISIBILITY) > 5e-4:
        problems.append(
            "median delta visibility: "
            f"{median_visibility} != approximately {EXPECTED_MEDIAN_VISIBILITY}"
        )
    if visibility_higher != EXPECTED_VISIBILITY_HIGHER:
        problems.append(
            f"visibility-higher count: {visibility_higher} "
            f"!= {EXPECTED_VISIBILITY_HIGHER}"
        )
    if positive_depth != len(rows):
        bad_indices = np.flatnonzero(delta_depth <= 0).tolist()
        problems.append(
            f"delta_depth <= 0 in {len(bad_indices)} frames: {bad_indices[:5]}"
        )
    if len(demo_indices) != 1:
        problems.append(f"Demo frame count: {len(demo_indices)} != 1")
    elif not (
        abs(nearest_depth[demo_indices[0]] - 814.0) <= 1e-9
        and abs(full_depth[demo_indices[0]] - 827.0) <= 1e-9
        and abs(nearest_visibility[demo_indices[0]] - 0.6321) <= 5e-4
        and abs(full_visibility[demo_indices[0]] - 0.7894) <= 5e-4
    ):
        problems.append("Demo frame absolute coordinates do not match")

    if problems:
        print("[ERROR] Absolute trade-off validation failed:")
        for problem in problems:
            print(f"  - {problem}")
        raise RuntimeError("Aborting before figure generation")

    return (
        delta_depth,
        delta_visibility,
        median_depth,
        median_visibility,
        visibility_higher,
    )


def make_figure(
    nearest_depth,
    full_depth,
    nearest_visibility,
    full_visibility,
    demo_index,
):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axis = plt.subplots(figsize=(7.0, 5.5))
    axis.scatter(
        nearest_depth,
        nearest_visibility,
        s=34,
        alpha=0.58,
        marker="o",
        facecolors="none",
        edgecolors="#C0392B",
        linewidths=1.2,
        label="Nearest baseline",
        zorder=3,
    )
    axis.scatter(
        full_depth,
        full_visibility,
        s=34,
        alpha=0.58,
        marker="^",
        facecolors="none",
        edgecolors="#1F6F8B",
        linewidths=1.2,
        label="Proposed (Full)",
        zorder=3,
    )

    # Highlight both absolute coordinates of the selected paper demo.
    axis.scatter(
        [nearest_depth[demo_index]],
        [nearest_visibility[demo_index]],
        s=82,
        marker="o",
        facecolors="none",
        edgecolors="#8E241A",
        linewidths=2.2,
        zorder=5,
    )
    axis.scatter(
        [full_depth[demo_index]],
        [full_visibility[demo_index]],
        s=88,
        marker="^",
        facecolors="none",
        edgecolors="#124A5E",
        linewidths=2.2,
        zorder=5,
    )

    all_depth = np.concatenate([nearest_depth, full_depth])
    all_visibility = np.concatenate([nearest_visibility, full_visibility])
    x_padding = max(8.0, float(np.ptp(all_depth)) * 0.04)
    y_padding = max(0.025, float(np.ptp(all_visibility)) * 0.055)
    axis.set_xlim(
        float(all_depth.min()) - x_padding,
        float(all_depth.max()) + x_padding,
    )
    axis.set_ylim(
        max(0.0, float(all_visibility.min()) - y_padding),
        min(1.0, float(all_visibility.max()) + y_padding),
    )

    axis.set_xlabel("GT-mask median depth (mm)")
    axis.set_ylabel("Visibility")
    axis.set_title(
        f"Selected target: nearest vs proposed "
        f"(n={len(nearest_depth)} frames)"
    )
    axis.legend(frameon=False, loc="lower right")
    axis.grid(color="#D0D0D0", alpha=0.30, linewidth=0.6)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    figure.tight_layout(pad=0.8)
    figure.savefig(
        OUTPUT_PNG,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        OUTPUT_PDF,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def main():
    rows = load_csv_rows()
    usable_frames = validate_reconstruction(rows)
    (
        nearest_depth,
        full_depth,
        nearest_visibility,
        full_visibility,
        demo_indices,
    ) = extract_plot_data(rows)
    (
        _,
        _,
        median_depth,
        median_visibility,
        visibility_higher,
    ) = validate_statistics(
        rows,
        nearest_depth,
        full_depth,
        nearest_visibility,
        full_visibility,
        demo_indices,
    )

    make_figure(
        nearest_depth,
        full_depth,
        nearest_visibility,
        full_visibility,
        demo_indices[0],
    )

    print("[GT-mask absolute trade-off scatter]")
    print(f"usable frames                  : {usable_frames}")
    print(f"disagreement frames            : {len(rows)}")
    print(f"nearest representative points  : {len(nearest_depth)}")
    print(f"Full points                    : {len(full_depth)}")
    print(f"median delta depth             : {median_depth:.1f} mm")
    print(f"median delta visibility        : {median_visibility:.4f}")
    print(
        f"Full visibility higher         : {visibility_higher}/{len(rows)} "
        f"({visibility_higher / len(rows) * 100:.1f}%)"
    )
    print(f"PNG                            : {OUTPUT_PNG}")
    print(f"PDF                            : {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
