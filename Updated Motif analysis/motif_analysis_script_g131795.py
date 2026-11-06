import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.stats import spearmanr

REFERENCE_MEME = 'reference_pwms.meme'
CANDIDATE_TSV = 'g131795_ranked_seq.tsv'

PWM_WIDTH_DEFAULT = 6  # all four reference motifs are 6bp wide
CANONICAL_SPACER_RANGE = (15, 19)  # typical sigma70 -35-to--10 spacer, bp
SD_PURINE_FRACTION_THRESHOLD = 0.75
SD_CORE_REGEX = re.compile(r'GGAGG|AGGAG')


def parse_meme_file(filepath):
    """Parses the MEME file to extract probability matrices."""
    motifs = {}
    current_motif = None
    matrix = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('MOTIF'):
                if current_motif:
                    motifs[current_motif] = np.array(matrix)
                current_motif = line.split()[1]
                matrix = []
            elif re.match(r'^[0-9.]+\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+$', line):
                matrix.append([float(x) for x in line.split()])

        if current_motif:
            motifs[current_motif] = np.array(matrix)

    return motifs


def organism_tag(motif_name):
    """First underscore-delimited token of a motif name, used as a short label (Ecoli / Bsubtilis)."""
    return motif_name.split('_')[0]


def split_motifs_by_element(motifs_dict):
    """Splits motifs into -10 and -35 groups based on naming convention (_m10_ / _m35_)."""
    m10_motifs = {name: pwm for name, pwm in motifs_dict.items() if '_m10_' in name}
    m35_motifs = {name: pwm for name, pwm in motifs_dict.items() if '_m35_' in name}
    return m10_motifs, m35_motifs


def score_sequence(sequence, pwm, bg_freq=0.25):
    """Calculates log-odds score of a fixed-width sequence against a PWM."""
    nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    score = 0.0
    epsilon = 1e-10

    for i, nuc in enumerate(sequence.upper()):
        if nuc in nuc_to_idx and i < len(pwm):
            prob = pwm[i][nuc_to_idx[nuc]]
            score += np.log2((prob + epsilon) / bg_freq)
    return score


def max_achievable_score(pwm, bg_freq=0.25):
    """The highest log-odds score any sequence could get against this PWM (picks the best base per position)."""
    epsilon = 1e-10
    return sum(np.log2((row.max() + epsilon) / bg_freq) for row in pwm)


def best_pwm_match(sequence, pwm, bg_freq=0.25):
    """
    Slides a PWM-width window across `sequence` and returns the best-scoring
    position. Needed because minus10_seq / minus35_seq are wider context
    windows (mostly 14bp), not pre-cut hexamers.
    Returns (best_score, best_start_index, best_window_sequence, n_windows_tried).
    """
    if not isinstance(sequence, str):
        return np.nan, None, None, 0

    w = len(pwm)
    n_windows = len(sequence) - w + 1
    if n_windows < 1:
        return np.nan, None, None, 0

    best_score = -np.inf
    best_start = None
    for start in range(n_windows):
        window = sequence[start:start + w]
        s = score_sequence(window, pwm, bg_freq)
        if s > best_score:
            best_score = s
            best_start = start

    best_window = sequence[best_start:best_start + w] if best_start is not None else None
    return best_score, best_start, best_window, n_windows


def best_match_in_group(sequence, motif_group, bg_freq=0.25):
    """Best score/motif/window across every PWM in a group. Returns dict of results."""
    best_score = -np.inf
    best_name, best_start, best_window = None, None, None
    n_windows = 0

    for motif_name, pwm in motif_group.items():
        score, start, window, nw = best_pwm_match(sequence, pwm, bg_freq)
        n_windows = max(n_windows, nw)
        if pd.notna(score) and score > best_score:
            best_score, best_name, best_start, best_window = score, motif_name, start, window

    if best_name is None:
        return {'motif': None, 'score': np.nan, 'window': None, 'start': None, 'n_windows': n_windows}
    return {'motif': best_name, 'score': best_score, 'window': best_window, 'start': best_start, 'n_windows': n_windows}


def is_likely_shine_dalgarno(sequence):
    """
    Heuristic flag for ribosome-binding-site contamination rather than a genuine
    AT-rich promoter box: SD sites are purine-rich and often contain a GGAGG/AGGAG
    core. Both conditions are approximate - this is a screening flag, not a proof.
    """
    if not isinstance(sequence, str) or len(sequence) == 0:
        return False
    seq = sequence.upper()
    purine_fraction = sum(c in 'AG' for c in seq) / len(seq)
    has_sd_core = bool(SD_CORE_REGEX.search(seq))
    return (purine_fraction >= SD_PURINE_FRACTION_THRESHOLD) or has_sd_core


def score_candidates(candidate_df, seq_col, assigned_group, opposite_group, label):
    """
    For each candidate sequence, scores it against its assigned motif group
    (e.g. minus10_seq vs m10 PWMs) AND the opposite group (minus10_seq vs m35
    PWMs), per organism, plus SD-contamination and swap flags.
    """
    rows = []
    for seq in candidate_df[seq_col]:
        assigned = best_match_in_group(seq, assigned_group)
        opposite = best_match_in_group(seq, opposite_group)

        row = {
            f'{label}_best_motif': assigned['motif'],
            f'{label}_best_score': assigned['score'],
            f'{label}_best_window': assigned['window'],
            f'{label}_best_start': assigned['start'],
            f'{label}_n_windows_tried': assigned['n_windows'],
            f'{label}_opposite_group_best_motif': opposite['motif'],
            f'{label}_opposite_group_best_score': opposite['score'],
        }

        # per-organism scores within the assigned group, for side-by-side comparison
        for motif_name, pwm in assigned_group.items():
            org = organism_tag(motif_name)
            score, start, window, _ = best_pwm_match(seq, pwm)
            row[f'{label}_score_{org}'] = score

        # flags
        if pd.notna(assigned['score']) and pd.notna(opposite['score']):
            row[f'{label}_flag_looks_like_opposite_box'] = opposite['score'] > assigned['score']
        else:
            row[f'{label}_flag_looks_like_opposite_box'] = False
        row[f'{label}_flag_possible_shine_dalgarno'] = is_likely_shine_dalgarno(seq)

        rows.append(row)

    return pd.DataFrame(rows, index=candidate_df.index)


def report_ceiling_effect(motif_group, scored_df, label):
    """
    Prints, per motif, the max achievable score and what fraction of candidates
    that were assigned to that motif are scoring at (or essentially at) the
    ceiling - i.e. how much of the score distribution is being flattened by
    the PWM's own strongly-conserved positions rather than reflecting real
    variation in the candidates.
    """
    print(f"\n--- {label} PWM ceiling-effect check ---")
    for motif_name, pwm in motif_group.items():
        ceiling = max_achievable_score(pwm)
        subset = scored_df[scored_df[f'{label}_best_motif'] == motif_name]
        if len(subset) == 0:
            print(f"  {motif_name}: max possible score = {ceiling:.2f} bits (no candidates assigned)")
            continue
        at_ceiling = np.isclose(subset[f'{label}_best_score'], ceiling, atol=0.05).sum()
        frac = at_ceiling / len(subset)
        print(f"  {motif_name}: max possible score = {ceiling:.2f} bits; "
              f"{at_ceiling}/{len(subset)} assigned candidates ({frac:.0%}) are at/near the ceiling")


def report_cross_checks(scored_df):
    """Spearman correlations between our re-derived scores and the file's own upstream metrics."""
    print("\n--- Cross-check against upstream pipeline metrics ---")
    pairs = [
        ('minus10_best_score', 'minus10_pvalue', 'expect negative correlation (higher score = lower p-value)'),
        ('minus10_best_score', 'peak_score', 'no strong expectation, informative either way'),
        ('minus10_best_score', 'priority_rank', 'expect negative correlation (better score = lower/better rank)'),
    ]
    for col_a, col_b, note in pairs:
        paired = scored_df[[col_a, col_b]].dropna()
        if len(paired) < 3:
            print(f"  {col_a} vs {col_b}: not enough overlapping non-null rows to test")
            continue
        rho, pval = spearmanr(paired[col_a], paired[col_b])
        print(f"  {col_a} vs {col_b}: Spearman rho = {rho:.3f} (p = {pval:.2g}, n = {len(paired)}) - {note}")


def plot_data_coverage(candidate_df):
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = pd.Series({
        'has_minus10 only': ((candidate_df['has_minus10']) & (~candidate_df['has_minus35'])).sum(),
        'has_minus35 only': ((~candidate_df['has_minus10']) & (candidate_df['has_minus35'])).sum(),
        'has both': ((candidate_df['has_minus10']) & (candidate_df['has_minus35'])).sum(),
        'has neither': ((~candidate_df['has_minus10']) & (~candidate_df['has_minus35'])).sum(),
    })
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax.set_title(f'Candidate Data Coverage (n={len(candidate_df)} total)')
    ax.set_ylabel('Number of candidates')
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values) * 0.01, str(v), ha='center')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig('data_coverage.png')
    plt.show()


def plot_reference_heatmaps(motifs_dict):
    nucleotides = ['A', 'C', 'G', 'T']
    n_motifs = len(motifs_dict)
    ncols = 2
    nrows = int(np.ceil(n_motifs / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))
    fig.suptitle('Reference Motif Position Weight Matrices (A, C, G, T)', fontsize=16)
    axes = np.array(axes).flatten()

    for ax, (motif_name, matrix) in zip(axes, motifs_dict.items()):
        sns.heatmap(matrix.T, annot=True, cmap='YlGnBu',
                    yticklabels=nucleotides, xticklabels=range(1, len(matrix) + 1), ax=ax)
        ax.set_title(motif_name)
        ax.set_xlabel('Position')

    for ax in axes[n_motifs:]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('reference_motifs_heatmap.png')
    plt.show()


def plot_score_distributions(scored_df, label):
    fig, ax = plt.subplots(figsize=(8, 6))
    col = f'{label}_best_score'
    motif_col = f'{label}_best_motif'
    sub = scored_df[[col, motif_col]].dropna()

    if len(sub) == 0:
        plt.close(fig)
        return

    # build n= labeled legend
    counts = sub[motif_col].value_counts()
    sub = sub.copy()
    sub['legend_label'] = sub[motif_col].map(lambda m: f'{m} (n={counts[m]})')

    sns.kdeplot(data=sub, x=col, hue='legend_label', fill=True, alpha=0.5, ax=ax)
    ax.set_title(f'Best {label} Match Score Distribution (n={len(sub)} scored candidates)')
    ax.set_xlabel('Log-Odds Score (Higher = Better Match)')
    plt.tight_layout()
    plt.savefig(f'{label}_score_distribution.png')
    plt.show()


if __name__ == "__main__":
    reference_motifs = parse_meme_file(REFERENCE_MEME)
    m10_motifs, m35_motifs = split_motifs_by_element(reference_motifs)

    candidate_data = pd.read_csv(CANDIDATE_TSV, sep='\t')

    m10_results = score_candidates(candidate_data, 'minus10_seq', m10_motifs, m35_motifs, 'minus10')
    m35_results = score_candidates(candidate_data, 'minus35_seq', m35_motifs, m10_motifs, 'minus35')

    scored_df = pd.concat([candidate_data, m10_results, m35_results], axis=1)

    # spacer sanity check against the file's own spacer_bp column
    lo, hi = CANONICAL_SPACER_RANGE

    def spacer_flag(x):
        if pd.isna(x):
            return np.nan
        return not (lo <= x <= hi)

    scored_df['flag_spacer_out_of_canonical_range'] = scored_df['spacer_bp'].apply(spacer_flag)

    scored_df.to_csv('g131795_pwm_scores.tsv', sep='\t', index=False)
    print(f"Wrote scored output for {len(scored_df)} candidates to g131795_pwm_scores.tsv")

    n_swap_10 = scored_df['minus10_flag_looks_like_opposite_box'].sum()
    n_swap_35 = scored_df['minus35_flag_looks_like_opposite_box'].sum()
    n_sd_10 = scored_df['minus10_flag_possible_shine_dalgarno'].sum()
    n_sd_35 = scored_df['minus35_flag_possible_shine_dalgarno'].sum()
    n_spacer_bad = scored_df['flag_spacer_out_of_canonical_range'].sum()
    print(f"\nFlagged: {n_swap_10} minus10_seq entries score better as -35 boxes; "
          f"{n_swap_35} minus35_seq entries score better as -10 boxes")
    print(f"Flagged: {n_sd_10} minus10_seq / {n_sd_35} minus35_seq entries look like Shine-Dalgarno sites, not promoter boxes")
    print(f"Flagged: {n_spacer_bad:.0f} candidates with spacer_bp outside the canonical {lo}-{hi}bp range")

    report_ceiling_effect(m10_motifs, scored_df, 'minus10')
    report_ceiling_effect(m35_motifs, scored_df, 'minus35')
    report_cross_checks(scored_df)

    plot_data_coverage(candidate_data)
    plot_reference_heatmaps(reference_motifs)
    plot_score_distributions(scored_df, 'minus10')
    plot_score_distributions(scored_df, 'minus35')
