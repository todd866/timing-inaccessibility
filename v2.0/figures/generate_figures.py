#!/usr/bin/env python3
"""
Timing Inaccessibility v2.0 - Figure Generation
================================================

Generates all figures for "Timing Inaccessibility and the Projection Bound"

Paper: Version 2.0 (2025)
Author: Ian Todd
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge
from matplotlib.patches import ConnectionPatch, Rectangle
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

# Set consistent style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight'
})

# Color palette
COLORS = {
    'primary': '#2A9D8F',      # Teal
    'secondary': '#E76F51',    # Coral
    'accent': '#E63946',       # Red
    'dark': '#264653',         # Dark blue
    'light': '#F4A261',        # Orange
    'gray': '#888888',
    'bg': '#F8F9FA'
}


def fig1_path_degeneracy():
    """
    Figure 1: Path Degeneracy Visualization

    Shows how one macro-outcome (e.g., "protein folded") corresponds to
    exponentially many micro-trajectories that are thermodynamically
    indistinguishable below the Landauer threshold.
    """
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # Panel A: Many paths, one outcome
    ax1 = fig.add_subplot(gs[0, 0])
    np.random.seed(42)

    # Start and end points
    start = np.array([0, 0.5])
    end = np.array([1, 0.5])

    # Generate multiple random paths
    n_paths = 50
    n_points = 100
    t = np.linspace(0, 1, n_points)

    for i in range(n_paths):
        # Random walk with drift toward end
        path_y = 0.5 + np.cumsum(np.random.randn(n_points) * 0.03)
        path_y = path_y - path_y[-1] + 0.5  # End at 0.5
        path_y[0] = 0.5  # Start at 0.5

        alpha = 0.3 if i > 0 else 0.8
        color = COLORS['primary'] if i > 0 else COLORS['accent']
        lw = 0.5 if i > 0 else 1.5
        ax1.plot(t, path_y, color=color, alpha=alpha, lw=lw)

    # Mark start and end
    ax1.scatter([0], [0.5], s=100, c=COLORS['dark'], zorder=5, marker='o')
    ax1.scatter([1], [0.5], s=100, c=COLORS['accent'], zorder=5, marker='s')
    ax1.text(0, 0.28, 'START\n(unfolded)', ha='center', fontsize=8)
    ax1.text(1, 0.28, 'END\n(folded)', ha='center', fontsize=8)

    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Conformational coordinate')
    ax1.set_title('A. Many Micro-Trajectories, One Macro-Outcome',
                  fontweight='bold', pad=10)
    ax1.text(0.5, 0.9, f'~10$^{{48}}$–10$^{{100}}$ paths', ha='center',
             fontsize=11, color=COLORS['accent'], fontweight='bold',
             transform=ax1.transAxes)

    # Panel B: Degeneracy scaling
    ax2 = fig.add_subplot(gs[0, 1])

    D_eff = np.array([10, 20, 50, 100, 200])
    tau_ratio = 200  # tau_c / delta_t
    kappa = 2

    log_omega = (D_eff / kappa) * np.log10(tau_ratio)

    bars = ax2.bar(range(len(D_eff)), log_omega, color=COLORS['primary'],
                   alpha=0.8, width=0.6)
    ax2.set_xticks(range(len(D_eff)))
    ax2.set_xticklabels([f'{d}' for d in D_eff])
    ax2.set_xlabel('Effective dimensionality $D_{eff}$')
    ax2.set_ylabel('$\\log_{10}\\Omega$ (path degeneracy)')
    ax2.set_title('B. Degeneracy Scales Exponentially with Dimension',
                  fontweight='bold', pad=10)

    # Add annotation - positioned to avoid legend
    ax2.annotate('$\\Omega \\sim \\exp\\left[\\frac{D_{eff}}{\\kappa}\\ln\\frac{\\tau_c}{\\Delta t}\\right]$',
                 xy=(0.55, 0.92), xycoords='axes fraction',
                 fontsize=11, ha='center',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=COLORS['gray']))

    # Highlight neural range
    ax2.axhspan(50, 100, alpha=0.15, color=COLORS['secondary'])
    ax2.text(0.3, 75, 'Neural\nrange', fontsize=8, ha='center', va='center',
             color=COLORS['secondary'], fontweight='bold', alpha=0.8)

    # Panel C: Sub-Landauer regime - horizontal bar chart style
    ax3 = fig.add_subplot(gs[1, 0])

    landauer = 2.87e-21  # kT ln 2 at 300K

    # Signals with energies - use horizontal bars for cleaner look
    signals = [
        ('Ephaptic coupling', 1e-23, COLORS['accent']),
        ('Weak synapse', 1e-22, COLORS['accent']),
        ('Molecular fluctuation', 2e-21, COLORS['accent']),
        ('Ion channel', 5e-20, COLORS['primary']),
        ('Action potential', 1e-13, COLORS['primary']),
    ]

    y_positions = np.arange(len(signals))
    energies = [s[1] for s in signals]
    colors = [s[2] for s in signals]
    labels = [s[0] for s in signals]

    ax3.barh(y_positions, energies, color=colors, alpha=0.7, height=0.6)
    ax3.axvline(x=landauer, color=COLORS['accent'], linestyle='--',
                linewidth=2, label='Landauer limit')
    ax3.axvspan(0, landauer, alpha=0.15, color=COLORS['accent'])

    ax3.set_xscale('log')
    ax3.set_xlim(1e-24, 1e-12)
    ax3.set_yticks(y_positions)
    ax3.set_yticklabels(labels, fontsize=8)
    ax3.set_xlabel('Energy (J)')
    ax3.set_title('C. Sub-Landauer Domain', fontweight='bold', pad=10)
    ax3.legend(loc='upper right', fontsize=8)

    # Add "timing inaccessible" annotation - positioned in empty space below bars
    ax3.text(1e-22, -0.7, 'Timing inaccessible', fontsize=8, ha='center',
             va='top', color=COLORS['accent'], fontweight='bold')
    ax3.set_ylim(-1.0, 4.8)

    # Panel D: Key insight text - condensed to avoid collision
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    ax4.text(0.5, 0.95, 'PATH DEGENERACY', ha='center', va='top',
             fontsize=14, fontweight='bold', transform=ax4.transAxes)

    # Condensed points - fewer lines, more spacing
    points = [
        'Below Landauer: stabilizing a reusable',
        'record requires $\\geq k_BT\\ln 2$ per order bit',
        '',
        'Exponentially many micro-trajectories',
        'map to same macro-outcome',
        '',
        'Digital: pay per bit | Continuous: pay at projection',
    ]

    for i, point in enumerate(points):
        ax4.text(0.08, 0.80 - i*0.09, point, ha='left', va='top',
                fontsize=9, transform=ax4.transAxes)

    # Key equation box - with more vertical space
    ax4.text(0.5, 0.12,
             '$E_{\\min}^{time} \\geq k_BT\\ln 2 \\cdot \\log_2(M!)$',
             ha='center', va='center', fontsize=10,
             transform=ax4.transAxes,
             bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['bg'],
                      edgecolor=COLORS['accent'], linewidth=2))
    ax4.text(0.5, 0.02, 'Temporal Registration Bound',
             ha='center', va='center', fontsize=8, style='italic',
             transform=ax4.transAxes)

    plt.savefig('fig1_path_degeneracy.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('fig1_path_degeneracy.png', dpi=150, bbox_inches='tight')
    print("Saved: fig1_path_degeneracy.pdf/png")
    plt.close()


def fig3_camera_engine():
    """
    Figure 3: Camera-Engine Duality

    Illustrates the three phases of the biological demon:
    1. Camera (sensing) - weak coupling to environment
    2. Engine (steering) - coordinated back-coupling
    3. Collapse (payment) - dimensional projection
    """
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)

    # Panel A: Camera phase
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')

    # Environment (left side)
    env_circle = Circle((0.2, 0.5), 0.15, fill=True,
                        facecolor=COLORS['light'], alpha=0.5,
                        edgecolor=COLORS['dark'], linewidth=2)
    ax1.add_patch(env_circle)
    ax1.text(0.2, 0.5, 'ENV', ha='center', va='center', fontweight='bold')

    # System (right side)
    sys_circle = Circle((0.7, 0.5), 0.2, fill=True,
                        facecolor=COLORS['primary'], alpha=0.3,
                        edgecolor=COLORS['primary'], linewidth=2)
    ax1.add_patch(sys_circle)
    ax1.text(0.7, 0.5, 'HIGH-D\nSYSTEM', ha='center', va='center',
             fontsize=9, fontweight='bold')

    # Multiple weak arrows (sub-Landauer coupling)
    for offset in [-0.1, -0.05, 0, 0.05, 0.1]:
        ax1.annotate('', xy=(0.5, 0.5 + offset), xytext=(0.35, 0.5 + offset),
                    arrowprops=dict(arrowstyle='->', color=COLORS['gray'],
                                   lw=1, alpha=0.6))

    ax1.text(0.42, 0.75, 'Weak coupling\n$E_{link} \\ll k_BT\\ln 2$',
             ha='center', fontsize=9, color=COLORS['gray'])
    ax1.set_title('A. CAMERA PHASE\n(Sensing)', fontweight='bold',
                  color=COLORS['primary'], pad=10)

    # Panel B: Engine phase
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    # Environment
    env_circle2 = Circle((0.3, 0.5), 0.15, fill=True,
                         facecolor=COLORS['light'], alpha=0.5,
                         edgecolor=COLORS['dark'], linewidth=2)
    ax2.add_patch(env_circle2)
    ax2.text(0.3, 0.5, 'ENV', ha='center', va='center', fontweight='bold')

    # System
    sys_circle2 = Circle((0.7, 0.5), 0.2, fill=True,
                         facecolor=COLORS['secondary'], alpha=0.3,
                         edgecolor=COLORS['secondary'], linewidth=2)
    ax2.add_patch(sys_circle2)
    ax2.text(0.7, 0.5, 'INTERNAL\nMAP', ha='center', va='center',
             fontsize=9, fontweight='bold')

    # Back-coupling arrows
    for offset in [-0.08, 0, 0.08]:
        ax2.annotate('', xy=(0.45, 0.5 + offset), xytext=(0.5, 0.5 + offset),
                    arrowprops=dict(arrowstyle='->', color=COLORS['secondary'],
                                   lw=1.5))

    ax2.text(0.5, 0.75, 'Coordinated\nback-coupling', ha='center',
             fontsize=9, color=COLORS['secondary'])
    ax2.set_title('B. ENGINE PHASE\n(Steering)', fontweight='bold',
                  color=COLORS['secondary'], pad=10)

    # Panel C: Collapse phase
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')

    # High-D system (before)
    sys_before = Circle((0.3, 0.5), 0.2, fill=True,
                        facecolor=COLORS['primary'], alpha=0.3,
                        edgecolor=COLORS['primary'], linewidth=2)
    ax3.add_patch(sys_before)
    ax3.text(0.3, 0.5, '$D_{eff}$\n~100', ha='center', va='center',
             fontsize=9, fontweight='bold')

    # Arrow
    ax3.annotate('', xy=(0.6, 0.5), xytext=(0.5, 0.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent'],
                               lw=3))

    # Low-D output (after)
    sys_after = Circle((0.75, 0.5), 0.08, fill=True,
                       facecolor=COLORS['accent'], alpha=0.8,
                       edgecolor=COLORS['accent'], linewidth=2)
    ax3.add_patch(sys_after)
    ax3.text(0.75, 0.5, "D'", ha='center', va='center',
             fontsize=9, fontweight='bold', color='white')

    ax3.text(0.5, 0.75, 'Dimensional\ncollapse', ha='center',
             fontsize=9, color=COLORS['accent'])
    ax3.text(0.5, 0.25, '$E_{collapse} \\geq k_BT_{eff}\\ln(N_{pre}/N_{post})$',
             ha='center', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor=COLORS['accent']))
    ax3.set_title('C. COLLAPSE PHASE\n(Payment)', fontweight='bold',
                  color=COLORS['accent'], pad=10)

    # Panel D: Timeline
    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 3)
    ax4.axis('off')

    # Timeline bar
    ax4.axhline(y=1.5, xmin=0.05, xmax=0.95, color=COLORS['dark'], lw=3)

    # Phases
    phases = [
        (1.5, 'CAMERA', COLORS['primary'], '0-100 ms\nWeak sensing'),
        (4, 'ENGINE', COLORS['secondary'], 'Concurrent\nSteering'),
        (7, 'COLLAPSE', COLORS['accent'], '~150 ms\nDecision'),
    ]

    for x, label, color, desc in phases:
        ax4.scatter([x], [1.5], s=200, c=color, zorder=5, edgecolor='white', lw=2)
        ax4.text(x, 2.2, label, ha='center', fontweight='bold', color=color, fontsize=11)
        ax4.text(x, 0.8, desc, ha='center', fontsize=9, color=COLORS['gray'])

    # Energy annotation
    ax4.annotate('No bits written\n(structural correlation only)',
                 xy=(2.75, 1.5), xytext=(2.75, 2.6),
                 ha='center', fontsize=9, color=COLORS['primary'],
                 arrowprops=dict(arrowstyle='->', color=COLORS['primary']))

    ax4.annotate('$\\sim D_{eff} \\cdot k_BT\\ln 2$\ndissipated here',
                 xy=(7, 1.5), xytext=(8.5, 2.5),
                 ha='center', fontsize=9, color=COLORS['accent'],
                 arrowprops=dict(arrowstyle='->', color=COLORS['accent']))

    ax4.set_title('D. Temporal Structure: Cost Appears Only at Collapse',
                  fontweight='bold', pad=20, fontsize=12)

    plt.savefig('fig3_camera_engine.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('fig3_camera_engine.png', dpi=150, bbox_inches='tight')
    print("Saved: fig3_camera_engine.pdf/png")
    plt.close()


def fig5_framework_timing():
    """
    Figure 5: Framework Dependence of Timing

    Shows how two observers with different temporal resolutions
    see different causal structures from the same underlying events.
    """
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.25)

    # Generate event data - use fixed well-spaced times to avoid label overlap
    n_events = 6
    event_times = np.array([5, 20, 40, 55, 75, 95])  # Well-spaced events
    event_labels = [chr(65 + i) for i in range(n_events)]  # A, B, C, D, E, F

    # Panel A: Fine resolution (Observer 1)
    ax1 = fig.add_subplot(gs[0, 0])
    delta_t_fine = 5  # 5 ms bins

    for i, (t, label) in enumerate(zip(event_times, event_labels)):
        ax1.scatter([t], [0.5], s=80, c=COLORS['primary'], zorder=5)
        ax1.text(t, 0.72, label, ha='center', fontweight='bold', fontsize=9)

    # Draw bin boundaries
    for x in np.arange(0, 105, delta_t_fine):
        ax1.axvline(x, color=COLORS['gray'], alpha=0.3, lw=0.5)

    ax1.set_xlim(-2, 102)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('Time (ms)')
    ax1.set_yticks([])
    ax1.set_title('A. Observer 1: $\\Delta t = 5$ ms\n(Fine Resolution)',
                  fontweight='bold', color=COLORS['primary'], pad=10)
    ax1.text(50, 0.18, 'Events are SEQUENTIAL\nA → B → C → D → E → F',
             ha='center', fontsize=8, color=COLORS['primary'],
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor=COLORS['primary']))

    # Panel B: Coarse resolution (Observer 2)
    ax2 = fig.add_subplot(gs[0, 1])
    delta_t_coarse = 25  # 25 ms bins

    for i, (t, label) in enumerate(zip(event_times, event_labels)):
        ax2.scatter([t], [0.5], s=80, c=COLORS['secondary'], zorder=5)
        ax2.text(t, 0.72, label, ha='center', fontweight='bold', fontsize=9)

    # Draw bin boundaries and shade
    for x in np.arange(0, 105, delta_t_coarse):
        ax2.axvline(x, color=COLORS['secondary'], alpha=0.5, lw=1.5)

    # Shade bins
    for i, x in enumerate(np.arange(0, 100, delta_t_coarse)):
        alpha = 0.1 if i % 2 == 0 else 0.2
        ax2.axvspan(x, x + delta_t_coarse, alpha=alpha, color=COLORS['secondary'])

    ax2.set_xlim(-2, 102)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Time (ms)')
    ax2.set_yticks([])
    ax2.set_title('B. Observer 2: $\\Delta t = 25$ ms\n(Coarse Resolution)',
                  fontweight='bold', color=COLORS['secondary'], pad=10)
    ax2.text(50, 0.18, 'Some events SIMULTANEOUS\n{A,B} → {C,D} → {E,F}',
             ha='center', fontsize=8, color=COLORS['secondary'],
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor=COLORS['secondary']))

    # Panel C: Causal implications
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)

    ax3.text(0.5, 0.95, 'CAUSAL IMPLICATIONS', ha='center', va='top',
             fontsize=12, fontweight='bold')

    # Observer 1 view
    ax3.text(0.5, 0.78, 'Observer 1 sees:', ha='center', fontsize=10,
             color=COLORS['primary'], fontweight='bold')
    ax3.text(0.5, 0.68, '"A caused B"\n"C caused D"', ha='center', fontsize=9,
             color=COLORS['primary'])

    # Observer 2 view
    ax3.text(0.5, 0.50, 'Observer 2 sees:', ha='center', fontsize=10,
             color=COLORS['secondary'], fontweight='bold')
    ax3.text(0.5, 0.40, '"A and B co-occurred"\n"C and D co-occurred"',
             ha='center', fontsize=9, color=COLORS['secondary'])

    # Key point
    ax3.text(0.5, 0.18, 'Same physical events\nDifferent causal structures\n'
             'Neither observer is "wrong"',
             ha='center', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['bg'],
                      edgecolor=COLORS['accent'], linewidth=2))

    ax3.set_title('C. Framework-Relative Causality', fontweight='bold', pad=10)

    # Panel D: The regress problem
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis('off')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)

    # Draw nested frameworks
    circles = [
        (0.5, 0.5, 0.4, COLORS['gray'], 'Framework 3\n$\\Delta t_3$'),
        (0.5, 0.5, 0.28, COLORS['secondary'], 'Framework 2\n$\\Delta t_2$'),
        (0.5, 0.5, 0.16, COLORS['primary'], 'Framework 1\n$\\Delta t_1$'),
    ]

    for x, y, r, color, label in circles:
        circle = Circle((x, y), r, fill=False, edgecolor=color, linewidth=2)
        ax4.add_patch(circle)

    ax4.text(0.5, 0.5, '?', ha='center', va='center', fontsize=24,
             color=COLORS['accent'], fontweight='bold')

    ax4.text(0.5, 0.02, 'No "true" resolution\nto adjudicate between frameworks',
             ha='center', fontsize=9, color=COLORS['gray'])
    ax4.set_title('D. The Regress Problem', fontweight='bold', pad=10)

    # Panel E: Landauer terminates regress
    ax5 = fig.add_subplot(gs[1, 1])

    resolutions = np.logspace(-4, -1, 50)  # 0.1 ms to 100 ms
    landauer_time = 1e-3  # ~1 ms (illustrative)

    # Information accessible
    info = np.where(resolutions > landauer_time,
                    np.log2(100 / resolutions),
                    np.log2(100 / landauer_time) * (resolutions / landauer_time)**0.5)

    ax5.semilogx(resolutions * 1000, info, color=COLORS['primary'], lw=2)
    ax5.axvline(landauer_time * 1000, color=COLORS['accent'], linestyle='--',
                lw=2, label='Landauer limit')
    ax5.axvspan(0.1, landauer_time * 1000, alpha=0.2, color=COLORS['accent'])

    ax5.set_xlabel('Resolution $\\Delta t$ (ms)', fontsize=9)
    ax5.set_ylabel('Accessible info (bits)', fontsize=9)
    ax5.set_title('E. Landauer Terminates the Regress',
                  fontweight='bold', pad=10)
    ax5.legend(loc='upper right', fontsize=7)
    # Position "Sub-Landauer" in axes fraction to avoid data overlap
    ax5.text(0.18, 0.35, 'Sub-\nLandauer', fontsize=7, color=COLORS['accent'],
             fontweight='bold', ha='center', transform=ax5.transAxes)

    # Panel F: Key insight - more compact
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    ax6.text(0.5, 0.98, 'FRAMEWORK DEPENDENCE\nOF TIMING', ha='center', va='top',
             fontsize=10, fontweight='bold')

    insights = [
        '• $\\Delta t$ is a framework choice',
        '',
        '• Different $\\Delta t$ → different',
        '  causal structures',
        '',
        '• Below Landauer: regress',
        '  physically terminates',
        '',
        '• "When" created at projection'
    ]

    for i, line in enumerate(insights):
        ax6.text(0.05, 0.75 - i*0.075, line, ha='left', va='top',
                fontsize=8, transform=ax6.transAxes)

    ax6.set_title(' ', pad=10)  # Spacer

    plt.savefig('fig5_framework_timing.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('fig5_framework_timing.png', dpi=150, bbox_inches='tight')
    print("Saved: fig5_framework_timing.pdf/png")
    plt.close()


def fig4_energy_scaling():
    """
    Figure 4: Digital vs Continuous Energy Scaling

    Compares the exponential cost of enumerative digital tracking
    with the linear (in D) cost of projection.
    """
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # Panel A: Scaling comparison
    ax1 = fig.add_subplot(gs[0, 0])

    D = np.arange(10, 210, 10)
    kT_ln2 = 2.87e-21  # J at 300K

    # Digital: exponential (log2 of state space)
    # State space ~ k^D where k ~ 10 states per dimension
    k = 10
    E_digital = D * np.log2(k) * kT_ln2  # bits * kT ln 2

    # Continuous: linear in D
    L_over_eps = 100  # resolution ratio
    E_continuous = D * np.log(L_over_eps) * kT_ln2 / np.log(2)

    ax1.semilogy(D, E_digital, color=COLORS['secondary'], lw=2,
                 label='Digital (enumerative)')
    ax1.semilogy(D, E_continuous, color=COLORS['primary'], lw=2,
                 label='Continuous (projection)')

    ax1.set_xlabel('Effective dimensionality $D_{eff}$')
    ax1.set_ylabel('Minimum energy cost (J)')
    ax1.set_title('A. Energy Scaling: Digital vs Continuous',
                  fontweight='bold', pad=10)
    ax1.legend(loc='lower right', fontsize=8)
    ax1.set_xlim(10, 200)

    # Highlight gap
    ax1.fill_between(D, E_continuous, E_digital, alpha=0.2,
                     color=COLORS['accent'])

    # Panel B: Practical gap (CMOS)
    ax2 = fig.add_subplot(gs[0, 1])

    # Practical CMOS is 10^4 - 10^6 x Landauer limit
    cmos_factor = np.array([1e4, 1e5, 1e6])
    D_example = 100

    E_landauer = D_example * np.log2(k) * kT_ln2
    E_cmos = E_landauer * cmos_factor
    E_bio = D_example * np.log(L_over_eps) * kT_ln2 / np.log(2)

    bars = ax2.bar(['Landauer\n(theoretical)', 'CMOS\n(practical)', 'Biological\n(projection)'],
                   [E_landauer, E_cmos[1], E_bio],
                   color=[COLORS['gray'], COLORS['secondary'], COLORS['primary']])

    ax2.set_yscale('log')
    ax2.set_ylabel('Energy per decision (J)')
    ax2.set_title(f'B. Practical Gap at $D_{{eff}}$ = {D_example}',
                  fontweight='bold', pad=10)

    # Annotate ratios - position above bars
    ax2.text(1.5, E_cmos[1] * 3, f'~{E_cmos[1]/E_bio:.0e}×',
             fontsize=9, ha='center', color=COLORS['accent'],
             fontweight='bold')

    # Panel C: Why digital can't integrate sub-Landauer
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)

    ax3.text(0.5, 0.98, 'WHY DIGITAL CANNOT INTEGRATE', ha='center', va='top',
             fontsize=11, fontweight='bold', color=COLORS['secondary'])

    # Digital architecture box
    rect1 = FancyBboxPatch((0.02, 0.52), 0.44, 0.38, boxstyle="round,pad=0.02",
                           facecolor='white', edgecolor=COLORS['secondary'], lw=2)
    ax3.add_patch(rect1)
    ax3.text(0.24, 0.86, 'DIGITAL', ha='center', fontweight='bold',
             color=COLORS['secondary'], fontsize=10)
    ax3.text(0.24, 0.74, '• Each input → bit', ha='center', fontsize=8)
    ax3.text(0.24, 0.64, '• Bit ≥ $k_BT\\ln 2$', ha='center', fontsize=8)
    ax3.text(0.24, 0.54, '• Amplify before process', ha='center', fontsize=8)

    # Continuous architecture box
    rect2 = FancyBboxPatch((0.54, 0.52), 0.44, 0.38, boxstyle="round,pad=0.02",
                           facecolor='white', edgecolor=COLORS['primary'], lw=2)
    ax3.add_patch(rect2)
    ax3.text(0.76, 0.86, 'CONTINUOUS', ha='center', fontweight='bold',
             color=COLORS['primary'], fontsize=10)
    ax3.text(0.76, 0.74, '• Weak coupling to many', ha='center', fontsize=8)
    ax3.text(0.76, 0.64, '• Integrate before collapse', ha='center', fontsize=8)
    ax3.text(0.76, 0.54, '• Pay only at output', ha='center', fontsize=8)

    # Key distinction
    ax3.text(0.5, 0.40, '↓', ha='center', fontsize=16, color=COLORS['accent'])
    ax3.text(0.5, 0.20, 'Digital: supra-Landauer at input\nCannot access sub-Landauer regime',
             ha='center', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['bg'],
                      edgecolor=COLORS['accent'], linewidth=2))

    # Panel D: Efficiency table - simplified
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    ax4.text(0.5, 0.98, 'EFFICIENCY COMPARISON', ha='center', va='top',
             fontsize=11, fontweight='bold')

    # Table data - fewer rows, clearer
    table_data = [
        ['', 'Digital', 'Continuous'],
        ['Energy', '$\\sim D\\log k$', '$\\sim D\\ln(L/\\varepsilon)$'],
        ['Tracking', 'Explicit', 'Implicit'],
        ['Input', '≥ $k_BT\\ln 2$', 'Sub-Landauer OK'],
        ['Output', 'Bit-exact', 'Approximate'],
    ]

    for i, row in enumerate(table_data):
        y = 0.82 - i * 0.12
        weight = 'bold' if i == 0 else 'normal'
        for j, cell in enumerate(row):
            x = 0.18 + j * 0.28
            color = COLORS['dark'] if j == 0 else (COLORS['secondary'] if j == 1 else COLORS['primary'])
            ax4.text(x, y, cell, ha='center', fontsize=8, fontweight=weight, color=color)

    # Divider line
    ax4.axhline(y=0.76, xmin=0.08, xmax=0.92, color=COLORS['gray'], lw=1)

    ax4.text(0.5, 0.25, 'Neither is "better"\nThey solve different problems',
             ha='center', fontsize=8, style='italic', color=COLORS['gray'])

    plt.savefig('fig4_energy_scaling.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('fig4_energy_scaling.png', dpi=150, bbox_inches='tight')
    print("Saved: fig4_energy_scaling.pdf/png")
    plt.close()


def fig2_projection_bound():
    """
    Figure 2: The Projection Bound

    Visualizes dimensional collapse from high-D to low-D
    and the associated thermodynamic cost.
    """
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # Panel A: 3D to 2D projection visualization
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')

    np.random.seed(42)
    n_points = 500

    # Generate points on a sphere (high-D manifold proxy)
    theta = np.random.uniform(0, 2*np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)
    r = 1 + 0.1 * np.random.randn(n_points)

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    ax1.scatter(x, y, z, c=z, cmap='viridis', s=5, alpha=0.6)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('A. High-D State Space\n($N_{\\varepsilon,pre}$ states)',
                  fontweight='bold', pad=10)
    ax1.view_init(elev=20, azim=45)

    # Panel B: Projected (collapsed)
    ax2 = fig.add_subplot(gs[0, 1])

    ax2.scatter(x, y, c=z, cmap='viridis', s=5, alpha=0.6)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_aspect('equal')
    ax2.set_title('B. Collapsed to Low-D\n($N_{\\varepsilon,post}$ states)',
                  fontweight='bold', pad=10)

    # Draw projection arrow between panels (conceptual)

    # Panel C: Energy cost
    ax3 = fig.add_subplot(gs[0, 2])

    # Ratio of states
    ratios = np.logspace(1, 6, 50)
    kT = 4.1e-21  # kT at 300K

    E_collapse = kT * np.log(ratios)

    ax3.loglog(ratios, E_collapse / 1e-21, color=COLORS['accent'], lw=2)
    ax3.set_xlabel('$N_{pre}/N_{post}$ (state ratio)')
    ax3.set_ylabel('$E_{collapse}$ ($\\times 10^{-21}$ J)')
    ax3.set_title('C. Projection Bound\n$E \\geq k_BT_{eff}\\ln(N_{pre}/N_{post})$',
                  fontweight='bold', pad=10)

    # Mark examples - use smaller markers, labels to the side
    examples = [
        (1e2, 'Binary', 12, 0),
        (1e4, 'Percept.', 12, 0),
        (1e6, 'Motor', 12, 0),
    ]

    for ratio, label, xoff, yoff in examples:
        E = kT * np.log(ratio)
        ax3.scatter([ratio], [E / 1e-21], s=50, c=COLORS['primary'], zorder=5)
        ax3.annotate(label, (ratio, E / 1e-21), textcoords="offset points",
                    xytext=(xoff, yoff), fontsize=6, ha='left', va='center')

    ax3.axhline(y=2.87, color=COLORS['gray'], linestyle='--', alpha=0.5)
    ax3.text(1.5e1, 3.5, '$k_BT\\ln 2$', fontsize=6, color=COLORS['gray'])

    # Panel D: Full bound with KL terms
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis('off')

    ax4.text(0.5, 0.98, 'FULL PROJECTION BOUND', ha='center', va='top',
             fontsize=10, fontweight='bold')

    # Main equation - split for readability
    ax4.text(0.5, 0.80,
             '$E \\geq k_BT_{eff}[\\ln(N_{pre}/N_{post})'
             ' - KL_{pre} + KL_{post}]$',
             ha='center', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor=COLORS['accent'], linewidth=2))

    # Term explanations - compact single column
    ax4.text(0.08, 0.58, '$\\ln(N_{pre}/N_{post})$: geometric compression',
             fontsize=7, color=COLORS['dark'])
    ax4.text(0.08, 0.45, '$-KL_{pre}$: pre-constraint lowers bound',
             fontsize=7, color=COLORS['dark'])
    ax4.text(0.08, 0.32, '$+KL_{post}$: output specificity raises bound',
             fontsize=7, color=COLORS['dark'])

    # Panel E: Typical-set simplification
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis('off')

    ax5.text(0.5, 0.98, 'TYPICAL-SET APPROXIMATION', ha='center', va='top',
             fontsize=11, fontweight='bold')

    ax5.text(0.5, 0.78, 'Near-uniform distributions\n(high-D, many modes):',
             ha='center', fontsize=9)

    ax5.text(0.5, 0.58, 'KL terms $\\approx$ 0', ha='center', fontsize=10,
             color=COLORS['primary'], fontweight='bold')

    ax5.text(0.5, 0.45, '↓', ha='center', fontsize=16, color=COLORS['accent'])

    ax5.text(0.5, 0.30, '$E \\geq k_BT_{eff}\\ln(N_{pre}/N_{post})$',
             ha='center', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['bg'],
                      edgecolor=COLORS['primary'], linewidth=2))

    ax5.text(0.5, 0.12, 'Pure geometric bound',
             ha='center', fontsize=8, style='italic', color=COLORS['gray'])

    # Panel F: Power-law scaling
    ax6 = fig.add_subplot(gs[1, 2])

    D_eff = np.arange(10, 210, 5)
    D_prime = 1  # Binary output
    L_over_eps = 100

    E = kT * (D_eff - D_prime) * np.log(L_over_eps)
    E_scaled = E / 1e-20

    ax6.plot(D_eff, E_scaled, color=COLORS['primary'], lw=2)
    ax6.fill_between(D_eff, 0, E_scaled, alpha=0.2, color=COLORS['primary'])

    ax6.set_xlabel('$D_{eff}$ (pre-collapse dimension)')
    ax6.set_ylabel('$E_{collapse}$ ($\\times 10^{-20}$ J)')
    ax6.set_title("F. Linear Scaling\n$E \\sim (D_{eff} - D')\\ln(L/\\varepsilon)$",
                  fontweight='bold', pad=10)

    # Mark neural range - position text based on actual y range
    ax6.axvspan(50, 200, alpha=0.1, color=COLORS['secondary'])
    y_max = E_scaled.max()
    ax6.text(125, y_max * 0.15, 'Neural range', ha='center', fontsize=8,
             color=COLORS['secondary'])

    plt.savefig('fig2_projection_bound.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('fig2_projection_bound.png', dpi=150, bbox_inches='tight')
    print("Saved: fig2_projection_bound.pdf/png")
    plt.close()


if __name__ == '__main__':
    print("Generating Timing Inaccessibility v2.0 figures...")
    print("=" * 50)

    # Figures generated in manuscript order:
    #   Figure 1: Path Degeneracy (Section 2)
    #   Figure 2: Projection Bound (Section 3)
    #   Figure 3: Camera-Engine Duality (Section 6)
    #   Figure 4: Energy Scaling (Section 6)
    #   Figure 5: Framework Timing (Section 7)

    fig1_path_degeneracy()
    fig2_projection_bound()
    fig3_camera_engine()
    fig4_energy_scaling()
    fig5_framework_timing()

    print("=" * 50)
    print("All figures generated successfully!")
    print("\nFigures created:")
    print("  fig1_path_degeneracy.pdf/png")
    print("  fig3_camera_engine.pdf/png")
    print("  fig5_framework_timing.pdf/png")
    print("  fig4_energy_scaling.pdf/png")
    print("  fig2_projection_bound.pdf/png")
