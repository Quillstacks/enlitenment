"""Shared plot style for all enlitenment notebooks."""

import matplotlib.pyplot as plt

# --- Palette (matches enlitenment frontend) ---
_BG       = '#222222'
_SURFACE  = '#264653'
_BORDER   = '#2e5a6a'
_ACCENT   = '#2a9d8f'
_GOLDEN   = '#e9c46a'
_ORANGE   = '#f4a261'
_TERRA    = '#e76f51'
_TEXT     = '#c4bfb9'

# --- Extended palette (10-variety coffee datasets) ---
_SAGE     = '#a8d8a8'
_STEEL    = '#6c9ebf'
_LAVENDER = '#c7b8ea'
_CORAL    = '#f4978e'
_PEACH    = '#f7c59f'

# --- Variety → colour mapping (shared by UDL notebooks) ---
_variety_colors = {
    'espresso'    : _TERRA,
    'ristretto'   : _ORANGE,
    'lungo'       : _GOLDEN,
    'americano'   : _ACCENT,
    'cappuccino'  : _TEXT,
    'latte'       : _SAGE,
    'flat_white'  : _STEEL,
    'cold_brew'   : _LAVENDER,
    'pour_over'   : _CORAL,
    'french_press': _PEACH,
}

__all__ = [
    '_BG', '_SURFACE', '_BORDER', '_ACCENT', '_GOLDEN', '_ORANGE', '_TERRA', '_TEXT',
    '_SAGE', '_STEEL', '_LAVENDER', '_CORAL', '_PEACH',
    '_variety_colors', 'tufte_axis',
]

plt.rcParams.update({
    'figure.facecolor':    _BG,
    'axes.facecolor':      _BG,
    'axes.edgecolor':      _BORDER,
    'axes.labelcolor':     _TEXT,
    'axes.titlesize':      10,
    'text.color':          _TEXT,
    'xtick.color':         _TERRA,
    'ytick.color':         _TERRA,
    'legend.facecolor':    _BG,
    'legend.edgecolor':    _BORDER,
    'legend.labelcolor':   _TEXT,
    'savefig.facecolor':   _BG,
    'font.family':         'sans-serif',
    'font.size':           11,
})


def tufte_axis(ax):
    """Minimal axis styling — visible bottom/left spines in border color."""
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('bottom', 'left'):
        ax.spines[spine].set_color(_BORDER)
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(axis='both', which='both', direction='out',
                   length=4, width=0.8, colors=_TERRA,
                   top=False, right=False)
    ax.set_title(ax.get_title(), color=_GOLDEN)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
