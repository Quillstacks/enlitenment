# Frontend Style Guide — Agent Reference

> This document is the single source of truth for visual styling in the YawningSquirrel frontend.
> When generating or modifying components, follow these rules exactly.

---

## Stack

- **Vite + React Router** (entry: `src/main.tsx`)
- **React 19**, TypeScript (strict)
- **Tailwind CSS v4** — CSS-first config via `@theme {}` in `src/index.css`
- **Recharts** for all data visualizations
- Fonts loaded via CSS `@font-face` / npm packages

---

## Typography

| Role | Font | Weight | Notes |
|---|---|---|---|
| **Body / headings** | `EB Garamond` | 300–400 | Tufte-style old-style serif. Default for all text. |
| **Mono (labels, chips, axes, insights)** | `Geist Mono` | 400 | Via `geist` npm package. Used for small metadata, chart labels, chip text. |

**CSS variables:**
```
--font-sans: 'EB Garamond', Palatino, "Book Antiqua", Georgia, serif;
--font-mono: 'Geist Mono', ui-monospace, monospace;
```

**Tailwind usage:** `font-sans` (body/headings), `font-mono` (labels/metadata).

**Font features enabled on `<body>`:**
```css
font-feature-settings: "kern" 1, "liga" 1, "calt" 1, "onum" 1;
```

**Body text:** `font-size: 1.0625rem`, `line-height: 1.7`.

**Heading pattern:** `text-4xl sm:text-5xl font-light leading-tight tracking-tight`.

---

## Color Palette

Base palette from [coolors.co/palette/264653-2a9d8f-e9c46a-f4a261-e76f51](https://coolors.co/palette/264653-2a9d8f-e9c46a-f4a261-e76f51).

### Semantic tokens

| Token | Hex | Tailwind class | Role |
|---|---|---|---|
| `--color-bg` | `#222222` | `bg-bg` | Page background base |
| `--color-surface` | `#264653` | `bg-surface` | Card/panel backgrounds, tooltip bg |
| `--color-border` | `#2e5a6a` | `border-border` | All borders, dividers, axis lines |
| `--color-accent` | `#2a9d8f` | `bg-accent` / `text-accent` | CTA buttons, active dots, focus rings |
| `--color-near-black` | `#e9c46a` | `text-near-black` | Primary text (golden — high contrast on dark bg) |
| `--color-gray-mid` | `#f4a261` | `text-gray-mid` | Secondary text, trend indicators (sandy orange) |
| `--color-gray-tertiary` | `#e76f51` | `text-gray-tertiary` | Muted/tertiary text, labels, chips (terracotta) |
| `--color-text` | `#c4bfb9` | `text-text` | Body text, insight footnotes (warm gray) |

### Shared palette constants (`lib/palette.ts`)

```ts
CHART_PRIMARY = "#e9c46a"   // golden — chart entity strokes, fills, scores
CHART_MUTED   = "#3a6878"   // desaturated teal — secondary/competitor bars, avg overlays
ACCENT_COLORS = ["#e76f51", "#f4a261", "#e9c46a", "#2a9d8f", "#264653", "#e76f51"]
```

### Usage rules

- **Body text** uses `--color-text` (`#c4bfb9`). Headings use `text-near-black` (golden).
- **Never use teal (`#2a9d8f`) as body or heading text.** It is the same hue family as the background — low contrast. Teal is exclusively for interactive elements (buttons, active states, focus rings, chart entity highlights).
- **Warm text on cool background** is the core contrast principle: golden/orange/terracotta text on dark charcoal.
- **Borders** are always `border-border` (`#2e5a6a`), 1px. No rounded corners unless explicitly required.
- **Surface panels** (cards, tooltips, sticky bars) use `bg-surface` (`#264653`) or `rgba(29,60,71,0.92)` for translucent variants.
- When in doubt, use `text-gray-tertiary` for small labels and `text-text` for longer-form content.

---

## Spacing & Layout

- **No border-radius.** Cards, inputs, buttons, tooltips all use sharp corners (`rounded-none` / `radius: 0`). This is a deliberate design choice — do not add rounding.
- **Generous whitespace.** Sections separated by `mt-12 sm:mt-14` or more. Don't crowd elements.
- **Max content width:** `max-w-3xl` for hero text, `max-w-[600px]` for inputs.
- **Small metadata text:** `text-[11px]` with `tracking-[0.16em]` or `tracking-[0.18em]` uppercase for section labels.
- **Dividers:** Use the editorial pattern — `<div className="flex-1 h-px bg-border" />` flanking a centered label.

---

## Component Patterns

### Buttons

**Primary CTA:**
```
bg-accent text-white text-sm font-medium px-5 py-3
hover:opacity-90 active:scale-95 transition-all duration-150
```

**Chip / tag button:**
```
text-[11px] font-mono text-gray-tertiary border border-border px-2.5 py-1
hover:border-accent hover:text-accent transition-colors duration-150
```

### Cards

```
bg-surface border border-border px-4 py-3
hover:scale-[1.03] hover:shadow-[0_4px_16px_rgba(0,0,0,0.4)]
active:scale-[0.98] transition-all duration-150
```
- Optional colored left border via `style={{ borderLeft: "3px solid ${accent}" }}`.
- Title: `text-sm font-medium text-near-black truncate`.
- Subtitle: `text-[11px] font-mono text-gray-tertiary`.

### Inputs

```
border border-border bg-surface px-5 h-[64px]
focus-within:shadow-[0_0_0_1.5px_#2a9d8f] focus-within:border-gray-mid
box-shadow: 0 1px 20px rgba(0,0,0,0.3)
```
- Input text: `text-near-black`, placeholder: `placeholder:text-gray-tertiary`.
- No border-radius.

### Tooltips (chart & general)

```css
background: #264653;
border: 1px solid #2e5a6a;
border-radius: 0;
padding: 6px 10px;
```
- Label: `9px`, mono, `#e76f51`.
- Value: `13px`, mono, `#e9c46a`, `font-weight: 300`, `tabular-nums`.

### Section labels / eyebrows

```
text-[11px] font-medium tracking-[0.18em] uppercase text-gray-tertiary
```

### Insight footnotes (below charts)

```
text-[10px] font-mono text-text mt-3 animate-fade-slide-in
style={{ animationDelay: "800ms"–"1000ms" }}
```

---

## Animations

### Available tokens (defined in `src/index.css`)

| Tailwind class | Keyframes | Duration | Use case |
|---|---|---|---|
| `animate-fade-slide-in` | Fade in + translateY(14px→0) | 0.7s ease-out | Entrance of sections, insights, staggered elements |
| `animate-shimmer` | Opacity pulse 1→0.35→1 | 1.8s ease-in-out infinite | Loading/skeleton states |
| `animate-scroll` | translateX(0→-50%) | 30s linear infinite | Infinite carousel |
| `animate-spin` | rotate(0→360deg) | 0.8s linear infinite | Loading spinners |

### Philosophy

- **Animate entrances only.** Elements fade/slide in once when they enter the viewport. No looping animations except carousel scroll and loading states.
- **Stagger with `animationDelay`** on sibling elements (e.g. `180ms`, `800ms`, `1000ms`). Use inline `style={{ animationDelay: "Xms" }}`.
- **Charts animate on `inView`** — pass `isAnimationActive={inView}` to Recharts components. Duration: 700–1200ms. Easing: `ease-out`.
- **Micro-interactions:** `hover:scale-[1.03]`, `active:scale-[0.98]`, `transition-all duration-150`. Keep hover effects subtle.
- **Crossfade** for text rotation: dual absolute slots, opacity transition over 500ms.

---

## Charts (General Rules)

All charts use **Recharts** with `<ResponsiveContainer width="100%" height={N}>`.

### Colors

- **Entity / primary data:** `CHART_PRIMARY` (`#e9c46a` golden) — strokes, fills, dot fills, highlighted bars.
- **Secondary / competitor data:** `CHART_MUTED` (`#3a6878`) — dimmed bars, average overlays.
- **Axis lines:** `#2e5a6a`, strokeWidth 1.
- **Axis tick text:** `9–12px`, mono font (`ui-monospace, monospace`), color `#e76f51` (terracotta). Highlighted tick uses `CHART_PRIMARY`.
- **Gradient fills:** Use `linearGradient` from `CHART_PRIMARY` at 15% opacity → 0% opacity.
- **Dots on data points:** `fill: CHART_PRIMARY`, `stroke: #264653`, `strokeWidth: 1.5`, `r: 3–4`.
- **Active dot (hover):** `r: 5`, `fill: #264653`, `stroke: CHART_PRIMARY`, `strokeWidth: 2` (inverted from resting state).

### Axis formatting

- X-axis: show `axisLine` with border color, hide `tickLine`.
- Y-axis (when shown): hide `axisLine` and `tickLine`.
- Reference lines: `stroke: #2e5a6a`, `strokeWidth: 1`. Dashed variant: `strokeDasharray="3 3"`, `strokeOpacity: 0.35`.

### Layout conventions

- **Header above chart:** Large value (`text-4xl font-light tabular-nums`, colored `CHART_PRIMARY`) with small `/100` suffix (`text-[11px] font-mono text-gray-tertiary`). Trend delta on the right.
- **Insight below chart:** `text-[10px] font-mono text-text`, fades in with delay. One sentence summarizing the data. Often includes an upsell hint on the free plan.

### Responsive

- Check `window.innerWidth < 768` for mobile adaptations.
- Reduce bar sizes, outer radii, font sizes, and axis widths on mobile.
- Use `barSize: 20` mobile / `30` desktop as a baseline.

---

## Responsive Breakpoints

| Breakpoint | Tailwind prefix | Usage |
|---|---|---|
| < 640px | (default) | Mobile: single column, smaller text, `br` line breaks in headings |
| 640px+ | `sm:` | Slightly larger text (`sm:text-5xl`, `sm:text-[17px]`) |
| 768px+ | `md:` | Chart layout switches (detected via JS, not Tailwind) |
| 1024px+ | `lg:` | Two-column plot sections with alternating order |

---

## Anti-patterns (Do NOT)

- Add `border-radius` or `rounded-*` classes to cards, inputs, buttons, or tooltips.
- Use teal (`#2a9d8f`) for text. It is only for interactive accents.
- Use `React.CSSProperties` — import `CSSProperties` as a named type instead.
- Put components that use hooks inside conditional render paths.
- Add decorative gradients, glows, or shadows beyond the established `box-shadow` values.
- Use UUIDs for tokens — the project uses `crypto.getRandomValues` three-word tokens.
- Import fonts via `next/font` — this is a Vite project, not Next.js.

---

## Quick Reference: New Component Checklist

When creating a new component, verify:

1. **No border-radius** anywhere.
2. **Colors** only from the semantic tokens above — no arbitrary hex values outside the palette.
3. **Font:** body text in `font-sans` (EB Garamond), metadata/labels in `font-mono` (Geist Mono).
4. **Entrance animation:** `animate-fade-slide-in` with appropriate `animationDelay`.
5. **Hover states:** subtle scale + shadow, `transition-all duration-150`.
6. **Responsive:** mobile-first, test at `< 640px`.
7. **Types:** import React types as named imports (`import type { CSSProperties } from "react"`).
