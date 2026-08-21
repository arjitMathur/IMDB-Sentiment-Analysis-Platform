# Design

## Theme
Beautiful UI dark product surface. Soft charcoal canvas, raised cards, blue accent for chrome. Green/red only for sentiment polarity.

## Color
| Token | Value | Role |
|---|---|---|
| `--page` | `#17181A` | Page background |
| `--canvas` | `#1C1D1F` | Nested canvas |
| `--surface` | `#232427` | Cards / composer |
| `--ink` / `--ink-2` / `--ink-3` | `#F2F3F4` / `#A5A8AD` / `#6C6F75` | Text hierarchy |
| `--line` | `#2E3033` | Hairlines |
| `--accent` | `#3D9AFF` | Primary CTA chrome |
| `--green` / `--red` | `#3DDC97` / `#FF5A5F` | Positive / negative only |

## Typography
- **UI:** Geist Sans
- **Data:** Geist Mono
- Instrument Serif kept available but unused in BUI primitives

## Layout
Centered workspace (~720px). Prompt Bar composer, Recommendation Card for verdict, Diff Table for compare, Loading State while pending.

## Motion
BUI primitives: pixel-on, shimmer-text, pop-in, fade-in. Honor `prefers-reduced-motion`.
