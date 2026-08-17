# Brand assets

| File | What | Use |
|------|------|-----|
| `mind-mark.svg` | The MIND mark, true vector, single flat `#4F46E5` | Any size — favicon through billboard |
| `mind-nerve-title-card.png` | 1920×1080 title card | Video open, thumbnail, slide |

## Colors

| Token | Hex | Where |
|-------|-----|-------|
| Mark / primary | `#4F46E5` | The mark, accents |
| Wordmark navy | `#293857` | "MIND" wordmark on light backgrounds |
| Card background | `#0A0A0E` | Video grade, dark surfaces |

## Type

Headings **Manrope**, body **Inter** — matching mindlang.dev
(`src/app/globals.css`: `--font-heading`, `--font-body`).

## Provenance of `mind-mark.svg`

Every prior copy of this mark was a **PNG base64'd inside an `<image>` tag** —
an SVG wrapper around a raster, so it did not scale:

- `mindlang.dev/public/favicon.svg` — 323×289 embedded PNG (mark only)
- `mind/assets/logo/mind-logo.svg` — 499×541 embedded PNG (mark + wordmark)

`mind-mark.svg` is a real vector traced from the higher-resolution source with
the wordmark masked off, then verified by rasterizing the result and diffing it
against the original alpha mask:

```
IoU = 0.99627   (0.22% of canvas mismatched — anti-aliased edge pixels only)
```

Fill is exactly `#4F46E5`, sampled from the source raster rather than eyeballed.

If the mark is ever redrawn, re-run that same rasterize-and-diff check rather
than trusting a visual comparison — a trace that looks right at card size can
still be wrong at favicon size.
