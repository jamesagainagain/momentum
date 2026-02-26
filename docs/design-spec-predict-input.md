# Design spec: Predict sticky input — “Paste a headline or describe a news event”

This spec applies **only** to the single text field with placeholder **“Paste a headline or describe a news event…”** on the Predict page (sticky at the bottom when there is history or a loading state). Not the empty-state card input or any other field.

Single source of truth for that control: layout, dimensions, typography, and liquid-glass treatment.

---

## 1. Overall behavior

- **No bar.** No background panel, no border, no visible container around the field. Only the **liquid-glass pill** (WebGL) defines the shape; everything else is transparent.
- **Sticky** at the bottom of the Predict content, inside the main content column.
- **Alignment:** Same horizontal padding and content width as the rest of the Predict page (`max-w-3xl` column + `px-page-x`).

---

## 2. Dimensions

### 2.1 Content column (page width)

| Token / class | Value | Notes |
|----------------|--------|--------|
| Content max width | `max-w-3xl` | 768px |
| Horizontal padding | `px-page-x` | `--space-page-x` = **2.5rem (40px)** each side |
| **Usable width for input row** | 768 − 80 = **688px** | At max breakpoint |

So the input row (glass pill + overlay) lives in a **688px**-wide band when the viewport is wide enough.

### 2.2 Sticky strip (vertical)

| Element | Class / value | Pixel |
|--------|----------------|--------|
| Top padding | `pt-6` | 24px |
| Bottom padding | `pb-8` | 32px |
| **Input row container** | `h-[4.5rem]` | **72px** height |

So the “bar” (the liquid-glass + input row) is **72px** tall.

### 2.3 Glass pill (WebGL canvas)

- **Canvas size:** Same as the input row container: **width = container width (e.g. 688px at max), height = 72px** (sized by `barContainerRef`).
- **Pill shape (when `fillShape={false}`):**
  - Width: **85%** of canvas width → half-width in shader terms: `(width * 0.85 * dpr) / 2`.
  - Height: **70%** of canvas height → half-height: `(height * 0.7 * dpr) / 2`.
  - Corner radius: **50%** of the smaller half-dimension → pill/capsule.
- **Position:** Centered in the canvas (no offset).
- **Outside pill:** Fully **transparent** (no rect, no bar).

### 2.4 Overlay (input + button)

| Area | Value | Pixels |
|------|--------|--------|
| Horizontal padding | `px-4` | 16px each side |
| Vertical padding | `py-3` | 12px top/bottom |
| Gap between input and button | `gap-3` | 12px |

So the **text field** starts 16px from the left edge of the 72px-tall row and has 12px gap to the send button.

### 2.5 Text field (“Paste a headline…”)

| Spec | Value | Notes |
|------|--------|--------|
| Min height | `min-h-[2.75rem]` | **44px** (matches `--touch-min`) |
| Width | `flex-1` | Fills space between left padding and button |
| Background / border | None | Fully transparent, no box |
| Placeholder | “Paste a headline or describe a news event…” | Single line, ellipsis at end |

### 2.6 Send button

| Spec | Value | Pixels |
|------|--------|--------|
| Size | `h-10 w-10` | **40×40px** |
| Border radius | `rounded-xl` | 12px |
| Icon | Send (or Loader) | `h-5 w-5` (20px) |

---

## 3. Typography

| Element | Token / class | Value |
|--------|----------------|--------|
| Placeholder & input text | `text-[15px]` | **15px** |
| Color (text) | `text-foreground` | `hsl(var(--foreground))` → 60 4% 8% |
| Placeholder color | `placeholder:text-muted-foreground` | `hsl(var(--muted-foreground))` → 45 8% 42% |
| Font | Inherit (body) | **Lora** (from `index.css` body) |

No bold/uppercase for the placeholder; same weight as body.

---

## 4. Session count (below the row)

| Spec | Value |
|------|--------|
| Margin top | `mt-2` | 8px |
| Font size | `text-[11px]` |
| Color | `text-muted-foreground` |
| Alignment | `text-center` |
| Copy | “N prediction(s) this session” |

---

## 5. Spacing summary (design tokens)

From `index.css`:

- `--space-page-x`: **2.5rem (40px)** — horizontal padding of content column.
- `--touch-min`: **2.75rem (44px)** — minimum touch height (input min-height).
- `--input-height`: **3.5rem (56px)** — optional reference; current row is 72px.

Row height **72px** is intentional so the pill has enough vertical room and the overlay padding fits.

---

## 6. Liquid-glass preset

The bar uses **`LIQUID_GLASS_BAR_PRESET`** from `@/lib/liquid-glass/preset` (refraction, Fresnel, glare, blur, etc.). No separate “bar” UI — only the WebGL pill and the overlaid input/button.

---

## 7. Quick reference (pixels)

| Item | Pixels |
|------|--------|
| Content column max width | 768 |
| Page horizontal padding (each side) | 40 |
| Input row height | 72 |
| Input min height | 44 |
| Send button | 40×40 |
| Overlay padding (horizontal) | 16 |
| Overlay padding (vertical) | 12 |
| Gap (input ↔ button) | 12 |
| Placeholder / input font size | 15 |

---

## 8. Do not

- Add a background or border to the sticky strip or the input.
- Add a visible “bar” or panel behind the pill.
- Use a fixed width for the text field; keep it `flex-1` within the padded row.
- Make the pill fill the full canvas (`fillShape` must stay `false` for the bar so only the pill is visible).
