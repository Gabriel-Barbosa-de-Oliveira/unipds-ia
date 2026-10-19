---
name: Pix Precision Banking
colors:
  surface: '#f8f9ff'
  surface-dim: '#cbdbf5'
  surface-bright: '#f8f9ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#eff4ff'
  surface-container: '#e5eeff'
  surface-container-high: '#dce9ff'
  surface-container-highest: '#d3e4fe'
  on-surface: '#0b1c30'
  on-surface-variant: '#44474c'
  inverse-surface: '#213145'
  inverse-on-surface: '#eaf1ff'
  outline: '#74777d'
  outline-variant: '#c4c6cc'
  surface-tint: '#525f71'
  primary: '#000000'
  on-primary: '#ffffff'
  primary-container: '#0f1c2c'
  on-primary-container: '#778598'
  inverse-primary: '#bac8dc'
  secondary: '#006d35'
  on-secondary: '#ffffff'
  secondary-container: '#3fff8b'
  on-secondary-container: '#007237'
  tertiary: '#000000'
  on-tertiary: '#ffffff'
  tertiary-container: '#101b30'
  on-tertiary-container: '#79849d'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#d6e4f9'
  primary-fixed-dim: '#bac8dc'
  on-primary-fixed: '#0f1c2c'
  on-primary-fixed-variant: '#3a4859'
  secondary-fixed: '#62ff96'
  secondary-fixed-dim: '#00e475'
  on-secondary-fixed: '#00210b'
  on-secondary-fixed-variant: '#005226'
  tertiary-fixed: '#d7e2ff'
  tertiary-fixed-dim: '#bbc6e2'
  on-tertiary-fixed: '#101b30'
  on-tertiary-fixed-variant: '#3c475d'
  background: '#f8f9ff'
  on-background: '#0b1c30'
  surface-variant: '#d3e4fe'
typography:
  display-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 40px
    fontWeight: '700'
    lineHeight: 48px
    letterSpacing: -0.03em
  display-lg-mobile:
    fontFamily: Plus Jakarta Sans
    fontSize: 30px
    fontWeight: '700'
    lineHeight: 38px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 28px
    fontWeight: '700'
    lineHeight: 36px
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Plus Jakarta Sans
    fontSize: 22px
    fontWeight: '600'
    lineHeight: 30px
    letterSpacing: -0.015em
  headline-sm:
    fontFamily: Plus Jakarta Sans
    fontSize: 18px
    fontWeight: '600'
    lineHeight: 26px
    letterSpacing: -0.01em
  body-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-lg-bold:
    fontFamily: Plus Jakarta Sans
    fontSize: 16px
    fontWeight: '600'
    lineHeight: 24px
  body-md:
    fontFamily: Plus Jakarta Sans
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  body-md-medium:
    fontFamily: Plus Jakarta Sans
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 20px
  body-sm:
    fontFamily: Plus Jakarta Sans
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
  label-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 14px
    fontWeight: '600'
    lineHeight: 20px
    letterSpacing: 0.01em
  label-md:
    fontFamily: Plus Jakarta Sans
    fontSize: 12px
    fontWeight: '600'
    lineHeight: 16px
    letterSpacing: 0.02em
  label-sm:
    fontFamily: Plus Jakarta Sans
    fontSize: 10px
    fontWeight: '700'
    lineHeight: 14px
    letterSpacing: 0.05em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  gutter: 1.5rem
  gutter-desktop: 2rem
  margin: 1rem
  margin-tablet: 2rem
  margin-desktop: auto
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 1rem
  space-lg: 1.5rem
  space-xl: 2.5rem
---

## Brand & Style

The design system projects absolute financial authority paired with lightning-fast transactional clarity. Designed for a high-end fintech banking context focusing on Pix instant payments, it evokes security, decisive action, and pristine clarity. The emotional atmosphere centers on confidence and frictionless validation: users must feel entirely safe during high-stakes verification steps, followed by an immediate sense of accomplishment upon completion.

The visual style blends **Modern Minimalism** with **Subtle High-Tech Glass & Tonal Layering**. It rejects noisy visual clutter in favor of ample negative space, sharp typography, and deliberate atmospheric contrast. Deep, executive-grade navy grounds the experience, providing the structural reassurance of a Tier-1 institution, while an electric mint-neon pulse animates real-time verification status, high-priority CTAs, and instant Pix confirmation receipts.

## Colors

The palette establishes an architectural hierarchy using deep navy blues balanced against pristine, cool-tinted surfaces and high-impact mint luminescence.

- **Primary (`#0D1B2A`)**: Midnight Navy. The core foundation for headings, primary buttons, card structures, and critical authentication barriers.
- **Secondary (`#00E676`)**: Electric Mint Glow. Used purposefully for successful Pix state validations, biometric check indicators, key transaction metrics, and accent signals. Never over-applied to large backgrounds; strictly treated as a precision feedback light.
- **Tertiary (`#1B263B`)**: Slate Midnight. Deployed for subtle tonal elevation, high-contrast borders within dark cards, secondary buttons, and active interactive borders.
- **Neutrals**:
  - `#FFFFFF`: Pure White for the transaction card containers and crisp surface planes.
  - `#F8FAFC`: Soft Cool Gray for the primary canvas/viewport backdrop.
  - `#E2E8F0`: Structural dividers, hairline card outlines, and unselected control rings.
  - `#64748B`: Mid-tone neutral for descriptive metadata, secondary timestamps, and input labels.
  - `#0F172A`: High-contrast body text for maximum legibility.
- **Feedback & Validation**:
  - Success/Confirmation: `#00E676` (Electric Mint) / `#10B981` (Deeper emerald for text contrast on light backgrounds).
  - Destructive/Cancel: `#EF4444`.
  - Warning/Attention: `#F59E0B`.

## Typography

The typographic hierarchy utilizes **Plus Jakarta Sans** throughout all scales to deliver a balance of geometric modernism and warm humanist touchpoints.

- Numbers and monetary amounts (e.g., Pix transfer values) leverage tabular figure settings (`tnum`) to eliminate proportional jitter during real-time fee calculation or animated counts.
- `display-lg` is dedicated exclusively to the central currency transfer value (e.g., `R$ 4.850,00`) on the review and voucher screens.
- `label-sm` utilizes uppercase casing and tracking (`letterSpacing: 0.05em`) for Pix payment tags, verification badges, and recipient account classifications.

## Layout & Spacing

The spatial architecture prioritizes single-task cognitive focus through a centered-card modality on desktop screens, minimizing eye travel and avoiding distraction during irreversible monetary actions.

- **Desktop (1024px and above)**: The main transactional viewport centers a fixed-width card container (max-width `520px` to `640px`) suspended within an expansive `#F8FAFC` viewport. Inner margins expand to `2.5rem` (`space-xl`), grouping the recipient verification, authentication trigger, and confirmation breakdown.
- **Tablet (768px – 1023px)**: Fixed-width center card remains constrained to `540px` with outer view margins set to `2rem`.
- **Mobile (Below 768px)**: Card borders fall away into a full-bleed surface layout with sticky bottom action panels for one-thumb biometric approval. Left and right gutter margins lock at `1rem` (`16px`).
- **Internal Component Spacing**: Follows a strict 4px/8px incremental grid. Input form rows maintain `1rem` separation; summary detail tables use `0.75rem` vertical spacing with hairline separators.

## Elevation & Depth

Visual hierarchy combines low-contrast hairline outlines with soft ambient shadows, avoiding heavy drops to preserve a light, clean finish.

- **Level 0 (Canvas Base)**: `#F8FAFC`. Completely matte without depth.
- **Level 1 (Card Surface / Confirmation Shell)**: Solid white `#FFFFFF` paired with an ultra-thin 1px perimeter border (`rgba(15, 23, 42, 0.06)`) and a diffused ambient shadow: `0 8px 30px -4px rgba(13, 27, 42, 0.04), 0 2px 6px -1px rgba(13, 27, 42, 0.02)`.
- **Level 2 (Active Drawer / Biometric Pin Modal)**: Elevated card state hovering over canvas: `0 20px 40px -8px rgba(13, 27, 42, 0.08), 0 4px 12px -2px rgba(13, 27, 42, 0.03)`.
- **Luminescent Accent Layer (Electric Mint Aura)**: Used strictly under primary confirmation state indicators and the final "Pix Confirmado" check glyph: a concentrated radial glow `0 0 24px rgba(0, 230, 118, 0.35)`.

## Shapes

The design system maintains a balanced, approachable visual signature through a geometric corner radius:

- **Base Corner Radius (`0.5rem` / `8px`)**: Applied to standard interactive elements including form inputs, secondary badges, action chips, and nested recipient preview items.
- **Medium Corner Radius (`rounded-lg`, `1rem` / `16px`)**: Applied to the primary transaction cards, modal sheets, and authentication keypad keys.
- **Large Corner Radius (`rounded-xl`, `1.5rem` / `24px`)**: Applied to high-level parent layout panels and desktop confirmation receipt shells.
- **Pill Formats**: Used exclusively for dynamic status chips, recipient Pix key pill tags, and full-width primary CTA action bars.

## Components

### Buttons
- **Primary Action (Confirm Pix)**: Navy base `#0D1B2A` with white `#FFFFFF` text, high-hit radius (`height: 52px`), and subtle hover transition to `#1B263B`. When entering "Authorizing" state, a subtle neon mint spinner activates.
- **Secondary Action (Cancel / Return)**: Transparent background, 1px outline in `#E2E8F0`, typography in `#0D1B2A`. Hover: `#F1F5F9`.
- **Electric Accent Button (Voucher Download / Share)**: Soft mint background `rgba(0, 230, 118, 0.12)`, text `#065F46`, hover: `rgba(0, 230, 118, 0.20)`.

### Cards (Pix Confirmation Hub)
- Desktop layout is built around a centralized single card with a `#FFFFFF` fill, 1px border (`#E2E8F0`), and `1.5rem` border-radius.
- Internal hierarchy divides the card into three distinct sections:
  1. *Transfer Metric Header*: Centered large value typography (`R$ 0,00`) above small recipient identity.
  2. *Data Grid / Summary*: Key-value rows detailing recipient institution, CPF/CNPJ mask, Pix key, scheduled date, and zero-fee disclaimer separated by `1px solid #F1F5F9`.
  3. *Action Footer*: Stacked primary confirmation CTA and tertiary cancel link.

### Chips & Badges
- **Status Badges**: Pill-shaped with `padding: 4px 12px`.
  - Instant Transfer: Light slate `#F1F5F9` with `#1E293B` text.
  - Success/Validated: Surface `rgba(0, 230, 118, 0.15)`, text `#047857`, preceded by a solid 6px mint indicator dot.

### Lists & Row Details
- Clean horizontal key-value rows. Left label set in `body-md` muted slate (`#64748B`), right value set in `body-md-medium` dark navy (`#0D1B2A`).
- Copy-to-clipboard inline icon triggers with immediate micro-feedback checkmark in `#00E676`.

### Input Fields & Verification PIN
- Inputs feature a crisp `1px solid #CBD5E1` border on a `#FFFFFF` or `#F8FAFC` background. 
- Focus state switches border to `#0D1B2A` with an ambient glow ring `0 0 0 3px rgba(13, 27, 42, 0.08)`.
- Numeric PIN entries utilize discrete individual rounded squares (`56px x 56px`), with typed digits rendered as deep navy discs.

### Checkboxes & Toggle Controls
- Selected state fills with `#0D1B2A` displaying a pure white checkmark.
- Biometric toggles (FaceID/TouchID prompt) show a mint toggle track (`#00E676`) when activated.