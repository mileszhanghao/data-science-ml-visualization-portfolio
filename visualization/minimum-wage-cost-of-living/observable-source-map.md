# Observable Source Map

Portfolio-safe reconstruction of the Observable notebook structure for the minimum wage and cost-of-living dashboard. The original collaborative notebook remains private; this file documents the source architecture and reusable Vega-Lite pattern without publishing course/team scaffolding.

## Notebook Structure

- Data-loading cells for minimum-wage, housing, food, and regional cost indicators.
- Cleaning cells that normalize geography names, year fields, and currency values.
- Derived metrics for affordability gaps, wage-to-cost ratios, and year-over-year change.
- Linked chart cells for regional comparison, selected-state drilldown, and cost-category breakdown.
- Markdown narrative cells explaining where minimum wage fails to track essential living costs.

## Interaction Model

- Region selector controls which state or metro area receives focus.
- Year selector compares affordability at a fixed point in time.
- Linked bars and lines reuse the same filtered dataset so readers can connect wage levels with cost pressure.
- Tooltip fields expose raw values, normalized affordability gaps, and source category.

## Public Vega-Lite Artifact

- `specs/affordability_gap_linked_view.vl.json` contains a cleaned example of the linked-view design pattern used by the Observable prototype.

## Portfolio Boundary

This source map keeps the implementation logic and dashboard design visible while excluding private notebook URLs, team-only files, and course submission text.
