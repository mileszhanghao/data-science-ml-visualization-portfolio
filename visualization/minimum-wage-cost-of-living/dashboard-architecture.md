# Dashboard Architecture

Collaborative policy analytics project comparing minimum wage against cost-of-living indicators across U.S. regions.

## Data Model

The dashboard workflow is organized around three table families:

| Table | Role |
| --- | --- |
| Minimum wage by state and year | Baseline policy input for wage-floor comparison |
| Living-cost and essential-expense indicators | Housing, food, transportation, and related affordability pressure |
| Merged region-year analytic table | Shared table used by chart views and filters |

## Interaction Model

- State selector for regional comparison.
- Year selector for time-window control.
- Linked line chart for wage trend analysis.
- Bar chart for cost component comparison.
- Scatter plot for wage-to-cost relationship screening.
- Dashboard panel that keeps the selected region and year synchronized across views.

## Engineering Pattern

1. Ingest public economic datasets.
2. Standardize region names, years, and numeric fields.
3. Join wage and cost indicators into a single analytic table.
4. Build chart components from the merged table rather than duplicating data logic.
5. Add controls that filter all views consistently.

## Portfolio Boundary

This repo keeps the report and architecture summary. Private review notes and scaffold text remain excluded.
