# Observable Prototype Notes

This project also had an Observable notebook prototype in the CSE 412 workspace. The notebook is not published here because it was a team/course workspace artifact and included attached course-stage data files.

## Source Notebook Observations

- Title: `Comparing Minimum Wage to the Cost of Living`
- Platform: Observable
- Visibility in source workspace: team/private
- Attached files observed in the notebook:
  - `inflation_data.csv`
  - `Minimum Wage Data.csv`

## Implemented Prototype Ideas

The Observable prototype explored:

- A line chart for food and beverage inflation index values over time.
- A year-threshold slider for filtering the displayed years.
- A planned U.S. state map comparing living-wage and minimum-wage gaps.
- A dashboard framing around regional affordability, wage growth, and essential expenses.

## Publication Boundary

The parsed notebook appeared to be an early prototype rather than a fully standalone public source artifact. In particular, the map cell referenced intermediate values such as `combinedData`, `states`, and `selectedYear` that were not defined in the visible parsed cell list. For the public repo, this file records the implementation provenance while `dashboard-architecture.md` captures the cleaned portfolio-safe design.
