# Observable Source Notes

The deceptive visualization was also present as an Observable notebook draft named `Zhang_AB_Visualization2`.

## Source Details

- Platform: Observable
- Source data endpoint used in the notebook:
  `https://data.seattle.gov/resource/2khk-5ukd.csv`
- Source data theme: City of Seattle wage data.
- Notebook visibility observed in source workspace: unlisted.

## Core Observable Logic

The notebook loaded Seattle wage rows directly from the Socrata CSV endpoint:

```js
seattleWages = d3.csv("https://data.seattle.gov/resource/2khk-5ukd.csv", d3.autoType)
```

The intentionally misleading chart used a bar chart by department and aggregated hourly rate with `max`:

```js
render({
  title: "Hourly Rates in Seattle: Are Some Departments Overpaid?",
  data: { values: seattleWages },
  mark: "bar",
  encoding: {
    x: {
      field: "department",
      type: "nominal",
      title: "Department",
      axis: { labelAngle: -45 }
    },
    y: {
      field: "hourly_rate",
      type: "quantitative",
      title: "Hourly Rate (USD)",
      aggregate: "max"
    },
    color: {
      field: "department",
      type: "nominal",
      legend: null
    },
    tooltip: [
      { field: "department", type: "nominal" },
      { field: "hourly_rate", type: "quantitative", aggregate: "max" }
    ]
  },
  width: 600,
  height: 400
})
```

## Portfolio Interpretation

This source supports the public repo's critique: the misleading version uses the maximum hourly rate by department, which can overstate typical compensation and make departments look unusually overpaid. The cleaned public Vega-Lite specs in `specs/` preserve that design as a critique artifact while separating it from the private course notebook.
