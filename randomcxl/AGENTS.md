# Project Development Conventions

## Curve-fit implementation placement

- Do not create ad hoc Python helper, edit, or runner files in the
  `project_randomcxl/` root.
- Keep logic that is specific to the Pecora comparison workflow inside
  `smpl/curvefit/cf_compare_pecora.ipynb`.
- Put reusable procedures for loading, extracting, transforming, comparing,
  or rendering information from curve-fit results in
  `smpl/curvefit/cf_tools.py`.
- Use temporary-system locations for one-off automation and remove those
  temporary files when the task is complete.
