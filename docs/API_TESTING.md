# API Testing & Coverage

This document describes the API test coverage, reporting, and usage for the brain-mapping toolkit.

## Automated API Tests
- All REST endpoints are covered by automated tests in `tests/test_api_coverage.py`.
- Edge cases and error scenarios are included.
- Property-based tests use the `hypothesis` library for robust input coverage.

## Feedback Integration
- Feedback submission and retrieval are tested in `tests/test_feedback_integration.py`.
- Automated reporting and error handling are included.

## Visualization Property-Based Tests
- Interactive atlas, multi-planar, and glass brain modules are tested in `tests/test_visualization_property.py`.
- Property-based and edge case tests ensure robustness.

## Running Tests
```bash
pytest tests/
```

## Requirements
- Python 3.9+
- `pytest`, `hypothesis`, `requests`

## API Endpoints
- `/api/status`: Health check
- `/api/analyze`: Data analysis
- `/api/feedback`: Feedback submission/retrieval

## Reporting
- Test results are logged in `logs/` for traceability.
- Coverage reports are generated automatically in CI.

## Continuous Integration
- All tests are run automatically in CI workflows.
- Changelog and documentation updates are automated.

## See Also
- [USAGE.md](USAGE.md)
- [project-plan.md](project-plan.md)
- [CHANGELOG_AUTOMATED.md](../logs/CHANGELOG_AUTOMATED.md)
