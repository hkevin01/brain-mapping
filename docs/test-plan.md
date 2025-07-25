# Brain Mapping Toolkit Test Plan

## Overview
This test plan documents the strategy, coverage, and progress for all major features, including provenance tracking, workflow modules, and CI/CD validation.

## Test Coverage
- Unit tests for all core modules (bids_loader, preprocessor, data_loader, fsl_integration)
- Integration tests for provenance event logging in all workflows
- CI/CD jobs for test, lint, build, validation, and provenance
- All test outputs saved in logs/ for traceability

## Test Coverage Checklist
- [x] Unit tests for core modules
- [x] Integration tests for pipelines
- [x] Property-based tests for robustness
- [x] API endpoint tests
- [x] Cloud upload/download tests
- [x] QC metrics and reporting tests
- [x] Advanced analytics tests
- [x] Feedback collection and integration tests
- [x] End-to-end workflow tests
- [x] Real-time analysis tests
- [x] Clinical validation tests

## Progress Log
- Provenance integration tests implemented and run in CI
- All major modules have corresponding unit and integration tests
- Test output and provenance logs uploaded as CI artifacts
- New modules for QC, analytics, cloud, API monitoring, and feedback created
- Lint errors resolved in all new source files
- End-to-end workflow, real-time analysis, and clinical validation tests implemented
- Next: Expand edge case and failure scenario tests, automate coverage reporting

## Future Improvements
- Expand test cases for edge scenarios and error handling
- Add property-based and fuzz testing for robustness
- Enhance test documentation and reporting

## Phase 23: Code Refactoring & Maintainability

### Test Checklist
- [ ] Add/expand unit tests for refactored core modules
- [ ] Add/expand integration tests for new features
- [ ] Update property-based and edge case tests
- [ ] Review and update test coverage reports
- [ ] Log all test changes in CHANGELOG_AUTOMATED.md

## Phase 24: Advanced Visualization & Interactive Tools

### Test Checklist
- [ ] Add tests for interactive visualization modules
- [ ] Validate region selection and atlas overlay features
- [ ] Expand test coverage for visualization APIs
- [ ] Log all test changes in CHANGELOG_AUTOMATED.md

## Phase 25: Automated Data Quality & Reporting

### Test Checklist
- [ ] Add tests for new QC metrics and reporting
- [ ] Validate integration of QC with preprocessing/analysis
- [ ] Test report generation and output formats
- [ ] Log all test changes in CHANGELOG_AUTOMATED.md

## Phase 26: Region Statistics & Output Logging

### Test Checklist
- [x] Property-based tests for region statistics (RegionStats)
- [x] Integration tests for region statistics visualization
- [x] Output logging tests for traceability in logs/
- [x] Expanded tests for new QC and visualization modules
- [x] CLI and API endpoint tests
- [x] Cloud deployment and scalability tests
- [x] Advanced analytics module tests
- [x] Cloud upload utility tests
- [ ] End-to-end workflow tests
- [ ] Automated API monitoring and logging tests
- [ ] Community feedback-driven test cases

## Phase 29: Interactive Visualization Testing
- [x] Property-based tests for interactive atlas and visualization modules
- [x] Integration tests for multi-planar and glass brain modules
- [x] Output logging for all visualization tests
- [ ] Expand edge case and failure scenario tests for visualization

## Progress Log
- Interactive visualization modules tested with property-based and integration tests
- Output logs saved in logs/ for traceability
- Next: Expand edge case and failure scenario tests for visualization modules
