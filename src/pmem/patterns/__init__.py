"""pattern-mining layer pattern-mining helpers.

Pattern modules are local-first analysis layers. They read canonical project
memory, emit evidence-backed audit candidates, and avoid causal claims.
"""

from pmem.patterns.anomaly import (
    ANOMALY_DETECTION_METHOD,
    ANOMALY_DETECTION_SCHEMA_VERSION,
    AnomalyRunOutcome,
    anomaly_detection_from_outcomes,
    anomaly_detection_payload,
)
from pmem.patterns.config_failure import (
    CONFIG_FAILURE_CORRELATION_METHOD,
    CONFIG_FAILURE_CORRELATION_SCHEMA_VERSION,
    DEFAULT_MIN_FEATURE_GROUP_RUNS,
    DEFAULT_MIN_TOTAL_RUNS,
    RunFailureOutcome,
    config_failure_correlation_from_outcomes,
    config_failure_correlation_payload,
)
from pmem.patterns.dataset_failure import (
    DATASET_FAILURE_CORRELATION_METHOD,
    DATASET_FAILURE_CORRELATION_SCHEMA_VERSION,
    DatasetIdentity,
    DatasetRunOutcome,
    dataset_failure_correlation_from_outcomes,
    dataset_failure_correlation_payload,
)
from pmem.patterns.recurring_failures import (
    RECURRING_FAILURE_METHOD,
    RECURRING_FAILURE_SCHEMA_VERSION,
    recurring_failure_report_from_inputs,
    recurring_failure_report_payload,
)
from pmem.patterns.temporal import (
    DEFAULT_MIN_DECISION_SIDE_RUNS,
    TEMPORAL_ANALYSIS_METHOD,
    TEMPORAL_ANALYSIS_SCHEMA_VERSION,
    DecisionEvent,
    TemporalRunOutcome,
    temporal_analysis_from_outcomes,
    temporal_analysis_payload,
)

__all__ = [
    "ANOMALY_DETECTION_METHOD",
    "ANOMALY_DETECTION_SCHEMA_VERSION",
    "CONFIG_FAILURE_CORRELATION_METHOD",
    "CONFIG_FAILURE_CORRELATION_SCHEMA_VERSION",
    "DATASET_FAILURE_CORRELATION_METHOD",
    "DATASET_FAILURE_CORRELATION_SCHEMA_VERSION",
    "DEFAULT_MIN_DECISION_SIDE_RUNS",
    "DEFAULT_MIN_FEATURE_GROUP_RUNS",
    "DEFAULT_MIN_TOTAL_RUNS",
    "RECURRING_FAILURE_METHOD",
    "RECURRING_FAILURE_SCHEMA_VERSION",
    "TEMPORAL_ANALYSIS_METHOD",
    "TEMPORAL_ANALYSIS_SCHEMA_VERSION",
    "AnomalyRunOutcome",
    "DatasetIdentity",
    "DatasetRunOutcome",
    "DecisionEvent",
    "RunFailureOutcome",
    "TemporalRunOutcome",
    "anomaly_detection_from_outcomes",
    "anomaly_detection_payload",
    "config_failure_correlation_from_outcomes",
    "config_failure_correlation_payload",
    "dataset_failure_correlation_from_outcomes",
    "dataset_failure_correlation_payload",
    "recurring_failure_report_from_inputs",
    "recurring_failure_report_payload",
    "temporal_analysis_from_outcomes",
    "temporal_analysis_payload",
]
