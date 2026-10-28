"""Execution engine for schema-3 empirical and composable-policy studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

from .adapters.registry import create_adapter
from .analysis_m3 import estimator_agreement, mask_composition, paired_policy_comparisons, ranking_quality
from .empirical import SingleBitMeasurement, measure_single_bits, select_all, stratified_candidates, uniform_candidates
from .estimators_m3 import EstimatorOutput, basic_estimator, compute_estimators
from .experiment_engine import TrialMeasurement, _arrays_digest, _evaluate_faulted
from .faults import FaultModel, FaultSchedule
from .identities import BitIdentity
from .manifests import digest_object, sha256_file
from .ordering import compose_ordering
from .parallel import ExecutionResources, LogicalTask, emit_worker_progress, run_process_tasks
from .protection import ProtectionMask, ProtectionPolicy
from .progress import ProgressTracker, forward_worker_event, study_operation_plan
from .pruning import checkpoint_pruning_contract
from .quantization import BitCodec, FixedPointFormat
from .ranking import Ranking
from .samples import SamplePartition, require_disjoint
from .statistics import summarize
from .study_config import StudyConfiguration, StudyReplications


@dataclass(frozen=True)
class StudyOutcome:
    configuration: StudyConfiguration
    scientific_payload: Mapping[str, Any]
    partitions: Mapping[str, SamplePartition]
    estimator_outputs: tuple[EstimatorOutput, ...]
    empirical_measurements: tuple[SingleBitMeasurement, ...]
    ranking_metrics: tuple[Mapping[str, Any], ...]
    estimator_agreement: tuple[Mapping[str, Any], ...]
    trials: tuple[TrialMeasurement, ...]
    protection_results: tuple[Mapping[str, Any], ...]
    paired_comparisons: Mapping[str, Any]
    composition: tuple[Mapping[str, Any], ...]
    rankings: Mapping[str, tuple[tuple[Any, ...], ...]]
    protection_masks: Mapping[str, tuple[tuple[Any, ...], ...]]
    fault_schedules: tuple[Mapping[str, Any], ...]
    candidate_selection: tuple[Mapping[str, Any], ...]
    replication_name: str = "replication_0"
    replication_aggregate: tuple[Mapping[str, Any], ...] = ()
    replications: tuple["StudyOutcome", ...] = ()
    execution_metrics: Mapping[str, Any] | None = None

    @property
    def scientific_payload_sha256(self) -> str:
        return digest_object(self.scientific_payload)


def _finite(value: Any) -> Any:
    """Make computed data strict-JSON compatible without altering finite values."""

    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_finite(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    return value


def _summaries(rows: Sequence[TrialMeasurement], masks: Mapping[str, ProtectionMask], confidence: float):
    result = []
    for name, mask in masks.items():
        selected = [row for row in rows if row.policy == name]
        accuracy = summarize([row.accuracy for row in selected], confidence=confidence, bounds=(0.0, 1.0)).to_dict()
        loss = summarize([row.categorical_crossentropy for row in selected], confidence=confidence).to_dict()
        result.append({
            "policy": name,
            "protected_bits": len(mask.protected),
            "coverage": mask.coverage,
            "mask_sha256": mask.digest(),
            "accuracy": accuracy,
            "categorical_crossentropy": loss,
            "mean_applied_events": float(np.mean([row.applied_events for row in selected])),
            "mean_suppressed_events": float(np.mean([row.suppressed_events for row in selected])),
            "minimum_accuracy": float(min(row.accuracy for row in selected)),
            "minimum_accuracy_trial": int(min(selected, key=lambda row: row.accuracy).trial),
        })
    return tuple(result)


def _execute_single_study(
    configuration: StudyConfiguration, *, adapter=None,
    replication_name: str = "replication_0", sample_offset: int = 0,
    tracker: ProgressTracker | None = None, replication_index: int = 1,
    replication_count: int = 1,
) -> StudyOutcome:
    """Execute a validated M3 study without writing any files."""

    codec = BitCodec(FixedPointFormat.parse(configuration.benchmark.quantization))
    context = {"replication": replication_index, "replications": replication_count}
    if tracker is not None:
        tracker.update("Model loading", **context)
    benchmark = adapter or create_adapter(configuration, codec)
    benchmark.load()
    if tracker is not None:
        tracker.update("Model loading", advance=1, **context)
        tracker.update("Sample loading", **context)
    ranking_x, ranking_y, ranking_partition = benchmark.sample_partition("ranking", configuration.samples.ranking_count, offset=sample_offset)
    empirical_x, empirical_y, empirical_partition = benchmark.sample_partition("empirical_validation", configuration.samples.empirical_count, offset=sample_offset)
    evaluation_x, evaluation_y, evaluation_partition = benchmark.sample_partition("evaluation", configuration.samples.evaluation_count, offset=sample_offset)
    partitions = {"ranking": ranking_partition, "empirical_validation": empirical_partition, "evaluation": evaluation_partition}
    if configuration.samples.require_disjoint:
        require_disjoint(tuple(partitions.values()))
    if tracker is not None:
        tracker.update("Sample loading", advance=1, **context)

    original = benchmark.values()
    identities = tuple(benchmark.bit_identities())
    if len(set(identities)) != len(identities):
        raise RuntimeError("eligible bit identities are not unique")
    outputs: list[EstimatorOutput] = []
    if tracker is not None:
        tracker.update("Estimator computation", **context)
    advanced_specifications = tuple(
        value for value in configuration.estimators if value.name not in {"msb_first", "random"}
    )
    advanced_outputs = iter(compute_estimators(
        advanced_specifications, benchmark, ranking_x, ranking_y, identities, codec,
    ))
    for specification in configuration.estimators:
        output = basic_estimator(specification.name, identities, seed=specification.seed) if specification.name in {"msb_first", "random"} else next(advanced_outputs)
        if output.implementation_version != specification.implementation_version:
            raise ValueError(
                f"estimator {specification.name} implementation version mismatch: "
                f"configuration requested {specification.implementation_version!r}, code provides {output.implementation_version!r}"
            )
        if output.name != specification.result_name:
            output = replace(output, name=specification.result_name, metadata={**output.metadata, "estimator_definition": specification.name})
        outputs.append(output)
    by_name = {output.name: output for output in outputs}
    agreement = estimator_agreement(outputs)
    if tracker is not None:
        tracker.update("Estimator computation", advance=1, **context)

    empirical: tuple[SingleBitMeasurement, ...] = ()
    candidate_rows: tuple[Mapping[str, Any], ...] = ()
    metrics: tuple[Mapping[str, Any], ...] = ()
    if configuration.study_type in {"single_bit", "estimator_validation"}:
        if tracker is not None:
            tracker.update("Candidate selection", **context)
        if configuration.candidates.strategy == "all":
            candidates = select_all(identities)
        elif configuration.candidates.strategy == "uniform":
            candidates = uniform_candidates(
                identities, configuration.candidates.count, seed=configuration.candidates.seed,
            )
        else:
            score_output = by_name.get(configuration.candidates.score_estimator or "")
            candidates = stratified_candidates(
                identities, configuration.candidates.count, seed=configuration.candidates.seed,
                scores=None if score_output is None else score_output.scores,
                quantile_count=configuration.candidates.score_quantiles,
            )
        candidate_rows = tuple({
            "parameter": value.identity.parameter, "flat_index": value.identity.flat_index,
            "bit_index": value.identity.bit_index, "inclusion_rule": value.inclusion_rule,
            "sampling_probability": value.sampling_probability, "score_quantile": value.score_quantile,
        } for value in candidates)
        if tracker is not None:
            tracker.adjust_total(len(candidates) - configuration.candidates.count)
            tracker.update("Candidate selection", advance=1, **context)

        def empirical_progress(index, total, measurement):
            if tracker is None:
                return
            stage_completed = index + 1
            tracker.update(
                "Empirical bit injection", advance=1,
                stage_completed=stage_completed, stage_total=total + 1, **context,
            )

        if tracker is not None:
            tracker.update(
                "Empirical bit injection", stage_completed=0,
                stage_total=len(candidates) + 1, **context,
            )
        empirical = measure_single_bits(
            benchmark, empirical_x, empirical_y, candidates, codec,
            evaluation_sample_sha256=empirical_partition.sample_sha256,
            progress=empirical_progress,
        )
        if tracker is not None:
            tracker.update("Ranking analysis", **context)
        metrics = ranking_quality(empirical, outputs)
        if tracker is not None:
            tracker.update("Ranking analysis", advance=1, **context)

    trial_rows: list[TrialMeasurement] = []
    masks: dict[str, ProtectionMask] = {}
    schedules: list[FaultSchedule] = []
    compositions: tuple[Mapping[str, Any], ...] = ()
    protection_results: tuple[Mapping[str, Any], ...] = ()
    comparisons: Mapping[str, Any] = {}
    if configuration.study_type == "protection":
        if tracker is not None:
            tracker.update("Protection-mask construction", **context)
        universe = frozenset(identities)
        masks["no_protection"] = ProtectionMask("control_none", frozenset(), len(identities), 0.0, {"control": True})
        masks["full_protection"] = ProtectionMask("control_full", universe, len(identities), float(len(identities)), {"control": True})
        for policy in configuration.policies:
            output = by_name[policy.estimator]
            ranking = Ranking(output.name, output.scores, output.metadata)
            ordering = compose_ordering(policy.name, ranking, identities, benchmark.parameter_specs(), policy.priority, seed=policy.seed)
            for coverage in configuration.protection.coverages:
                name = f"{policy.name}@{coverage:.6f}"
                mask = ProtectionPolicy.ordered(ordering, coverage)
                masks[name] = ProtectionMask(name, mask.protected, mask.universe_size, mask.budget, {**mask.metadata, "prespecified": policy.prespecified})
        probabilities = configuration.protection.probabilities or (configuration.fault.probability,)
        multi_ber = len(probabilities) > 1
        evaluation_masks = {
            (f"ber={probability:.8g}/{name}" if multi_ber else name): mask
            for probability in probabilities for name, mask in masks.items()
        }
        if tracker is not None:
            tracker.update("Protection-mask construction", advance=1, **context)
        state_before = _arrays_digest(benchmark.values())
        fault_stage_total = len(probabilities) * configuration.execution.trials * (1 + len(masks))
        fault_stage_completed = 0
        running: dict[str, list[tuple[float, float]]] = {}
        if tracker is not None:
            tracker.update("Fault evaluation", stage_completed=0, stage_total=fault_stage_total, **context)
        for probability_index, probability in enumerate(probabilities):
            fault_model = FaultModel(
                target=configuration.fault.target, location=configuration.fault.location,
                persistence=configuration.fault.persistence, distribution=configuration.fault.distribution,
                probability=probability,
            )
            for trial in range(configuration.execution.trials):
                schedule = FaultSchedule.generate(
                    fault_model, identities,
                    seed=configuration.execution.fault_seed + probability_index * configuration.execution.trials + trial,
                    inference_count=configuration.samples.evaluation_count,
                )
                schedules.append(schedule)
                fault_stage_completed += 1
                if tracker is not None:
                    tracker.update(
                        "Fault evaluation", advance=1,
                        stage_completed=fault_stage_completed, stage_total=fault_stage_total,
                        logical_ber=probability, policy="schedule",
                        trial=trial + 1, trials=configuration.execution.trials, **context,
                    )
                for name, mask in masks.items():
                    condition_name = f"ber={probability:.8g}/{name}" if multi_ber else name
                    accuracy, loss, accounting = _evaluate_faulted(benchmark, evaluation_x, evaluation_y, original, schedule, mask, codec)
                    trial_rows.append(TrialMeasurement(
                        trial, condition_name, schedule.digest(), accuracy, loss, accounting.schedule_hash,
                        accounting.proposed_events, accounting.applied_events,
                        accounting.suppressed_events, accounting.unique_applied_bits,
                    ))
                    running.setdefault(condition_name, []).append((accuracy, loss))
                    fault_stage_completed += 1
                    if tracker is not None:
                        values = running[condition_name]
                        tracker.update(
                            "Fault evaluation", advance=1,
                            stage_completed=fault_stage_completed, stage_total=fault_stage_total,
                            logical_ber=probability, coverage=mask.coverage, policy=name,
                            trial=trial + 1, trials=configuration.execution.trials,
                            mean_accuracy=float(np.mean([value[0] for value in values])),
                            mean_loss=float(np.mean([value[1] for value in values])),
                            **context,
                        )
        if _arrays_digest(benchmark.values()) != state_before:
            raise RuntimeError("model state was not restored after protection trials")
        if tracker is not None:
            tracker.update("Statistical aggregation", **context)
        protection_results = _summaries(trial_rows, evaluation_masks, configuration.execution.confidence)
        accuracy_comparisons, loss_comparisons = {}, {}
        for probability in probabilities:
            names = [f"ber={probability:.8g}/{name}" if multi_ber else name for name in masks]
            accuracy_vectors = {name: [row.accuracy for row in trial_rows if row.policy == name] for name in names}
            loss_vectors = {name: [row.categorical_crossentropy for row in trial_rows if row.policy == name] for name in names}
            accuracy_comparisons.update(paired_policy_comparisons(accuracy_vectors, confidence=configuration.execution.confidence, seed=configuration.execution.permutation_seed))
            loss_comparisons.update(paired_policy_comparisons(loss_vectors, confidence=configuration.execution.confidence, seed=configuration.execution.permutation_seed))
        comparisons = {
            "accuracy": accuracy_comparisons,
            "categorical_crossentropy": loss_comparisons,
            "multiplicity": "Holm adjustment within each BER and reported metric family",
        }
        compositions = mask_composition({name: tuple(mask.protected) for name, mask in masks.items()}, identities)
        if tracker is not None:
            tracker.update("Statistical aggregation", advance=1, **context)

    exact_rankings = {
        output.name: tuple((identity.parameter, identity.flat_index, identity.bit_index, float(output.scores[identity])) for identity in sorted(identities))
        for output in outputs
    }
    exact_masks = {name: tuple((identity.parameter, identity.flat_index, identity.bit_index) for identity in sorted(mask.protected)) for name, mask in masks.items()}
    exact_schedules = tuple(({
        "trial": trial, "seed": schedule.seed, "sha256": schedule.digest(),
        "events": tuple((event.inference_index, event.bit.parameter, event.bit.flat_index, event.bit.bit_index) for event in schedule.events),
    } if configuration.schema_version == "3.0.0" else {
        "schedule_index": trial, "seed": schedule.seed, "logical_ber": schedule.model.probability, "sha256": schedule.digest(),
        "events": tuple((event.inference_index, event.bit.parameter, event.bit.flat_index, event.bit.bit_index) for event in schedule.events),
    }) for trial, schedule in enumerate(schedules))
    if tracker is not None:
        tracker.update("Clean evaluation", **context)
    clean = benchmark.evaluate(evaluation_x, evaluation_y)
    if tracker is not None:
        tracker.update("Clean evaluation", advance=1, **context)
    quantized = benchmark.quantized_values()
    empirical_payload = [{key: value for key, value in measurement.to_dict().items() if key != "runtime_seconds"} for measurement in empirical]
    metric_payload = [{key: value for key, value in row.items() if key != "runtime_seconds"} for row in metrics]
    if tracker is not None:
        tracker.update("Scientific payload", **context)
    benchmark_payload = {
        "id": configuration.benchmark.id,
        "checkpoint_sha256": sha256_file(configuration.benchmark.checkpoint),
        "dataset_sha256": sha256_file(configuration.benchmark.dataset),
        "eligible_parameters": sum(value.size for value in benchmark.parameter_specs()),
        "eligible_bits": len(identities),
        "partitions": {name: value.to_dict() for name, value in partitions.items()},
    }
    pruning_contract = checkpoint_pruning_contract(benchmark)
    if pruning_contract is not None:
        benchmark_payload["structured_pruning"] = pruning_contract
    representation = getattr(benchmark, "representation_manifest", None)
    if callable(representation):
        benchmark_payload["representation"] = representation()
    scientific_payload = _finite({
        "schema": "netsurf-m3-study/v1",
        "replication": {"name": replication_name, "sample_offset": sample_offset},
        "scientific_protocol": configuration.scientific_protocol,
        "benchmark": benchmark_payload,
        "quantization": codec.format.to_dict(),
        "estimator_definitions": {output.name: {"implementation_version": output.implementation_version, **dict(output.metadata)} for output in outputs},
        "candidate_selection": list(candidate_rows),
        "empirical_measurements": empirical_payload,
        "ranking_metrics": metric_payload,
        "estimator_agreement": list(agreement),
        "clean_evaluation": asdict(clean),
        "fault_model": asdict(FaultModel(configuration.fault.target, configuration.fault.location, configuration.fault.persistence, configuration.fault.distribution, configuration.fault.probability)),
        "protection_results": list(protection_results),
        "paired_comparisons": comparisons,
        "mask_composition": list(compositions),
        "exact_artifact_sha256": {
            "rankings": digest_object(exact_rankings), "protection_masks": digest_object(exact_masks), "fault_schedules": digest_object(exact_schedules),
        },
        "checks": {
            "partitions_disjoint": configuration.samples.require_disjoint,
            "model_state_restored": _arrays_digest(benchmark.values()) == _arrays_digest(original),
            "matched_schedules": all(
                len({row.schedule_sha256 for row in trial_rows if row.trial == trial and (not (configuration.protection.probabilities and len(configuration.protection.probabilities) > 1) or row.policy.startswith(f"ber={probability:.8g}/"))}) == 1
                for probability in (configuration.protection.probabilities or (configuration.fault.probability,))
                for trial in range(configuration.execution.trials)
            ) if trial_rows else "not_applicable",
            "quantized_state_exact": all(np.array_equal(original[name], quantized[name]) for name in original),
        },
    })
    if configuration.schema_version == "3.0.0":
        # Preserve the established schema-3.0 scientific payload digest.
        scientific_payload.pop("estimator_agreement", None)
        scientific_payload.pop("replication", None)
    else:
        scientific_payload["fault_model"]["evaluated_probabilities"] = list(
            configuration.protection.probabilities or (configuration.fault.probability,)
        )
    if _arrays_digest(benchmark.values()) != _arrays_digest(original):
        raise RuntimeError("model state changed during study")
    if tracker is not None:
        tracker.update("Scientific payload", advance=1, **context)
    return StudyOutcome(
        configuration, scientific_payload, partitions, tuple(outputs), empirical, tuple(_finite(metrics)),
        tuple(_finite(agreement)), tuple(trial_rows), tuple(_finite(protection_results)), _finite(comparisons), tuple(_finite(compositions)),
        exact_rankings, exact_masks, exact_schedules, candidate_rows,
        replication_name,
    )


def _aggregate_replication_metrics(outcomes: Sequence[StudyOutcome]):
    rows = []
    metric_names = (
        "spearman_positive_damage", "spearman_absolute_change",
        "kendall_positive_damage", "top_5_recall", "top_10_recall",
        "top_20_recall", "zero_score_fraction", "tied_score_fraction",
    )
    keys = sorted({(row["estimator"], row["scope"]) for outcome in outcomes for row in outcome.ranking_metrics})
    for estimator, scope in keys:
        selected = [row for outcome in outcomes for row in outcome.ranking_metrics if row["estimator"] == estimator and row["scope"] == scope]
        for metric in metric_names:
            values = np.asarray([row[metric] for row in selected if row.get(metric) is not None], dtype=np.float64)
            if not len(values):
                continue
            rows.append({
                "estimator": estimator, "scope": scope, "metric": metric,
                "replication_count": len(values), "mean": float(np.mean(values)),
                "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "minimum": float(np.min(values)), "maximum": float(np.max(values)),
                "statistical_unit": "independent configured replication",
            })
    runtime_keys = sorted({output.name for outcome in outcomes for output in outcome.estimator_outputs})
    for estimator in runtime_keys:
        values = np.asarray([output.runtime_seconds for outcome in outcomes for output in outcome.estimator_outputs if output.name == estimator], dtype=np.float64)
        rows.append({
            "estimator": estimator, "scope": "execution", "metric": "runtime_seconds",
            "replication_count": len(values), "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "minimum": float(np.min(values)), "maximum": float(np.max(values)),
            "statistical_unit": "independent configured replication",
        })
    return tuple(rows)


def execute_study(
    configuration: StudyConfiguration, *, adapter=None, progress=None,
    _tracker: ProgressTracker | None = None,
    resources: ExecutionResources | None = None,
) -> StudyOutcome:
    """Execute one or more explicit replications through the same study engine."""

    plan = configuration.replications
    own_tracker = _tracker is None
    tracker = _tracker or ProgressTracker(
        configuration.name, study_operation_plan(configuration), progress
    )
    resources = resources or ExecutionResources()
    if plan.count == 1 or resources.workers == 1:
        if plan.count == 1:
            outcome = _execute_single_study(
                configuration, adapter=adapter, tracker=tracker, replication_index=1,
                replication_count=1,
            )
            if own_tracker:
                tracker.finish()
            return outcome
    if adapter is not None:
        raise ValueError("injecting one adapter instance is unsupported for replicated studies")

    def child_configuration(index: int) -> StudyConfiguration:
        estimators = tuple(replace(value, seed=value.seed + index * plan.hutchinson_seed_stride) for value in configuration.estimators)
        return replace(
            configuration,
            candidates=replace(configuration.candidates, seed=configuration.candidates.seed + index * plan.candidate_seed_stride),
            estimators=estimators,
            execution=replace(configuration.execution, fault_seed=configuration.execution.fault_seed + index * plan.fault_seed_stride),
            replications=StudyReplications(1, 0, 0, 0, 0),
        )

    execution_metrics = None
    if resources.workers == 1:
        outcomes = []
        for index in range(plan.count):
            outcomes.append(_execute_single_study(
                child_configuration(index), replication_name=f"replication_{index}",
                sample_offset=index * plan.sample_offset_stride,
                tracker=tracker, replication_index=index + 1,
                replication_count=plan.count,
            ))
    else:
        tasks = tuple(LogicalTask(
            f"replication:{index:06d}",
            (child_configuration(index), index, plan.count, index * plan.sample_offset_stride),
        ) for index in range(plan.count))

        def completed(_completion):
            # Worker events provide bounded stage detail only. Advance each
            # declared work class once its authoritative result is available;
            # scientific results are still merged by logical task identity.
            child_plan = study_operation_plan(child_configuration(0))
            for stage, count in child_plan.items():
                if count:
                    tracker.update(stage, advance=count)

        def pool_changed(active, total):
            tracker.update("Parallel replication execution", active_workers=active, total_workers=total)

        parallel = run_process_tasks(
            tasks, _study_replication_worker, resources,
            on_completed=completed, on_pool_changed=pool_changed,
            on_progress=lambda _task_id, event: forward_worker_event(tracker, event),
        )
        outcomes = list(parallel.values)
        execution_metrics = {
            "worker_elapsed_seconds": parallel.elapsed_seconds,
            "peak_aggregate_worker_rss_mb": parallel.peak_aggregate_worker_rss_mb,
            "peak_worker_rss_mb": parallel.peak_worker_rss_mb,
            "logical_task_count": len(tasks),
        }

    tracker.update("Replication aggregation")
    aggregate = _aggregate_replication_metrics(outcomes)
    tracker.update("Replication aggregation", advance=1)
    primary = outcomes[0]
    payload = {
        "schema": "netsurf-m3-replicated-study/v1",
        "scientific_protocol": configuration.scientific_protocol,
        "replication_plan": asdict(plan),
        "replications": [outcome.scientific_payload for outcome in outcomes],
        # Wall-clock runtime is reported, but is intentionally excluded from
        # the reproducibility-critical scientific digest.
        "replication_aggregate": [row for row in aggregate if row["metric"] != "runtime_seconds"],
    }
    result = replace(
        primary, configuration=configuration, scientific_payload=_finite(payload),
        replication_aggregate=tuple(_finite(aggregate)), replications=tuple(outcomes),
        execution_metrics=execution_metrics,
    )
    if own_tracker:
        tracker.finish()
    return result


def _study_replication_worker(payload) -> StudyOutcome:
    """Spawn entry point: one independently seeded schema-3 replication."""

    configuration, index, count, sample_offset = payload
    tracker = ProgressTracker(configuration.name, study_operation_plan(configuration), emit_worker_progress)
    return _execute_single_study(
        configuration,
        replication_name=f"replication_{index}",
        sample_offset=sample_offset,
        tracker=tracker,
        replication_index=index + 1,
        replication_count=count,
    )
