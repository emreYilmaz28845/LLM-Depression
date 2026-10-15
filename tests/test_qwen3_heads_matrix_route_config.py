from tools.qwen3_heads_matrix import HEAD_SEED, HEAD_VARIANTS, SCHEMA_VERSION, check_matrix

CONFIG = "configs/experiments/androids_mergedfolds/androids_audio_only_harmonized_selmacrof1_likelihood_v1_mergedfolds.yaml"


def _matrix(route_configs, routes):
    return {
        "schema_version": SCHEMA_VERSION,
        "head_variants": list(HEAD_VARIANTS),
        "head_seed": HEAD_SEED,
        "route_configs": route_configs,
        "routes": routes,
    }


def test_route_config_subset_passes_the_route_count_check():
    route = {"route_id": "androids_interview_audio_only_native", "config": CONFIG, "language": "native", "jobs": []}
    failures = check_matrix(_matrix({route["route_id"]: CONFIG}, [route]))
    assert not any("23 standalone routes" in failure for failure in failures)
    assert not any("--route-config" in failure for failure in failures)


def test_route_config_mismatch_is_reported():
    route = {"route_id": "androids_interview_audio_only_native", "config": "configs/main/other.yaml", "language": "native", "jobs": []}
    failures = check_matrix(_matrix({route["route_id"]: CONFIG}, [route]))
    assert any("differs from its --route-config" in failure for failure in failures)


def test_without_route_config_the_full_route_set_is_required():
    failures = check_matrix(_matrix(None, []))
    assert any("23 standalone routes" in failure for failure in failures)
