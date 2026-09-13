from __future__ import annotations

import hashlib

from scripts.experiments.setup.icl_arch_matrix.evaluate_prodigy import (
    load_external_models,
    load_reference_fingerprints,
    verify_external_checkpoint,
)


def test_external_model_list_preserves_selected_step_seed_and_provenance(tmp_path):
    checkpoint = tmp_path / "state_dict_6000.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model_list = tmp_path / "models.tsv"
    model_list.write_text(
        "model_id\tcheckpoint\tsources\ttraining_seed\tcheckpoint_step\t"
        "training_revision\tcheckpoint_sha256\n"
        f"nmi_baseline_r8_s2\t{checkpoint}\ta,b\t2\t6000\tdeadbeef\t{digest}\n",
        encoding="utf-8",
    )

    model, = load_external_models(model_list)
    assert model.model_id == "nmi_baseline_r8_s2"
    assert model.sources == ("a", "b")
    assert model.training_seed == 2
    assert model.checkpoint_step == 6000
    assert model.training_revision == "deadbeef"
    verify_external_checkpoint(model)


def test_reference_fingerprints_accept_published_tsv(tmp_path):
    reference = tmp_path / "classification_long.tsv"
    reference.write_text(
        "dataset\tepisode_fingerprint\n"
        "covid_political\tabc\n"
        "covid_political\tabc\n"
        "twibot20\tdef\n",
        encoding="utf-8",
    )
    assert load_reference_fingerprints(reference) == {
        "covid_political": "abc",
        "twibot20": "def",
    }
