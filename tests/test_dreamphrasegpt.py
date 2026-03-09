from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from dreamphrasegpt import artifacts, interactive
from dreamphrasegpt import cli as app
from dreamphrasegpt import runtime as runtime_module
from dreamphrasegpt.config import (
    Dataset,
    GenerationConfig,
    ModelConfig,
    RuntimeSettings,
    TrainingConfig,
    TrainingResult,
    format_section_title,
)
from dreamphrasegpt.runtime import BatchProvider, build_model
from dreamphrasegpt.source_filter import build_bloom_source_filter

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_training_config(*, dataset_path: str | None = None, seed: int = 42) -> TrainingConfig:
    return TrainingConfig(
        dataset_path=dataset_path,
        seed=seed,
        steps=2,
        batch_size=2,
        model=ModelConfig.from_dimensions(
            vocab_size=1,
            block_size=4,
            n_layer=2,
            n_embd=96,
            n_head=3,
        ),
        learning_rate=3e-4,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=0.01,
        requested_device="cpu",
        requested_dtype="auto",
        amp_requested=None,
        compile_requested=False,
        print_every=1,
    )


def make_generation_config(*, samples: int = 0) -> GenerationConfig:
    return GenerationConfig(
        num_samples=samples,
        temperature=0.8,
        requested_block_size=4,
    )


def make_dummy_result(training_config: TrainingConfig) -> TrainingResult:
    model = build_model(training_config.model, None, torch.device("cpu"))
    return TrainingResult(
        model=model,
        elapsed=0.01,
        total_tokens=8,
        tok_s=100.0,
        steps_s=10.0,
        final_loss=0.1,
        completed_steps=training_config.steps,
        optimizer_state={},
        scaler_state=None,
        runtime=RuntimeSettings(
            requested_device=training_config.requested_device,
            resolved_device="cpu",
            requested_dtype=training_config.requested_dtype,
            amp_requested=training_config.amp_requested,
            amp_enabled=False,
            amp_dtype=None,
            compile_requested=training_config.compile_requested,
            compile_enabled=False,
        ),
    )


def create_artifact_fixture(root: Path) -> tuple[Path, Path, Path]:
    dataset = Dataset(
        data=torch.tensor([3, 0, 1, 3, 1, 2, 3], dtype=torch.long),
        id_to_char=["a", "b", "c"],
        bos_id=3,
        vocab_size=4,
    )
    model_config = ModelConfig.from_dimensions(
        vocab_size=dataset.vocab_size,
        block_size=4,
        n_layer=1,
        n_embd=6,
        n_head=3,
    )
    training_config = TrainingConfig(
        dataset_path="fixture.txt",
        seed=7,
        steps=1,
        batch_size=2,
        model=model_config,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=0.01,
        requested_device="cpu",
        requested_dtype="auto",
        amp_requested=None,
        compile_requested=False,
        print_every=1,
    )
    runtime = RuntimeSettings(
        requested_device="cpu",
        resolved_device="cpu",
        requested_dtype="auto",
        amp_requested=None,
        amp_enabled=False,
        amp_dtype=None,
        compile_requested=False,
        compile_enabled=False,
    )
    model = build_model(model_config, None, torch.device("cpu"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)
    checkpoint = artifacts.build_training_checkpoint(
        model,
        optimizer.state_dict(),
        None,
        dataset,
        training_config,
        runtime,
        completed_steps=1,
        total_tokens=10,
        final_loss=1.0,
    )

    model_path = root / "fixture.model.pt"
    resume_path = root / "fixture.resume.pt"
    combined_path = root / "fixture_combined.pt"

    artifacts.save_artifact_file(
        artifacts.build_model_artifact(checkpoint, source_path=resume_path),
        model_path,
    )
    resume_artifact = artifacts.build_resume_artifact(checkpoint, source_path=model_path)
    artifacts.save_artifact_file(resume_artifact, resume_path)
    artifacts.save_artifact_file(
        artifacts.merge_model_and_resume_artifacts(
            artifacts.build_model_artifact(checkpoint, source_path=resume_path),
            resume_artifact,
            model_path=model_path,
            resume_path=resume_path,
        ),
        combined_path,
    )
    return model_path, resume_path, combined_path


class ValidationParityTests(unittest.TestCase):
    def test_cli_rejects_invalid_embedding_head_combo(self) -> None:
        parser = app.build_arg_parser()
        args = parser.parse_args(["--n-embd", "100", "--n-head", "3"])
        with self.assertRaises(SystemExit):
            app.validate_args(args)

    def test_cli_rejects_removed_novelty_flags(self) -> None:
        parser = app.build_arg_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--label-smoothing", "0.2"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--loss-drop-prob", "0.2"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--reject-source-matches"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--max-retries-per-sample", "10"])

    def test_interactive_prompt_retries_invalid_embedding_head_combo(self) -> None:
        training = make_training_config(dataset_path="dataset.txt")
        generation = make_generation_config()
        responses = [
            "auto",
            "2",
            "y",
            "2",
            "4",
            "2",
            "100",
            "3",
            "0.001",
            "auto",
            "2",
            "y",
            "2",
            "4",
            "2",
            "96",
            "3",
            "0.001",
            "n",
            "0",
        ]

        with mock.patch("dreamphrasegpt.interactive.prompt_user", side_effect=responses):
            prompted_training, prompted_generation, prompted_save = (
                interactive.prompt_train_settings(
                    training,
                    generation,
                    "auto",
                )
            )

        self.assertEqual(prompted_training.model.n_embd, 96)
        self.assertEqual(prompted_training.model.n_head, 3)
        self.assertEqual(prompted_generation.num_samples, 0)
        self.assertIsNone(prompted_save)


class SectionTitleTests(unittest.TestCase):
    def test_product_header_uses_brand_capitalization(self) -> None:
        self.assertEqual(format_section_title("dreamphrasegpt"), "DreamPhraseGPT")

    def test_lowercase_headers_are_title_cased(self) -> None:
        self.assertEqual(format_section_title("benchmark settings"), "Benchmark Settings")


class FlowTests(unittest.TestCase):
    def test_run_training_flow_seeds_before_loading_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "seeded.txt"
            dataset_path.write_text(
                "\n".join(f"entry_{index}" for index in range(12)),
                encoding="utf-8",
            )
            training = make_training_config(dataset_path=str(dataset_path), seed=123)
            generation = make_generation_config()
            captured_streams: list[list[int]] = []

            def fake_train_once(
                training_config: TrainingConfig, dataset: Dataset, device: torch.device
            ):
                captured_streams.append(dataset.data.tolist())
                return make_dummy_result(training_config)

            with mock.patch("dreamphrasegpt.cli.train_once", side_effect=fake_train_once):
                app.run_training_flow(
                    training,
                    generation,
                    save_arg=None,
                    models_dir=Path(temp_dir),
                    should_generate=False,
                )
                app.run_training_flow(
                    training,
                    generation,
                    save_arg=None,
                    models_dir=Path(temp_dir),
                    should_generate=False,
                )

        self.assertEqual(captured_streams[0], captured_streams[1])


class RuntimeTests(unittest.TestCase):
    def test_batch_provider_includes_last_valid_window(self) -> None:
        torch.manual_seed(0)
        batcher = BatchProvider(
            torch.arange(6, dtype=torch.long),
            block_size=4,
            batch_size=256,
            device=torch.device("cpu"),
        )
        x, _ = batcher.get()
        windows = {tuple(row.tolist()) for row in x}
        self.assertIn((0, 1, 2, 3), windows)
        self.assertIn((1, 2, 3, 4), windows)

    def test_generate_samples_retries_source_matches(self) -> None:
        dataset = Dataset(
            data=torch.tensor([2, 0, 1, 2], dtype=torch.long),
            id_to_char=["a", "b"],
            bos_id=2,
            vocab_size=3,
        )
        generation = GenerationConfig(
            num_samples=1,
            temperature=1.0,
            requested_block_size=4,
        )
        source_filter = build_bloom_source_filter(["ab"])

        with mock.patch(
            "dreamphrasegpt.runtime.generate_sample_once",
            side_effect=["ab", "aba"],
        ):
            samples = runtime_module.generate_samples(
                mock.Mock(),
                dataset,
                torch.device("cpu"),
                generation,
                source_filter=source_filter,
            )

        self.assertEqual(samples, ["aba"])


class ArtifactTests(unittest.TestCase):
    def test_inference_policy_ignores_resume_companion_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path, _, _ = create_artifact_fixture(Path(temp_dir))
            inference_bundle = artifacts.load_artifact_bundle(
                model_path,
                artifacts.ArtifactRuntimePolicy.for_inference(torch.device("cpu")),
            )
            resume_bundle = artifacts.load_artifact_bundle(
                model_path,
                artifacts.ArtifactRuntimePolicy.for_resume(torch.device("cpu")),
            )

        self.assertEqual(inference_bundle.dataset.data.numel(), 0)
        self.assertEqual(inference_bundle.training_metadata, {})
        self.assertGreater(resume_bundle.dataset.data.numel(), 0)
        self.assertIn("optimizer_state", resume_bundle.raw_artifact)
        self.assertIsNotNone(inference_bundle.source_filter)
        self.assertTrue(inference_bundle.source_filter.matches("ab"))

    def test_resume_only_artifact_loads_companion_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _, resume_path, _ = create_artifact_fixture(Path(temp_dir))
            bundle = artifacts.load_artifact_bundle(
                resume_path,
                artifacts.ArtifactRuntimePolicy.for_resume(torch.device("cpu")),
            )

        self.assertIn("state_dict", bundle.raw_artifact)
        self.assertGreater(bundle.dataset.data.numel(), 0)
        self.assertIn("completed_steps", bundle.resume_state)
        self.assertIsNotNone(bundle.source_filter)

    def test_combined_artifact_loads_in_one_step(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _, _, combined_path = create_artifact_fixture(Path(temp_dir))
            bundle = artifacts.load_artifact_bundle(
                combined_path,
                artifacts.ArtifactRuntimePolicy.for_resume(torch.device("cpu")),
            )

        self.assertIn("optimizer_state", bundle.raw_artifact)
        self.assertIn("state_dict", bundle.raw_artifact)
        self.assertIsNotNone(bundle.source_filter)

    def test_corrupt_artifact_fails_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bad_path = Path(temp_dir) / "bad.model.pt"
            artifacts.save_artifact_file({"model_config": {}}, bad_path)
            with self.assertRaises(SystemExit):
                artifacts.load_artifact_bundle(
                    bad_path,
                    artifacts.ArtifactRuntimePolicy.for_inference(torch.device("cpu")),
                )

    def test_resolve_save_paths_and_related_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir) / "models"
            paths = artifacts.resolve_save_paths("myrun", models_dir)
            self.assertEqual(paths.directory, models_dir / "myrun")
            self.assertEqual(paths.model.name, "myrun.model.pt")

            model_path, resume_path, _ = create_artifact_fixture(models_dir / "fixture")
            js_bundle = model_path.with_suffix("").with_suffix("")
            js_bundle = js_bundle.parent / f"{js_bundle.name}.model"
            js_bundle.write_bytes(b"bundle")

            related = {path.name for path in artifacts.related_artifact_paths(model_path)}
            self.assertIn(model_path.name, related)
            self.assertIn(resume_path.name, related)
            self.assertIn(js_bundle.name, related)

    def test_format_artifact_display_name_collapses_numbered_standard_run_directory(self) -> None:
        models_dir = Path("models")
        model_path = models_dir / "english_words_2" / "english_words.model.pt"

        display_name = artifacts.format_artifact_display_name(model_path, models_dir)

        self.assertEqual(display_name, "english_words_2")


@unittest.skipUnless(shutil.which("node"), "Node.js is required for JS bundle compatibility test")
class JsBundleCompatibilityTests(unittest.TestCase):
    def test_exported_bundle_runs_with_node_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path, _, _ = create_artifact_fixture(Path(temp_dir))
            bundle_path = Path(temp_dir) / "fixture.model"
            artifacts.export_js_model_bundle(model_path, bundle_path)

            result = subprocess.run(
                ["node", "run_js_bundle.js", str(bundle_path), "--samples", "1"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stderr.strip(), "")


if __name__ == "__main__":
    unittest.main()
