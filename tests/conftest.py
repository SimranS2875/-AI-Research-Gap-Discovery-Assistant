"""Shared test helpers: synthetic PDFs and a fake LLM (no network, no API key needed)."""
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pymupdf
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gap_assistant.config import Settings  # noqa: E402
from gap_assistant.llm import LLMClient  # noqa: E402
from gap_assistant.schemas import (LLMClaim, LLMEvidence, LLMExtraction, LLMGap, LLMGapList,  # noqa: E402
                                   LLMNoveltyItem, LLMNoveltyList)
from gap_assistant.embeddings import tfidf_embed  # noqa: E402

FILLER = (
    "Retinal blood vessel segmentation supports the diagnosis of diabetic retinopathy and glaucoma. "
    "Accurate maps of the vasculature help clinicians measure vessel width, tortuosity and branching. "
    "Deep convolutional networks have replaced hand-crafted filters in most recent studies. "
)

# name -> (title, body sentences). Every sentence quoted by the fake LLM appears verbatim here.
PAPERS: Dict[str, tuple] = {
    "paper_a.pdf": ("UNetVessel: A U-Net for Retinal Vessel Segmentation", [
        "We train a U-Net on the DRIVE and STARE datasets and report the Dice score and F1 score.",
        "Our U-Net improves Dice by two points over the previous baseline.",
        "We only evaluated on adult fundus images captured with a single camera type.",
        "Future work will study pediatric patients and other imaging devices.",
    ]),
    "paper_b.pdf": ("SwinVessel: Transformers for Vessel Segmentation", [
        "We compare Swin-UNet with a U-Net baseline on the DRIVE dataset using Dice and accuracy.",
        "Swin-UNet outperforms the U-Net baseline on thin vessels.",
        "Our model was tested only on adult fundus images from healthy and diabetic subjects.",
        "No statistical significance tests were performed in this study.",
    ]),
    "paper_c.pdf": ("SegFormer for Retinal Vasculature", [
        "We evaluate SegFormer and U-Net on the STARE and CHASE_DB1 datasets with the F1 score.",
        "SegFormer reaches the highest F1 score on CHASE_DB1.",
        "The training sets are small, which may limit generalization to other populations.",
        "In future work we plan to pretrain on larger unlabeled collections.",
    ]),
    "paper_d.pdf": ("Lightweight Vessel Segmentation on Edge Devices", [
        "We compare SegFormer and Swin-UNet on the CHASE_DB1 dataset using accuracy and Dice.",
        "Our lightweight variant runs in real time on a mobile processor.",
        "Only adult fundus images were used and no statistical significance tests were reported.",
        "Future work will investigate pediatric patients.",
    ]),
}

EXTRACTIONS: Dict[str, LLMExtraction] = {
    "paper_a.pdf": LLMExtraction(
        title="UNetVessel: A U-Net for Retinal Vessel Segmentation", authors=["A. Author"], year=2021,
        domain="retinal vessel segmentation", research_problem="Segment retinal vessels with a U-Net.",
        methods=["U-Net"], datasets=["DRIVE", "STARE"], metrics=["Dice", "F1"],
        key_results=[LLMClaim(statement="U-Net improves Dice by two points.",
                              quote="Our U-Net improves Dice by two points over the previous baseline.")],
        limitations=[LLMClaim(statement="Only adult images from one camera.",
                              quote="We only evaluated on adult fundus images captured with a single camera type.")],
        future_work=[LLMClaim(statement="Study pediatric patients.",
                              quote="Future work will study pediatric patients and other imaging devices.")],
        assumptions=["adult fundus images"]),
    "paper_b.pdf": LLMExtraction(
        title="SwinVessel: Transformers for Vessel Segmentation", authors=[], year=2022,
        domain="retinal vessel segmentation", research_problem="Use Swin transformers for vessels.",
        methods=["Swin-UNet", "U-Net"], datasets=["DRIVE"], metrics=["Dice", "accuracy"],
        key_results=[LLMClaim(statement="Swin-UNet is better on thin vessels.",
                              quote="Swin-UNet outperforms the U-Net baseline on thin vessels.")],
        limitations=[LLMClaim(statement="Tested only on adult images.",
                              quote="Our model was tested only on adult fundus images from healthy and diabetic subjects."),
                     LLMClaim(statement="No significance tests.",
                              quote="No statistical significance tests were performed in this study.")],
        future_work=[], assumptions=["adult fundus images"]),
    "paper_c.pdf": LLMExtraction(
        title="SegFormer for Retinal Vasculature", authors=[], year=2023,
        domain="retinal vessel segmentation", research_problem="Apply SegFormer to retinal vessels.",
        methods=["SegFormer", "U-Net"], datasets=["STARE", "CHASE_DB1"], metrics=["F1"],
        key_results=[LLMClaim(statement="SegFormer has best F1 on CHASE_DB1.",
                              quote="SegFormer reaches the highest F1 score on CHASE_DB1.")],
        limitations=[LLMClaim(statement="Small training sets limit generalization.",
                              quote="The training sets are small, which may limit generalization to other populations.")],
        future_work=[LLMClaim(statement="Pretrain on larger unlabeled data.",
                              quote="In future work we plan to pretrain on larger unlabeled collections.")],
        assumptions=[]),
    "paper_d.pdf": LLMExtraction(
        title="Lightweight Vessel Segmentation on Edge Devices", authors=[], year=2024,
        domain="retinal vessel segmentation", research_problem="Run vessel segmentation on mobile devices.",
        methods=["SegFormer", "Swin-UNet"], datasets=["CHASE_DB1"], metrics=["accuracy", "Dice"],
        key_results=[LLMClaim(statement="Real-time on mobile.",
                              quote="Our lightweight variant runs in real time on a mobile processor.")],
        limitations=[LLMClaim(statement="Adult images only; no significance tests.",
                              quote="Only adult fundus images were used and no statistical significance tests were reported.")],
        future_work=[LLMClaim(statement="Investigate pediatric patients.",
                              quote="Future work will investigate pediatric patients.")],
        assumptions=[]),
}


def make_pdf(path: Path, title: str, sentences: List[str]) -> Path:
    doc = pymupdf.open()
    body = "\n\n".join([title, FILLER * 2] + sentences + [FILLER * 2, "References", "[1] Some Reference."])
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(40, 40, 555, 800), body, fontsize=10)
    doc.set_metadata({"title": title})
    doc.save(str(path))
    doc.close()
    return path


class FakeLLM(LLMClient):
    """Deterministic stand-in for Gemini. Records how many calls were made."""

    def __init__(self):
        self.calls: Dict[str, int] = {"extract": 0, "gaps": 0, "novelty": 0, "text": 0}
        self.last_gap_prompt = ""

    def default_model(self) -> str:
        return "fake-flash"

    def generate_json(self, prompt, schema, *, model=None, system=None):
        if schema is LLMExtraction:
            self.calls["extract"] += 1
            for name, ex in EXTRACTIONS.items():
                if f'source="{name}"' in prompt:
                    return ex
            raise AssertionError("unknown paper in prompt")
        if schema is LLMGapList:
            self.calls["gaps"] += 1
            self.last_gap_prompt = prompt
            return LLMGapList(gaps=[
                LLMGap(title="Pediatric retinal images are unstudied", gap_type="scope",
                       description="All papers use adult images.", why_it_matters="Screening of children.",
                       evidence=[LLMEvidence(paper_id="P1", quote="We only evaluated on adult fundus images captured with a single camera type."),
                                 LLMEvidence(paper_id="p2", quote="Our model was tested only on adult fundus images from healthy and diabetic subjects."),
                                 LLMEvidence(paper_id="P9", quote="made up paper")],
                       suggested_direction="Build a pediatric benchmark.", search_keywords="pediatric retinal vessel segmentation",
                       novelty=5, feasibility=3),
                LLMGap(title="Missing significance testing", gap_type="methodological",
                       description="Papers lack statistical tests.", why_it_matters="Reliability.",
                       evidence=[LLMEvidence(paper_id="P2", quote="No statistical significance tests were performed in this study."),
                                 LLMEvidence(paper_id="P4", quote="This sentence was invented by the model and is not in the paper.")],
                       suggested_direction="Re-run comparisons with paired tests.", search_keywords="statistical significance vessel segmentation",
                       novelty=3, feasibility=5),
                LLMGap(title="Hallucinated gap", gap_type="coverage", description="x", why_it_matters="x",
                       evidence=[LLMEvidence(paper_id="P77", quote="")], suggested_direction="x",
                       search_keywords="x", novelty=9, feasibility=-3),
            ])
        if schema is LLMNoveltyList:
            self.calls["novelty"] += 1
            return LLMNoveltyList(items=[LLMNoveltyItem(gap_number=1, status="partly_addressed", reason="Closest: related paper."),
                                         LLMNoveltyItem(gap_number=2, status="bogus_status", reason="ignored")])
        raise AssertionError(f"unexpected schema {schema}")

    def generate_text(self, prompt, *, model=None, system=None):
        self.calls["text"] += 1
        return "Answer from the fake model."

    def embed(self, texts):
        return tfidf_embed(texts)


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def settings(tmp_path):
    return Settings(api_key="test", cache_dir=tmp_path / "cache", novelty_check=False)


@pytest.fixture
def pdf_dir(tmp_path):
    d = tmp_path / "pdfs"
    d.mkdir()
    for name, (title, sentences) in PAPERS.items():
        make_pdf(d / name, title, sentences)
    return d
