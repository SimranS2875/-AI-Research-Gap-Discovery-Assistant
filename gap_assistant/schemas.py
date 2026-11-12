"""Data models.

`LLM*` classes are the JSON shapes we ask Gemini to return. The other classes are the
records the rest of the app works with (they add things the code computes, such as
whether a quote was verified against the paper text).
"""
from typing import List, Optional

from pydantic import BaseModel, Field

GAP_TYPES = [
    "coverage",  # a method/dataset/setting combination nobody in the set tried
    "stated_limitation",  # limitation or future work repeated across papers
    "contradiction",  # papers disagree or results do not replicate
    "methodological",  # weak evaluation, missing baselines, small data, etc.
    "scope",  # restricted population/domain/language/scale; generalization unknown
]

NOVELTY_STATUSES = ["likely_open", "partly_addressed", "likely_addressed"]


# --------------------------------------------------------------------------- LLM shapes
class LLMClaim(BaseModel):
    statement: str = Field(description="One-sentence paraphrase of the claim.")
    quote: str = Field(
        description="A single sentence copied word-for-word from the paper that supports "
        "the statement. Empty string if no sentence supports it."
    )


class LLMExtraction(BaseModel):
    title: str = Field(description="Paper title.")
    authors: List[str] = Field(description="Author names; empty list if not stated.")
    year: Optional[int] = Field(description="Publication year, or null if not stated.")
    domain: str = Field(description="Short field / application area, e.g. 'medical image segmentation'.")
    research_problem: str = Field(description="The problem the paper addresses, 1-2 sentences.")
    methods: List[str] = Field(
        description="Names of the proposed method and the main compared methods/baselines."
    )
    datasets: List[str] = Field(description="Datasets / benchmarks / corpora used in experiments.")
    metrics: List[str] = Field(description="Evaluation metrics used.")
    key_results: List[LLMClaim] = Field(description="The main findings (max 5).")
    limitations: List[LLMClaim] = Field(
        description="Limitations the authors acknowledge or that are explicit in the text."
    )
    future_work: List[LLMClaim] = Field(description="Future directions the authors propose.")
    assumptions: List[str] = Field(
        description="Scope restrictions / assumptions: data type, population, language, setting, scale."
    )


class LLMEvidence(BaseModel):
    paper_id: str = Field(description="Paper ID such as P3.")
    quote: str = Field(
        description="Quote copied exactly from a 'quote' field in the digest. "
        "Empty string only for coverage gaps."
    )


class LLMGap(BaseModel):
    title: str = Field(description="Short title of the research gap (max 12 words).")
    gap_type: str = Field(description="One of: " + ", ".join(GAP_TYPES))
    description: str = Field(description="What is missing or unresolved, in 2-3 sentences.")
    why_it_matters: str = Field(description="Why filling this gap would matter, 1-2 sentences.")
    evidence: List[LLMEvidence] = Field(description="Supporting evidence from the digest.")
    suggested_direction: str = Field(description="A concrete research question or study to address it.")
    search_keywords: str = Field(
        description="4-8 plain keywords (no quotes/operators) to search the literature for this topic."
    )
    novelty: int = Field(description="1-5. 5 = none of the provided papers address it.")
    feasibility: int = Field(description="1-5. 5 = a student/small team could address it with common resources.")


class LLMGapList(BaseModel):
    gaps: List[LLMGap]


class LLMNoveltyItem(BaseModel):
    gap_number: int = Field(description="The gap number as given in the input.")
    status: str = Field(description="One of: " + ", ".join(NOVELTY_STATUSES))
    reason: str = Field(description="1-2 sentences, naming the closest paper if any.")


class LLMNoveltyList(BaseModel):
    items: List[LLMNoveltyItem]


# --------------------------------------------------------------------------- app records
class Claim(BaseModel):
    statement: str
    quote: str = ""
    verified: bool = False  # True if the quote was found in the paper text


class Paper(BaseModel):
    paper_id: str
    source: str  # file name or URL
    source_type: str = "full_text"  # "full_text" or "abstract"
    url: str = ""
    title: str
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    domain: str = ""
    research_problem: str = ""
    methods: List[str] = Field(default_factory=list)
    datasets: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)
    key_results: List[Claim] = Field(default_factory=list)
    limitations: List[Claim] = Field(default_factory=list)
    future_work: List[Claim] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)


class Evidence(BaseModel):
    paper_id: str
    paper_title: str = ""
    quote: str = ""
    verified: bool = False


class RelatedWork(BaseModel):
    title: str
    year: Optional[int] = None
    url: str = ""
    citations: Optional[int] = None
    snippet: str = ""


class NoveltyAssessment(BaseModel):
    status: str = "unchecked"  # unchecked | likely_open | partly_addressed | likely_addressed
    reason: str = ""
    query: str = ""
    related: List[RelatedWork] = Field(default_factory=list)


class Gap(BaseModel):
    rank: int = 0
    title: str
    gap_type: str
    description: str
    why_it_matters: str = ""
    suggested_direction: str = ""
    search_keywords: str = ""
    evidence: List[Evidence] = Field(default_factory=list)
    novelty: int = 3
    feasibility: int = 3
    support: int = 0  # number of distinct papers cited as evidence
    score: float = 0.0
    novelty_check: NoveltyAssessment = Field(default_factory=NoveltyAssessment)
