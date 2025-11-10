"""Data loading helpers for resume and job description datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class ResumeRecord:
    """Structured representation of a single resume annotation."""

    resume_id: str
    candidate_name: str
    text: str
    job_id: str
    match_label: int


@dataclass
class JobRecord:
    """Structured representation of a single job description."""

    job_id: str
    title: str
    text: str


def _load_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_resumes(path: Path) -> List[ResumeRecord]:
    records = []
    for payload in _load_jsonl(path):
        summary = payload.get("summary", "").strip()
        skills = payload.get("skills", [])
        skills_text = ", ".join(skills)
        text_segments = [summary]
        if skills_text:
            text_segments.append(f"Key skills: {skills_text}")
        text = "\n".join(segment for segment in text_segments if segment)
        records.append(
            ResumeRecord(
                resume_id=payload["resume_id"],
                candidate_name=payload.get("candidate_name", ""),
                text=text,
                job_id=payload["job_id"],
                match_label=int(payload.get("match_label", 0)),
            )
        )
    return records


def load_jobs(path: Path) -> List[JobRecord]:
    records = []
    for payload in _load_jsonl(path):
        description = payload.get("description", "").strip()
        requirements = payload.get("requirements", [])
        requirements_text = "; ".join(requirements)
        text_segments = [description]
        if requirements_text:
            text_segments.append(f"Requirements: {requirements_text}")
        text = "\n".join(segment for segment in text_segments if segment)
        records.append(
            JobRecord(
                job_id=payload["job_id"],
                title=payload.get("title", ""),
                text=text,
            )
        )
    return records


def load_labeled_pairs(resume_path: str | Path, job_path: str | Path) -> pd.DataFrame:
    """Load resume and job records and return a merged labeled dataframe."""

    resume_records = load_resumes(Path(resume_path))
    job_records = {record.job_id: record for record in load_jobs(Path(job_path))}

    rows = []
    for resume in resume_records:
        job = job_records.get(resume.job_id)
        if not job:
            continue
        text = (
            f"[RESUME] {resume.candidate_name}\n{resume.text}\n\n"
            f"[JOB] {job.title}\n{job.text}"
        )
        rows.append(
            {
                "resume_id": resume.resume_id,
                "job_id": job.job_id,
                "candidate_name": resume.candidate_name,
                "job_title": job.title,
                "text": text,
                "label": resume.match_label,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No labeled pairs were produced from the provided files.")
    return df


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a stratified train/validation split."""

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
