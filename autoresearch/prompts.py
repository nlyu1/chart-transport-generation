from __future__ import annotations

from pathlib import Path


ROLE_PROMPT_FILES = {
    "metastudy_agent": "metastudy_agent.md",
    "metastudy_reviewer": "metastudy_reviewer.md",
    "study_agent": "study_agent.md",
    "study_reviewer": "study_reviewer.md",
    "substudy_agent": "substudy_agent.md",
    "substudy_reviewer": "substudy_reviewer.md",
    "substudy_synthesizer": "substudy_synthesizer.md",
    "reporter": "reporter.md",
}


class PromptLibrary:
    def __init__(
        self,
        *,
        prompt_root: Path,
    ) -> None:
        self.prompt_root = prompt_root

    @classmethod
    def build_default(cls) -> "PromptLibrary":
        return cls(prompt_root=Path(__file__).with_name("prompts"))

    def load(self, *, role: str) -> str:
        if role not in ROLE_PROMPT_FILES:
            raise KeyError(f"Unknown role prompt: {role}")
        prompt_path = self.prompt_root / ROLE_PROMPT_FILES[role]
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        return prompt_path.read_text(encoding="utf-8")


__all__ = ["PromptLibrary", "ROLE_PROMPT_FILES"]
