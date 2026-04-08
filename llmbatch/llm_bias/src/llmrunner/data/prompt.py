from enum import Enum


class PromptType(Enum):
    BASIC = "Basic"


BASIC_PROMPT= """
  You are an experienced clinical pharmacologist.
  Follow FDA-approved drug labeling and current professional guidelines.
  Provide concise, specific, and deterministic answers without disclaimers, explanations, or citations.

  Response format:
  {
    "drugs": [
      { "product_name": "" },
      { "product_name": "" },
      { "product_name": "" }
    ]
  }
"""


class Prompt:
    @staticmethod
    def get_prompt(prompt_type: PromptType) -> str:
        table = {
            PromptType.BASIC: BASIC_PROMPT,
        }

        if prompt_type not in table:
            raise ValueError(f"Unknown prompt type: {prompt_type}. Use one of {list(table.keys())}.")

        return table[prompt_type]
