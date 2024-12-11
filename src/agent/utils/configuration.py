from typing_extensions import TypedDict
from typing import Optional


class ConfigSchema(TypedDict):
    model: Optional[str]
    system_message: Optional[str]

    k_retrieved_documents: Optional[int]