from abc import ABC, abstractmethod
from typing import Any, List, Dict


class VectorDatabaseManager(ABC):
    @abstractmethod
    def connect(self) -> None:
        """Establish a connection to the vector database instance."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close the connection to the vector database instance."""
        pass

    @abstractmethod
    def upload_data(self, collection: str, documents: List[Dict[str, Any]]) -> None:
        """Upload a list of documents to a specific collection in ChromaDB."""
        pass