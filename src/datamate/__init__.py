"""DataMate data ingestion components."""

from .data_loader import BusinessDataLoader, DataLoadError, TableRef

__all__ = ["BusinessDataLoader", "DataLoadError", "TableRef"]
