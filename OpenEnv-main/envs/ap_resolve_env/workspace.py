"""
DocumentWorkspace — manages the document set for a single episode.

Tracks which documents have been opened and supports field extraction
from structured document content.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .models import CaseFixture


class DocumentWorkspace:
    """Manages documents available in the current episode."""

    def __init__(self, fixture: CaseFixture) -> None:
        self._documents: Dict[str, str] = dict(fixture.documents)
        self._opened: Dict[str, str] = {}          # doc_name -> content
        self._current_doc: Optional[str] = None    # last opened doc name
        self._extracted: Dict[str, Any] = {}       # field -> value

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open(self, doc_name: str) -> Optional[str]:
        """Open a document by name. Returns content or None if not found."""
        if doc_name not in self._documents:
            return None
        content = self._documents[doc_name]
        self._opened[doc_name] = content
        self._current_doc = doc_name
        return content

    def extract_field(self, field: str, doc_name: Optional[str] = None) -> Optional[Any]:
        """
        Extract a named field from the currently open (or specified) document.

        Returns the extracted value or None if not found / no doc open.
        Also caches the result in self._extracted.
        """
        target_doc = doc_name or self._current_doc
        if target_doc is None:
            return None
        content = self._opened.get(target_doc)
        if content is None:
            return None

        value = self._parse_field(field, content, target_doc)
        if value is not None:
            self._extracted[field] = value
        return value

    def list_available(self) -> List[str]:
        """Return all document names available in this episode."""
        return list(self._documents.keys())

    def already_opened(self, doc_name: str) -> bool:
        """Return True if the document has been opened at least once."""
        return doc_name in self._opened

    def get_extracted(self) -> Dict[str, Any]:
        """Return all extracted facts so far."""
        return dict(self._extracted)

    def get_current_content(self) -> Optional[str]:
        """Return the content of the currently open document."""
        if self._current_doc is None:
            return None
        return self._opened.get(self._current_doc)

    def get_opened_names(self) -> List[str]:
        """Return names of all documents opened so far."""
        return list(self._opened.keys())

    # ------------------------------------------------------------------
    # Field extraction helpers
    # ------------------------------------------------------------------

    def _parse_field(self, field: str, content: str, doc_name: str) -> Optional[Any]:
        """
        Extract a field value from document content using heuristic patterns.

        Supports common AP document fields by name.
        """
        field_lower = field.lower()

        # Map field names to extraction strategies
        extractors = {
            "invoice_amount": self._extract_total_amount,
            "po_amount": self._extract_total_amount,
            "total_due": self._extract_total_amount,
            "total_authorized": self._extract_total_amount,
            "vendor_id": self._extract_vendor_id,
            "invoice_number": self._extract_invoice_number,
            "po_number": self._extract_po_number,
            "grn_number": self._extract_grn_number,
            "received_qty": self._extract_received_qty,
            "billed_qty": self._extract_billed_qty,
            "ssd_qty_invoiced": self._extract_ssd_qty_invoiced,
            "ssd_qty_ordered": self._extract_ssd_qty_ordered,
            "service_completion_pct": self._extract_service_completion_pct,
            "duplicate_flag": self._extract_duplicate_flag,
            "early_payment_discount": self._extract_early_payment_discount,
            "vendor_status": self._extract_vendor_status,
            "payment_terms": self._extract_payment_terms,
            "invoice_date": self._extract_date,
        }

        extractor = extractors.get(field_lower)
        if extractor:
            return extractor(content)

        # Generic key: value pattern fallback
        return self._generic_extract(field, content)

    def _extract_total_amount(self, content: str) -> Optional[float]:
        """Extract the total/due amount from a document."""
        patterns = [
            r"Total Due:\s*\$?([\d,]+\.?\d*)",
            r"Total Authorized:\s*\$?([\d,]+\.?\d*)",
            r"Total:\s*\$?([\d,]+\.?\d*)",
            r"Early Payment Amount:\s*\$?([\d,]+\.?\d*)",
        ]
        for pattern in patterns:
            m = re.search(pattern, content, re.IGNORECASE)
            if m:
                return float(m.group(1).replace(",", ""))
        return None

    def _extract_vendor_id(self, content: str) -> Optional[str]:
        m = re.search(r"Vendor(?:\s+ID)?:\s*(VENDOR-\d+)", content, re.IGNORECASE)
        if m:
            return m.group(1)
        m = re.search(r"\((VENDOR-\d+)\)", content)
        if m:
            return m.group(1)
        return None

    def _extract_invoice_number(self, content: str) -> Optional[str]:
        m = re.search(r"Invoice Number:\s*(INV-[\w-]+)", content, re.IGNORECASE)
        return m.group(1) if m else None

    def _extract_po_number(self, content: str) -> Optional[str]:
        m = re.search(r"PO Number:\s*(PO-[\w-]+)", content, re.IGNORECASE)
        if not m:
            m = re.search(r"PO Reference:\s*(PO-[\w-]+)", content, re.IGNORECASE)
        return m.group(1) if m else None

    def _extract_grn_number(self, content: str) -> Optional[str]:
        m = re.search(r"GRN Number:\s*(GRN-[\w-]+)", content, re.IGNORECASE)
        return m.group(1) if m else None

    def _extract_received_qty(self, content: str) -> Optional[int]:
        """Extract total received quantity from GRN."""
        matches = re.findall(r"(\d+)\s+units?\s+received", content, re.IGNORECASE)
        if matches:
            return sum(int(q) for q in matches)
        return None

    def _extract_billed_qty(self, content: str) -> Optional[int]:
        """Extract total billed quantity from invoice line items."""
        matches = re.findall(r"x(\d+)\s+@", content, re.IGNORECASE)
        if matches:
            return sum(int(q) for q in matches)
        return None

    def _extract_ssd_qty_invoiced(self, content: str) -> Optional[int]:
        m = re.search(r"SSD\s+1TB\s+x(\d+)", content, re.IGNORECASE)
        return int(m.group(1)) if m else None

    def _extract_ssd_qty_ordered(self, content: str) -> Optional[int]:
        m = re.search(r"SSD\s+1TB\s+x(\d+)", content, re.IGNORECASE)
        return int(m.group(1)) if m else None

    def _extract_service_completion_pct(self, content: str) -> Optional[int]:
        m = re.search(r"(\d+)%\s+complete", content, re.IGNORECASE)
        return int(m.group(1)) if m else None

    def _extract_duplicate_flag(self, content: str) -> Optional[bool]:
        if "duplicate" in content.lower():
            return True
        return False

    def _extract_early_payment_discount(self, content: str) -> Optional[float]:
        m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:if paid|early payment discount)", content, re.IGNORECASE)
        if not m:
            m = re.search(r"(\d+)/\d+\s+Net", content)
        return float(m.group(1)) if m else None

    def _extract_vendor_status(self, content: str) -> Optional[str]:
        m = re.search(r"Status:\s*(\w+)", content, re.IGNORECASE)
        return m.group(1) if m else None

    def _extract_payment_terms(self, content: str) -> Optional[str]:
        m = re.search(r"Payment Terms:\s*(.+)", content, re.IGNORECASE)
        return m.group(1).strip() if m else None

    def _extract_date(self, content: str) -> Optional[str]:
        m = re.search(r"Date:\s*(\d{4}-\d{2}-\d{2})", content, re.IGNORECASE)
        return m.group(1) if m else None

    def _generic_extract(self, field: str, content: str) -> Optional[str]:
        """Generic key: value extraction."""
        pattern = re.compile(
            rf"{re.escape(field.replace('_', ' '))}[:\s]+([^\n]+)",
            re.IGNORECASE,
        )
        m = pattern.search(content)
        return m.group(1).strip() if m else None
