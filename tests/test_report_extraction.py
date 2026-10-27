import tempfile
import unittest
from pathlib import Path

import medical_report_analyzer as analyzer


class ReportExtractionTests(unittest.TestCase):
    def test_txt_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.txt"
            path.write_text("Hemoglobin 13.5 g/dL\nWBC 7000 /uL", encoding="utf-8")
            text = analyzer.extract_text_from_file(path)
            values = analyzer.extract_values(text)
            self.assertAlmostEqual(values["hemoglobin"], 13.5)
            self.assertAlmostEqual(values["wbc"], 7000)

    def test_empty_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.txt"
            path.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "empty"):
                analyzer.extract_text_from_file(path)

    def test_unsupported_extension_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.docx"
            path.write_text("test", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                analyzer.extract_text_from_file(path)

    def test_empty_parser_text_is_safe(self):
        self.assertEqual(analyzer.extract_values("   \n"), {})

    def test_ocr_failure_has_actionable_message(self):
        original = analyzer.pytesseract
        class BrokenTesseract:
            @staticmethod
            def get_tesseract_version():
                raise RuntimeError("missing executable")
        analyzer.pytesseract = BrokenTesseract
        try:
            with self.assertRaisesRegex(RuntimeError, "Tesseract OCR executable"):
                analyzer._ocr_available()
        finally:
            analyzer.pytesseract = original

    def test_text_pdf_prefers_embedded_text(self):
        if analyzer.fitz is None:
            self.skipTest("PyMuPDF is not installed")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.pdf"
            doc = analyzer.fitz.open()
            page = doc.new_page()
            page.insert_text((72, 72), "Hemoglobin 13.5 g/dL")
            doc.save(str(path))
            doc.close()
            text = analyzer.extract_text_from_file(path)
            self.assertIn("Hemoglobin", text)


if __name__ == "__main__":
    unittest.main()
