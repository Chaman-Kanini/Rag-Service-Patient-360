import re
from typing import List, Tuple


class TextNormalizer:
    """
    Normalize extracted PDF text by fixing spacing issues while preserving original meaning.
    
    This preprocessor handles common PDF extraction artifacts like:
    - Missing spaces between words
    - Broken line breaks
    - Concatenated headers and values
    - Inconsistent spacing around punctuation, dates, phone numbers, and units
    """
    
    def __init__(self):
        # Common medical/clinical terms that often get concatenated
        self.common_terms = [
            'patient', 'diagnosis', 'treatment', 'medication', 'history', 'symptoms',
            'doctor', 'nurse', 'hospital', 'clinic', 'medical', 'health', 'care',
            'blood', 'pressure', 'heart', 'rate', 'temperature', 'weight', 'height',
            'allergies', 'procedure', 'surgery', 'prescription', 'dosage', 'mg', 'ml',
            'table', 'contents', 'page', 'date', 'time', 'name', 'address', 'phone',
            'states', 'quit', 'years', 'ago', 'received', 'work', 'home', 'cell'
        ]
        
        # Compile patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for text normalization."""
        
        # Pattern for camelCase or concatenated words (e.g., TableofContents)
        self.camel_case_pattern = re.compile(r'([a-z])([A-Z])')
        
        # Pattern for lowercase followed by uppercase (missing space)
        self.lower_upper_pattern = re.compile(r'([a-z])([A-Z])')
        
        # Pattern for digit followed by letter without space
        self.digit_letter_pattern = re.compile(r'(\d)([A-Za-z])')
        
        # Pattern for letter followed by digit without space (except units)
        self.letter_digit_pattern = re.compile(r'([A-Za-z])(\d)')
        
        # Pattern for concatenated common words
        self.word_concat_patterns = []
        for term in self.common_terms:
            # Match term followed by another word
            pattern = re.compile(rf'\b({term})([a-z]+)\b', re.IGNORECASE)
            self.word_concat_patterns.append((term, pattern))
    
    def normalize(self, text: str) -> str:
        """
        Normalize text by fixing spacing issues.
        
        Args:
            text: Raw extracted text from PDF
            
        Returns:
            Normalized text with corrected spacing
        """
        if not text or not text.strip():
            return text
        
        # Step 1: Fix dates and times
        text = self._fix_dates_and_times(text)
        
        # Step 2: Fix phone numbers
        text = self._fix_phone_numbers(text)
        
        # Step 3: Fix units and measurements
        text = self._fix_units_and_measurements(text)
        
        # Step 4: Fix concatenated words
        text = self._fix_concatenated_words(text)
        
        # Step 5: Fix punctuation spacing
        text = self._fix_punctuation_spacing(text)
        
        # Step 6: Fix line breaks and paragraphs
        text = self._fix_line_breaks(text)
        
        # Step 7: Clean up excessive whitespace
        text = self._clean_whitespace(text)
        
        return text
    
    def _fix_dates_and_times(self, text: str) -> str:
        """Fix spacing in dates and times."""
        
        # Fix date formats: 03/20/202408:24PM -> 03/20/2024 08:24PM
        text = re.sub(r'(\d{2}/\d{2}/\d{4})(\d{2}:\d{2})', r'\1 \2', text)
        
        # Fix time formats: 08:24PM -> 08:24 PM
        text = re.sub(r'(\d{1,2}:\d{2})([AP]M)', r'\1 \2', text)
        
        # Fix date with missing space: Received:03/20/2024 -> Received: 03/20/2024
        text = re.sub(r'([A-Za-z]):(\d)', r'\1: \2', text)
        
        # Fix date separators without spaces: 03/20/2024-05/15/2024 -> 03/20/2024 - 05/15/2024
        text = re.sub(r'(\d{2}/\d{2}/\d{4})-(\d{2}/\d{2}/\d{4})', r'\1 - \2', text)
        
        return text
    
    def _fix_phone_numbers(self, text: str) -> str:
        """Fix spacing around phone numbers."""
        
        # Fix phone with label: 228-206-7054(Work) -> 228-206-7054 (Work)
        text = re.sub(r'(\d{3}-\d{3}-\d{4})\(([A-Za-z]+)\)', r'\1 (\2)', text)
        
        # Fix phone with extension: 555-1234ext123 -> 555-1234 ext 123
        text = re.sub(r'(\d{3}-\d{4})ext(\d+)', r'\1 ext \2', text, flags=re.IGNORECASE)
        
        return text
    
    def _fix_units_and_measurements(self, text: str) -> str:
        """Fix spacing around units and measurements."""
        
        # Common medical units
        units = ['mg', 'ml', 'mcg', 'g', 'kg', 'lb', 'oz', 'cm', 'mm', 'in', 'ft',
                'bpm', 'mmHg', 'F', 'C', 'IU', 'mEq', 'units']
        
        for unit in units:
            # Fix missing space before unit: 500mg -> 500 mg
            text = re.sub(rf'(\d)({unit})\b', rf'\1 \2', text, flags=re.IGNORECASE)
        
        # Fix ranges: 120-140mmHg -> 120-140 mmHg
        text = re.sub(r'(\d+-\d+)([a-zA-Z]+)', r'\1 \2', text)
        
        # Fix percentages: 98.6%O2 -> 98.6% O2
        text = re.sub(r'(\d+\.?\d*)%([A-Za-z])', r'\1% \2', text)
        
        return text
    
    def _fix_concatenated_words(self, text: str) -> str:
        """Fix concatenated words like 'TableofContents' or 'patientstatesquit'."""
        
        # Fix camelCase: TableOfContents -> Table Of Contents
        text = self.camel_case_pattern.sub(r'\1 \2', text)
        
        # Fix specific common concatenations using dictionary
        for term, pattern in self.word_concat_patterns:
            # This will match things like "patientstates" and split to "patient states"
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    original = ''.join(match)
                    fixed = ' '.join(match)
                    text = text.replace(original, fixed)
        
        # Fix lowercase followed by uppercase: patientStates -> patient States
        text = self.lower_upper_pattern.sub(r'\1 \2', text)
        
        # Fix digit followed by letter: 30yearsago -> 30 yearsago
        text = self.digit_letter_pattern.sub(r'\1 \2', text)
        
        # Fix letter followed by digit (except common units): patient30 -> patient 30
        # But preserve: 5mg, 10ml, etc.
        text = re.sub(r'([A-Za-z])(\d+)(?![a-z]{1,4}\b)', r'\1 \2', text)
        
        return text
    
    def _fix_punctuation_spacing(self, text: str) -> str:
        """Fix spacing around punctuation."""
        
        # Fix missing space after punctuation: word.Another -> word. Another
        text = re.sub(r'([.!?;,])([A-Z])', r'\1 \2', text)
        
        # Fix missing space after colon: Label:Value -> Label: Value
        text = re.sub(r':([A-Za-z0-9])', r': \1', text)
        
        # Fix multiple spaces around punctuation
        text = re.sub(r'\s+([.!?;,:])', r'\1', text)
        
        # Fix space before opening parenthesis: word (text) -> word (text)
        text = re.sub(r'([A-Za-z0-9])\(', r'\1 (', text)
        
        # Fix space after closing parenthesis: (text)word -> (text) word
        text = re.sub(r'\)([A-Za-z])', r') \1', text)
        
        return text
    
    def _fix_line_breaks(self, text: str) -> str:
        """Fix broken line breaks and preserve logical paragraph structure."""
        
        # Preserve intentional line breaks (double newlines)
        text = text.replace('\n\n', '<<PARAGRAPH>>')
        
        # Fix broken sentences (lowercase after newline suggests continuation)
        text = re.sub(r'\n([a-z])', r' \1', text)
        
        # Preserve line breaks before headers (all caps or title case)
        text = re.sub(r'\n([A-Z][A-Z\s]+:)', r'\n\n\1', text)
        
        # Preserve line breaks before numbered/bulleted lists
        text = re.sub(r'\n(\d+\.|\*|-)\s', r'\n\n\1 ', text)
        
        # Restore paragraph breaks
        text = text.replace('<<PARAGRAPH>>', '\n\n')
        
        return text
    
    def _clean_whitespace(self, text: str) -> str:
        """Clean up excessive whitespace."""
        
        # Replace multiple spaces with single space
        text = re.sub(r' {2,}', ' ', text)
        
        # Replace multiple newlines with max two
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove trailing/leading whitespace from lines
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Remove leading/trailing whitespace from entire text
        text = text.strip()
        
        return text


# Singleton instance for easy import
normalizer = TextNormalizer()


def normalize_text(text: str) -> str:
    """
    Convenience function to normalize text.
    
    Args:
        text: Raw extracted text from PDF
        
    Returns:
        Normalized text with corrected spacing
    """
    return normalizer.normalize(text)
