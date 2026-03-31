"""
Test cases for text normalization.
Run with: python -m pytest app/utils/test_normalizer.py
Or manually: python app/utils/test_normalizer.py
"""

from text_normalizer import normalize_text


def test_table_of_contents():
    """Test: TableofContents -> Table of Contents"""
    input_text = "TableofContents"
    expected = "Table of Contents"
    result = normalize_text(input_text)
    assert result == expected, f"Expected '{expected}', got '{result}'"
    print(f"✓ TableofContents -> {result}")


def test_patient_states_quit():
    """Test: patientstatesquit30yearsago -> patient states quit 30 years ago"""
    input_text = "patientstatesquit30yearsago"
    result = normalize_text(input_text)
    # Should at least separate the number
    assert "30 years" in result or "30 y" in result, f"Got: {result}"
    print(f"✓ patientstatesquit30yearsago -> {result}")


def test_date_time():
    """Test: Received:03/20/202408:24PM -> Received: 03/20/2024 08:24 PM"""
    input_text = "Received:03/20/202408:24PM"
    expected = "Received: 03/20/2024 08:24 PM"
    result = normalize_text(input_text)
    assert result == expected, f"Expected '{expected}', got '{result}'"
    print(f"✓ Date/Time: {result}")


def test_phone_number():
    """Test: 228-206-7054(Work) -> 228-206-7054 (Work)"""
    input_text = "228-206-7054(Work)"
    expected = "228-206-7054 (Work)"
    result = normalize_text(input_text)
    assert result == expected, f"Expected '{expected}', got '{result}'"
    print(f"✓ Phone: {result}")


def test_medication_dosage():
    """Test: 500mg -> 500 mg"""
    input_text = "Patient takes 500mg daily"
    result = normalize_text(input_text)
    assert "500 mg" in result, f"Expected '500 mg' in result, got '{result}'"
    print(f"✓ Dosage: {result}")


def test_blood_pressure():
    """Test: 120/80mmHg -> 120/80 mmHg"""
    input_text = "Blood pressure: 120/80mmHg"
    result = normalize_text(input_text)
    assert "120/80 mmHg" in result or "120/ 80 mmHg" in result, f"Got: {result}"
    print(f"✓ Blood Pressure: {result}")


def test_camel_case():
    """Test: PatientHistory -> Patient History"""
    input_text = "PatientHistory"
    result = normalize_text(input_text)
    assert "Patient History" in result, f"Got: {result}"
    print(f"✓ CamelCase: {result}")


def test_punctuation_spacing():
    """Test: word.Another -> word. Another"""
    input_text = "First sentence.Second sentence"
    result = normalize_text(input_text)
    assert ". S" in result, f"Got: {result}"
    print(f"✓ Punctuation: {result}")


def test_complex_medical_text():
    """Test complex medical text with multiple issues"""
    input_text = """PatientName:JohnDoe
DOB:01/15/198508:30AM
Phone:555-123-4567(Cell)
Diagnosis:Type2Diabetes
Medication:Metformin500mg2xdaily
BloodPressure:130/85mmHg
Notes:Patientstatesquit smoking10yearsago.FollowUp:03/20/2024"""
    
    result = normalize_text(input_text)
    
    # Check various fixes
    checks = [
        ("Patient Name" in result or "Patient name" in result, "Patient Name separation"),
        ("John Doe" in result, "Name separation"),
        ("08:30 AM" in result, "Time formatting"),
        ("555-123-4567 (Cell)" in result, "Phone formatting"),
        ("Type 2" in result or "Type2" in result, "Type2 handling"),
        ("500 mg" in result, "Medication dosage"),
        ("mmHg" in result, "Blood pressure unit"),
        ("10 years" in result, "Years separation"),
    ]
    
    print(f"\n✓ Complex Medical Text:")
    print(f"  Input length: {len(input_text)} chars")
    print(f"  Output length: {len(result)} chars")
    print(f"  Sample output:\n{result[:200]}...")
    
    for check, description in checks:
        status = "✓" if check else "✗"
        print(f"  {status} {description}")


def run_all_tests():
    """Run all test cases"""
    print("=" * 60)
    print("TEXT NORMALIZATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_table_of_contents,
        test_patient_states_quit,
        test_date_time,
        test_phone_number,
        test_medication_dosage,
        test_blood_pressure,
        test_camel_case,
        test_punctuation_spacing,
        test_complex_medical_text,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
