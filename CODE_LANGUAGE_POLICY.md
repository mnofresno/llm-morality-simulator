# Code Language Policy

## Policy

**ALL code, documentation, and technical content in this repository MUST be written in English.**

## Scope

This policy applies to:

1. **Source Code**
   - Variable names
   - Function names
   - Class names
   - Comments
   - Docstrings
   - Type hints
   - Error messages
   - Log messages

2. **Tests**
   - Test function names
   - Test descriptions
   - Assert messages
   - Test output/print statements

3. **Documentation**
   - README files
   - API documentation
   - Inline documentation
   - Comments in code

4. **Configuration Files**
   - Configuration comments
   - Configuration descriptions
   - CI/CD workflow descriptions

5. **User Interface**
   - UI labels (unless part of experiment content)
   - Error messages
   - Help text
   - Tooltips

## Exception

The ONLY exception to this policy is:
- **Experiment Content**: Text content within scenario prompts, user prompts, and AI responses that are part of the experimental data may be in any language as required by the experiment design.

## Enforcement

- Pre-commit hooks should check for compliance
- Code reviews must verify English-only code
- CI/CD may include language checks
- Automated tools may flag non-English content

## Rationale

- Consistency across the codebase
- Easier collaboration with international developers
- Better tooling support
- Standard practice in open source projects
- Improved maintainability

## Examples

### ✅ Correct (English)
```python
def calculate_statistics(results: List[Dict]) -> Dict:
    """Calculate statistics from experiment results."""
    # Process results
    return stats
```

### ❌ Incorrect (Non-English)
```python
def calcular_estadisticas(resultados: List[Dict]) -> Dict:
    """Calcula estadísticas de los resultados."""
    # Procesar resultados
    return estadisticas
```

## Review Checklist

When reviewing code, verify:
- [ ] All variable names are in English
- [ ] All function/class names are in English
- [ ] All comments are in English
- [ ] All docstrings are in English
- [ ] All error messages are in English
- [ ] All test descriptions are in English
- [ ] All user-facing messages are in English

