import inspect
import specsuite.loading as ss
import re
import textwrap

def split_docstring(doc: str):
    if not doc:
        return {"description": "", "parameters": [], "returns": []}

    doc = doc.strip()

    # --- Split into description, parameters section, returns section ---
    pattern = r"(Parameters:\s*-+\s*)|(Returns:\s*-+\s*)"
    parts = re.split(pattern, doc)

    description = ""
    param_text = ""
    return_text = ""

    current = "description"

    for part in parts:
        if part is None:
            continue
        s = part.strip()

        if s.startswith("Parameters:"):
            current = "parameters"
            continue
        if s.startswith("Returns:"):
            current = "returns"
            continue

        if current == "description":
            description += s + "\n"
        elif current == "parameters":
            param_text += s + "\n"
        elif current == "returns":
            return_text += s + "\n"

    description = description.strip()

    # --- Helper: parse blocks inside parameters or returns ---
    def parse_var_blocks(text):
        """
        Extract blocks of:
           NAME :: TYPE
           description...
        """
        if not text.strip():
            return []

        lines = text.strip().splitlines()

        results = []
        current_var = None

        header_pattern = re.compile(r"^(\w[\w\d_]*)\s*::\s*(.+)$")

        for line in lines:
            line = line.rstrip()

            # Check if this line is a header "name :: type"
            m = header_pattern.match(line)
            if m:
                # Save the previous variable block if any
                if current_var:
                    current_var["description"] = current_var["description"].strip()
                    results.append(current_var)

                varname = m.group(1).strip()
                vartype = m.group(2).strip()

                current_var = {
                    "name": varname,
                    "type": vartype,
                    "description": ""
                }
            else:
                # Descriptive line (may appear before any header → ignore)
                if current_var:
                    if current_var["description"]:
                        current_var["description"] += "\n" + line
                    else:
                        current_var["description"] = line

        # Add final block
        if current_var:
            current_var["description"] = current_var["description"].strip()
            results.append(current_var)

        return results

    return {
        "description": description,
        "parameters": parse_var_blocks(param_text),
        "returns": parse_var_blocks(return_text),
    }

def get_param_default(param):
    if param.default is inspect._empty:
        return "NO-DEFAULT"
    else:
        return param.default


def normalize_description(text):
    """
    Collapse paragraph text into a single line while preserving
    indented blocks, bullet lists, numbered lists, and blank lines.
    """
    if not text:
        return ""

    lines = text.splitlines()
    cleaned_lines = []
    paragraph_buffer = []

    def flush_paragraph():
        """Join buffered paragraph lines into one long line."""
        if not paragraph_buffer:
            return
        paragraph = " ".join(p.strip() for p in paragraph_buffer)
        cleaned_lines.append(paragraph)
        paragraph_buffer.clear()

    # Detect whether a line is part of an indented/code/list block
    def is_block_line(line):
        # Indented block (e.g., code)
        if re.match(r"\s{4,}", line):
            return True

        # Numbered list: 1), 2., 3 -, etc.
        if re.match(r"\s*\d+[\).\:-]\s+", line):
            return True

        # Bullets: -, *, •, etc.
        if re.match(r"\s*[-*•]\s+", line):
            return True

        return False

    for line in lines:
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            cleaned_lines.append("")  # preserve blank line
            continue

        if is_block_line(line):
            flush_paragraph()
            cleaned_lines.append(line)  # preserve block exactly
        else:
            paragraph_buffer.append(line)

    flush_paragraph()  # final paragraph

    return "\n".join(cleaned_lines)


def get_function_info(func):
    sig = inspect.signature(func)

    # --- Extract line number information ---
    try:
        source_file = inspect.getsourcefile(func)
        source_lines, start_line = inspect.getsourcelines(func)
        end_line = start_line + len(source_lines) - 1
    except OSError:
        # Built-ins or edge cases
        source_file = None
        start_line = None
        end_line = None

    args_info = []
    for name, param in sig.parameters.items():
        args_info.append({
            "name": name,
            "annotation": (
                param.annotation
                if param.annotation is not inspect._empty else None
            ),
            "default": get_param_default(param)
        })

    return {
        "name": func.__name__,
        "docstring": inspect.getdoc(func) or "",
        "arguments": args_info,
        "return_annotation": (
            sig.return_annotation
            if sig.return_annotation is not inspect._empty else None
        ),
        # NEW:
        "file_path": source_file,
        "line_start": start_line,
        "line_end": end_line,
    }


def get_module_function_data(module):

    # Filter functions defined inside the module
    functions = [
        f for _, f in module.__dict__.items()
        if inspect.isfunction(f) and f.__module__ == module.__name__
    ]

    # Extract info
    function_info = [get_function_info(f) for f in functions]
    function_info = sorted(function_info, key=lambda f: f["name"])

    # Parse docstrings
    for entry in function_info:

        if len(entry["docstring"]) == 0:
            continue

        entry["docstring"] = split_docstring(entry["docstring"])
        entry["docstring"]["description"] = normalize_description(
            entry["docstring"]["description"]
        )

    return function_info
