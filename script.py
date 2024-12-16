import re
import sys


def generate_impls(input_file, output_file):
    with open(input_file, "r") as f:
        content = f.read()

    # Find the content within the specified mod
    mod_pattern = r"pub mod dnnl_format_tag_t \{\s*((?:[^{}]|\{[^{}]*\})*)\s*\}"
    mod_match = re.search(mod_pattern, content, re.DOTALL)

    if not mod_match:
        print(f"Error: Could not find mod dnnl_format_tag_t in the input file.")
        return

    mod_content = mod_match.group(1)

    # Use a regular expression to find the enum variants and their doc comments within the mod
    pattern = r"#\[doc = \"(.*?)\"\]\n\s*pub const (\w+):"
    matches = re.findall(pattern, mod_content)

    with open(output_file, "w") as f:
        for comment, variant in matches:
            tag_struct = variant.replace("dnnl_", "")
            # Escape any backslashes in the comment
            escaped_comment = comment.replace("\\", "\\\\")
            # Remove newlines and extra whitespace from the comment
            cleaned_comment = " ".join(escaped_comment.split())

            num_pattern = r".*?([0-9]+)D.*"
            ma = re.search(num_pattern, cleaned_comment)
            num = int(ma.group(1)) if ma else 6

            f.write(
                f'impl_format_tag!({tag_struct}, {variant}, {num}, "{cleaned_comment}");\n'
            )


if __name__ == "__main__":
    generate_impls(sys.argv[1], "output.rs")
