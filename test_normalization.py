import re

def normalize_name(name):
    if not isinstance(name, str):
        return ""

    name = name.lower()
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    return name


examples = [
    "ABC Technologies Pvt. Ltd.",
    "ABC TECHNOLOGIES PVT LTD",
    "ABC-Technologies Pvt Ltd"
]

for name in examples:
    print(name, " ---> ", normalize_name(name))