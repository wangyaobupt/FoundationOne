from setuptools import setup, find_packages
from pathlib import Path

# Read the README for long_description
this_directory = Path(__file__).parent
readme_path = this_directory / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "Foundation One (f1)"

# Read install requirements from requirements.txt
req_path = this_directory / "requirements.txt"
if req_path.exists():
    def _parse_requirements(lines: list[str]) -> list[str]:
        reqs: list[str] = []
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('-e ') or line.startswith('--editable'):
                continue
            reqs.append(line)
        return reqs
    install_requires = _parse_requirements(req_path.read_text(encoding="utf-8").splitlines())
else:
    install_requires = [
    ]

setup(
    name="foundation-one",
    version="0.1.0",
    description="Foundation One (f1): base infrastructure, shared services, and common LLM helpers",
    author="Yao WANG",
    author_email="wangyao_bupt@hotmail.com",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Topic :: Software Development :: Libraries",
    ],
    keywords=["llm", "openai", "volcano", "sdk", "agent", "chat"],
)
