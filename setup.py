"""
Setup configuration for Phase-Based Tactical Analytics Toolkit
SkillCorner × PySport Analytics Cup Submission
"""

from setuptools import setup, find_packages  # type: ignore

setup(
    name="gamestate-analytics",
    version="1.0.0",
    author="Hamza Adhnan Shakir",
    author_email="hamzashakir149@gmail.com",
    description="Context-aware tactical analytics toolkit for football tracking data",
    url="https://github.com/hamza-shakir/analytics_cup_analyst",
    
    # CRITICAL LINES
    package_dir={"": "src"},              # Look in src/ folder
    packages=find_packages(where="src"),  # Auto-find gamestate/
    python_requires=">=3.9",
    
    # DEPENDENCIES
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "kloppy>=3.18.0",
        "matplotlib>=3.7.0",
        "mplsoccer>=1.3.0",
        "scipy>=1.11.0",
        "requests>=2.31.0",
    ],
)